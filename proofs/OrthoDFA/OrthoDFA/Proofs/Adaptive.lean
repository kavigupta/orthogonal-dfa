import OrthoDFA.Proofs.Basics
import OrthoDFA.Proofs.Schedule
import OrthoDFA.Proofs.BinomTail
import OrthoDFA.Proofs.BinomTest
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

The union bound over `stoppable` is a finite sum: a rung carries `δ·npref/(8·N)`, and
`δ·V/(16·npref)` more once it is at or above `V = validCount`, and the ladder's counts halve,
so no summable weight over all budgets is needed and no state has to be encoded as a number.
Validity charges the screen at `validFlip`, not at the finer `flipBudget` the round's own
arithmetic asks for, so `V` is far below the top count and every rung above it is stoppable
(`rung_mem_stoppable`).  `vote_mem_grid` is what lets the state be a `State` at all — a
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
variable {uni : J} {nveto : ℕ}
variable {η₀ bnd : ℝ}
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
noncomputable def solvedState (η₀ bnd : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) : State :=
  solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)

lemma ladderLen_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
  rw [ladderLen]; omega

lemma solvedState_mem_schedule (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) :
    solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit
      ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit := by
  rw [schedule, solvedState]
  refine Finset.mem_image.2 ⟨0, Finset.mem_range.2 (ladderLen_pos _ _ _ _ _ _), ?_⟩
  norm_num

/-- Every rung draws the same pool: only the prefix count halves along the ladder. -/
lemma nsuff_of_mem_schedule {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit) :
    B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  rfl

/-- A budget that covers the top rung covers every rung, so every rung reads its whole table. -/
lemma certSize_of_mem_schedule {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    {a : ℕ}
    (ha : prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit
        * (famCount η₀ populations indecisionLimit εcov δ crossLimit + 1)
      ≤ a * poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
    (hB : B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit) :
    certSize a B = B.npref := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  have hle : prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i
      ≤ a * poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
        / (famCount η₀ populations indecisionLimit εcov δ crossLimit + 1) :=
    le_trans (Nat.div_le_self _ _) ((Nat.le_div_iff_mul_le (by omega)).2 ha)
  exact min_eq_left (le_trans hle (le_max_right _ _))

lemma prefCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
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
    ∑ B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) := by
  classical
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set L : ℕ := ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit with hLdef
  calc ∑ B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit, (B.npref : ℝ)
      ≤ ∑ i ∈ Finset.range L,
          ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit
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
    (s : Finset State)
    (hs : s ⊆ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit) :
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
    (lt_min (div_pos (sig_pos η₀ hsig) (by norm_num)) (by linarith))

lemma cutBudget_le (η₀ indecisionLimit εcov : ℝ) :
    cutBudget η₀ indecisionLimit εcov ≤ εcov / 8
    ∧ cutBudget η₀ indecisionLimit εcov ≤ sig η₀ / 10
    ∧ cutBudget η₀ indecisionLimit εcov ≤ indecisionLimit / 2 :=
  ⟨min_le_left _ _, (min_le_right _ _).trans (min_le_left _ _),
    (min_le_right _ _).trans (min_le_right _ _)⟩

lemma voteSig_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < voteSig η₀ := by
  rw [voteSig]
  have := sig_pos η₀ hsig
  positivity

lemma voteSlack_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < voteSlack η₀ := by
  rw [voteSlack]
  have := voteSig_pos η₀ hsig
  positivity

lemma flipFrac_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < flipFrac η₀ := by
  rw [flipFrac]
  have := voteSig_pos η₀ hsig
  positivity

lemma validFrac_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < validFrac η₀ := by
  rw [validFrac]
  have := voteSig_pos η₀ hsig
  positivity

lemma flipFrac_le_validFrac (η₀ : ℝ) (hsig : η₀ < 1 / 2) : flipFrac η₀ ≤ validFrac η₀ := by
  rw [flipFrac, validFrac]
  have := voteSig_pos η₀ hsig
  linarith

lemma famCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    0 < famCount η₀ populations indecisionLimit εcov δ crossLimit := by
  rw [famCount]; omega

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
      Nat.cast_nonneg (α := ℝ) ⌈4 / voteSlack η₀⌉₊]
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

/-- The seed's term in `famCount`: the family outnumbers `8/voteSlack`. -/
lemma famCount_seed (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    8 ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ := by
  have hs := voteSlack_pos η₀ hsig
  have hle : 4 / voteSlack η₀
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
  rw [decSlack, sub_nonneg, div_le_iff₀ hk]
  nlinarith

lemma decSlack_le (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    decSlack η₀ populations indecisionLimit εcov δ crossLimit ≤ voteSlack η₀ / 2 := by
  rw [decSlack]
  have : (0 : ℝ) ≤ 2 / ((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
    := by positivity
  linarith

/-- `famCount_tail` at the returned family's slack: the seed's term pays for the two reads. -/
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
  have hA : Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2 + 4 / v ≤ κ / 2 := by
    have h1 := Nat.le_ceil (Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2)
    have h2 := Nat.le_ceil (4 / v)
    have hf : κ / 2 = (⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2⌉₊ : ℝ)
        + (⌈Real.log (1 / crossLimit) / v ^ 2⌉₊ : ℝ) + (⌈4 / v⌉₊ : ℝ) + 1 := by
      rw [hκdef, famCount]; push_cast; ring
    linarith [Nat.cast_nonneg (α := ℝ) ⌈Real.log (1 / crossLimit) / v ^ 2⌉₊]
  have hA' : Real.log (2 / cutBudget η₀ indecisionLimit εcov) + 4 * v ≤ κ * v ^ 2 / 2 := by
    have e1 : (Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2 + 4 / v) * v ^ 2
        = Real.log (2 / cutBudget η₀ indecisionLimit εcov) + 4 * v := by
      field_simp
    have := mul_le_mul_of_nonneg_right hA (by positivity : (0 : ℝ) ≤ v ^ 2)
    rw [e1] at this
    linarith
  have hd : decSlack η₀ populations indecisionLimit εcov δ crossLimit = v / 2 - 2 / (κ + 1) := by
    rw [decSlack]
  have hk1 : (0 : ℝ) < κ + 1 := by linarith
  -- `2κ(v/2 − 2/(κ+1))² ≥ κv²/2 − 4v`
  have hq : κ * v ^ 2 / 2 - 4 * v
      ≤ 2 * κ * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2 := by
    rw [hd]
    have hr : κ / (κ + 1) ≤ 1 := by rw [div_le_one hk1]; linarith
    have hr0 : 0 ≤ κ / (κ + 1) := div_nonneg hκ0 hk1.le
    have e : 2 * κ * (v / 2 - 2 / (κ + 1)) ^ 2
        = κ * v ^ 2 / 2 - 4 * v * (κ / (κ + 1)) + 8 * (κ / (κ + 1)) / (κ + 1) := by
      field_simp
      ring
    rw [e]
    have : 0 ≤ 8 * (κ / (κ + 1)) / (κ + 1) := by positivity
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
      Nat.cast_nonneg (α := ℝ) ⌈4 / voteSlack η₀⌉₊]
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
lemma bandHalf_lt (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2)
    (hb : voteSig η₀ ≤ bnd) :
    bandHalf η₀ populations indecisionLimit εcov δ crossLimit + 1
      ≤ ⌈bnd * famCount η₀ populations indecisionLimit εcov δ crossLimit⌉₊ := by
  have hκ0 : (0 : ℝ) ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) :=
    Nat.cast_nonneg _
  have hv : voteSlack η₀ = 3 * voteSig η₀ / 10 := rfl
  have hs := voteSig_pos η₀ hsig
  have hb' := bandHalf_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hsig
  have hseed := famCount_seed (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hsig
  have hc : bnd * (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
      ≤ (⌈bnd * famCount η₀ populations indecisionLimit εcov δ crossLimit⌉₊ : ℝ) :=
    Nat.le_ceil _
  have hsb := mul_le_mul_of_nonneg_right hb hκ0
  have hlt : (bandHalf η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1
      ≤ (⌈bnd * famCount η₀ populations indecisionLimit εcov δ crossLimit⌉₊ : ℝ) := by
    rw [hv] at hb' hseed
    nlinarith
  exact_mod_cast hlt

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
  positivity

lemma validFlip_pos (η₀ : ℝ) (populations : Finset J) {εcov : ℝ} (hsig : η₀ < 1 / 2)
    (hε : 0 < εcov) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < validFlip η₀ populations εcov := by
  rw [validFlip]
  have := validFrac_pos η₀ hsig
  positivity

/-- The screen's cutoff sits under an eighth of the flip validity charges it at. -/
lemma flipBudget_le_validFlip (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 ≤ εcov) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    flipBudget η₀ populations indecisionLimit εcov δ ≤ validFlip η₀ populations εcov / 8 := by
  have hff : flipFrac η₀ = 7 * validFrac η₀ / 9 := by
    rw [flipFrac, validFrac]
    ring
  rw [flipBudget, validFlip, hff]
  have hf := validFrac_pos η₀ hsig
  have hc := (cutBudget_le η₀ indecisionLimit εcov).1
  rw [div_div, div_le_div_iff₀ (by positivity) (by positivity)]
  have hw : (0 : ℝ) ≤ validFrac η₀ * (populations.card : ℝ) := by positivity
  nlinarith [mul_le_mul_of_nonneg_right hc hw, mul_nonneg hε hw]

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
    (rule.subset _ _ _ _ _ B.k (one_mem_screenedAt O populations B x) (Finset.mem_of_mem_erase hv))

lemma prefixWeight_nonneg (populations : Finset J) (m : ℕ) (x : Run Ω S J) (p : S) :
    0 ≤ prefixWeight populations m x p :=
  Finset.sum_nonneg (fun _ _ => by positivity)

open scoped Classical in
lemma drawOrder_of_drawn {M : ℕ} (sd : Fin M → S) {v : S} (h : ∃ i, sd i = v) :
    ∃ i, sd i = v ∧ drawOrder sd v = i.val := by
  have hne : (Finset.univ.filter (fun i => sd i = v)).Nonempty := by
    obtain ⟨i, hi⟩ := h; exact ⟨i, Finset.mem_filter.2 ⟨Finset.mem_univ _, hi⟩⟩
  refine ⟨(Finset.univ.filter (fun i => sd i = v)).min' hne, ?_, by rw [drawOrder, dif_pos hne]⟩
  exact (Finset.mem_filter.1 (Finset.min'_mem _ hne)).2

open scoped Classical in
lemma drawOrder_of_not_drawn {M : ℕ} (sd : Fin M → S) {v : S} (h : ¬ ∃ i, sd i = v) :
    drawOrder sd v = M := by
  have hne : ¬ (Finset.univ.filter (fun i => sd i = v)).Nonempty := by
    rintro ⟨i, hi⟩; exact h ⟨i, (Finset.mem_filter.1 hi).2⟩
  rw [drawOrder, dif_neg hne]

/-- Distinct suffixes of the pool sit at distinct positions: a drawn one at its first draw,
and only the seed can be undrawn. -/
lemma poolOrder_injOn (M : ℕ) (x : Run Ω S J) : Set.InjOn (poolOrder M x) ↑(poolAt M x) := by
  classical
  set sd : Fin M → S := fun i => suffixDraw i.val x with hsd
  have hpool : ∀ v ∈ poolAt M x, v = 1 ∨ ∃ i, sd i = v := by
    intro v hv
    rcases Finset.mem_insert.1 hv with h | h
    · exact Or.inl h
    · obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 h
      exact Or.inr ⟨⟨i, Finset.mem_range.1 hi⟩, rfl⟩
  intro u hu v hv huv
  change drawOrder sd u = drawOrder sd v at huv
  by_cases hdu : ∃ i, sd i = u <;> by_cases hdv : ∃ i, sd i = v
  · obtain ⟨i, hi, hiu⟩ := drawOrder_of_drawn sd hdu
    obtain ⟨i', hi', hiv⟩ := drawOrder_of_drawn sd hdv
    rw [← hi, ← hi', Fin.ext (hiu.symm.trans (huv.trans hiv))]
  · obtain ⟨i, -, hiu⟩ := drawOrder_of_drawn sd hdu
    rw [hiu, drawOrder_of_not_drawn sd hdv] at huv
    exact absurd huv (ne_of_lt i.isLt)
  · obtain ⟨i, -, hiv⟩ := drawOrder_of_drawn sd hdv
    rw [hiv, drawOrder_of_not_drawn sd hdu] at huv
    exact absurd huv.symm (ne_of_lt i.isLt)
  · rcases hpool u hu with rfl | h1
    · rcases hpool v hv with rfl | h2
      · rfl
      · exact absurd h2 hdv
    · exact absurd h1 hdu

lemma poolOrder_injOn_screened (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : Set.InjOn (poolOrder B.nsuff x) ↑(screenedAt O.mq populations B x) :=
  (poolOrder_injOn B.nsuff x).mono (fun _ hv => screenedAt_subset O populations B x hv)

/-- The family with its seed. -/
noncomputable def clusterFull (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  rule.pick (fun w => O.mq w (oracleNoise x) = 1) (prefixWeight populations B.npref x)
    (poolOrder B.nsuff x) (prefixesAt populations B.npref x) (screenedAt O.mq populations B x) B.k

lemma clusterAt_card_add_one (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) :
    (clusterBy rule O.mq populations x B).card + 1 = (clusterFull rule O populations x B).card :=
  Finset.card_erase_add_one (rule.seed_mem _ _ _ _ _ B.k (one_mem_screenedAt O populations B x))

lemma one_not_mem_clusterAt (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : (1 : S) ∉ clusterBy rule O.mq populations x B :=
  Finset.notMem_erase 1 _

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
  exact rule.congr _ _ _ _ _ _ _ (one_mem_screenedAt O populations B (ω, d))
    (fun p hp v hv => hbit _ (mem_readSet hp (screenedAt_subset O populations B (ω, d) hv)))

/-- The gate's own query strings are not read by the clustering.  A prefix is never
`p · v` for a prefix `p` and a pool suffix `v ≠ ε`, on a flat alphabet — so the bits the gate scores
are untouched by everything that decides which side each prefix falls on. -/
lemma disjoint_readSet {Pre Suf : Set S} (hflat : Flat Pre Suf) {P cands C : Finset S}
    (hP : ∀ p ∈ P, p ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (hC : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) : Disjoint C (readSet P cands) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  have hv1 : v = 1 := flat_eq_one hflat (hP p hp) (hC _ hz) (hV v hv) rfl
  rw [hv1, mul_one] at hz
  exact (Finset.disjoint_left.1 hPC hp) hz

/-- The same, for the gate's own prefixes against their own votes: those read `p · v` with
`v ≠ ε`, which is never a prefix. -/
lemma disjoint_readSet_erase {Pre Suf : Set S} (hflat : Flat Pre Suf) {cands C : Finset S}
    (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (hC : ∀ p ∈ C, p ∈ Pre) :
    Disjoint C (readSet C (cands.erase 1)) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact (Finset.mem_erase.1 hv).1 (flat_eq_one hflat (hC p hp) (hC _ hz)
    (hV v (Finset.mem_of_mem_erase hv)) rfl)

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
lemma measurableSet_filter_pred₂ (O : Oracle μ S) {T : Set S} (A : Finset S)
    (Pr₁ Pr₂ : S → Ω → Prop) [∀ ω, DecidablePred (fun p => Pr₁ p ω)]
    [∀ ω, DecidablePred (fun p => Pr₂ p ω)]
    (h₁ : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr₁ p ω})
    (h₂ : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr₂ p ω}) (Q : Finset S → Finset S → Prop) :
    MeasurableSet[noiseAlg O T]
      {ω | Q (A.filter (fun p => Pr₁ p ω)) (A.filter (fun p => Pr₂ p ω))} := by
  classical
  have hcover : {ω | Q (A.filter (fun p => Pr₁ p ω)) (A.filter (fun p => Pr₂ p ω))}
      = ⋃ U ∈ A.powerset, ⋃ V ∈ A.powerset.filter (Q U),
          ({ω | A.filter (fun p => Pr₁ p ω) = U} ∩ {ω | A.filter (fun p => Pr₂ p ω) = V}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe,
      Finset.mem_filter, Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, Finset.filter_subset _ _, _, ⟨Finset.filter_subset _ _, h⟩, rfl, rfl⟩, ?_⟩
    rintro ⟨U, -, V, ⟨-, hQ⟩, rfl, rfl⟩
    exact hQ
  rw [hcover]
  exact Finset.measurableSet_biUnion _ (fun U _ => Finset.measurableSet_biUnion _ (fun V _ =>
    (measurableSet_filter_pred' O Pr₁ h₁ (fun W => W = U)).inter
      (measurableSet_filter_pred' O Pr₂ h₂ (fun W => W = V))))

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
is read off `ε` and why `Clusterer.seed_mem` matters.  A family is admitted only when
each side reads as its own class (`drift_verdict`).

The cutoffs are `B.lo` and `B.hi`, and the binomial nulls are the rates they cut at over the
family without its seed, `(lo + 1)/#F` and `(hi + 1)/#F`; `PassableAt` is where the budgets
they have to keep are written out. -/

/-- The prefixes the cut accepts, and all the prefixes it decides. -/
noncomputable def cutSides (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    Finset S × Finset S :=
  (P.filter (fun p => hi < voteCount mq F p ω),
    P.filter (fun p => hi < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo))

/-- The accept side's hits plus the reject side's misses. -/
noncomputable def agreeOf (A Dset U : Finset S) : ℕ :=
  (A ∩ U).card + ((Dset \ A) \ U).card

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
/-- The agreement statistic's mean: at least `r` per decided prefix the cut gets right, where `r`
is at most the agreeing rate of each correct prefix's class.  A wrong one is charged its whole
`r`, since at a rate near `0` it reads as agreeing almost never. -/
lemma agree_mean_ge (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset) (r : ℝ)
    (hrA : ∀ p ∈ A, O.label p = 1 → r ≤ 1 - O.ηIn)
    (hrR : ∀ p ∈ Dset \ A, O.label p = 0 → r ≤ 1 - O.ηOut) :
    (Dset.card : ℝ) * r - r * ((miscutOf O A Dset : ℕ) : ℝ)
      ≤ ∑ p ∈ Dset, μ[agreeVar O A p] := by
  classical
  have hnn : ∀ p, 0 ≤ μ[agreeVar O A p] := fun p =>
    integral_nonneg_of_ae ((agreeVar_icc O A p).mono fun ω h => h.1)
  have hA : r * ((A.filter (fun p => O.label p = 1)).card : ℝ)
      ≤ ∑ p ∈ A, μ[agreeVar O A p] := by
    refine le_trans ?_ (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset (fun p => O.label p = 1) A)
      (fun p _ _ => hnn p))
    rw [mul_comm, ← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun p hp => ?_)
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    have hfun : agreeVar O A p = O.mq p := by funext ω; simp [agreeVar, hpA]
    have hin : p ∈ O.L := by
      by_contra hout
      simp [Oracle.label, Set.indicator_apply, hout] at hl
    have hr : O.rate p = O.ηIn := by simp [Oracle.rate, hin]
    rw [hfun, mq_mean, hl, hr]
    linarith [hrA p hpA hl]
  have hR : r * (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
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
    have hout : p ∉ O.L := by
      intro hin
      simp [Oracle.label, Set.indicator_apply, hin] at hl
    have hr : O.rate p = O.ηOut := by simp [Oracle.rate, hout]
    rw [hr]
    linarith [hrR p hpR hl]
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
  have hid : r * ((A.filter (fun p => O.label p = 1)).card : ℝ)
      + r * (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
      = (Dset.card : ℝ) * r - r * ((miscutOf O A Dset : ℕ) : ℝ) := by
    rw [hmis, ← hcards, ← hcA, ← hcR]; ring
  linarith [hA, hR, hid, hsplit]

/-- A cut that is mostly right reads as agreeing often enough.  Unlike the two
side-wise tests this replaces, the denominator is the whole decided set, so the bound does
not degrade when one side of the cut is small. -/
lemma agree_sound_of_wrong (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset)
    (θ τ w r : ℝ) (hτ : 0 ≤ τ) (hr0 : 0 ≤ r)
    (hrA : ∀ p ∈ A, O.label p = 1 → r ≤ 1 - O.ηIn)
    (hrR : ∀ p ∈ Dset \ A, O.label p = 0 → r ≤ 1 - O.ηOut)
    (hw : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w)
    (hθ : (Dset.card : ℝ) * (θ + τ)
      ≤ (Dset.card : ℝ) * r - r * w) :
    μ.real {ω | ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
        ≤ (Dset.card : ℝ) * θ}
      ≤ Real.exp (-2 * (Dset.card : ℝ) * τ ^ 2) := by
  classical
  have hmean : (Dset.card : ℝ) * (θ + τ) ≤ ∑ p ∈ Dset, μ[agreeVar O A p] := by
    nlinarith [agree_mean_ge O A Dset hAD r hrA hrR, mul_le_mul_of_nonneg_left hw hr0]
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
/-- The gate's bound on a side the votes choose.  `agree_sound_of_wrong` prices a *fixed*
side; this prices the one the run produces, by conditioning on the votes, which read `Q`,
disjoint from the prefixes the statistic scores.  `sel ω` is the side as an (accept part, whole
side) pair, so an accept side is `(A, A)` and a reject side `(∅, R)`. -/
theorem side_agree_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (sel : Ω → Finset S × Finset S) (hsel : ∀ ω, sel ω ∈ C.powerset ×ˢ C.powerset)
    (hAD : ∀ ω, (sel ω).1 ⊆ (sel ω).2)
    (hselcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω')
    (θ τ w r : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hr0 : 0 ≤ r)
    (hrA : r ≤ 1 - O.ηIn ∨ ∀ ω, (sel ω).1 = ∅)
    (hrR : r ≤ 1 - O.ηOut ∨ ∀ ω, (sel ω).2 = (sel ω).1)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (θ + τ) ≤ (n : ℝ) * r - r * w) :
    μ.real {ω | n₀ ≤ (sel ω).2.card
        ∧ ((miscutOf O (sel ω).1 (sel ω).2 : ℕ) : ℝ) ≤ w
        ∧ ((agreeOf (sel ω).1 (sel ω).2 (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
            ≤ ((sel ω).2.card : ℝ) * θ}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  set Pr : (Finset S × Finset S) → Finset S → Prop := fun t U =>
    t.1 ⊆ t.2 ∧ (r ≤ 1 - O.ηIn ∨ t.1 = ∅) ∧ (r ≤ 1 - O.ηOut ∨ t.2 = t.1) ∧ n₀ ≤ t.2.card
      ∧ ((miscutOf O t.1 t.2 : ℕ) : ℝ) ≤ w
      ∧ ((agreeOf t.1 t.2 U : ℕ) : ℝ) ≤ (t.2.card : ℝ) * θ with hPr
  have hbad : ∀ t ∈ C.powerset ×ˢ C.powerset,
      μ.real {ω | Pr t (C.filter (fun p => O.mq p ω = 1))}
        ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
    rintro ⟨A, Dset⟩ hmem
    by_cases hADt : A ⊆ Dset ∧ (r ≤ 1 - O.ηIn ∨ A = ∅) ∧ (r ≤ 1 - O.ηOut ∨ Dset = A)
    · obtain ⟨hADt, hA', hR'⟩ := hADt
      have hrA' : ∀ p ∈ A, O.label p = 1 → r ≤ 1 - O.ηIn := by
        intro p hp _
        rcases hA' with h | h
        · exact h
        · rw [h] at hp; simp at hp
      have hrR' : ∀ p ∈ Dset \ A, O.label p = 0 → r ≤ 1 - O.ηOut := by
        intro p hp _
        rcases hR' with h | h
        · exact h
        · rw [h] at hp; simp at hp
      by_cases hn : n₀ ≤ Dset.card
      · by_cases hwc : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w
        · have hDC : Dset ⊆ C := Finset.mem_powerset.1 (Finset.mem_product.1 hmem).2
          have hDcard : Dset.card ≤ C.card := Finset.card_le_card hDC
          have hsub : {ω | Pr (A, Dset) (C.filter (fun p => O.mq p ω = 1))}
              ⊆ {ω | ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
                    ≤ (Dset.card : ℝ) * θ} := by
            intro ω hω
            have h4 := hω.2.2.2.2.2
            rwa [agreeOf_filter_of_subset O A Dset C hADt hDC ω] at h4
          refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
          refine le_trans (agree_sound_of_wrong O A Dset hADt θ τ w r hτ hr0 hrA' hrR' hwc
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
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        exact hADt ⟨h1, h2, h3⟩
      rw [hz]; simpa using Real.exp_nonneg _
  have hmain := selection_read_bound O C Q hdisj (C.powerset ×ˢ C.powerset) (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset C, Finset.empty_mem_powerset C⟩)
    sel hsel hselcongr Pr _ (Real.exp_nonneg _) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  exact ⟨hAD ω, hrA.imp_right (fun h => h ω), hrR.imp_right (fun h => h ω), hω.1, hω.2.1, hω.2.2⟩

open scoped Classical in
/-- Both of the gate's wrong-counts are charged to the same mis-cut set.  A prefix the
cut accepts but the oracle rejects, and one the cut rejects but the oracle accepts, are each
a prefix where the cut is wrong — and the two kinds are disjoint. -/
lemma miscutOf_le_cutWrong (O : Oracle μ S) (lo hi : ℕ) (F C : Finset S) (ω : Ω) :
    miscutOf O (cutSides O.mq lo hi F C ω).1 (cutSides O.mq lo hi F C ω).2
      ≤ (C.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card := by
  classical
  have hsubA : ((cutSides O.mq lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo hi F p ω) := by
    intro q hq
    obtain ⟨hqA, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqA
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.1 hv)⟩
  have hsubR : (((cutSides O.mq lo hi F C ω).2 \ (cutSides O.mq lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0)))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo hi F p ω) := by
    intro q hq
    obtain ⟨hqR, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqD, hqA⟩ := Finset.mem_sdiff.1 hqR
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqD
    have hnot : ¬ (hi < voteCount O.mq F q ω) := fun h =>
      hqA (Finset.mem_filter.2 ⟨hqC, h⟩)
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.2 (hv.resolve_left hnot))⟩
  have hdj : Disjoint (((cutSides O.mq lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1)))
      (((cutSides O.mq lo hi F C ω).2 \ (cutSides O.mq lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0))) := by
    refine Finset.disjoint_left.2 (fun q hq hq' => ?_)
    exact (Finset.mem_sdiff.1 (Finset.mem_filter.1 hq').1).2 (Finset.mem_filter.1 hq).1
  rw [miscutOf, ← Finset.card_union_of_disjoint hdj]
  exact Finset.card_le_card (Finset.union_subset hsubA hsubR)

/-- The family at a reachable state is invalid: on some population its cut is wrong on more
than an `εcov` fraction, or leaves more than `2·indecisionLimit` undecided. -/
def FailAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (indecisionLimit εcov : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
      ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1)
          (familyBy rule O.mq populations x B) p (oracleNoise x)} ≤ 2 * indecisionLimit}

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

/-- Hoeffding's bound on the binomial lower tail, as a fact about `binomCdfLe`. -/
theorem binomCdfLe_le (n j : ℕ) (r τ : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hτ : 0 ≤ τ)
    (hj : j ≤ n) (h : (j : ℝ) ≤ (n : ℝ) * (r - τ)) :
    binomCdfLe n r j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  rw [binomCdfLe_eq n j r hj]
  refine binomSfGe_le n (n - j) (1 - r) τ (by linarith) (by linarith) hτ ?_
  rw [Nat.cast_sub hj]
  linarith

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
    (εcov δ α pAP ρ : ℝ) :
    ∑ B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) := by
  classical
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun _ _ _ => Nat.cast_nonneg _))
    (schedule_npref_sum_le (bnd := bnd) η₀ populations εcov δ α pAP)
  rw [stoppable]
  exact Finset.filter_subset _ _

lemma capped_of_mem_stoppable {η₀ : ℝ} {populations : Finset J}
    {εcov δ α pAP ρ : ℝ} {B : State}
    (hB : B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ) :
    Capped η₀ populations indecisionLimit εcov δ ρ pAP
      (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)
      (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit) B := by
  classical
  exact (Finset.mem_filter.1 hB).2

lemma mem_schedule_of_mem_stoppable {η₀ : ℝ} {populations : Finset J}
    {εcov δ α pAP ρ : ℝ} {B : State}
    (hB : B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ) :
    B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit := by
  classical
  exact (Finset.mem_filter.1 hB).1

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
that never flips from one that flips on a `Δ` fraction by at least `Δ(1−2η)²`.  The
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
    (seedLoss_indep hflat O cn cd hP v hvS) (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
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
    ∀ w ∈ clusterBy rule O.mq populations x B,
      ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip w p) :=
  fun w hw => hscreen w (rule.subset _ _ _ _ _ B.k
    (one_mem_screenedAt O populations B x) (Finset.mem_of_mem_erase hw))

/-! ### From flip mass to a correct cut

The band is what turns "few members flip" into "the cut is right".  A rejecting prefix is
accepted only when the vote clears the centre, and members preserving at `p` contribute only
through noise, so with a fraction `f` flipping the vote's mean is at most `ηOut + (1 − ηOut)·f`
— a flip can cost a whole read when the rates are lopsided.  The vote stays decisive with a
flipping `flipFrac` of the family, but it only cuts wrong once its mean passes the centre, so
validity absorbs a flipping `validFrac`, and Markov over the members' flip masses charges the
misclassified mass at `Δ/validFrac`. -/

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

/-- Upper tail on a rejecting prefix.  A clean member reads accepting at most at the bound `β₀`
on the rate off the language, and a flipping one at most always, so a flip fraction of `f` lifts
the vote's mean only to `β₀ + (1 − β₀)·f`. -/
theorem voteSum_upper (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ β₀ : ℝ) (hβ : O.ηOut ≤ β₀) (hβ1 : β₀ ≤ 1)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * ((β₀ + (1 - β₀) * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, μ[O.mq (p * v)] ≤ β₀ + (1 - β₀) * O.flip v p := by
    intro v _
    have hlab : O.label (p * v) = O.flip v p := by
      show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
      rw [hp]; ring
    rw [mq_mean, hlab]
    have hr0 := O.rate_nonneg (p * v)
    rcases O.flip_bit v p with h | h
    · have hl : O.label (p * v) = 0 := by rw [hlab, h]
      have hout : p * v ∉ O.L := by
        intro hin
        simp [Oracle.label, Set.indicator_apply, hin] at hl
      have hr : O.rate (p * v) = O.ηOut := by simp [Oracle.rate, hout]
      rw [h, hr]
      linarith
    · rw [h]
      linarith
  have hmean : ∑ v ∈ F, μ[O.mq (p * v)] ≤ (F.card : ℝ) * (β₀ + (1 - β₀) * f) := by
    refine le_trans (Finset.sum_le_sum hper) ?_
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ 1 - β₀ := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumUpper_le (fun v : S => O.mq (p * v)) F (β₀ + (1 - β₀) * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- Lower tail on an accepting prefix.  Mirror of `voteSum_upper`: a clean member reads
accepting at rate at least `1 − β₁`, for `β₁` a bound on the rate on the language, and a
flipping one possibly never, so the vote's mean only falls to `(1 − β₁)(1 − f)`. -/
theorem voteSum_lower (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ β₁ : ℝ) (hβ : O.ηIn ≤ β₁) (hβ1 : β₁ ≤ 1)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * (((1 - β₁) * (1 - f)) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, (1 - β₁) * (1 - O.flip v p) ≤ μ[O.mq (p * v)] := by
    intro v _
    have hlab : O.label (p * v) = 1 - O.flip v p := by
      show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
      rw [hp]; ring
    rw [mq_mean, hlab]
    have hr0 := O.rate_nonneg (p * v)
    rcases O.flip_bit v p with h | h
    · have hl : O.label (p * v) = 1 := by rw [hlab, h]; ring
      have hin : p * v ∈ O.L := by
        by_contra hout
        simp [Oracle.label, Set.indicator_apply, hout] at hl
      have hr : O.rate (p * v) = O.ηIn := by simp [Oracle.rate, hin]
      rw [h, hr]
      linarith
    · rw [h]
      linarith
  have hmean : (F.card : ℝ) * ((1 - β₁) * (1 - f)) ≤ ∑ v ∈ F, μ[O.mq (p * v)] := by
    refine le_trans ?_ (Finset.sum_le_sum hper)
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ 1 - β₁ := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumLower_le (fun v : S => O.mq (p * v)) F ((1 - β₁) * (1 - f)) γ
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
    (β₀ β₁ : ℝ) (hβ₀ : O.ηOut ≤ β₀) (hβ₁ : O.ηIn ≤ β₁) (hβ₀1 : β₀ ≤ 1) (hβ₁1 : β₁ ≤ 1)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * ((β₀ + (1 - β₀) * f) + γ) ≤ (hi : ℝ) + 1)
    (hlo : (lo : ℝ) ≤ (F.card : ℝ) * (((1 - β₁) * (1 - f)) - γ)) :
    μ.real {ω | ¬ cutCorrect O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ β₀ hβ₀ hβ₀1 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hacc : ¬ (hi < voteCount O.mq F p ω → O.label p = 1) := by
      intro hacc
      exact hbad ⟨hacc, fun _ => hp⟩
    have hgt : hi < voteCount O.mq F p ω := by
      by_contra hc
      exact hacc (fun h => absurd h hc)
    have : (hi : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((β₀ + (1 - β₀) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]; linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ β₁ hβ₁ hβ₁1 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hrej : ¬ (voteCount O.mq F p ω ≤ lo → O.label p = 0) := by
      intro hrej
      exact hbad ⟨fun _ => hp, hrej⟩
    have hle : voteCount O.mq F p ω ≤ lo := by
      by_contra hc
      exact hrej (fun h => absurd h hc)
    have : (voteCount O.mq F p ω : ℝ) ≤ (lo : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((1 - β₁) * (1 - f)) - γ)
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
    ∀ v ∈ insert 1 ((Finset.range M).image s), v ∈ insert 1 Suf := by
  intro v hv
  rcases Finset.mem_insert.1 hv with rfl | hv
  · exact Set.mem_insert _ _
  · obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hv
    exact Set.mem_insert_of_mem _ (hs i)

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

open scoped Classical in
/-- Events about the table, the pool and further draws are measurable, down to the weights and
order the clustering reads: all of them are functions of countably many draws. -/
lemma measurableSet_of_draws {α : Type*} [Countable α] (populations : Finset J) (B : State)
    (g : Run Ω S J → α) (hg : ∀ a, MeasurableSet {x | g x = a})
    (R : Finset S → Finset S → (S → ℝ) → (S → ℕ) → α → Set (Run Ω S J))
    (hR : ∀ P C wt ord a, MeasurableSet (R P C wt ord a)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (prefixWeight populations B.npref x) (poolOrder B.nsuff x) (g x)} := by
  classical
  have hpre : ∀ x : Run Ω S J, ∀ j, prefixesOf j B.npref x
      = (Finset.univ : Finset (Fin B.npref)).image (fun i => prefixDraw j i.val x) :=
    fun x j => image_range_eq_image_univ B.npref (fun i => prefixDraw j i x)
  have hpool : ∀ x : Run Ω S J, poolAt B.nsuff x
      = insert 1 ((Finset.univ : Finset (Fin B.nsuff)).image (fun i => suffixDraw i.val x)) :=
    fun x => congrArg (insert 1) (image_range_eq_image_univ B.nsuff (fun i => suffixDraw i x))
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x) (g x)}
      = ⋃ z : (J → Finset S) × (Fin B.nsuff → S) × α,
          ((((⋂ j : J, {x : Run Ω S J | prefixesOf j B.npref x = z.1 j})
            ∩ ⋂ i : Fin B.nsuff, {x : Run Ω S J | suffixDraw i.val x = z.2.1 i})
            ∩ {x : Run Ω S J | g x = z.2.2})
          ∩ R (populations.biUnion z.1) (insert 1 ((Finset.univ : Finset (Fin B.nsuff)).image
              z.2.1))
              (popWeight populations z.1) (drawOrder z.2.1) z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_iInter]
    constructor
    · intro h
      refine ⟨((fun j => prefixesOf j B.npref x), (fun i => suffixDraw i.val x), g x),
        ⟨⟨fun _ => rfl, fun _ => rfl⟩, rfl⟩, ?_⟩
      rw [← hpool x]
      exact h
    · rintro ⟨⟨Pj, sd, a⟩, ⟨⟨hP, hs⟩, ha⟩, hx⟩
      simp only at hP hs ha hx
      have hPj : (fun j => prefixesOf j B.npref x) = Pj := funext hP
      have hsd : (fun i : Fin B.nsuff => suffixDraw i.val x) = sd := funext hs
      have e1 : prefixesAt populations B.npref x = populations.biUnion Pj := by
        rw [← hPj]; rfl
      have e2 : poolAt B.nsuff x = insert 1 ((Finset.univ : Finset (Fin B.nsuff)).image sd) := by
        rw [hpool x, hsd]
      have e3 : prefixWeight populations B.npref x = popWeight populations Pj := by
        rw [← hPj]; rfl
      have e4 : poolOrder B.nsuff x = drawOrder sd := by rw [← hsd]; rfl
      rw [e1, e2, e3, e4, ha]
      exact hx
  rw [hcov]
  refine MeasurableSet.iUnion (fun z => MeasurableSet.inter ?_ (hR _ _ _ _ _))
  refine MeasurableSet.inter (MeasurableSet.inter (MeasurableSet.iInter (fun j => ?_))
    (MeasurableSet.iInter (fun i => measurableSet_eq_fun (measurable_sfx i.val) measurable_const)))
    (hg _)
  simp only [hpre]
  exact measurableSet_finData (fun i : Fin B.npref => prefixDraw j i.val)
    (fun i => measurable_prf j i.val) (fun t => (Finset.univ : Finset (Fin B.npref)).image t) (z.1
        j)

lemma measurableSet_certFiber (j : J) (m : ℕ) (a : Fin m → S) :
    MeasurableSet {x : Run Ω S J | (fun i : Fin m => certPrefix j i.val x) = a} := by
  have h : {x : Run Ω S J | (fun i : Fin m => certPrefix j i.val x) = a}
      = ⋂ i : Fin m, {x : Run Ω S J | certPrefix j i.val x = a i} := by
    ext x; simp only [Set.mem_setOf_eq, Set.mem_iInter, funext_iff]
  rw [h]
  exact MeasurableSet.iInter (fun i => measurableSet_eq_fun (measurable_cert j i.val)
      measurable_const)

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
fraction of `f` moves each class's mean toward the other by up to `(1 − β)·f`. -/
theorem decided_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ) (hγ : 0 ≤ γ)
    (β₀ β₁ : ℝ) (hβ₀ : O.ηOut ≤ β₀) (hβ₁ : O.ηIn ≤ β₁) (hβ₀1 : β₀ ≤ 1) (hβ₁1 : β₁ ≤ 1)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * (((1 - β₁) * (1 - f)) - γ))
    (hlo : (F.card : ℝ) * ((β₀ + (1 - β₀) * f) + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O.mq lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ β₀ hβ₀ hβ₀1 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O.mq F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((β₀ + (1 - β₀) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ β₁ hβ₁ hβ₁1 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O.mq F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O.mq F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((1 - β₁) * (1 - f)) - γ)
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

lemma measurableSet_certNotInj (j : J) (m : ℕ) :
    MeasurableSet {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => certPrefix j i.val x)} := by
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => certPrefix j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_cert j i.val)
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
  exact hmeasPrf hmeasSet

open scoped Classical in
/-- The runs whose first `m` certification draws from `j` hit `W` at most `r` times. -/
def classShort (j : J) (m : ℕ) (W : Set S) (r : ℝ) : Set (Run Ω S J) :=
  {x | (((Finset.range m).filter (fun i => certPrefix j i x ∈ W)).card : ℝ) ≤ r}

open scoped Classical in
lemma measurableSet_classShort (j : J) (m : ℕ) (W : Set S) (r : ℝ) :
    MeasurableSet (classShort (Ω := Ω) (J := J) j m W r) := by
  classical
  have hWm : MeasurableSet W := measurableSet_of_countable W
  have hcnt : Measurable (fun x : Run Ω S J =>
      ((Finset.range m).filter (fun i => certPrefix j i x ∈ W)).card) := by
    simp_rw [Finset.card_filter]
    refine Finset.measurable_sum _ (fun i _ => Measurable.ite ?_ measurable_const measurable_const)
    exact (measurable_cert j i) hWm
  exact hcnt (Set.to_countable {n : ℕ | (n : ℝ) ≤ r}).measurableSet

open scoped Classical in
/-- `cert_hits_wrongSet` on the run: the certification sample holds its share of a set. -/
theorem cert_class_short_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) (W : Set S)
    (l t : ℝ) (hl : 0 ≤ l) (ht : 0 ≤ t) (hW : l ≤ (D j).real W) :
    (runMeasure μ D Dsf).real (classShort j m W ((m : ℝ) * (l - t)))
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  have hmap : Measure.map (fun x : Run Ω S J => x.2.2) (runMeasure μ D Dsf)
      = Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j := by
    rw [show (fun x : Run Ω S J => x.2.2) = Prod.snd ∘ Prod.snd from rfl,
      ← Measure.map_map measurable_snd measurable_snd, runMeasure_eq_prod, Measure.map_snd_prod]
    simp only [measure_univ, one_smul]
    rw [drawMeasure, Measure.map_snd_prod]
    simp
  have hWm : MeasurableSet W := measurableSet_of_countable W
  have hcnt : Measurable
      (fun c : J → ℕ → S => ((Finset.range m).filter (fun i => c j i ∈ W)).card) := by
    simp_rw [Finset.card_filter]
    refine Finset.measurable_sum _ (fun i _ => Measurable.ite ?_ measurable_const measurable_const)
    exact ((measurable_pi_apply i).comp (measurable_pi_apply j)) hWm
  have hset : MeasurableSet {c : J → ℕ → S |
      (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) ≤ (m : ℝ) * (l - t)} :=
    hcnt (Set.to_countable {n : ℕ | (n : ℝ) ≤ (m : ℝ) * (l - t)}).measurableSet
  have hpre : classShort j m W ((m : ℝ) * (l - t))
      = (fun x : Run Ω S J => x.2.2) ⁻¹' {c : J → ℕ → S |
          (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) ≤ (m : ℝ) * (l - t)} := rfl
  rw [hpre, measureReal_def, ← Measure.map_apply (by fun_prop) hset, hmap, ← measureReal_def]
  exact cert_hits_wrongSet D j m W l t hl ht hW

/-- A stream of i.i.d. draws, read from its `a`-th draw on, is the same stream. -/
lemma measurePreserving_streamShift (ν : Measure S) [IsProbabilityMeasure ν] (a : ℕ) :
    MeasurePreserving (fun s : ℕ → S => fun i => s (a + i))
      (Measure.infinitePi fun _ : ℕ => ν) (Measure.infinitePi fun _ : ℕ => ν) := by
  classical
  refine ⟨by fun_prop, ?_⟩
  refine Measure.eq_infinitePi _ fun s t ht => ?_
  have hpre : (fun s' : ℕ → S => fun i => s' (a + i)) ⁻¹' ((s : Set ℕ).pi t)
      = ((s.image (fun i => a + i) : Finset ℕ) : Set ℕ).pi (fun k => t (k - a)) := by
    ext x
    simp only [Set.mem_preimage, Set.mem_pi, Finset.mem_coe, Finset.coe_image, Set.mem_image,
      forall_exists_index, and_imp]
    constructor
    · rintro h k i hi rfl
      simpa using h i hi
    · intro h i hi
      simpa using h (a + i) i hi rfl
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.pi s.countable_toSet (fun i _ => ht i)),
    hpre, Measure.infinitePi_pi _ (fun i _ => ht _),
    Finset.prod_image (fun i _ j _ h => by simpa using h)]
  simp

/-- The run with every certification stream read from its `a`-th draw on. -/
def certShift (a : ℕ) (x : Run Ω S J) : Run Ω S J :=
  (x.1, (x.2.1, fun j i => x.2.2 j (a + i)))

lemma measurePreserving_certShift (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (a : ℕ) :
    MeasurePreserving (certShift a) (runMeasure μ D Dsf) (runMeasure μ D Dsf) := by
  have hC : MeasurePreserving (fun (c : J → ℕ → S) j => (fun i => c j (a + i)))
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) :=
    measurePreserving_pi _ _ (fun j => measurePreserving_streamShift (D j) a)
  exact (MeasurePreserving.id μ).prod ((MeasurePreserving.id _).prod hC)

/-- An event read at the shifted certification streams is as likely as the event itself. -/
lemma measureReal_certShift (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (a : ℕ)
    (E : Set (Run Ω S J)) (hE : MeasurableSet E) :
    (runMeasure μ D Dsf).real (certShift a ⁻¹' E) = (runMeasure μ D Dsf).real E := by
  rw [measureReal_def, measureReal_def,
    (measurePreserving_certShift (μ := μ) D Dsf a).measure_preimage hE.nullMeasurableSet]

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
lemma disjoint_gateReads {Pre Suf : Set S} (hflat : Flat Pre Suf) (P C A : Finset S)
    (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ C, v ∈ insert 1 Suf) (hA : ∀ p ∈ A, p ∈ Pre)
    (hPA : Disjoint P A) :
    Disjoint (↑A : Set S) (↑(readSet P C ∪ readSet A (C.erase 1)) : Set S) := by
  classical
  rw [Finset.coe_union, Set.disjoint_union_right]
  exact ⟨Finset.disjoint_coe.2 (disjoint_readSet hflat hP hV hA hPA),
    Finset.disjoint_coe.2 (disjoint_readSet_erase hflat hV hA)⟩

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

open scoped Classical in
/-- At a fixed table, the FNR test passes.  The undecided count splits into the prefixes the
family is light for, bounded by `frac_selected_le`, and the heavy ones, whose number is itself a
fraction.  Lightness at a *fresh* prefix only holds for most of them, never all, so carrying it
as a per-prefix condition rather than a global guard is what makes the argument available. -/
theorem ret_at_whp {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C)
    (lo hi : ℕ) (l lcut : ℝ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (hone : (1 : S) ∈ cands) (hT1 : ∀ t ∈ T, (1 : S) ∉ t)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo (hi + 1) (insert 1 A₀) p ω} ≤ E) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ (((C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _))
    (indecision_frac_le_seed hflat O P cands C hP hV hCPre hPC hone lo (hi + 1) T good t₀ ht₀
      hTC hT1 fam hfam hcongr E l hE (le_trans hElcut hlcl) hdec)
  obtain ⟨hheavy, hbad⟩ := hω
  show l * (C.card : ℝ) < ((C.filter (fun p => fam ω ∈ good p
      ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
  have hsplit : (C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card
      ≤ (C.filter (fun p => fam ω ∈ good p
          ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card
        + (C.filter (fun p => fam ω ∉ good p)).card := by
    refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
    intro p hp
    obtain ⟨hpC, hq⟩ := Finset.mem_filter.1 hp
    by_cases hg : fam ω ∈ good p
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
  have hsplitR : ((C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
      ≤ ((C.filter (fun p => fam ω ∈ good p
          ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
        + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast hsplit
  have hscale : lcut * (C.card : ℝ) ≤ l * (C.card : ℝ) :=
    mul_le_mul_of_nonneg_right hlcl (Nat.cast_nonneg _)
  push_neg at hbad
  linarith

set_option maxHeartbeats 1000000 in
open scoped Classical in
/-- At a fixed table, the gate certifies the uniform pool's sample `C`, and the family decides
some prefix of it each way.  Each side holds most of its class, because the family is light
and its misses on each class are a small fraction of that class; each side is nearly pure,
because a wrong member of it is a miscut; and a large, nearly pure side reads its class past the
rate the cut splits at, which `side_agree_bound` prices. -/
theorem gate_at_whp {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (Q : Finset S) (hQsup : readSet P cands ⊆ Q)
    (hdisjQ : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi κ : ℕ) (α lcut τ q : ℝ) (n₁ : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (hone : (1 : S) ∈ cands) (hT1 : ∀ t ∈ T, (1 : S) ∉ t)
    (hgoodcard : ∀ p, ∀ A₀ ∈ good p, A₀.card = κ)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (E : ℝ) (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo (hi + 1) (insert 1 A₀) p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo hi A₀ p ω} ≤ E)
    (hCL : q * (C.card : ℝ) ≤ ((C.filter (fun p => p ∈ O.L)).card : ℝ))
    (hCN : q * (C.card : ℝ) ≤ ((C.filter (fun p => p ∉ O.L)).card : ℝ))
    (hlcut1 : lcut < 1) (hn₁ : (n₁ : ℝ) ≤ q * (C.card : ℝ) - 3 * lcut * (C.card : ℝ))
    (hn₁pos : 0 < n₁) (hκ : 0 < κ)
    (hlohi : lo + 1 ≤ κ) (hhi : hi + 1 ≤ κ)
    (hga : ∀ n : ℕ, n₁ ≤ n → n ≤ C.card →
      (n : ℝ) * ((((hi : ℝ) + 1) / κ + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηIn) - (1 - O.ηIn) * (2 * lcut * (C.card : ℝ)))
    (hgr : ∀ n : ℕ, n₁ ≤ n → n ≤ C.card →
      (n : ℝ) * (((1 - ((lo : ℝ) + 1) / (κ + 1)) + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηOut) - (1 - O.ηOut) * (2 * lcut * (C.card : ℝ)))
    (hα : Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ (certified O.mq lo hi α (fam ω) C ω
          ∧ (∃ p ∈ C, hi + 1 < voteCount O.mq (insert 1 (fam ω)) p ω)
          ∧ (∃ p ∈ C, voteCount O.mq (insert 1 (fam ω)) p ω ≤ lo))}
      ≤ 2 * Real.exp (-2 * (C.card : ℝ) * (lcut - E) ^ 2)
        + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := by
  classical
  set CL := C.filter (fun p => p ∈ O.L) with hCLdef
  set CN := C.filter (fun p => p ∉ O.L) with hCNdef
  have hCLsub : CL ⊆ C := Finset.filter_subset _ _
  have hCNsub : CN ⊆ C := Finset.filter_subset _ _
  -- the two count events and the two side events
  set B1 := {ω | lcut * (C.card : ℝ) < ((C.filter (fun p => fam ω ∈ good p
      ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)} with hB1
  set B3 := {ω | lcut * (C.card : ℝ) < ((C.filter (fun p => fam ω ∈ good p
      ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)} with hB3
  set Aω : Ω → Finset S := fun ω => C.filter (fun p => hi < voteCount O.mq (fam ω) p ω)
    with hAω
  set Rω : Ω → Finset S := fun ω => C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo)
    with hRω
  have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ C,
      voteCount O.mq (fam ω) p ω = voteCount O.mq (fam ω') p ω' := by
    intro ω ω' h p hp
    rw [← hcongr ω ω' (fun w hw => h w (hQsup hw))]
    exact voteCount_congr O _ p (fun v hv => by rw [mq_congr O (h _ (hQ ω p hp v hv))])
  have hAcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → Aω ω = Aω ω' := by
    intro ω ω' h
    exact Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp])
  have hRcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → Rω ω = Rω ω' := by
    intro ω ω' h
    exact Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp])
  set B5 := {ω | n₁ ≤ (Aω ω, Aω ω).2.card
      ∧ ((miscutOf O (Aω ω, Aω ω).1 (Aω ω, Aω ω).2 : ℕ) : ℝ) ≤ 2 * lcut * (C.card : ℝ)
      ∧ ((agreeOf (Aω ω, Aω ω).1 (Aω ω, Aω ω).2 (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
          ≤ (((Aω ω, Aω ω).2.card : ℕ) : ℝ) * (((hi : ℝ) + 1) / κ + τ)} with hB5
  set B6 := {ω | n₁ ≤ ((∅ : Finset S), Rω ω).2.card
      ∧ ((miscutOf O ((∅ : Finset S), Rω ω).1 ((∅ : Finset S), Rω ω).2 : ℕ) : ℝ)
          ≤ 2 * lcut * (C.card : ℝ)
      ∧ ((agreeOf ((∅ : Finset S), Rω ω).1 ((∅ : Finset S), Rω ω).2
            (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
          ≤ ((((∅ : Finset S), Rω ω).2.card : ℕ) : ℝ) * ((1 - ((lo : ℝ) + 1) / (κ + 1)) + τ)}
    with hB6
  have hmB1 := indecision_frac_le_seed hflat O P cands C hP hV hCPre hPC hone lo (hi + 1) T
    good t₀ ht₀ hTC hT1 fam hfam hcongr E lcut hE hElcut hdec
  have hmB3 := miscut_frac_le hflat O P cands C hP hV hCPre hPC lo hi T good t₀ ht₀ hTC
    fam hfam hcongr E lcut hE hElcut hcut
  have hmB5 := side_agree_bound O C Q hdisjQ (fun ω => (Aω ω, Aω ω))
    (fun ω => Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩)
    (fun ω => le_rfl) (fun ω ω' h => by rw [hAcongr ω ω' h])
    (((hi : ℝ) + 1) / κ + τ) τ (2 * lcut * (C.card : ℝ)) (1 - O.ηIn) n₁ hτ
    (by linarith [le_trans (le_max_left O.ηIn O.ηOut) hsig]) (Or.inl le_rfl)
    (Or.inr (fun ω => rfl)) hga
  have hmB6 := side_agree_bound O C Q hdisjQ (fun ω => ((∅ : Finset S), Rω ω))
    (fun ω => Finset.mem_product.2 ⟨Finset.empty_mem_powerset _,
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩)
    (fun ω => Finset.empty_subset _) (fun ω ω' h => by rw [hRcongr ω ω' h])
    ((1 - ((lo : ℝ) + 1) / (κ + 1)) + τ) τ (2 * lcut * (C.card : ℝ)) (1 - O.ηOut) n₁ hτ
    (by linarith [le_trans (le_max_right O.ηIn O.ηOut) hsig]) (Or.inr (fun ω => rfl))
    (Or.inl le_rfl) hgr
  -- off the six events, a light family passes
  have hsub : {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ (certified O.mq lo hi α (fam ω) C ω
          ∧ (∃ p ∈ C, hi + 1 < voteCount O.mq (insert 1 (fam ω)) p ω)
          ∧ (∃ p ∈ C, voteCount O.mq (insert 1 (fam ω)) p ω ≤ lo))}
      ⊆ (B1 ∪ B3) ∪ (B5 ∪ B6) := by
    intro ω ⟨hlight, hfail⟩
    by_contra hout
    simp only [Set.mem_union, not_or] at hout
    obtain ⟨⟨h1, h3⟩, ⟨h5, h6⟩⟩ := hout
    simp only [hB1, hB3, Set.mem_setOf_eq, not_lt] at h1 h3
    -- a class's share of a count is at most the count
    have hsubC : ∀ (D₀ : Finset S), D₀ ⊆ C → ∀ (Pr : S → Prop) [DecidablePred Pr],
        ((D₀.filter Pr).card : ℝ) ≤ ((C.filter Pr).card : ℝ) := by
      intro D₀ hD Pr _
      exact_mod_cast Finset.card_le_card (fun p hp =>
        Finset.mem_filter.2 ⟨hD (Finset.mem_filter.1 hp).1, (Finset.mem_filter.1 hp).2⟩)
    have hf1 : (1 : S) ∉ fam ω := hT1 _ (hfam ω)
    have hCpos : 0 < C.card := by
      rcases Nat.eq_zero_or_pos C.card with h0 | h0
      · exfalso
        rw [h0] at hn₁
        have : (0 : ℝ) < (n₁ : ℝ) := by exact_mod_cast hn₁pos
        push_cast at hn₁
        linarith
      · exact h0
    -- the family has its round's size: some prefix is light
    have hcardF : (fam ω).card = κ := by
      have hlt : ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) < (C.card : ℝ) := by
        have : (0 : ℝ) < (C.card : ℝ) := by exact_mod_cast hCpos
        nlinarith
      have hlt' : (C.filter (fun p => fam ω ∉ good p)).card < C.card := by exact_mod_cast hlt
      obtain ⟨p, hpC, hpg⟩ : ∃ p ∈ C, fam ω ∈ good p := by
        by_contra hno
        push_neg at hno
        have hC : C.filter (fun p => fam ω ∉ good p) = C := Finset.filter_true_of_mem hno
        rw [hC] at hlt'
        exact absurd hlt' (lt_irrefl _)
      exact hgoodcard p _ hpg
    -- the bad prefixes of each class
    have hgoodL : ∀ p ∈ CL, fam ω ∈ good p → decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω →
        cutCorrect O lo hi (fam ω) p ω → p ∈ Aω ω ∧ hi + 1 < voteCount O.mq (insert 1 (fam ω)) p ω := by
      intro p hp _ hd hcc
      have hpL : O.label p = 1 := by
        simp only [Oracle.label, Set.indicator_apply]
        rw [if_pos (Finset.mem_filter.1 hp).2]; rfl
      have hvd := decided_of_insert_one O.mq lo hi (fam ω) hf1 p ω hd
      rcases hd with hacc | hrej
      · refine ⟨Finset.mem_filter.2 ⟨hCLsub hp, ?_⟩, hacc⟩
        rcases hvd with h | h
        · exact h
        · exfalso; have := hcc.2 h; rw [hpL] at this; norm_num at this
      · exfalso
        have hv := voteCount_insert_one O.mq (fam ω) hf1 p ω
        have hle : voteCount O.mq (fam ω) p ω ≤ lo := by omega
        have := hcc.2 hle
        rw [hpL] at this; norm_num at this
    have hgoodN : ∀ p ∈ CN, fam ω ∈ good p → decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω →
        cutCorrect O lo hi (fam ω) p ω → p ∈ Rω ω ∧ voteCount O.mq (insert 1 (fam ω)) p ω ≤ lo := by
      intro p hp _ hd hcc
      have hpN : O.label p = 0 := by
        simp only [Oracle.label, Set.indicator_apply]
        rw [if_neg (Finset.mem_filter.1 hp).2]
      have hvd := decided_of_insert_one O.mq lo hi (fam ω) hf1 p ω hd
      rcases hd with hacc | hrej
      · exfalso
        have hv := voteCount_insert_one O.mq (fam ω) hf1 p ω
        have hs : (if O.mq p ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
        have hgt : hi < voteCount O.mq (fam ω) p ω := by omega
        have := hcc.1 hgt
        rw [hpN] at this; norm_num at this
      · refine ⟨Finset.mem_filter.2 ⟨hCNsub hp, ?_⟩, hrej⟩
        have hv := voteCount_insert_one O.mq (fam ω) hf1 p ω
        omega
    -- counting: the well-read part of a class is most of it
    have hcountL : (CL.card : ℝ) - 3 * lcut * (C.card : ℝ)
        ≤ ((CL.filter (fun p => fam ω ∈ good p ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by
      have hc : CL.card ≤ (CL.filter (fun p => fam ω ∈ good p
            ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card
          + (C.filter (fun p => fam ω ∉ good p)).card
          + (CL.filter (fun p => fam ω ∈ good p
              ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card
          + (CL.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card := by
        refine le_trans (Finset.card_le_card (s := CL) ?_)
          (le_trans (Finset.card_union_le _ _) (Nat.add_le_add_right
            (le_trans (Finset.card_union_le _ _) (Nat.add_le_add_right
              (Finset.card_union_le _ _) _)) _))
        intro p hp
        by_cases hg : fam ω ∈ good p
        · by_cases hd : decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
          · by_cases hcc : cutCorrect O lo hi (fam ω) p ω
            · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_left _
                (Finset.mem_filter.2 ⟨hp, hg, hd, hcc⟩)))
            · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hp, hg, hcc⟩)
          · exact Finset.mem_union_left _ (Finset.mem_union_right _
              (Finset.mem_filter.2 ⟨hp, hg, hd⟩))
        · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _
            (Finset.mem_filter.2 ⟨hCLsub hp, hg⟩)))
      have hcR : (CL.card : ℝ) ≤ ((CL.filter (fun p => fam ω ∈ good p
            ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
          + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ)
          + ((CL.filter (fun p => fam ω ∈ good p
              ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
          + ((CL.filter (fun p => fam ω ∈ good p
              ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by exact_mod_cast hc
      have hd' := hsubC CL hCLsub (fun p => fam ω ∈ good p
        ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)
      have hc' := hsubC CL hCLsub (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)
      linarith
    have hcountN : (CN.card : ℝ) - 3 * lcut * (C.card : ℝ)
        ≤ ((CN.filter (fun p => fam ω ∈ good p ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by
      have hc : CN.card ≤ (CN.filter (fun p => fam ω ∈ good p
            ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card
          + (C.filter (fun p => fam ω ∉ good p)).card
          + (CN.filter (fun p => fam ω ∈ good p
              ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card
          + (CN.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card := by
        refine le_trans (Finset.card_le_card (s := CN) ?_)
          (le_trans (Finset.card_union_le _ _) (Nat.add_le_add_right
            (le_trans (Finset.card_union_le _ _) (Nat.add_le_add_right
              (Finset.card_union_le _ _) _)) _))
        intro p hp
        by_cases hg : fam ω ∈ good p
        · by_cases hd : decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
          · by_cases hcc : cutCorrect O lo hi (fam ω) p ω
            · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_left _
                (Finset.mem_filter.2 ⟨hp, hg, hd, hcc⟩)))
            · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hp, hg, hcc⟩)
          · exact Finset.mem_union_left _ (Finset.mem_union_right _
              (Finset.mem_filter.2 ⟨hp, hg, hd⟩))
        · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _
            (Finset.mem_filter.2 ⟨hCNsub hp, hg⟩)))
      have hcR : (CN.card : ℝ) ≤ ((CN.filter (fun p => fam ω ∈ good p
            ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
            ∧ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
          + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ)
          + ((CN.filter (fun p => fam ω ∈ good p
              ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
          + ((CN.filter (fun p => fam ω ∈ good p
              ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by exact_mod_cast hc
      have hd' := hsubC CN hCNsub (fun p => fam ω ∈ good p
        ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)
      have hc' := hsubC CN hCNsub (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)
      linarith
    have hCLle : (CL.card : ℝ) ≤ (C.card : ℝ) := by exact_mod_cast Finset.card_le_card hCLsub
    have hCNle : (CN.card : ℝ) ≤ (C.card : ℝ) := by exact_mod_cast Finset.card_le_card hCNsub
    have hlc0 : 0 ≤ lcut := le_trans hE hElcut
    set WL := CL.filter (fun p => fam ω ∈ good p ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
      ∧ cutCorrect O lo hi (fam ω) p ω) with hWL
    set WN := CN.filter (fun p => fam ω ∈ good p ∧ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω
      ∧ cutCorrect O lo hi (fam ω) p ω) with hWN
    have hWLn : (n₁ : ℝ) ≤ (WL.card : ℝ) := by linarith [hcountL, hn₁, hCL]
    have hWNn : (n₁ : ℝ) ≤ (WN.card : ℝ) := by linarith [hcountN, hn₁, hCN]
    have hWLA : WL ⊆ Aω ω := fun p hp => by
      obtain ⟨hpL, hg, hd, hcc⟩ := Finset.mem_filter.1 hp
      exact (hgoodL p hpL hg hd hcc).1
    have hWNR : WN ⊆ Rω ω := fun p hp => by
      obtain ⟨hpN, hg, hd, hcc⟩ := Finset.mem_filter.1 hp
      exact (hgoodN p hpN hg hd hcc).1
    have hAn : n₁ ≤ (Aω ω).card := by
      have := Finset.card_le_card hWLA
      have : (n₁ : ℝ) ≤ ((Aω ω).card : ℝ) := le_trans hWLn (by exact_mod_cast this)
      exact_mod_cast this
    have hRn : n₁ ≤ (Rω ω).card := by
      have := Finset.card_le_card hWNR
      have : (n₁ : ℝ) ≤ ((Rω ω).card : ℝ) := le_trans hWNn (by exact_mod_cast this)
      exact_mod_cast this
    -- a wrong member of a side is a miscut
    have hmisA : ((miscutOf O (Aω ω) (Aω ω) : ℕ) : ℝ) ≤ 2 * lcut * (C.card : ℝ) := by
      have hsub' : (Aω ω).filter (fun p => ¬ (O.label p = 1))
          ⊆ C.filter (fun p => fam ω ∉ good p)
            ∪ CN.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω) := by
        intro p hp
        obtain ⟨hpA, hlab⟩ := Finset.mem_filter.1 hp
        obtain ⟨hpC, hv⟩ := Finset.mem_filter.1 hpA
        by_cases hg : fam ω ∈ good p
        · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpC, ?_⟩,
            hg, fun hcc => hlab (hcc.1 hv)⟩)
          intro hpL
          apply hlab
          simp only [Oracle.label, Set.indicator_apply, if_pos hpL]; rfl
        · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
      have hc := le_trans (Finset.card_le_card hsub') (Finset.card_union_le _ _)
      have hm : miscutOf O (Aω ω) (Aω ω) = ((Aω ω).filter (fun p => ¬ (O.label p = 1))).card := by
        simp [miscutOf]
      rw [hm]
      have hcR : (((Aω ω).filter (fun p => ¬ (O.label p = 1))).card : ℝ)
          ≤ ((C.filter (fun p => fam ω ∉ good p)).card : ℝ)
            + ((CN.filter (fun p => fam ω ∈ good p
                ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by exact_mod_cast hc
      linarith [hcR, hlight, h3, hsubC CN hCNsub (fun p => fam ω ∈ good p
        ∧ ¬ cutCorrect O lo hi (fam ω) p ω)]
    have hmisR : ((miscutOf O ∅ (Rω ω) : ℕ) : ℝ) ≤ 2 * lcut * (C.card : ℝ) := by
      have hsub' : (Rω ω).filter (fun p => ¬ (O.label p = 0))
          ⊆ C.filter (fun p => fam ω ∉ good p)
            ∪ CL.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω) := by
        intro p hp
        obtain ⟨hpR, hlab⟩ := Finset.mem_filter.1 hp
        obtain ⟨hpC, hv⟩ := Finset.mem_filter.1 hpR
        by_cases hg : fam ω ∈ good p
        · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpC, ?_⟩,
            hg, fun hcc => hlab (hcc.2 hv)⟩)
          by_contra hpL
          apply hlab
          simp only [Oracle.label, Set.indicator_apply, if_neg hpL]
        · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
      have hc := le_trans (Finset.card_le_card hsub') (Finset.card_union_le _ _)
      have hm : miscutOf O ∅ (Rω ω) = ((Rω ω).filter (fun p => ¬ (O.label p = 0))).card := by
        simp [miscutOf]
      rw [hm]
      have hcR : (((Rω ω).filter (fun p => ¬ (O.label p = 0))).card : ℝ)
          ≤ ((C.filter (fun p => fam ω ∉ good p)).card : ℝ)
            + ((CL.filter (fun p => fam ω ∈ good p
                ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ) := by exact_mod_cast hc
      linarith [hcR, hlight, h3, hsubC CL hCLsub (fun p => fam ω ∈ good p
        ∧ ¬ cutCorrect O lo hi (fam ω) p ω)]
    -- the sides' counts on the seed's column
    have hagA : ((Aω ω).card : ℝ) * (((hi : ℝ) + 1) / κ + τ)
        < ((C.filter (fun p => hi < voteCount O.mq (fam ω) p ω ∧ O.mq p ω = 1)).card : ℝ) := by
      have h5' : ¬ (((agreeOf (Aω ω) (Aω ω) (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
          ≤ (((Aω ω).card : ℕ) : ℝ) * (((hi : ℝ) + 1) / κ + τ)) := fun h => h5 ⟨hAn, hmisA, h⟩
      have heq : agreeOf (Aω ω) (Aω ω) (C.filter (fun p => O.mq p ω = 1))
          = (C.filter (fun p => hi < voteCount O.mq (fam ω) p ω ∧ O.mq p ω = 1)).card := by
        rw [agreeOf, Finset.sdiff_self, Finset.empty_sdiff, Finset.card_empty, add_zero,
          Finset.filter_and] <;> rfl
      rw [heq] at h5'
      exact lt_of_not_ge h5'
    have hagR : ((Rω ω).card : ℝ) * ((1 - ((lo : ℝ) + 1) / (κ + 1)) + τ)
        < ((Rω ω).card : ℝ)
          - ((C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo ∧ O.mq p ω = 1)).card : ℝ) := by
      have h6' : ¬ (((agreeOf ∅ (Rω ω) (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
          ≤ (((Rω ω).card : ℕ) : ℝ) * ((1 - ((lo : ℝ) + 1) / (κ + 1)) + τ)) :=
        fun h => h6 ⟨hRn, hmisR, h⟩
      have hsplitR : ((Rω ω) \ (C.filter (fun p => O.mq p ω = 1))).card
          + (C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo ∧ O.mq p ω = 1)).card
          = (Rω ω).card := by
        rw [Finset.filter_and, show Rω ω ∩ C.filter (fun p => O.mq p ω = 1)
          = (C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo)) ∩ C.filter (fun p => O.mq p ω = 1)
          from rfl]
        exact Finset.card_sdiff_add_card_inter _ _
      have heq : agreeOf ∅ (Rω ω) (C.filter (fun p => O.mq p ω = 1))
          = ((Rω ω) \ (C.filter (fun p => O.mq p ω = 1))).card := by
        simp [agreeOf]
      rw [heq] at h6'
      have hsR : (((Rω ω) \ (C.filter (fun p => O.mq p ω = 1))).card : ℝ)
          + ((C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo ∧ O.mq p ω = 1)).card : ℝ)
          = ((Rω ω).card : ℝ) := by exact_mod_cast hsplitR
      linarith [lt_of_not_ge h6']
    have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκ
    have ha1 : ((hi : ℝ) + 1) / κ ≤ 1 := by
      rw [div_le_one hκR]; exact_mod_cast hhi
    have hr1 : ((lo : ℝ) + 1) / (κ + 1) ≤ 1 := by
      rw [div_le_one (by positivity)]
      have : ((lo : ℝ) + 1) ≤ (κ : ℝ) := by exact_mod_cast hlohi
      linarith
    have hexpn : ∀ m : ℕ, n₁ ≤ m → Real.exp (-2 * (m : ℝ) * τ ^ 2) ≤ α := fun m hm =>
      le_trans (Real.exp_le_exp.2 (by
        have : (n₁ : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
        nlinarith [sq_nonneg τ])) hα
    apply hfail
    refine ⟨⟨Or.inl ⟨_, (Finset.card_pos.1 (lt_of_lt_of_le hn₁pos hAn))
        |>.choose_spec⟩, fun _ => ?_, fun _ => ?_⟩, ?_, ?_⟩
    · rw [hcardF]
      refine le_trans (binomSfGe_le _ _ _ τ (by positivity) ha1 hτ hagA.le) (hexpn _ hAn)
    · rw [hcardF]
      have hj : (C.filter (fun p => voteCount O.mq (fam ω) p ω ≤ lo ∧ O.mq p ω = 1)).card
          ≤ (Rω ω).card := by
        refine Finset.card_le_card (fun p hp => ?_)
        obtain ⟨hpC, hv, -⟩ := Finset.mem_filter.1 hp
        exact Finset.mem_filter.2 ⟨hpC, hv⟩
      refine le_trans (binomCdfLe_le _ _ _ τ (by positivity) hr1 hτ hj ?_) (hexpn _ hRn)
      linarith
    · obtain ⟨p, hp⟩ := Finset.card_pos.1 (lt_of_lt_of_le (by exact_mod_cast hn₁pos : (0 : ℝ) < n₁)
        hWLn |> fun h => by exact_mod_cast h : 0 < WL.card)
      obtain ⟨hpL, hg, hd, hcc⟩ := Finset.mem_filter.1 hp
      exact ⟨p, hCLsub hpL, (hgoodL p hpL hg hd hcc).2⟩
    · obtain ⟨p, hp⟩ := Finset.card_pos.1 (lt_of_lt_of_le (by exact_mod_cast hn₁pos : (0 : ℝ) < n₁)
        hWNn |> fun h => by exact_mod_cast h : 0 < WN.card)
      obtain ⟨hpN, hg, hd, hcc⟩ := Finset.mem_filter.1 hp
      exact ⟨p, hCNsub hpN, (hgoodN p hpN hg hd hcc).2⟩
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h13 := measureReal_union_le (μ := μ) B1 B3
  have h56 := measureReal_union_le (μ := μ) B5 B6
  have e1 : μ.real B1 ≤ Real.exp (-2 * (C.card : ℝ) * (lcut - E) ^ 2) := hmB1
  have e3 : μ.real B3 ≤ Real.exp (-2 * (C.card : ℝ) * (lcut - E) ^ 2) := hmB3
  have e5 : μ.real B5 ≤ Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := hmB5
  have e6 : μ.real B6 ≤ Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := hmB6
  linarith

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
noncomputable def clusterOf (rule : Clusterer S) (O : Oracle μ S) (wt : S → ℝ) (ord : S → ℕ)
    (sc scd : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω) : Finset S :=
  (rule.pick (fun w => O.mq w ω = 1) wt ord P (screened O.mq sc scd P cands ω) k).erase 1

lemma clusterAt_eq_clusterOf (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) :
    clusterBy rule O.mq populations x B
      = clusterOf rule O (prefixWeight populations B.npref x) (poolOrder B.nsuff x) B.sc B.scd
          (prefixesAt populations B.npref x) (poolAt B.nsuff x) B.k (oracleNoise x) :=
  rfl

lemma clusterOf_subset (O : Oracle μ S) (wt : S → ℝ) (ord : S → ℕ) (sc scd : ℕ) (P cands : Finset
    S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) : clusterOf rule O wt ord sc scd P cands k ω ⊆ cands :=
  fun v hv => screened_subset O sc scd P cands ω
    (rule.subset _ _ _ _ _ k (one_mem_screened O sc scd P cands ω hone) (Finset.mem_of_mem_erase
        hv))

lemma clusterOf_congr_mq (O : Oracle μ S) (wt : S → ℝ) (ord : S → ℕ) (sc scd : ℕ) (P cands : Finset
    S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterOf rule O wt ord sc scd P cands k ω = clusterOf rule O wt ord sc scd P cands k ω' := by
  classical
  have hscr : screened O.mq sc scd P cands ω = screened O.mq sc scd P cands ω' :=
    Finset.filter_congr (fun v hv => by
      rw [screenCount_congr O hone hv hbit, screenBase_congr O hone hbit])
  unfold clusterOf
  rw [hscr]
  exact congrArg (fun t : Finset S => t.erase 1) (rule.congr _ _ _ _ _ _ _
    (one_mem_screened O sc scd P cands ω' hone)
    (fun p hp v hv => hbit _ (mem_readSet hp (screened_subset O sc scd P cands ω' hv))))

open scoped Classical in
lemma measurableSet_clusterOf (O : Oracle μ S) (wt : S → ℝ) (ord : S → ℕ) (sc scd : ℕ) (P cands :
    Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | clusterOf rule O wt ord sc scd P cands k ω = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => O.mq w ω' = 1) = U ∧ clusterOf rule O wt ord sc scd P cands k ω' = A₀ with hPred
  have hcov : {ω | clusterOf rule O wt ord sc scd P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => O.mq w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (clusterOf_congr_mq O wt ord sc scd P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
lemma measurableSet_insert_clusterOf (O : Oracle μ S) (wt : S → ℝ) (ord : S → ℕ) (sc scd : ℕ) (P
    cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | insert 1 (clusterOf rule O wt ord sc scd P cands k ω) = A₀} := by
  classical
  have hcov : {ω | insert 1 (clusterOf rule O wt ord sc scd P cands k ω) = A₀}
      = ⋃ A ∈ cands.powerset.filter (fun A => insert 1 A = A₀),
          {ω | clusterOf rule O wt ord sc scd P cands k ω = A} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_filter, Finset.mem_powerset,
      exists_prop]
    exact ⟨fun h => ⟨_, ⟨clusterOf_subset O wt ord sc scd P cands k ω hone, h⟩, rfl⟩,
      fun ⟨A, ⟨_, hA⟩, hc⟩ => hc ▸ hA⟩
  rw [hcov]
  exact MeasurableSet.biUnion (Finset.countable_toSet _)
    (fun A _ => measurableSet_clusterOf O wt ord sc scd P cands k hone A)

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
theorem screen_tail_rel {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P cands : Finset S} (hP : ∀ p ∈ P, p ∈ Pre)
    (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (v : S) (hvS : v ∈ insert 1 Suf) (hv : v ≠ 1)
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
  have h1 := screenCount_upper hflat O hcd hP v (hV v hvc) hv γ hγ hsig hclean
  have h2 : μ.real (⋃ w ∈ cands.erase 1,
        {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)})
      ≤ (cands.card : ℝ) * Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    have hper : ∀ w ∈ cands.erase 1,
        μ.real {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)}
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
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ)) :
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
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
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
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
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
        have htail := screen_pass_rel (cands := Cd) hflat O Nat.one_lt_two hP hV v hv1 hvC γ B.sc B.scd hγ
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
theorem measureReal_smallScreen_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
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
        · exact measureReal_screenFail_le hflat O populations D Dsf hsupp hsuppSf B γ hγ hscd
            hsig hsc

open scoped Classical in
/-- Every member the clustering keeps is clean.  Three things have to go right: the
population's draws distinct, the screen holding, and the drawn prefixes not understating a
candidate's flip mass.  The clustering's ranking does not appear — the family is a subset of what
the screen left, so it inherits the bound. -/
theorem measureReal_dirtyMember_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : State) (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.npref)
    (Δ γ g ρ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hρ0 : 0 ≤ ρ)
    (hρD : collisionMass (D j) ≤ ρ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ)) :
    (runMeasure μ D Dsf).real ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
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
      ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
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
    have hrank := clusterAt_flip_bound (rule := rule) O populations B x Δ hscreen
    obtain ⟨v, hv, hbad⟩ : ∃ v ∈ clusterBy rule O.mq populations x B, ¬ (flipMass O (D j) v ≤ Δp) := by
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
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ Δp})
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ E3) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2) + (runMeasure μ D Dsf).real E3 := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h123 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) E3
        linarith
    _ ≤ _ := by
        gcongr
        · exact prefix_not_injective_le D Dsf j B.npref ρ hρD hρ0
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

lemma certOf_mono (j : J) {m m' : ℕ} (h : m ≤ m') (x : Run Ω S J) :
    certOf j m x ⊆ certOf j m' x := by
  intro p hp
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
  exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 (by have := Finset.mem_range.1 hi; omega), rfl⟩

lemma prefixesAt_mono (populations : Finset J) {m m' : ℕ} (h : m ≤ m') (x : Run Ω S J) :
    prefixesAt populations m x ⊆ prefixesAt populations m' x := by
  intro p hp
  obtain ⟨j, hj, hp'⟩ := Finset.mem_biUnion.1 hp
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp'
  exact Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_image.2
    ⟨i, Finset.mem_range.2 (by have := Finset.mem_range.1 hi; omega), rfl⟩⟩

lemma certOf_eq_image (j : J) (m : ℕ) (x : Run Ω S J) :
    certOf j m x = (Finset.univ : Finset (Fin m)).image (fun i => certPrefix j i.val x) :=
  image_range_eq_image_univ m (fun i => certPrefix j i x)

open scoped Classical in
/-- An event about the table, a certification sample and the family the round clusters is
measurable when it is at each fixed table, sample and family. -/
lemma measurableSet_certFam (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (Φ : Finset S → Finset S → Finset S → Ω → Prop)
    (hΦ : ∀ P A F, MeasurableSet {ω | Φ P A F ω}) :
    MeasurableSet {x : Run Ω S J | Φ (prefixesAt populations B.npref x) (certOf j B.npref x)
      (clusterBy rule O.mq populations x B) (oracleNoise x)} := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin B.npref → S), MeasurableSet (if
      (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω : Ω | Φ P ((Finset.univ : Finset (Fin B.npref)).image tt)
        (clusterOf rule O wt ord B.sc B.scd P C B.k ω) ω} else (∅ : Set (Run Ω S J))) := by
    intro P C wt ord tt
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | Φ P ((Finset.univ : Finset (Fin B.npref)).image tt) A₀ ω})
        (fun A₀ => hΦ _ _ _))
    · exact MeasurableSet.empty
  have hrw : {x : Run Ω S J | Φ (prefixesAt populations B.npref x) (certOf j B.npref x)
        (clusterBy rule O.mq populations x B) (oracleNoise x)}
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω : Ω | Φ P ((Finset.univ : Finset (Fin B.npref)).image tt)
            (clusterOf rule O wt ord B.sc B.scd P C B.k ω) ω} else (∅ : Set (Run Ω S J)))
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin B.npref) => certPrefix j i.val x)
    (measurableSet_certFiber j B.npref) _ hR

lemma measurableSet_const_and {c : Prop} {s : Ω → Prop} (h : MeasurableSet {ω | s ω}) :
    MeasurableSet {ω | c ∧ s ω} := by
  by_cases hc : c <;> simp [hc, h]

open scoped Classical in
/-- The certification sample misses a wrong cut.  The family is fixed before the sample
is drawn, so a cut wrong on `εcov` of the population is hit `εcov·m` times up to the
Hoeffding slack `t`. -/
noncomputable def hitShort (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (Dj : Measure S) (j : J)
    (B : State) (εcov t : ℝ) : Set (Run Ω S J) :=
  {x | Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
        (clusterBy rule O.mq populations x B) p (oracleNoise x)}
    ∧ (((certOf j B.npref x).filter (fun p =>
        ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
      ≤ (B.npref : ℝ) * (εcov - t)}

open scoped Classical in
lemma measurableSet_hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (εcov t : ℝ) :
    MeasurableSet (hitShort rule O populations Dj j B εcov t) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin B.npref → S), MeasurableSet (if
      (1 : S) ∈ C then
      {x : Run Ω S J | Function.Injective tt
        ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
            (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
        ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))
                p (oracleNoise x))).card
              : ℝ)
          ≤ (B.npref : ℝ) * (εcov - t)} else ∅) := by
    intro P C wt ord tt
    split_ifs with hone
    · by_cases hinj : Function.Injective tt
      · have hω : MeasurableSet {ω : Ω |
            εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω)).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
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
                (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise
                    x))
                    p (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)}
            = oracleNoise ⁻¹' {ω : Ω |
              εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                  (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi
                    (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω)).card : ℝ)
                ≤ (B.npref : ℝ) * (εcov - t)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hinj, true_and]
        rw [hset]
        exact measurable_nz hω
      · simp [hinj]
    · exact MeasurableSet.empty
  have hrw : hitShort rule O populations Dj j B εcov t
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise
                    x))
                    p (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), hitShort,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin B.npref) => certPrefix j i.val x)
    (measurableSet_certFiber j B.npref) _ hR

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
    MeasurableSet {ω | A.filter (fun p => hi < voteCount O.mq F p ω
        ∨ voteCount O.mq F p ω ≤ lo) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => hi < voteCount O.mq F p ω ∨ voteCount O.mq F p ω ≤ lo)
    (fun p _ => (measurableSet_voteCount_gt O F hi p).union
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
    (runMeasure μ D Dsf).real (hitShort rule O populations (D j) j B εcov t)
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (B.npref : ℝ) * t ^ 2) with hEdef
  have hEnn : runMeasure μ D Dsf (hitShort rule O populations (D j) j B εcov t) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_hitShort O populations (D j) j B εcov t) _ ?_
    intro y
    set F : Finset S := clusterBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B
      with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : εcov ≤ (D j).real W
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ hitShort rule O populations (D j) j B εcov t}
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
          ∈ hitShort rule O populations (D j) j B εcov t} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨-, hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (hitShort rule O populations (D j) j B εcov t)).toReal
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
noncomputable def heavyHits (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (f q t : ℝ) : Set (Run Ω S J) :=
  {x | Dj.real {p | ¬ (flipCount O (clusterBy rule O.mq populations x B) p
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)} ≤ q
    ∧ (B.npref : ℝ) * (q + t)
        ≤ (((certOf j B.npref x).filter (fun p =>
          ¬ (flipCount O (clusterBy rule O.mq populations x B) p
            ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)}

open scoped Classical in
lemma measurableSet_heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (f q t : ℝ) :
    MeasurableSet (heavyHits rule O populations Dj j B f q t) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin B.npref → S), MeasurableSet (if
      (1 : S) ∈ C then
      {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
            (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
          ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) * f)} ≤ q
        ∧ (B.npref : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              ¬ (flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) *
                    f))).card
                  : ℝ)} else ∅) := by
    intro P C wt ord tt
    split_ifs with hone
    · refine measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {_ω : Ω |
          Dj.real {p | ¬ (flipCount O A₀ p ≤ (A₀.card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
              ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ (flipCount O A₀ p ≤ (A₀.card : ℝ) * f))).card : ℝ)})
        (fun A₀ => ?_))
      by_cases hcond : Dj.real {p | ¬ (flipCount O A₀ p
            ≤ (A₀.card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              ¬ (flipCount O A₀ p ≤ (A₀.card : ℝ) * f))).card : ℝ)
      · simpa [hcond] using MeasurableSet.univ
      · simpa [hcond] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : heavyHits rule O populations Dj j B f q t
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
                (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
              ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) * f)} ≤ q
            ∧ (B.npref : ℝ) * (q + t)
                ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ (flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                    ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)).card : ℝ)
                      * f))).card : ℝ)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), heavyHits,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin B.npref) => certPrefix j i.val x)
    (measurableSet_certFiber j B.npref) _ hR

open scoped Classical in
/-- The flip-heavy prefixes are a `q` fraction of the population, so the sample sees at most
`m(q + t)` of them. -/
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
/-- The certification sample misses the prefixes the family's vote leaves undecided:
`hitShort` with the FNR test's event in place of a wrong cut. -/
noncomputable def undecidedShort (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (l t : ℝ) : Set (Run Ω S J) :=
  {x | Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    ∧ l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p (oracleNoise x)}
    ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
      ≤ (B.npref : ℝ) * (l - t)}

open scoped Classical in
lemma measurableSet_undecidedShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (l t : ℝ) :
    MeasurableSet (undecidedShort rule O populations Dj j B l t) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin B.npref → S), MeasurableSet (if
      (1 : S) ∈ C then
      {x : Run Ω S J | Function.Injective tt
        ∧ l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))) p (oracleNoise
                x)}
        ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
            ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k
                (oracleNoise x))) p
              (oracleNoise x))).card : ℝ)
          ≤ (B.npref : ℝ) * (l - t)} else ∅) := by
    intro P C wt ord tt
    split_ifs with hone
    · by_cases hinj : Function.Injective tt
      · have hω : MeasurableSet {ω : Ω |
            l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
                (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k ω)) p ω}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ decided O.mq B.lo (B.hi + 1)
                  (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k ω)) p ω)).card : ℝ)
              ≤ (B.npref : ℝ) * (l - t)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2
              (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω | l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                ≤ (B.npref : ℝ) * (l - t)})
            (fun A₀ => MeasurableSet.inter
              (measurableSet_le measurable_const
                (measurable_undecidedMassReal O Dj B.lo (B.hi + 1) (insert 1 A₀)))
              (measurableSet_indecisionCount O B.lo (B.hi + 1) (insert 1 A₀) _ _))
        have hset : {x : Run Ω S J | Function.Injective tt
            ∧ l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
                (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))) p
                (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ decided O.mq B.lo (B.hi + 1)
                  (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))) p
                  (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (l - t)}
            = oracleNoise ⁻¹' {ω : Ω |
              l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
                  (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k ω)) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ decided O.mq B.lo (B.hi + 1)
                    (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k ω)) p ω)).card : ℝ)
                ≤ (B.npref : ℝ) * (l - t)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hinj, true_and]
        rw [hset]
        exact measurable_nz hω
      · simp [hinj]
    · exact MeasurableSet.empty
  have hrw : undecidedShort rule O populations Dj j B l t
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Function.Injective tt
            ∧ l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
                (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))) p
                (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ decided O.mq B.lo (B.hi + 1)
                  (insert 1 (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x))) p
                  (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (l - t)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), undecidedShort, familyAt,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
    rfl
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin B.npref) => certPrefix j i.val x)
    (measurableSet_certFiber j B.npref) _ hR

open scoped Classical in
/-- `measureReal_hitShort_le` for the undecided set: the family is fixed before the
certification draws, so a large undecided mass is seen on the sample. -/
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
    set F : Finset S := familyBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B
      with hF
    set W : Set S := {p | ¬ decided O.mq B.lo (B.hi + 1) F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        familyBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : l ≤ (D j).real W
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ undecidedShort rule O populations (D j) j B l t}
          ⊆ {c | (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
            ≤ (B.npref : ℝ) * (l - t)} := by
        rintro c ⟨hinj, -, hcount⟩
        change (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (B.npref : ℝ) * (l - t)
        calc (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
            = (((certOf j B.npref ((y.1, (y.2, c)) : Run Ω S J)).filter
                (fun p => ¬ decided O.mq B.lo (B.hi + 1) F p y.1)).card : ℝ) :=
              congrArg (fun n : ℕ => (n : ℝ)) (card_filter_certOf j B.npref
                ((y.1, (y.2, c)) : Run Ω S J)
                (fun p => ¬ decided O.mq B.lo (B.hi + 1) F p y.1) _ _ hinj)
          _ ≤ (B.npref : ℝ) * (l - t) := hcount
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
        rintro ⟨-, hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (undecidedShort rule O populations (D j) j B l t)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
lemma measurableSet_gateFixed (O : Oracle μ S) (lo hi : ℕ) (α : ℝ) (F A : Finset S) :
    MeasurableSet {ω | certified O.mq lo hi α F A ω
      ∧ (∃ p ∈ A, hi + 1 < voteCount O.mq (insert 1 F) p ω)
      ∧ (∃ p ∈ A, voteCount O.mq (insert 1 F) p ω ≤ lo)} := by
  classical
  have hgt : ∀ (G : Finset S) (n : ℕ), ∀ p ∈ A,
      MeasurableSet[noiseAlg O Set.univ] {ω | n < voteCount O.mq G p ω} :=
    fun G n p _ => measurableSet_voteCount_gt O G n p
  have hle : ∀ (G : Finset S) (n : ℕ), ∀ p ∈ A,
      MeasurableSet[noiseAlg O Set.univ] {ω | voteCount O.mq G p ω ≤ n} :=
    fun G n p _ => measurableSet_voteCount_le O G n p
  have hone : ∀ p ∈ A, MeasurableSet[noiseAlg O Set.univ] {ω | O.mq p ω = 1} :=
    fun p _ => measurableSet_mq_eq_one O (Set.mem_univ p)
  have hex : ∀ (Pr : S → Ω → Prop), (∀ p ∈ A, MeasurableSet[noiseAlg O Set.univ] {ω | Pr p ω}) →
      MeasurableSet {ω | ∃ p ∈ A, Pr p ω} := by
    intro Pr hPr
    have h := measurableSet_filter_pred' O Pr hPr (fun U => U.Nonempty)
    simp only [Finset.filter_nonempty_iff] at h
    exact noiseAlg_le O Set.univ _ h
  simp only [certified, Set.setOf_and]
  refine MeasurableSet.inter (MeasurableSet.inter ?_ (MeasurableSet.inter ?_ ?_))
    (MeasurableSet.inter (hex _ (hgt _ _)) (hex _ (hle _ _)))
  · exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred₂ O A _ _ (hgt F hi) (hle F lo)
      (fun U V => U.Nonempty ∨ V.Nonempty))
  · exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred₂ O A _
      (fun p ω => hi < voteCount O.mq F p ω ∧ O.mq p ω = 1) (hgt F hi)
      (fun p hp => (hgt F hi p hp).inter (hone p hp))
      (fun U V => U.Nonempty → binomSfGe U.card (((hi : ℝ) + 1) / F.card) V.card ≤ α))
  · exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred₂ O A _
      (fun p ω => voteCount O.mq F p ω ≤ lo ∧ O.mq p ω = 1) (hle F lo)
      (fun p hp => (hle F lo p hp).inter (hone p hp))
      (fun U V => U.Nonempty → binomCdfLe U.card (((lo : ℝ) + 1) / (F.card + 1)) V.card ≤ α))

/-- The family is usable at `p`: at most an `f` fraction of it flips there, and the family
is neither too small for the thresholds nor larger than the round allows. -/
def famGood (O : Oracle μ S) (f : ℝ) (kmin kmax : ℕ) (F : Finset S) (p : S) : Prop :=
  flipCount O F p ≤ (F.card : ℝ) * f ∧ kmin ≤ F.card ∧ F.card ≤ kmax

open scoped Classical in
/-- The draws were good and the FNR test still failed.  Everything the test needs of its draws
— the certification prefixes fresh and nonempty, and the family light on all but an `lcut`
fraction of them — holds.  This is the event `ret_at_whp` prices. -/
noncomputable def retMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (l lcut f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | (Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
      ∧ (certOf j B.npref x).card = B.npref
      ∧ 0 < (certOf j B.npref x).card
      ∧ (((certOf j B.npref x).filter (fun p =>
          ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
          ≤ lcut * ((certOf j B.npref x).card : ℝ))
    ∧ ¬ ((((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (clusterBy rule O.mq populations x B)) p (oracleNoise x))).card : ℝ)
          ≤ 2 * l * ((certOf j B.npref x).card : ℝ))}

open scoped Classical in
lemma measurableSet_retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (l lcut f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (retMiss rule O populations j B l lcut f kmin kmax) :=
  measurableSet_certFam O populations j B (fun P A F ω =>
    (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card
      ∧ ((A.filter (fun p => ¬ famGood O f kmin kmax F p)).card : ℝ) ≤ lcut * (A.card : ℝ))
    ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω)).card : ℝ)
      ≤ 2 * l * (A.card : ℝ)))
    (fun _ A F => measurableSet_const_and (measurableSet_indecisionCount O B.lo (B.hi + 1)
      (insert 1 F) A (2 * l * (A.card : ℝ))).compl)

open scoped Classical in
/-- The draws were good and the gate still refused: the uniform pool's sample is fresh, holds a
`q` fraction of each class, and the family is light on all but an `lcut` fraction of it. -/
noncomputable def gateMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (B : State) (α q lcut f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | (Disjoint (prefixesAt populations B.npref x) (certOf uni B.npref x)
      ∧ (certOf uni B.npref x).card = B.npref
      ∧ q * ((certOf uni B.npref x).card : ℝ)
        ≤ (((certOf uni B.npref x).filter (fun p => p ∈ O.L)).card : ℝ)
      ∧ q * ((certOf uni B.npref x).card : ℝ)
        ≤ (((certOf uni B.npref x).filter (fun p => p ∉ O.L)).card : ℝ)
      ∧ (((certOf uni B.npref x).filter (fun p =>
          ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
          ≤ lcut * ((certOf uni B.npref x).card : ℝ))
    ∧ ¬ (certified O.mq B.lo B.hi α (clusterBy rule O.mq populations x B) (certOf uni B.npref x)
          (oracleNoise x)
        ∧ (∃ p ∈ certOf uni B.npref x, B.hi + 1
            < voteCount O.mq (insert 1 (clusterBy rule O.mq populations x B)) p (oracleNoise x))
        ∧ (∃ p ∈ certOf uni B.npref x, voteCount O.mq
            (insert 1 (clusterBy rule O.mq populations x B)) p (oracleNoise x) ≤ B.lo))}

open scoped Classical in
lemma measurableSet_gateMiss (O : Oracle μ S) (populations : Finset J) (uni : J) (B : State)
    (α q lcut f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (gateMiss rule O populations uni B α q lcut f kmin kmax) :=
  measurableSet_certFam O populations uni B (fun P A F ω =>
    (Disjoint P A ∧ A.card = B.npref
      ∧ q * (A.card : ℝ) ≤ ((A.filter (fun p => p ∈ O.L)).card : ℝ)
      ∧ q * (A.card : ℝ) ≤ ((A.filter (fun p => p ∉ O.L)).card : ℝ)
      ∧ ((A.filter (fun p => ¬ famGood O f kmin kmax F p)).card : ℝ) ≤ lcut * (A.card : ℝ))
    ∧ ¬ (certified O.mq B.lo B.hi α F A ω
      ∧ (∃ p ∈ A, B.hi + 1 < voteCount O.mq (insert 1 F) p ω)
      ∧ (∃ p ∈ A, voteCount O.mq (insert 1 F) p ω ≤ B.lo)))
    (fun _ A F => measurableSet_const_and (measurableSet_gateFixed O B.lo B.hi α F A).compl)

open scoped Classical in
/-- The lift of `ret_at_whp`: at every table the FNR test fails only as often as the
undecided count allows; the draws are charged elsewhere. -/
theorem measureReal_retMiss_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (l lcut f E : ℝ) (kmin kmax : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E) :
    (runMeasure μ D Dsf).real (retMiss rule O populations j B l lcut f kmin kmax)
      ≤ Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2) := by
  classical
  set R : ℝ := Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2) with hR
  have hEnn : runMeasure μ D Dsf (retMiss rule O populations j B l lcut f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_retMiss O populations j B l lcut f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
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
    set fam : Ω → Finset S :=
      fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
    set T : Finset (Finset S) := (Cd.erase 1).powerset with hT
    set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
      flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
    have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
      (fun v hv => Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
        clusterAt_subset O populations B _ hv⟩)
    have hT1 : ∀ t ∈ T, (1 : S) ∉ t := fun t ht h1 =>
      (Finset.mem_erase.1 (Finset.mem_powerset.1 ht h1)).1 rfl
    have hlightEq : ∀ ω, ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ)
        = ((Ad.filter (fun p => ¬ famGood O f kmin kmax (fam ω) p)).card : ℝ) := by
      intro ω
      refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
      intro p _
      simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
      exact Iff.rfl
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref
    · obtain ⟨hdisj, hAdm⟩ := hdraw
      have hmain := ret_at_whp hflat O Pd Cd Ad hP hV hA hdisj B.lo B.hi l lcut T good ∅
        (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => (Finset.mem_powerset.1 ht).trans (Finset.erase_subset _ _))
        (Finset.mem_insert_self _ _) hT1 fam hfamT
        (fun ω ω' h => clusterAt_congr O populations B d h)
        E hE hElcut hlcl
        (fun p _ A₀ hA₀ hgp => hdec A₀ p (hT1 A₀ hA₀) (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ retMiss rule O populations j B l lcut f kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (Ad.card : ℝ)
            ∧ ¬ (((Ad.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
                  ≤ 2 * l * (Ad.card : ℝ))} := by
        rintro ω ⟨⟨-, -, -, hlight⟩, hbad⟩
        exact ⟨(hlightEq ω).trans_le hlight, hbad⟩
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq (by rw [hcardR])))
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ retMiss rule O populations j B l lcut f kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨⟨h1, h2, -⟩, -⟩
        exact hdraw ⟨h1, h2⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (retMiss rule O populations j B l lcut f kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
/-- The lift of `gate_at_whp`, at the uniform pool. -/
theorem measureReal_gateMiss_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (α τ q lcut f E : ℝ) (kmin kmax n₁ : ℕ) (hkk : kmax ≤ kmin)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E)
    (hlcut1 : lcut < 1)
    (hn₁ : (n₁ : ℝ) ≤ q * (B.npref : ℝ) - 3 * lcut * (B.npref : ℝ))
    (hn₁pos : 0 < n₁) (hκ : 0 < kmin) (hlohi : B.lo + 1 ≤ kmin) (hhi : B.hi + 1 ≤ kmin)
    (hga : ∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
      (n : ℝ) * ((((B.hi : ℝ) + 1) / kmin + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηIn) - (1 - O.ηIn) * (2 * lcut * (B.npref : ℝ)))
    (hgr : ∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
      (n : ℝ) * (((1 - ((B.lo : ℝ) + 1) / (kmin + 1)) + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηOut) - (1 - O.ηOut) * (2 * lcut * (B.npref : ℝ)))
    (hα : Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real (gateMiss rule O populations j B α q lcut f kmin kmax)
      ≤ 2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
        + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := by
  classical
  set R : ℝ := 2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
    + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : runMeasure μ D Dsf (gateMiss rule O populations j B α q lcut f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_gateMiss O populations j B α q lcut f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
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
    set fam : Ω → Finset S :=
      fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
    set T : Finset (Finset S) := (Cd.erase 1).powerset with hT
    set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
      flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
    have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
      (fun v hv => Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
        clusterAt_subset O populations B _ hv⟩)
    have hT1 : ∀ t ∈ T, (1 : S) ∉ t := fun t ht h1 =>
      (Finset.mem_erase.1 (Finset.mem_powerset.1 ht h1)).1 rfl
    have hlightEq : ∀ ω, ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ)
        = ((Ad.filter (fun p => ¬ famGood O f kmin kmax (fam ω) p)).card : ℝ) := by
      intro ω
      refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
      intro p _
      simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
      exact Iff.rfl
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref
        ∧ q * (Ad.card : ℝ) ≤ ((Ad.filter (fun p => p ∈ O.L)).card : ℝ)
        ∧ q * (Ad.card : ℝ) ≤ ((Ad.filter (fun p => p ∉ O.L)).card : ℝ)
    · obtain ⟨hdisj, hAdm, hCL, hCN⟩ := hdraw
      have hmain := gate_at_whp hflat O Pd Cd Ad hP hV hA hdisj
        (readSet Pd Cd ∪ readSet Ad (Cd.erase 1)) Finset.subset_union_left
        (disjoint_gateReads hflat Pd Cd Ad hP hV hA hdisj)
        B.lo B.hi kmin α lcut τ q n₁ T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => (Finset.mem_powerset.1 ht).trans (Finset.erase_subset _ _))
        (Finset.mem_insert_self _ _) hT1
        (fun p A₀ hA₀ => le_antisymm (le_trans (Finset.mem_filter.1 hA₀).2.2.2 hkk)
          (Finset.mem_filter.1 hA₀).2.2.1)
        fam hfamT (fun ω ω' h => clusterAt_congr O populations B d h)
        (fun ω p hp v hv => Finset.mem_union_right _ (mem_readSet hp
          (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, clusterAt_subset O populations B _ hv⟩)))
        E hE hElcut hτ hsig
        (fun p _ A₀ hA₀ hgp => hdec A₀ p (hT1 A₀ hA₀) (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        hCL hCN hlcut1 (by rw [hAdm]; exact hn₁) hn₁pos hκ hlohi hhi
        (fun n h1 h2 => by rw [hAdm] at h2 ⊢; exact hga n h1 h2)
        (fun n h1 h2 => by rw [hAdm] at h2 ⊢; exact hgr n h1 h2) hα
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ gateMiss rule O populations j B α q lcut f kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (Ad.card : ℝ)
            ∧ ¬ (certified O.mq B.lo B.hi α (fam ω) Ad ω
              ∧ (∃ p ∈ Ad, B.hi + 1 < voteCount O.mq (insert 1 (fam ω)) p ω)
              ∧ (∃ p ∈ Ad, voteCount O.mq (insert 1 (fam ω)) p ω ≤ B.lo))} := by
        rintro ω ⟨⟨-, -, -, -, hlight⟩, hbad⟩
        exact ⟨(hlightEq ω).trans_le hlight, hbad⟩
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq ?_))
      rw [hR, hAdm]
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ gateMiss rule O populations j B α q lcut f kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨⟨h1, h2, h3, h4, -⟩, -⟩
        exact hdraw ⟨h1, h2, h3, h4⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (gateMiss rule O populations j B α q lcut f kmin kmax)).toReal
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
noncomputable def validMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (lcut f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
    ∧ (certOf j B.npref x).card = B.npref
    ∧ lcut * ((certOf j B.npref x).card : ℝ)
      < (((certOf j B.npref x).filter (fun p =>
          famGood O f kmin kmax (clusterBy rule O.mq populations x B) p
          ∧ ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
            (oracleNoise x))).card : ℝ)}

open scoped Classical in
lemma measurableSet_validMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (lcut f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (validMiss rule O populations j B lcut f kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin B.npref → S), MeasurableSet (if
      (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
        ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
        ∧ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
          < ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              famGood O f kmin kmax (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
              ∧ ¬ cutCorrect O B.lo B.hi
                (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                (oracleNoise x))).card : ℝ)} else ∅) := by
    intro P C wt ord tt
    split_ifs with hone
    · set A : Finset S := (Finset.univ : Finset (Fin B.npref)).image tt with hA
      by_cases hdraw : Disjoint P A ∧ A.card = B.npref
      · have hω : MeasurableSet {ω : Ω | lcut * (A.card : ℝ)
            < ((A.filter (fun p => famGood O f kmin kmax
                (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p
              ∧ ¬ cutCorrect O B.lo B.hi
                (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω)).card : ℝ)} :=
          measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2
              (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω : Ω | lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax A₀ p
                ∧ ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)})
            (fun A₀ => measurableSet_miscutCount O f kmin kmax B.lo B.hi A₀ A
              (lcut * (A.card : ℝ)))
        have hset : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref
            ∧ lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                ∧ ¬ cutCorrect O B.lo B.hi
                  (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                  (oracleNoise x))).card : ℝ)}
            = oracleNoise ⁻¹' {ω : Ω | lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p
                ∧ ¬ cutCorrect O B.lo B.hi
                  (clusterOf rule O wt ord B.sc B.scd P C B.k ω) p ω)).card : ℝ)} := by
          ext x
          exact ⟨fun h => h.2.2, fun h => ⟨hdraw.1, hdraw.2, h⟩⟩
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref
            ∧ lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                ∧ ¬ cutCorrect O B.lo B.hi
                  (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                  (oracleNoise x))).card : ℝ)} = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro ⟨h1, h2, -⟩
          exact hdraw ⟨h1, h2⟩
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : validMiss rule O populations j B lcut f kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
            ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
            ∧ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
              < ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  famGood O f kmin kmax (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise
                      x)) p
                  ∧ ¬ cutCorrect O B.lo B.hi
                    (clusterOf rule O wt ord B.sc B.scd P C B.k (oracleNoise x)) p
                    (oracleNoise x))).card : ℝ)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), validMiss,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin B.npref) => certPrefix j i.val x)
    (measurableSet_certFiber j B.npref) _ hR

open scoped Classical in
/-- Hoeffding over the certification sample: the family is chosen from the table's reads, the
sample's reads are at other strings, so the light prefixes misfire independently, each at most
`E`. -/
theorem measureReal_validMiss_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (lcut f E : ℝ) (kmin kmax : ℕ) (hE : 0 ≤ E) (hElcut : E ≤ lcut)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real (validMiss rule O populations j B lcut f kmin kmax)
      ≤ Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2) := by
  classical
  set R : ℝ := Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2) with hR
  have hR0 : 0 ≤ R := Real.exp_nonneg _
  have hEnn : runMeasure μ D Dsf (validMiss rule O populations j B lcut f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_validMiss O populations j B lcut f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
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
        fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
      set T : Finset (Finset S) := Cd.powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => clusterAt_subset O populations B _ hv)
      have hmain := miscut_frac_le hflat O Pd Cd Ad hP hV hA hdisj B.lo B.hi T good ∅
        (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => Finset.mem_powerset.1 ht) fam hfamT
        (fun ω ω' h => clusterAt_congr O populations B d h)
        E lcut hE hElcut
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ validMiss rule O populations j B lcut f kmin kmax}
          ⊆ {ω : Ω | lcut * (Ad.card : ℝ) < ((Ad.filter (fun p => fam ω ∈ good p
              ∧ ¬ cutCorrect O B.lo B.hi (fam ω) p ω)).card : ℝ)} := by
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
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ validMiss rule O populations j B lcut f kmin kmax}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, -⟩
        exact hdraw ⟨h1, h2⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (validMiss rule O populations j B lcut f kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state and one population.  A family of the round's size that is wrong on
more than `εcov` of the population is seen to be by the certification sample, and the wrong
prefixes the sample holds are then either ones the family flips — few, once no member flips
much — or ones where the vote of a barely-flipped family misfires — few, by Hoeffding.  One
that leaves more than `2·indecisionLimit` undecided is seen to by the sample the FNR test
counted at `indecisionLimit`. -/
theorem measureReal_validFail_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (hkpos : 0 < B.k)
    (indecisionLimit εcov lcut f E Δp ρ th t : ℝ) (hil : 0 ≤ indecisionLimit) (hε0 : 0 ≤ εcov)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (ht : 0 ≤ t) (hf0 : 0 < f)
    (hbudget : Δp / f + th + lcut ≤ εcov - t)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real
        ({x : Run Ω S J | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
            ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
                  (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
                ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
            ∧ ¬ (1 - εcov ≤ (D j).real
              {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
              ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1)
                  (familyBy rule O.mq populations x B) p (oracleNoise x)}
                ≤ 2 * indecisionLimit)}
          \ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ≤ ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)} with hE2
  set E3 : Set (Run Ω S J) := hitShort rule O populations (D j) j B εcov t with hE3
  set E7 : Set (Run Ω S J) := heavyHits rule O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := validMiss rule O populations j B lcut f (B.k - 1) (B.k - 1) with hE8
  set E9 : Set (Run Ω S J) :=
    undecidedShort rule O populations (D j) j B (2 * indecisionLimit) indecisionLimit with hE9
  have hsub : ({x : Run Ω S J | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
        ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
              (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
            ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
        ∧ ¬ (1 - εcov ≤ (D j).real
          {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
          ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1)
              (familyBy rule O.mq populations x B) p (oracleNoise x)}
            ≤ 2 * indecisionLimit)}
      \ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ⊆ (E1 ∪ E2) ∪ (E3 ∪ (E7 ∪ (E8 ∪ E9))) := by
    rintro x ⟨⟨hsize, hfnr, hboth⟩, hclean⟩
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
    by_cases hcov : 1 - εcov ≤ (D j).real
        {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
    · refine Or.inr (Or.inr (Or.inr (Or.inr ⟨h1, (not_le.1 (fun h => hboth ⟨hcov, h⟩)).le, ?_⟩)))
      calc _ ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ) := hfnr
        _ = (B.npref : ℝ) * (2 * indecisionLimit - indecisionLimit) := by rw [hcard]; ring
    have hfail := hcov
    -- the family is exactly the round's size
    have hle : (clusterFull rule O populations x B).card ≤ B.k :=
      rule.card_le _ _ _ _ _ B.k (poolOrder_injOn_screened O populations B x) hkpos
    have herase : (clusterBy rule O.mq populations x B).card = B.k - 1 := by
      have := clusterAt_card_add_one (rule := rule) O populations B x
      omega
    have hpop : εcov ≤ (D j).real
        {p | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)} := by
      rw [show {p : S | ¬ cutCorrect O B.lo B.hi
            (clusterBy rule O.mq populations x B) p (oracleNoise x)}
          = {p : S | cutCorrect O B.lo B.hi
            (clusterBy rule O.mq populations x B) p (oracleNoise x)}ᶜ from rfl,
        measureReal_compl (measurableSet_of_countable _), measureReal_def, measure_univ,
        ENNReal.toReal_one]
      push_neg at hfail
      linarith
    by_cases hshort : (((certOf j B.npref x).filter (fun p =>
          ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ (B.npref : ℝ) * (εcov - t)
    · exact Or.inr (Or.inl ⟨h1, hpop, hshort⟩)
    push_neg at hshort
    -- the mass the family flips, by Markov over its members
    have hmass : (D j).real {p | ¬ (flipCount O (clusterBy rule O.mq populations x B) p
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)} ≤ Δp / f := by
      set F : Finset S := clusterBy rule O.mq populations x B with hFdef
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
          (fun v hv => hclean v hv)
    by_cases h7 : (B.npref : ℝ) * (Δp / f + th)
        ≤ (((certOf j B.npref x).filter (fun p =>
          ¬ (flipCount O (clusterBy rule O.mq populations x B) p
            ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)
    · exact Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩))
    replace h7 := not_le.1 h7
    refine Or.inr (Or.inr (Or.inr (Or.inl ⟨h2, ?_, ?_⟩)))
    · exact_mod_cast hcard
    · -- every wrong prefix is heavy, or light and misread by the vote
      have hcover : ((certOf j B.npref x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)))
          ⊆ ((certOf j B.npref x).filter (fun p =>
              ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)))
            ∪ ((certOf j B.npref x).filter (fun p =>
              famGood O f (B.k - 1) (B.k - 1) (clusterBy rule O.mq populations x B) p
              ∧ ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
                (oracleNoise x))) := by
        intro p hp
        obtain ⟨hpC, hbad⟩ := Finset.mem_filter.1 hp
        by_cases hh : flipCount O (clusterBy rule O.mq populations x B) p
            ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f
        · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, ⟨hh, ?_, ?_⟩, ?_⟩)
          · rw [herase]
          · rw [herase]
          · exact hbad
        · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hh⟩)
      have hcnt := le_trans (Finset.card_le_card hcover) (Finset.card_union_le _ _)
      have hcntR : (((certOf j B.npref x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
          ≤ (((certOf j B.npref x).filter (fun p =>
              ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)
            + (((certOf j B.npref x).filter (fun p =>
              famGood O f (B.k - 1) (B.k - 1) (clusterBy rule O.mq populations x B) p
              ∧ ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
                (oracleNoise x))).card : ℝ) := by exact_mod_cast hcnt
      rw [hcard]
      have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
      have hb := mul_le_mul_of_nonneg_left hbudget hm0
      linarith [hshort, h7, hcntR, hb]
  calc (runMeasure μ D Dsf).real
        ({x : Run Ω S J | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
            ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
                  (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
                ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
            ∧ ¬ (1 - εcov ≤ (D j).real
              {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
              ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1)
                  (familyBy rule O.mq populations x B) p (oracleNoise x)}
                ≤ 2 * indecisionLimit)}
          \ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ (E3 ∪ (E7 ∪ (E8 ∪ E9)))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
        + ((runMeasure μ D Dsf).real E3 + ((runMeasure μ D Dsf).real E7
          + ((runMeasure μ D Dsf).real E8 + (runMeasure μ D Dsf).real E9))) := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h89 := measureReal_union_le (μ := runMeasure μ D Dsf) E8 E9
        have h789 := measureReal_union_le (μ := runMeasure μ D Dsf) E7 (E8 ∪ E9)
        have h3789 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 (E7 ∪ (E8 ∪ E9))
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2)
          (E3 ∪ (E7 ∪ (E8 ∪ E9)))
        linarith
    _ ≤ ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.npref : ℝ) ^ 2 * ρ)
        + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.npref ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.npref ρ hρ (hρ j hj) hρ0
        · exact measureReal_hitShort_le D Dsf O populations j B εcov t hε0 ht
        · exact measureReal_heavyHits_le D Dsf O populations j B f (Δp / f) th hth
        · exact measureReal_validMiss_le hflat O populations D Dsf hsupp hsuppSf j hj B lcut f E
            (B.k - 1) (B.k - 1) hE hElcut hcut
        · exact measureReal_undecidedShort_le D Dsf O populations j B (2 * indecisionLimit)
            indecisionLimit (by linarith) hil
    _ = ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))) := by ring

open scoped Classical in
/-- The gate on a sample, and the FNR's rule that the family decide some prefix each way there. -/
def gateOK (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (α : ℝ) (B : State)
    (C : Finset S) (x : Run Ω S J) : Prop :=
  certified O.mq B.lo B.hi α (clusterBy rule O.mq populations x B) C (oracleNoise x)
    ∧ (∃ p ∈ C, B.hi + 1 < voteCount O.mq (familyBy rule O.mq populations x B) p (oracleNoise x))
    ∧ (∃ p ∈ C, voteCount O.mq (familyBy rule O.mq populations x B) p (oracleNoise x) ≤ B.lo)

open scoped Classical in
/-- The round's own test at one population: the family's size, the FNR count, and at the
uniform pool the gate. -/
noncomputable def retAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (indecisionLimit α : ℝ) (nveto : ℕ) (B : State) (j : J) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
    ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
    ∧ (j = uni → gateOK rule O populations α B (certOf uni B.npref x) x
      ∧ certified O.mq B.lo B.hi α (clusterBy rule O.mq populations x B)
        (gateOf uni B.npref 0 x) (oracleNoise x))
    ∧ Disjoint (prefixesAt populations B.npref x) (certOf j (2 * B.npref + nveto) x)
    ∧ (j ≠ uni → ∀ p ∈ vetoOf j B.npref nveto x,
        cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x))}

lemma mem_ret_of_retAt (O : Oracle μ S) (populations : Finset J) (huni : uni ∈ populations)
    (indecisionLimit α : ℝ) (B : State) (x : Run Ω S J)
    (h : ∀ j ∈ populations, x ∈ retAt rule O populations uni indecisionLimit α nveto B j)
    (hdrift : noDrift O.mq populations uni B.lo B.hi α (clusterBy rule O.mq populations x B)
      B.npref nveto 0 x) :
    x ∈ retBy rule O.mq populations uni indecisionLimit α nveto B.npref B := by
  obtain ⟨hsize, -, hgate, -⟩ := h uni huni
  obtain ⟨⟨-, ⟨p, hp, hacc⟩, ⟨q, hq, hrej⟩⟩, hcertG⟩ := hgate rfl
  exact ⟨hsize, fun j hj => (h j hj).2.1, ⟨uni, huni, p, hp, hacc⟩, ⟨uni, huni, q, hq, hrej⟩,
    hdrift, 0, hcertG, hdrift⟩

open scoped Classical in
/-- The runs where the clustering stalls on the seed, or overshoots the round's size.  The
gate refuses a stalled family through whichever side the population populates, so this is the liveness
half's obligation, not the round's. -/
noncomputable def stalled (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (B : State)
    (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | ¬ (kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)}

open scoped Classical in
/-- A pool of `k` screened candidates is a cluster of `k`.  The seed is one of them and
the ranking keeps it, so the family has `k − 1` members. -/
lemma stalled_subset (O : Oracle μ S) (populations : Finset J) (B : State) (hkpos : 0 < B.k) :
    stalled rule O populations B (B.k - 1) (B.k - 1)
      ⊆ {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)} := by
  intro x hx
  by_contra hk
  simp only [Set.mem_setOf_eq, Classical.not_not] at hk
  have hcard : (clusterFull rule O populations x B).card = B.k :=
    rule.card_eq _ _ _ _ _ B.k (prefixWeight_nonneg populations B.npref x)
      (poolOrder_injOn_screened O populations B x) (one_mem_screenedAt O populations B x) hk hkpos
  have herase : (clusterBy rule O.mq populations x B).card = B.k - 1 := by
    have := clusterAt_card_add_one (rule := rule) O populations B x
    omega
  exact hx ⟨le_of_eq herase.symm, le_of_eq herase⟩

open scoped Classical in
/-- The liveness half's obligation, discharged: the clustering has a family of the round's
own size except on the events `measureReal_smallScreen_le` prices. -/
theorem measureReal_stalled_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real (stalled rule O populations B (B.k - 1) (B.k - 1))
      ≤ (B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + ((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations D Dsf hsupp hsuppSf B j₀ hj₀ γ pAP t ρsf ρ
      hγ hpAP0 ht hpAPBound hscd hsig hsc hcount hρsf hρsf0 hρj hρ0)

open scoped Classical in
/-- A veto draw the family, clean and of its round's size, cuts the wrong way. -/
noncomputable def vetoBad (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (Dj : Measure S) (j : J) (B : State) (nveto : ℕ) (Δp : ℝ) (kmin kmax : ℕ) :
    Set (Run Ω S J) :=
  {x | (kmin ≤ (clusterBy rule O.mq populations x B).card
        ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)
    ∧ (∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp)
    ∧ Disjoint (prefixesAt populations B.npref x) (certOf j (B.npref + nveto) x)
    ∧ ∃ p ∈ vetoOf j B.npref nveto x,
        ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}

/-- One certification coordinate has its population's law. -/
lemma certCoord_real (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)] (j : J) (k : ℕ)
    (W : Set S) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j).real {c | c j k ∈ W}
      = (D j).real W := by
  have hmp : MeasurePreserving (fun c : J → ℕ → S => c j k)
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (D j) :=
    (measurePreserving_eval_infinitePi (fun _ : ℕ => D j) k).comp
      (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j)
  rw [measureReal_def, measureReal_def,
    show {c : J → ℕ → S | c j k ∈ W} = (fun c : J → ℕ → S => c j k) ⁻¹' W from rfl,
    ← Measure.map_apply hmp.measurable (measurableSet_of_countable W), hmp.map_eq]

open scoped Classical in
/-- The family is clean and flips heavily at the `k`-th certification draw. -/
noncomputable def vetoHeavy (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (Dj : Measure S) (j : J) (B : State) (k : ℕ) (Δp f : ℝ) : Set (Run Ω S J) :=
  {x | (∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp)
    ∧ ¬ (flipCount O (clusterBy rule O.mq populations x B) (certPrefix j k x)
      ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)}

open scoped Classical in
/-- The family is of its round's size and light at the `k`-th certification draw, which is
fresh, and still cuts it the wrong way. -/
noncomputable def vetoLight (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (j : J) (B : State) (m k : ℕ) (f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x) (certOf j m x)
    ∧ (kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)
    ∧ flipCount O (clusterBy rule O.mq populations x B) (certPrefix j k x)
      ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f
    ∧ ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) (certPrefix j k x)
      (oracleNoise x)}

open scoped Classical in
lemma measurableSet_vetoHeavy (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (k : ℕ) (Δp f : ℝ) :
    MeasurableSet (vetoHeavy rule O populations Dj j B k Δp f) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin (k + 1) → S), MeasurableSet (if
      (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω : Ω |
        (∀ v ∈ clusterOf rule O wt ord B.sc B.scd P C B.k ω, flipMass O Dj v ≤ Δp)
        ∧ ¬ (flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt (Fin.last k))
          ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k ω).card : ℝ) * f)}
      else (∅ : Set (Run Ω S J))) := by
    intro P C wt ord tt
    split_ifs with hone
    · refine measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | (∀ v ∈ A₀, flipMass O Dj v ≤ Δp)
          ∧ ¬ (flipCount O A₀ (tt (Fin.last k)) ≤ (A₀.card : ℝ) * f)})
        (fun A₀ => ?_))
      exact MeasurableSet.const _
    · exact MeasurableSet.empty
  have hrw : vetoHeavy rule O populations Dj j B k Δp f
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω : Ω |
            (∀ v ∈ clusterOf rule O wt ord B.sc B.scd P C B.k ω, flipMass O Dj v ≤ Δp)
            ∧ ¬ (flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt (Fin.last k))
              ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k ω).card : ℝ) * f)}
          else (∅ : Set (Run Ω S J)))
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin (k + 1) => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage,
      vetoHeavy, ← clusterAt_eq_clusterOf O populations B x, Fin.val_last]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin (k + 1)) => certPrefix j i.val x)
    (measurableSet_certFiber j (k + 1)) _ hR

open scoped Classical in
lemma measurableSet_vetoLight (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (m k : ℕ) (hk : k < m) (f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (vetoLight rule O populations j B m k f kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (tt : Fin m → S), MeasurableSet (if (1 :
      S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin m)).image tt)}
      ∩ oracleNoise ⁻¹' {ω : Ω |
        (kmin ≤ (clusterOf rule O wt ord B.sc B.scd P C B.k ω).card
          ∧ (clusterOf rule O wt ord B.sc B.scd P C B.k ω).card ≤ kmax)
        ∧ flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt ⟨k, hk⟩)
          ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k ω).card : ℝ) * f
        ∧ ¬ cutCorrect O B.lo B.hi (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt ⟨k, hk⟩) ω}
      else (∅ : Set (Run Ω S J))) := by
    intro P C wt ord tt
    split_ifs with hone
    · refine MeasurableSet.inter ?_ (measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | (kmin ≤ A₀.card ∧ A₀.card ≤ kmax)
          ∧ flipCount O A₀ (tt ⟨k, hk⟩) ≤ (A₀.card : ℝ) * f
          ∧ ¬ cutCorrect O B.lo B.hi A₀ (tt ⟨k, hk⟩) ω})
        (fun A₀ => measurableSet_const_and (measurableSet_const_and
          (measurableSet_cutCorrect O B.lo B.hi A₀ _)))))
      exact MeasurableSet.const _
    · exact MeasurableSet.empty
  have hrw : vetoLight rule O populations j B m k f kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C wt ord tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin m)).image tt)}
          ∩ oracleNoise ⁻¹' {ω : Ω |
            (kmin ≤ (clusterOf rule O wt ord B.sc B.scd P C B.k ω).card
              ∧ (clusterOf rule O wt ord B.sc B.scd P C B.k ω).card ≤ kmax)
            ∧ flipCount O (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt ⟨k, hk⟩)
              ≤ ((clusterOf rule O wt ord B.sc B.scd P C B.k ω).card : ℝ) * f
            ∧ ¬ cutCorrect O B.lo B.hi (clusterOf rule O wt ord B.sc B.scd P C B.k ω) (tt ⟨k, hk⟩)
                ω}
          else (∅ : Set (Run Ω S J)))
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x)
        (fun i : Fin m => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage,
      Set.mem_inter_iff, vetoLight, ← certOf_eq_image j m x,
      ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_draws populations B (fun x (i : Fin m) => certPrefix j i.val x)
    (measurableSet_certFiber j m) _ hR

open scoped Classical in
/-- A clean family flips heavily at a fresh draw no more often than Markov allows. -/
theorem measureReal_vetoHeavy_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) (k : ℕ) (Δp f : ℝ) (hΔp : 0 ≤ Δp)
    (hf0 : 0 < f) :
    (runMeasure μ D Dsf).real (vetoHeavy rule O populations (D j) j B k Δp f) ≤ Δp / f := by
  classical
  have hq0 : 0 ≤ Δp / f := div_nonneg hΔp hf0.le
  have hEnn : runMeasure μ D Dsf (vetoHeavy rule O populations (D j) j B k Δp f)
      ≤ ENNReal.ofReal (Δp / f) := by
    refine runMeasure_slice_cert_le D Dsf _ (measurableSet_vetoHeavy O populations (D j) j B k Δp f)
      _ (fun y => ?_)
    set F : Finset S :=
      clusterBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hF
    set W : Set S := {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hclean : ∀ v ∈ F, flipMass O (D j) v ≤ Δp
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ vetoHeavy rule O populations (D j) j B k Δp f} ⊆ {c | c j k ∈ W} := by
        rintro c ⟨-, hh⟩
        rw [hFeq c] at hh
        exact hh
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _),
        ← measureReal_def, certCoord_real D j k W]
      refine ENNReal.ofReal_le_ofReal ?_
      rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
      · have hz : W = (∅ : Set S) := by
          ext p
          simp only [hW, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_not]
          have hFe : F = ∅ := Finset.card_eq_zero.1 hF0
          simp [flipCount, hFe]
        rw [hz, measureReal_empty]
        exact hq0
      · have hsetle : W ⊆ {p | f * (F.card : ℝ) ≤ flipCount O F p} := by
          intro p hp
          have hlt := not_le.1 hp
          rw [Set.mem_setOf_eq, mul_comm]
          linarith
        exact le_trans (measureReal_mono hsetle (measure_ne_top _ _))
          (flipCount_mass_le O (D j) F Δp f hf0 hFpos hclean)
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ vetoHeavy rule O populations (D j) j B k Δp f} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hc, -⟩
        rw [hFeq c] at hc
        exact hclean hc
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (vetoHeavy rule O populations (D j) j B k Δp f)).toReal
      ≤ (ENNReal.ofReal (Δp / f)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = Δp / f := ENNReal.toReal_ofReal hq0

open scoped Classical in
/-- A light family misfires at a fresh draw at the vote's own rate: its reads there are disjoint
from the table's, so conditioning on the family leaves them untouched. -/
theorem measureReal_vetoLight_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State) (m k : ℕ) (hk : k < m)
    (f E : ℝ) (kmin kmax : ℕ) (hE : 0 ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real (vetoLight rule O populations j B m k f kmin kmax) ≤ E := by
  classical
  have hEnn : runMeasure μ D Dsf (vetoLight rule O populations j B m k f kmin kmax)
      ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_vetoLight O populations j B m k hk f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    set p : S := d.2 j k with hpdef
    set fam : Ω → Finset S :=
      fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
    by_cases hpP : p ∈ Pd
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ vetoLight rule O populations j B m k f kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hdisj, -⟩
        exact Finset.disjoint_left.1 hdisj hpP
          (Finset.mem_image.2 ⟨k, Finset.mem_range.2 hk, rfl⟩)
      simp [hsec]
    set R : Finset S := Cd.image (fun v => p * v) with hR
    have hdisjR := disjoint_image_readSet hflat hP hV (hdC j hj k) hpP
    set Bad : Finset S → Set Ω := fun A₀ =>
      if (kmin ≤ A₀.card ∧ A₀.card ≤ kmax) ∧ flipCount O A₀ p ≤ (A₀.card : ℝ) * f
      then {ω | ¬ cutCorrect O B.lo B.hi A₀ p ω} else ∅ with hBad
    have hmain := selection_block_bound O R (readSet Pd Cd) hdisjR Cd.powerset ∅
      (Finset.empty_mem_powerset _) fam
      (fun ω => Finset.mem_powerset.2 (fun v hv => Finset.mem_erase.1
        (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, clusterAt_subset O populations B _ hv⟩)
        |>.2))
      (fun ω ω' h => clusterAt_congr O populations B d h) Bad
      (fun A₀ => by
        simp only [hBad]
        split_ifs
        · exact measurableSet_cutCorrect O B.lo B.hi A₀ p
        · exact MeasurableSet.empty)
      (fun A₀ hA₀ => by
        simp only [hBad]
        split_ifs
        · exact measurableSet_filter_pred_map O (T := ↑R) (fun v => p * v)
            (fun v hv => Finset.mem_coe.2 (Finset.mem_image.2
              ⟨v, Finset.mem_powerset.1 hA₀ hv, rfl⟩))
            (fun U => ¬ ((B.hi < Finset.card U → O.label p = 1)
              ∧ (Finset.card U ≤ B.lo → O.label p = 0)))
        · exact @MeasurableSet.empty _ (noiseAlg O ↑R))
      E hE
      (fun A₀ _ => by
        simp only [hBad]
        split_ifs with h
        · exact hcut A₀ p h.2 h.1.1 h.1.2
        · simpa using hE)
    have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ vetoLight rule O populations j B m k f kmin kmax}
        ⊆ {ω | ω ∈ Bad (fam ω)} := by
      rintro ω ⟨-, hsize, hlight, hcc⟩
      show ω ∈ Bad (fam ω)
      simp only [hBad]
      rw [if_pos ⟨hsize, hlight⟩]
      exact hcc
    refine le_trans (measure_mono hsec) ?_
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
    exact ENNReal.ofReal_le_ofReal hmain
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (vetoLight rule O populations j B m k f kmin kmax)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE

open scoped Classical in
/-- Each of the veto's draws is cut wrongly at most as often as a fresh prefix: the family
flips at a heavy one with the probability Markov gives, and misfires at a light one at the
vote's own rate. -/
theorem measureReal_vetoBad_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State) (nveto : ℕ)
    (Δp f E : ℝ) (kmin kmax : ℕ) (hE : 0 ≤ E) (hΔp : 0 ≤ Δp) (hf0 : 0 < f)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real (vetoBad rule O populations (D j) j B nveto Δp kmin kmax)
      ≤ (nveto : ℝ) * (Δp / f + E) := by
  classical
  have hsub : vetoBad rule O populations (D j) j B nveto Δp kmin kmax
      ⊆ ⋃ i ∈ Finset.range nveto,
          (vetoLight rule O populations j B (B.npref + nveto) (B.npref + i) f kmin kmax
            ∪ vetoHeavy rule O populations (D j) j B (B.npref + i) Δp f) := by
    rintro x ⟨hsize, hclean, hdisj, p, hp, hcc⟩
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
    refine Set.mem_biUnion hi ?_
    by_cases hl : flipCount O (clusterBy rule O.mq populations x B) (certPrefix j (B.npref + i) x)
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f
    · exact Or.inl ⟨hdisj, hsize, hl, hcc⟩
    · exact Or.inr ⟨hclean, hl⟩
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _))
    (le_trans (measureReal_biUnion_finset_le _ _) ?_)
  have hper : ∀ i ∈ Finset.range nveto, (runMeasure μ D Dsf).real
      (vetoLight rule O populations j B (B.npref + nveto) (B.npref + i) f kmin kmax
        ∪ vetoHeavy rule O populations (D j) j B (B.npref + i) Δp f) ≤ Δp / f + E := by
    intro i hi
    refine le_trans (measureReal_union_le _ _) ?_
    have h1 := measureReal_vetoLight_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj B
      (B.npref + nveto) (B.npref + i) (by have := Finset.mem_range.1 hi; omega) f E kmin kmax hE hcut
    have h2 := measureReal_vetoHeavy_le (rule := rule) D Dsf O populations j B (B.npref + i) Δp f hΔp hf0
    linarith
  refine le_trans (Finset.sum_le_sum hper) (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- The round returns at one population.  Everything outside `retMiss` is a fact about
the draws: the certification prefixes repeat, or meet the table, or under-represent a class,
or the family is dirty and the sample sees it.  The cluster's own size is the liveness
half's business and is carried as `Estall`. -/
theorem measureReal_notRetAt_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (huni : uni ∈ populations)
    (B : State) (hmpos : 0 < B.npref) (hsig : O.η ≤ 1 / 2)
    (α τ l lcut f E Δp ρ th qmin : ℝ) (kmin kmax n₁ : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hkk : kmax ≤ kmin)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (hf0 : 0 < f)
    (hheavy : Δp / f + th ≤ lcut)
    (hq0 : 0 ≤ qmin) (hqL : qmin ≤ (D uni).real O.L) (hqN : qmin ≤ (D uni).real O.Lᶜ)
    (Estall : ℝ) (hstall : (runMeasure μ D Dsf).real (stalled rule O populations B kmin kmax) ≤ Estall)
    (Edirty : ℝ) (hdirty : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E)
    (hlcut1 : lcut < 1)
    (hn₁ : (n₁ : ℝ) ≤ qmin / 2 * (B.npref : ℝ) - 3 * lcut * (B.npref : ℝ))
    (hn₁pos : 0 < n₁) (hκ : 0 < kmin) (hlohi : B.lo + 1 ≤ kmin) (hhi : B.hi + 1 ≤ kmin)
    (hga : ∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
      (n : ℝ) * ((((B.hi : ℝ) + 1) / kmin + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηIn) - (1 - O.ηIn) * (2 * lcut * (B.npref : ℝ)))
    (hgr : ∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
      (n : ℝ) * (((1 - ((B.lo : ℝ) + 1) / (kmin + 1)) + τ) + τ)
        ≤ (n : ℝ) * (1 - O.ηOut) - (1 - O.ηOut) * (2 * lcut * (B.npref : ℝ)))
    (hα : Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations uni (2 * l) α nveto B j}
      ≤ ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (2 * (B.npref : ℝ) + nveto) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
              + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2)))))))
        + (nveto : ℝ) * (Δp / f + E)
        + ((B.npref : ℝ) ^ 2 * ρ + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
            + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2))))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.npref x) (certOf j (2 * B.npref + nveto) x)} with hE2
  set E5 : Set (Run Ω S J) := stalled rule O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits rule O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := retMiss rule O populations j B l lcut f kmin kmax with hE8
  set E9 : Set (Run Ω S J) := classShort uni B.npref O.L ((B.npref : ℝ) * (qmin - qmin / 2))
    with hE9
  set E10 : Set (Run Ω S J) := classShort uni B.npref O.Lᶜ ((B.npref : ℝ) * (qmin - qmin / 2))
    with hE10
  set E11 : Set (Run Ω S J) := gateMiss rule O populations uni B α (qmin / 2) lcut f kmin kmax
    with hE11
  set E13 : Set (Run Ω S J) := vetoBad rule O populations (D j) j B nveto Δp kmin kmax
    with hE13
  -- the gate's own draws from the uniform pool, read as the FNR's draws of a shifted run
  set E1g : Set (Run Ω S J) := certShift B.npref ⁻¹'
    {x | ¬ Function.Injective (fun i : Fin B.npref => certPrefix uni i.val x)} with hE1g
  set E7g : Set (Run Ω S J) := certShift B.npref ⁻¹'
    heavyHits rule O populations (D uni) uni B f (Δp / f) th with hE7g
  set E9g : Set (Run Ω S J) := certShift B.npref ⁻¹'
    classShort uni B.npref O.L ((B.npref : ℝ) * (qmin - qmin / 2)) with hE9g
  set E10g : Set (Run Ω S J) := certShift B.npref ⁻¹'
    classShort uni B.npref O.Lᶜ ((B.npref : ℝ) * (qmin - qmin / 2)) with hE10g
  set E11g : Set (Run Ω S J) := certShift B.npref ⁻¹'
    gateMiss rule O populations uni B α (qmin / 2) lcut f kmin kmax with hE11g
  set G : Set (Run Ω S J) := (E1g ∪ E7g) ∪ ((E9g ∪ E10g) ∪ E11g) with hGdef
  have hsub : {x : Run Ω S J | x ∉ retAt rule O populations uni (2 * l) α nveto B j}
      ⊆ (((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) ∪ E13) ∪ G := by
    intro x hx
    by_cases hG : x ∈ G
    · exact Or.inr hG
    refine Or.inl ?_
    by_cases h13 : x ∈ E13
    · exact Or.inr h13
    refine Or.inl ?_
    by_cases h1 : Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    · by_cases h2big : Disjoint (prefixesAt populations B.npref x)
          (certOf j (2 * B.npref + nveto) x)
      · have h2 : Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x) :=
          Finset.disjoint_of_subset_right (certOf_mono j (by omega) x) h2big
        have h2v : Disjoint (prefixesAt populations B.npref x) (certOf j (B.npref + nveto) x) :=
          Finset.disjoint_of_subset_right (certOf_mono j (by omega) x) h2big
        by_cases h5 : kmin ≤ (clusterBy rule O.mq populations x B).card
            ∧ (clusterBy rule O.mq populations x B).card ≤ kmax
        · by_cases h6 : ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp
          · -- the certification sample is good and the family is clean
            have hveto : j ≠ uni → ∀ p ∈ vetoOf j B.npref nveto x,
                cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x) :=
              fun _ p hp => by
                by_contra hc
                exact h13 ⟨h5, h6, h2v, p, hp, hc⟩
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
            have hcardN : (certOf j B.npref x).card = B.npref := by exact_mod_cast hcard
            have hCpos : 0 < (certOf j B.npref x).card := by
              have : (0 : ℝ) < ((certOf j B.npref x).card : ℝ) := by
                rw [hcard]; exact_mod_cast hmpos
              exact_mod_cast this
            -- Markov over the family: the mean flip fraction is at most `Δp`, so the mass
            -- where an `f` fraction flips is `Δp / f`, with no family-size factor.
            have hmass : (D j).real {p | ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)}
                ≤ Δp / f := by
              set F : Finset S := clusterBy rule O.mq populations x B with hFdef
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
                  (fun v hv => h6 v hv)
            by_cases h7 : (B.npref : ℝ) * (Δp / f + th)
                ≤ (((certOf j B.npref x).filter (fun p =>
                  ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                    ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)
            · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩)))
            · have hlight : (((certOf j B.npref x).filter (fun p =>
                    ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
                    ≤ lcut * ((certOf j B.npref x).card : ℝ) := by
                have hlt := not_le.1 h7
                have hle : (((certOf j B.npref x).filter (fun p =>
                    ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
                    = (((certOf j B.npref x).filter (fun p =>
                      ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                        ≤ ((clusterBy rule O.mq populations x B).card : ℝ)
                          * f))).card : ℝ) := by
                  refine congrArg (fun t : Finset S => (t.card : ℝ))
                    (Finset.filter_congr (fun p _ => ?_))
                  unfold famGood
                  simp only [h5.1, h5.2, and_true, true_and]
                rw [hle, hcard]
                have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
                nlinarith
              have hsize : B.k ≤ (clusterBy rule O.mq populations x B).card + 1 := by omega
              by_cases hfnr : (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo
                  (B.hi + 1) (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * ((certOf j B.npref x).card : ℝ)
              · by_cases hju : j = uni
                · subst hju
                  by_cases hgok : gateOK rule O populations α B (certOf j B.npref x) x
                  · -- the FNR's draws pass; what can still fail is the gate's own draws
                    exfalso
                    by_cases hcg : certified O.mq B.lo B.hi α (clusterBy rule O.mq populations x B)
                        (gateOf j B.npref 0 x) (oracleNoise x)
                    · exact hx ⟨hsize, hfnr, fun _ => ⟨hgok, hcg⟩, h2big, hveto⟩
                    apply hG
                    set y : Run Ω S J := certShift B.npref x with hy
                    by_cases h1g : Function.Injective (fun i : Fin B.npref => certPrefix j i.val y)
                    · have hcardG : ((certOf j B.npref y).card : ℝ) = (B.npref : ℝ) := by
                        have hinjOn : Set.InjOn (fun i => certPrefix j i y)
                            ↑(Finset.range B.npref) := by
                          intro a ha b hb hab
                          have := h1g (show (fun i : Fin B.npref => certPrefix j i.val y)
                              ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
                            = (fun i : Fin B.npref => certPrefix j i.val y)
                              ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
                          simpa using congrArg Fin.val this
                        unfold certOf
                        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
                      have hcardGN : (certOf j B.npref y).card = B.npref := by
                        exact_mod_cast hcardG
                      by_cases h7g : (B.npref : ℝ) * (Δp / f + th)
                          ≤ (((certOf j B.npref y).filter (fun p =>
                            ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                              ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)
                      · exact Or.inl (Or.inr ⟨hmass, h7g⟩)
                      have hlightG : (((certOf j B.npref y).filter (fun p =>
                          ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
                          ≤ lcut * ((certOf j B.npref y).card : ℝ) := by
                        have hlt := not_le.1 h7g
                        have hle : (((certOf j B.npref y).filter (fun p =>
                            ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
                            = (((certOf j B.npref y).filter (fun p =>
                              ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                                ≤ ((clusterBy rule O.mq populations x B).card : ℝ)
                                  * f))).card : ℝ) := by
                          refine congrArg (fun t : Finset S => (t.card : ℝ))
                            (Finset.filter_congr (fun p _ => ?_))
                          unfold famGood
                          simp only [h5.1, h5.2, and_true, true_and]
                        rw [hle, hcardG]
                        have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
                        nlinarith
                      by_cases hcLg : (((Finset.range B.npref).filter
                          (fun i => certPrefix j i y ∈ O.L)).card : ℝ)
                          ≤ (B.npref : ℝ) * (qmin - qmin / 2)
                      · refine Or.inr (Or.inl (Or.inl ?_))
                        show certShift B.npref x ∈ classShort j B.npref O.L _
                        simp only [classShort, Set.mem_setOf_eq]
                        convert hcLg using 4
                      by_cases hcNg : (((Finset.range B.npref).filter
                          (fun i => certPrefix j i y ∈ O.Lᶜ)).card : ℝ)
                          ≤ (B.npref : ℝ) * (qmin - qmin / 2)
                      · refine Or.inr (Or.inl (Or.inr ?_))
                        show certShift B.npref x ∈ classShort j B.npref O.Lᶜ _
                        simp only [classShort, Set.mem_setOf_eq]
                        convert hcNg using 4
                      have hdisjG : Disjoint (prefixesAt populations B.npref y)
                          (certOf j B.npref y) := by
                        refine Finset.disjoint_of_subset_right (fun p hp => ?_) h2big
                        obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
                        exact Finset.mem_image.2 ⟨B.npref + i, Finset.mem_range.2
                          (by have := Finset.mem_range.1 hi; omega), rfl⟩
                      refine Or.inr (Or.inr ⟨⟨hdisjG, hcardGN, ?_, ?_, hlightG⟩,
                        fun hc => hcg hc.1⟩)
                      · have heq : (((certOf j B.npref y).filter (fun p => p ∈ O.L)).card : ℝ)
                            = (((Finset.range B.npref).filter
                              (fun i => certPrefix j i y ∈ O.L)).card : ℝ) :=
                          congrArg (fun n : ℕ => (n : ℝ))
                            (card_filter_certOf j B.npref y (fun p => p ∈ O.L) _ _ h1g).symm
                        rw [heq, hcardG]
                        linarith [not_le.1 hcLg]
                      · have heq : (((certOf j B.npref y).filter (fun p => p ∉ O.L)).card : ℝ)
                            = (((Finset.range B.npref).filter
                              (fun i => certPrefix j i y ∈ O.Lᶜ)).card : ℝ) :=
                          congrArg (fun n : ℕ => (n : ℝ))
                            (card_filter_certOf j B.npref y (fun p => p ∈ O.Lᶜ) _ _ h1g).symm
                        rw [heq, hcardG]
                        linarith [not_le.1 hcNg]
                    · exact Or.inl (Or.inl h1g)
                  refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ?_))))
                  by_cases hcL : (((Finset.range B.npref).filter
                      (fun i => certPrefix j i x ∈ O.L)).card : ℝ)
                      ≤ (B.npref : ℝ) * (qmin - qmin / 2)
                  · refine Or.inl (Or.inl ?_)
                    simp only [hE9, classShort, Set.mem_setOf_eq]
                    convert hcL using 4
                  by_cases hcN : (((Finset.range B.npref).filter
                      (fun i => certPrefix j i x ∈ O.Lᶜ)).card : ℝ)
                      ≤ (B.npref : ℝ) * (qmin - qmin / 2)
                  · refine Or.inl (Or.inr ?_)
                    simp only [hE10, classShort, Set.mem_setOf_eq]
                    convert hcN using 4
                  refine Or.inr ⟨⟨h2, hcardN, ?_, ?_, hlight⟩, hgok⟩
                  · have heq : (((certOf j B.npref x).filter (fun p => p ∈ O.L)).card : ℝ)
                        = (((Finset.range B.npref).filter
                          (fun i => certPrefix j i x ∈ O.L)).card : ℝ) :=
                      congrArg (fun n : ℕ => (n : ℝ))
                        (card_filter_certOf j B.npref x (fun p => p ∈ O.L) _ _ h1).symm
                    rw [heq, hcard]
                    linarith [not_le.1 hcL]
                  · have heq : (((certOf j B.npref x).filter (fun p => p ∉ O.L)).card : ℝ)
                        = (((Finset.range B.npref).filter
                          (fun i => certPrefix j i x ∈ O.Lᶜ)).card : ℝ) :=
                      congrArg (fun n : ℕ => (n : ℝ))
                        (card_filter_certOf j B.npref x (fun p => p ∈ O.Lᶜ) _ _ h1).symm
                    rw [heq, hcard]
                    linarith [not_le.1 hcN]
                · exact absurd ⟨hsize, hfnr, fun h => absurd h hju, h2big, hveto⟩ hx
              · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
                  ⟨⟨h2, hcardN, hCpos, hlight⟩, hfnr⟩))))
          · exact Or.inr (Or.inr (Or.inl h6))
        · exact Or.inr (Or.inl h5)
      · exact Or.inl (Or.inr h2big)
    · exact Or.inl (Or.inl h1)
  have hgate : (runMeasure μ D Dsf).real E11
      ≤ 2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
        + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := by
    exact measureReal_gateMiss_le hflat O populations D Dsf hsupp hsuppSf uni huni B α τ
      (qmin / 2) lcut f E kmin kmax n₁ hkk hE hElcut hτ hsig hdec hcut hlcut1 hn₁ hn₁pos hκ
      hlohi hhi hga hgr hα
  have hGb : (runMeasure μ D Dsf).real G ≤ ((B.npref : ℝ) ^ 2 * ρ + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
            + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2))))) := by
    have hg1 := measureReal_union_le (μ := runMeasure μ D Dsf) E1g E7g
    have hg2 := measureReal_union_le (μ := runMeasure μ D Dsf) E9g E10g
    have hg3 := measureReal_union_le (μ := runMeasure μ D Dsf) (E9g ∪ E10g) E11g
    have hg4 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1g ∪ E7g) ((E9g ∪ E10g) ∪ E11g)
    have b1 : (runMeasure μ D Dsf).real E1g ≤ (B.npref : ℝ) ^ 2 * ρ := by
      rw [hE1g, measureReal_certShift D Dsf _ _ (measurableSet_certNotInj uni B.npref)]
      exact cert_not_injective_le D Dsf uni B.npref ρ (hρ uni huni) hρ0
    have b7 : (runMeasure μ D Dsf).real E7g ≤ Real.exp (-2 * (B.npref : ℝ) * th ^ 2) := by
      rw [hE7g, measureReal_certShift D Dsf _ _
        (measurableSet_heavyHits O populations (D uni) uni B f (Δp / f) th)]
      exact measureReal_heavyHits_le D Dsf O populations uni B f (Δp / f) th hth
    have b9 : (runMeasure μ D Dsf).real E9g ≤ Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2) := by
      rw [hE9g, measureReal_certShift D Dsf _ _ (measurableSet_classShort uni B.npref O.L _)]
      exact cert_class_short_le (μ := μ) D Dsf uni B.npref O.L qmin (qmin / 2) hq0
        (by linarith) hqL
    have b10 : (runMeasure μ D Dsf).real E10g
        ≤ Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2) := by
      rw [hE10g, measureReal_certShift D Dsf _ _ (measurableSet_classShort uni B.npref O.Lᶜ _)]
      exact cert_class_short_le (μ := μ) D Dsf uni B.npref O.Lᶜ qmin (qmin / 2) hq0
        (by linarith) hqN
    have b11 : (runMeasure μ D Dsf).real E11g
        ≤ 2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
          + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) := by
      rw [hE11g, measureReal_certShift D Dsf _ _
        (measurableSet_gateMiss O populations uni B α (qmin / 2) lcut f kmin kmax)]
      exact hgate
    linarith
  have hOld : (runMeasure μ D Dsf).real
      (((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) ∪ E13)
      ≤ ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (2 * (B.npref : ℝ) + nveto) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
              + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2)))))))
        + (nveto : ℝ) * (Δp / f + E) := by
   calc (runMeasure μ D Dsf).real
          (((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) ∪ E13)
    _ ≤ (((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
        + ((runMeasure μ D Dsf).real E5 + ((runMeasure μ D Dsf).real E6
          + ((runMeasure μ D Dsf).real E7 + ((runMeasure μ D Dsf).real E8
            + (((runMeasure μ D Dsf).real E9 + (runMeasure μ D Dsf).real E10)
              + (runMeasure μ D Dsf).real E11))))))
        + (runMeasure μ D Dsf).real E13 := by
        have h13u := measureReal_union_le (μ := runMeasure μ D Dsf)
          ((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) E13
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h910 := measureReal_union_le (μ := runMeasure μ D Dsf) E9 E10
        have h911 := measureReal_union_le (μ := runMeasure μ D Dsf) (E9 ∪ E10) E11
        have h8 := measureReal_union_le (μ := runMeasure μ D Dsf) E8 ((E9 ∪ E10) ∪ E11)
        have h78 := measureReal_union_le (μ := runMeasure μ D Dsf) E7
          (E8 ∪ ((E9 ∪ E10) ∪ E11))
        have h678 := measureReal_union_le (μ := runMeasure μ D Dsf) E6
          (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))
        have h5678 := measureReal_union_le (μ := runMeasure μ D Dsf) E5
          (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11))))
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2)
          (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))
        linarith
    _ ≤ (((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (2 * (B.npref : ℝ) + nveto) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
              + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2))))))))
        + (nveto : ℝ) * (Δp / f + E) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.npref ρ (hρ j hj) hρ0
        · refine le_trans (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _))
            (le_trans (prefix_cert_disjoint_le D Dsf populations j (2 * B.npref + nveto) ρ hρ
              (hρ j hj) hρ0) (le_of_eq (by push_cast; ring)))
          exact fun hd => hx (Finset.disjoint_of_subset_left
            (prefixesAt_mono populations (by omega) x) hd)
        · exact measureReal_heavyHits_le D Dsf O populations j B f (Δp / f) th hth
        · exact measureReal_retMiss_le hflat O populations D Dsf hsupp hsuppSf j hj B l lcut f E
            kmin kmax hE hElcut hlcl hdec
        · exact cert_class_short_le (μ := μ) D Dsf uni B.npref O.L qmin (qmin / 2) hq0 (by linarith) hqL
        · exact cert_class_short_le (μ := μ) D Dsf uni B.npref O.Lᶜ qmin (qmin / 2) hq0
            (by linarith) hqN
        · exact measureReal_vetoBad_le hflat O populations D Dsf hsupp hsuppSf j hj B nveto Δp f E
            kmin kmax hE hΔp hf0 hcut
    _ = ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (2 * (B.npref : ℝ) + nveto) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
              + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2)))))))
        + (nveto : ℝ) * (Δp / f + E) := by ring
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | x ∉ retAt rule O populations uni (2 * l) α nveto B j}
      ≤ (runMeasure μ D Dsf).real
          ((((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) ∪ E13) ∪ G) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runMeasure μ D Dsf).real
          (((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ (E8 ∪ ((E9 ∪ E10) ∪ E11)))))) ∪ E13)
        + (runMeasure μ D Dsf).real G := measureReal_union_le _ _
    _ ≤ _ := add_le_add hOld hGb

/-- Part 1, reduced to one state.  The ladder is finite, so the union over the states the
loop may stop at is a finite sum: a rung carries `δ·npref/(8·N)`, and the counts sum to at
most `2N`, plus `δ·V/(16·npref)` when it is at or above `V`, and those sum to at most `2`.  No
summable weight over all budgets is needed, and so no encoding of a budget as a number.

`G` is what the rungs share.  The pool's findability and its draws colliding do not mention
the prefixes, so each is one event for the whole ladder; charging them once rather than per
rung is what keeps the pool count free of the ladder's length, which the prefix count
determines, and keeps the pool's collisions off the low rungs, whose shares are small.

What remains of Part 1 is `hper`: at one rung, off `G`, a family that passes both gates is
valid on every population except with probability that rung's share. -/
theorem validity_of_ladder (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ : ℝ) (hδ : 0 ≤ δ) (s : Finset State) (N V : ℕ) (hN : 0 < N)
    (hsum : ∑ B ∈ s, (B.npref : ℝ) ≤ 2 * N)
    (hinv : ∑ B ∈ s, (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) ≤ 2)
    (G : Set (Run Ω S J)) (hG : (runMeasure μ D Dsf).real G ≤ δ / 8)
    (hper : ∀ B ∈ s, (runMeasure μ D Dsf).real
      ((retBy rule O.mq populations uni indecisionLimit α nveto B.npref B ∩ FailAt rule O populations D indecisionLimit εcov B) \ G)
        ≤ δ * (B.npref : ℝ) / (8 * N)
          + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0) :
    (runMeasure μ D Dsf).real (⋃ B : {B : State // B ∈ s},
        retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val
          ∩ FailAt rule O populations D indecisionLimit εcov B.val) ≤ δ / 2 := by
  classical
  have hNR : (0 : ℝ) < (N : ℕ) := by exact_mod_cast hN
  have hsplit : (⋃ B : {B : State // B ∈ s},
        retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val
          ∩ FailAt rule O populations D indecisionLimit εcov B.val)
      ⊆ (⋃ B : {B : State // B ∈ s},
          (retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val
            ∩ FailAt rule O populations D indecisionLimit εcov B.val) \ G) ∪ G := by
    intro x hx
    obtain ⟨B, hB⟩ := Set.mem_iUnion.1 hx
    by_cases hg : x ∈ G
    · exact Or.inr hg
    · exact Or.inl (Set.mem_iUnion.2 ⟨B, hB, hg⟩)
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hite : ∀ B : State, (if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0)
      = δ / 16 * (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) := by
    intro B; split_ifs <;> ring
  have hmain : (runMeasure μ D Dsf).real (⋃ B : {B : State // B ∈ s},
      (retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val
        ∩ FailAt rule O populations D indecisionLimit εcov B.val) \ G) ≤ 3 * δ / 8 := by
    rw [Set.iUnion_subtype]
    calc (runMeasure μ D Dsf).real (⋃ B, ⋃ (_ : B ∈ s),
          (retBy rule O.mq populations uni indecisionLimit α nveto B.npref B ∩ FailAt rule O populations D indecisionLimit εcov B) \ G)
        ≤ ∑ B ∈ s, (runMeasure μ D Dsf).real
            ((retBy rule O.mq populations uni indecisionLimit α nveto B.npref B
              ∩ FailAt rule O populations D indecisionLimit εcov B) \ G) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ s, (δ * (B.npref : ℝ) / (8 * (N : ℕ))
            + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0) :=
          Finset.sum_le_sum hper
      _ = (δ / (8 * (N : ℕ))) * ∑ B ∈ s, (B.npref : ℝ)
            + δ / 16 * ∑ B ∈ s, (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) := by
          rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
          congr 1
          · exact Finset.sum_congr rfl (fun B _ => by ring)
          · exact Finset.sum_congr rfl (fun B _ => hite B)
      _ ≤ (δ / (8 * (N : ℕ))) * (2 * (N : ℕ)) + δ / 16 * 2 := by
          gcongr
      _ = 3 * δ / 8 := by field_simp; ring
  linarith [hmain, hG]

lemma exp_tail_anti {m a b : ℝ} (hm : 0 ≤ m) (ha : 0 ≤ a) (hab : a ≤ b) :
    Real.exp (-2 * m * b ^ 2) ≤ Real.exp (-2 * m * a ^ 2) := by
  refine Real.exp_le_exp.2 ?_
  have h := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ ha hab 2) hm
  nlinarith

/-! ### How `per_state_le` gets its bound

Validity is read off how the family was built, not off the gate.  Where the noise rate
depends on the class, a correct cut's agreement with the seed's column sits anywhere from
`1 − η` to `1` according to the population's mix of classes, so no null separates it from a
cut drifted on an `εcov` fraction.  What does bound the drift is the screen: it keeps no
candidate that flips more than `validFlip` of the table, Markov over the family turns that
into a bound on the mass where many members flip, and the vote is right elsewhere.

`measureReal_validFail_le` carries that through the certification sample, which the family
never saw: a cut wrong on `εcov` of the population is wrong on `15εcov/16` of the sample
(`hitShort`), few of those can be prefixes the family flips (`heavyHits`), and few of the
rest can be misread by a family that barely flips there (`validMiss`, Hoeffding). -/

/-- The vote's two clean means, pushed by a flipping `f` of the family and read the rest of the
margin, `voteSig η₀ − f`, in, still straddle the centre `bnd·κ`.  A flip moves a member's read
toward the other class by up to a whole bit, which is why `f` is charged in full. -/
lemma vote_shifts (O : Oracle μ S) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hb0 : voteSig η₀ ≤ bnd) (hb1 : bnd ≤ 1 - voteSig η₀)
    {f : ℝ} (hf0 : 0 ≤ f) (hf : f ≤ voteSig η₀) {κ : ℝ} (hκ0 : 0 ≤ κ) :
    κ * ((O.ηOut + (1 - O.ηOut) * f) + (voteSig η₀ - f)) ≤ κ * bnd
      ∧ κ * bnd ≤ κ * (((1 - O.ηIn) * (1 - f)) - (voteSig η₀ - f)) := by
  have hf1 : 0 ≤ 1 - f := by linarith
  have hu : (O.ηOut + (1 - O.ηOut) * f) + (voteSig η₀ - f) ≤ bnd := by
    nlinarith [mul_le_mul_of_nonneg_right (show O.ηOut ≤ bnd - voteSig η₀ by linarith) hf1,
      mul_nonneg hf0 (sub_nonneg.2 hb0)]
  have hl : bnd ≤ ((1 - O.ηIn) * (1 - f)) - (voteSig η₀ - f) := by
    nlinarith [mul_le_mul_of_nonneg_right (show bnd + voteSig η₀ ≤ 1 - O.ηIn by linarith) hf1,
      mul_nonneg hf0 (show 0 ≤ 1 - bnd - voteSig η₀ by linarith)]
  exact ⟨mul_le_mul_of_nonneg_left hu hκ0, mul_le_mul_of_nonneg_left hl hκ0⟩

/-- What every rung of the ladder satisfies whatever its prefix count: only the prefix count
and the skip guard change down the ladder, and none of these read either. -/
lemma rung_facts (O : Oracle μ S) (populations : Finset J) {εcov δ pAP : ℝ} (mi : ℕ)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hbLo : voteSig η₀ ≤ bnd) (hbHi : bnd ≤ 1 - voteSig η₀)
    (hεcov : 0 < εcov) (hind : 0 < indecisionLimit) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < (solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k
    ∧ 0 < (solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).scd
    ∧ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).sc : ℝ)
      ≤ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).scd : ℝ)
        * (flipBudget η₀ populations indecisionLimit εcov δ * (1 - 2 * O.η) ^ 2
          - 2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2))
    ∧ (((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k - 1 : ℕ) : ℝ)
        * ((O.ηOut + (1 - O.ηOut) * flipFrac η₀) + voteSlack η₀ / 2)
      ≤ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).hi : ℝ)
    ∧ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).lo : ℝ)
      < (((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k - 1 : ℕ) :
      ℝ)
        * (((1 - O.ηIn) * (1 - flipFrac η₀)) - voteSlack η₀ / 2)
    ∧ Real.exp (-2
        * (((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k
            - 1 : ℕ) : ℝ)
        * (voteSlack η₀ / 2) ^ 2) ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  classical
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hη₀ hεcov hind hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hind hcard
  have hv := voteSlack_pos η₀ hη₀
  have hvs : voteSig η₀ - flipFrac η₀ = voteSlack η₀ := by rw [flipFrac, voteSlack]; ring
  have hF0 := (flipFrac_pos η₀ hη₀).le
  have hFs : flipFrac η₀ ≤ voteSig η₀ := by
    rw [flipFrac]; linarith [voteSig_pos η₀ hη₀]
  have hb := bandHalf_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀ hbLo
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  set c : ℕ := ⌈bnd * κ⌉₊ with hcdef
  have hcge : bnd * (κ : ℝ) ≤ (c : ℝ) := Nat.le_ceil _
  have hclt : (c : ℝ) < bnd * (κ : ℝ) + 1 :=
    Nat.ceil_lt_add_one (mul_nonneg (le_trans (voteSig_pos η₀ hη₀).le hbLo) hκ0)
  obtain ⟨hlowShift, hhiShift⟩ := vote_shifts O hbOut hbIn hbLo hbHi hF0 hFs hκ0
  rw [hvs] at hlowShift hhiShift
  have hb0' : (0 : ℝ) ≤ (b : ℝ) := Nat.cast_nonneg _
  have hκv : (0 : ℝ) ≤ (κ : ℝ) * voteSlack η₀ := mul_nonneg hκ0 hv.le
  set B : State := solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi
    with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
  have hBscd : B.scd = ⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  clear_value B b κ c
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hBk]; omega
  · rw [hBscd]; omega
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
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega]
    push_cast
    linarith [hlowShift, hcge, hb0', hκv]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.sub_sub, Nat.cast_sub hb1]
    push_cast
    linarith [hhiShift, hclt, hb0', hκv]
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail η₀ populations hη₀ hεcov hind

/-- A vote with a `validFrac` of the family flipping has its mean inside the tenth of the margin
`validFrac` leaves, and the band's far edge is still `voteSlack/2` of the family beyond that. -/
lemma rung_valid_band (O : Oracle μ S) (populations : Finset J) {εcov δ pAP : ℝ} (mi : ℕ)
    (hη₀ : η₀ < 1 / 2) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hb0 : voteSig η₀ ≤ bnd) (hb1 : bnd ≤ 1 - voteSig η₀) :
    (((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k - 1 : ℕ) : ℝ)
        * ((O.ηOut + (1 - O.ηOut) * validFrac η₀) + voteSlack η₀ / 2)
      ≤ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).hi : ℝ) + 1
    ∧ ((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).lo : ℝ)
      ≤ (((solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi).k - 1 : ℕ) :
      ℝ)
        * (((1 - O.ηIn) * (1 - validFrac η₀)) - voteSlack η₀ / 2) := by
  have hs := voteSig_pos η₀ hη₀
  have hV0 := (validFrac_pos η₀ hη₀).le
  have hVs : validFrac η₀ ≤ voteSig η₀ := by rw [validFrac]; linarith
  have hvs : voteSig η₀ - validFrac η₀ = voteSig η₀ / 10 := by rw [validFrac]; ring
  have hv : voteSlack η₀ = 3 * voteSig η₀ / 10 := rfl
  have hb1' := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀ hb0
  have hseed := famCount_seed (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hwide := Nat.lt_floor_add_one
    ((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ / 2)
  rw [← bandHalf] at hwide
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  set c : ℕ := ⌈bnd * κ⌉₊ with hcdef
  have hcge : bnd * (κ : ℝ) ≤ (c : ℝ) := Nat.le_ceil _
  have hclt : (c : ℝ) < bnd * (κ : ℝ) + 1 :=
    Nat.ceil_lt_add_one (mul_nonneg (le_trans hs.le hb0) hκ0)
  obtain ⟨hup, hdn⟩ := vote_shifts O hbOut hbIn hb0 hb1 hV0 hVs hκ0
  rw [hvs] at hup hdn
  set B : State := solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit mi
    with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
  clear_value B b κ c
  rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hBhi, hBlo, Nat.sub_sub, Nat.cast_sub hb1']
  push_cast
  rw [hv] at hseed hwide ⊢
  constructor
  · linarith [hup, hcge, hwide]
  · linarith [hdn, hclt, hwide, hseed]

/-- The band is wide enough for `cross_of_mem_schedule` to read the crossing bound off. -/
lemma band_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hη₀ : η₀ < 1 / 2) (hb0 : voteSig η₀ ≤ bnd)
    (hB : B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit) :
    B.lo ≤ B.hi ∧ 2 ≤ B.k
      ∧ ((B.k : ℝ) - 1) * Real.log (1 / crossLimit) ≤ 2 * ((B.hi : ℝ) - B.lo) ^ 2 := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  have hv := voteSlack_pos η₀ hη₀
  have hκpos := famCount_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hb := bandHalf_wide (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀ hb0
  have hL := famCount_cross (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ hη₀
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set c : ℕ := ⌈bnd * κ⌉₊ with hcdef
  set B : State := solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
  clear_value B b κ c
  rw [hBk, hBlo, hBhi]
  refine ⟨by omega, by omega, ?_⟩
  rw [Nat.sub_sub, Nat.cast_sub hb1]
  push_cast
  have hκ0 : (0 : ℝ) ≤ κ := Nat.cast_nonneg _
  have hy : 0 ≤ (κ : ℝ) * voteSlack η₀ / 2 := by positivity
  have hsq := pow_le_pow_left₀ hy hb 2
  nlinarith [mul_le_mul_of_nonneg_left hL hκ0]

/-- Every rung cuts at `bnd`: its band contains `bnd` of the family. -/
lemma centre_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hη₀ : η₀ < 1 / 2) (hb0 : voteSig η₀ ≤ bnd)
    (hB : B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit) :
    (B.lo : ℝ) < bnd * ((B.k : ℝ) - 1) ∧ bnd * ((B.k : ℝ) - 1) ≤ B.hi := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀ hb0
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  set c : ℕ := ⌈bnd * κ⌉₊ with hcdef
  have hcge : bnd * (κ : ℝ) ≤ (c : ℝ) := Nat.le_ceil _
  have hclt : (c : ℝ) < bnd * (κ : ℝ) + 1 :=
    Nat.ceil_lt_add_one (mul_nonneg (le_trans (voteSig_pos η₀ hη₀).le hb0) hκ0)
  set B : State := solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
  clear_value B b κ c
  rw [hBk, hBlo, hBhi, Nat.sub_sub, Nat.cast_sub hb1]
  push_cast
  have hb0' : (0 : ℝ) ≤ (b : ℝ) := Nat.cast_nonneg _
  constructor <;> linarith

/-- A vote over a family no larger than a rung's own, whose mean lies outside the rung's band,
reaches the band's far side at most `crossLimit` of the time. -/
theorem cross_of_mem_schedule (O : Oracle μ S) {populations : Finset J} {εcov δ α pAP : ℝ}
    {B : State} (hη₀ : η₀ < 1 / 2) (hb0 : voteSig η₀ ≤ bnd) (hcross : 0 < crossLimit)
    (hB : B ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit)
    (F : Finset S) (hF : F.card + 1 ≤ B.k) (p : S) :
    (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
    ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit) := by
  obtain ⟨hlohi, hk, hband⟩ := band_of_mem_schedule hη₀ hb0 hB
  have hnpos : (0 : ℝ) < ((B.k - 1 : ℕ) : ℝ) := by
    have : 0 < B.k - 1 := by omega
    exact_mod_cast this
  have hnR : ((B.k - 1 : ℕ) : ℝ) = (B.k : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega)]; simp
  have hexp : Real.exp (-2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)) ≤ crossLimit := by
    rw [show -2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)
        = -(2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)) by ring]
    refine exp_neg_le_of_log_le hcross ?_
    rw [le_div_iff₀ hnpos, hnR]
    linarith
  obtain ⟨h1, h2⟩ := vote_cross_le O F p hlohi (n := B.k - 1) (by omega) (by omega)
  exact ⟨fun h => (h1 h).trans hexp, fun h => (h2 h).trans hexp⟩

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state.  Off the pool's findability and its draws colliding, a returned
family is valid on every population except where the certification draws, the screen or the
vote fail, charged per population. -/
theorem per_state_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (hsig : O.η ≤ 1 / 2)
    (indecisionLimit εcov α ρ pAP : ℝ) (hil : 0 ≤ indecisionLimit) (hε0 : 0 ≤ εcov)
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (B : State) (hkpos : 0 < B.k) (hmpos : 0 < B.npref)
    (hscd : 0 < B.scd) (hcount : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP - pAP / 2))
    (Δ γ g th f lcut E t r : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hth : 0 ≤ th)
    (hf0 : 0 < f) (hE : 0 ≤ E) (ht : 0 ≤ t) (hr : 0 ≤ r) (hrE : r ≤ lcut - E)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ))
    (hbudget : ((populations.card : ℝ) * Δ + g) / f + th + lcut ≤ εcov - t)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real ((retBy rule O.mq populations uni indecisionLimit α nveto B.npref B
        ∩ FailAt rule O populations D indecisionLimit εcov B)
          \ (apShort O B.nsuff pAP (pAP / 2)
            ∪ {x | ¬ Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x)}))
      ≤ (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
          + (((B.npref : ℝ) ^ 2 * ρ
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
              + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2))
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + (Real.exp (-2 * (B.npref : ℝ) * r ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))))) := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + g with hΔp
  set CR : Set (Run Ω S J) := {x | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
      ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} with hCR
  set V : J → Set (Run Ω S J) := fun j => {x : Run Ω S J |
      B.k ≤ (clusterBy rule O.mq populations x B).card + 1
      ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
            (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
          ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
      ∧ ¬ (1 - εcov ≤ (D j).real
        {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x)}
        ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1)
            (familyBy rule O.mq populations x B) p (oracleNoise x)}
          ≤ 2 * indecisionLimit)}
    with hV
  set Dy : J → Set (Run Ω S J) := fun j =>
    {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp} with hDy
  have hsub : ((retBy rule O.mq populations uni indecisionLimit α nveto B.npref B ∩ FailAt rule O populations D indecisionLimit εcov B)
        \ (apShort O B.nsuff pAP (pAP / 2)
          ∪ {x | ¬ Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x)}))
      ⊆ ⋃ j ∈ populations, ((V j \ Dy j) ∪ (CR ∩ Dy j)) := by
    rintro x ⟨⟨hret, hfail⟩, hG⟩
    have hc : x ∈ CR := by
      by_contra hc
      exact hG (noCleanRef_subset O populations B pAP (pAP / 2) hcount hc)
    simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
    obtain ⟨j, hj, hfj⟩ := hfail
    refine Set.mem_biUnion hj ?_
    by_cases hd : x ∈ Dy j
    · exact Or.inr ⟨hc, hd⟩
    · exact Or.inl ⟨⟨hret.1, hret.2.1 j hj, hfj⟩, hd⟩
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have h2 : (runMeasure μ D Dsf).real (⋃ j ∈ populations, ((V j \ Dy j) ∪ (CR ∩ Dy j)))
      ≤ ∑ _j ∈ populations, (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
          + (((B.npref : ℝ) ^ 2 * ρ
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
              + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2))
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + (Real.exp (-2 * (B.npref : ℝ) * r ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))))) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) (Finset.sum_le_sum (fun j hj => ?_))
    refine le_trans (measureReal_union_le _ _) ?_
    have hv := measureReal_validFail_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj B hkpos
      indecisionLimit εcov lcut f E Δp ρ th t hil hε0 hE (by linarith) (by rw [hΔp]; positivity)
      hth ht hf0 hbudget hρ hρ0 hcut
    have htail : Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
        ≤ Real.exp (-2 * (B.npref : ℝ) * r ^ 2) :=
      exp_tail_anti (Nat.cast_nonneg _) hr hrE
    have hd := measureReal_dirtyMember_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj B hsig
      hmpos Δ γ g ρ hΔ hγ hg hρ0 (hρ j hj) hscd hsc
    have hv' : (runMeasure μ D Dsf).real (V j \ Dy j)
        ≤ ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
          + (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + (Real.exp (-2 * (B.npref : ℝ) * r ^ 2)
                + Real.exp (-2 * (B.npref : ℝ) * indecisionLimit ^ 2)))) :=
      le_trans hv (by linarith [htail])
    have hd' : (runMeasure μ D Dsf).real (CR ∩ Dy j)
        ≤ (B.npref : ℝ) ^ 2 * ρ + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2) := hd
    linarith [hv', hd']
  rw [Finset.sum_const, nsmul_eq_mul] at h2
  exact h2

open scoped Classical in
/-- Part 1 — whatever is returned is valid, whenever it is returned.

Except with probability `δ/2`, no rung the loop may stop at returns an invalid family. -/
theorem validity_of_returned {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α : ℝ) (hindLim : 0 < indecisionLimit)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hb0 : voteSig η₀ ≤ bnd) (hb1 : bnd ≤ 1 - voteSig η₀)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (pAP : ℝ) (hpAP0 : 0 ≤ pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hfind : Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (pAP / 2) ^ 2)
      ≤ δ / 16)
    (hcoll : (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2 * ρsf
      ≤ δ / 16) :
    (runMeasure μ D Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP
            crossLimit ρ},
          retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val
            ∩ FailAt rule O populations D indecisionLimit εcov B.val) ≤ δ / 2 := by
  classical
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρ0 : 0 ≤ ρ := le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ j₀ hj₀)
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 ⟨j₀, hj₀⟩
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  refine validity_of_ladder O populations D Dsf indecisionLimit εcov α δ hδ.le _
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
    (prefCount_pos _ _ _ _ _ _)
    (stoppable_npref_sum_le η₀ populations εcov δ α pAP ρ)
    (schedule_inv_sum_le η₀ populations εcov δ α pAP _ _
      (by rw [stoppable]; exact Finset.filter_subset _ _))
    (apShort O M pAP (pAP / 2)
      ∪ {x | ¬ Function.Injective (fun i : Fin M => suffixDraw i.val x)}) ?_ (fun B hB => ?_)
  · refine le_trans (measureReal_union_le _ _) ?_
    have h1 := le_trans
      (measureReal_apShort_le D Dsf O M pAP (pAP / 2) hpAP0 (by linarith) hpAPBound) hfind
    have h2 := le_trans (suffix_not_injective_le (μ := μ) D Dsf M ρsf hρsf hρsf0) hcoll
    linarith
  have hcap := capped_of_mem_stoppable hB
  have hsched := (Finset.mem_filter.1 hB).1
  have hM : B.nsuff = M := nsuff_of_mem_schedule hsched
  rw [schedule] at hsched
  obtain ⟨i, -, hBeq⟩ := Finset.mem_image.1 hsched
  obtain ⟨hk, hscd, hsc, -, -, hEc⟩ := rung_facts O populations
    (δ := δ) (pAP := pAP)
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i)
    hηle hη₀ hbOut hbIn hb0 hb1 hεcov hindLim hcard
  obtain ⟨hhi, hlo⟩ := rung_valid_band O populations (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) (εcov := εcov) (δ := δ) (pAP := pAP)
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) hη₀ hbOut hbIn
    hb0 hb1
  rw [hBeq] at hk hscd hsc hhi hlo hEc
  have hv : 0 < validFlip η₀ populations εcov := validFlip_pos η₀ populations hη₀ hεcov hcard
  have hflipv := flipBudget_le_validFlip (indecisionLimit := indecisionLimit) (δ := δ)
    η₀ populations hη₀ hεcov.le hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hindLim hcard
  have hf := validFrac_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  -- the solved cutoff is under the validity flip's separation, less both deviations
  have hsc' : (B.sc : ℝ) ≤ (B.scd : ℝ) * (validFlip η₀ populations εcov * (1 - 2 * O.η) ^ 2
      - 2 * validMargin η₀ populations εcov) := by
    refine le_trans hsc (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
    have hq : (1 - 2 * η₀) ^ 2 ≤ (1 - 2 * O.η) ^ 2 := by
      have h1 : (0 : ℝ) ≤ 1 - 2 * η₀ := by linarith
      nlinarith [eta_nonneg O]
    have hvm : 2 * validMargin η₀ populations εcov
        = 7 * validFlip η₀ populations εcov * (1 - 2 * η₀) ^ 2 / 8 := by
      rw [validMargin, hsval]; ring
    have hQ : (0 : ℝ) ≤ (1 - 2 * O.η) ^ 2 := sq_nonneg _
    rw [hvm]
    nlinarith [mul_le_mul_of_nonneg_right hflipv hQ, mul_le_mul_of_nonneg_left hq hv.le]
  have hbudget : ((populations.card : ℝ) * validFlip η₀ populations εcov
      + εcov * validFrac η₀ / 8) / validFrac η₀ + εcov / 32 + 3 * εcov / 32
        ≤ εcov - εcov / 16 := by
    have hid : ((populations.card : ℝ) * validFlip η₀ populations εcov
        + εcov * validFrac η₀ / 8) / validFrac η₀ = 13 * εcov / 16 := by
      rw [validFlip]
      field_simp
      ring
    rw [hid]
    linarith
  have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * validFrac η₀ →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω}
        ≤ Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (voteSlack η₀ / 2) ^ 2) := by
    intro F p hf' hmin hmax
    have hcardF : F.card = B.k - 1 := le_antisymm hmax hmin
    refine le_trans (cutCorrect_whp O F p _ _ (validFrac η₀) (voteSlack η₀ / 2) O.ηOut O.ηIn
      le_rfl le_rfl (by linarith [le_trans (le_max_right O.ηIn O.ηOut) hsig.le])
      (by linarith [le_trans (le_max_left O.ηIn O.ηOut) hsig.le]) hf'
      (half_pos (voteSlack_pos η₀ hη₀)).le
      ?_ ?_)
      (le_of_eq ?_)
    · rw [hcardF]; exact hhi
    · rw [hcardF]; exact hlo
    · rw [hcardF]
  have hmiss : εcov / 32
      ≤ 3 * εcov / 32 - Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (voteSlack η₀ / 2) ^ 2) := by
    linarith [(cutBudget_le η₀ indecisionLimit εcov).1]
  have h := per_state_le (rule := rule) (uni := uni) (nveto := nveto) hflat O populations D Dsf hsupp hsuppSf hsig.le indecisionLimit
    εcov α ρ pAP hindLim.le hεcov.le hρ hρ0 B hk hcap.mpos hscd
    (by have := hcap.found; linarith)
    (validFlip η₀ populations εcov) (validMargin η₀ populations εcov) (εcov * validFrac η₀ / 8)
    (εcov / 32) (validFrac η₀) (3 * εcov / 32)
    (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (voteSlack η₀ / 2) ^ 2)) (εcov / 16) (εcov / 32)
    hv (by rw [validMargin]; positivity) (by positivity) (by positivity) hf (Real.exp_nonneg _)
    (by positivity) (by positivity) hmiss hsc' hbudget hcut
  rw [← hM]
  exact le_trans h hcap.share

/-- What one round at one population can cost: the draws, the family's size and cleanliness,
the sample's two class counts and its heavy fraction, and the round's own two tests. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut τ th E γscr γdirty gdirty tap ρ ρsf qmin heavy : ℝ) (nveto n₁ : ℕ) (B : State) : ℝ :=
  ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (2 * (B.npref : ℝ) + nveto) ^ 2 * ρ)
    + ((((B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
          + ((B.npref : ℝ) ^ 2 * ρ + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
        + (((B.npref : ℝ) ^ 2 * ρ + (((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + ((B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2)
              + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2) + (B.nsuff : ℝ) ^ 2 * ρsf))))
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
              + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
                  + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
                + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                  + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2))))))))
    + (nveto : ℝ) * (heavy + E)
    + ((B.npref : ℝ) ^ 2 * ρ + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
        + ((Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2)
            + Real.exp (-2 * (B.npref : ℝ) * (qmin / 2) ^ 2))
          + (2 * Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
            + 2 * Real.exp (-2 * (n₁ : ℝ) * τ ^ 2)))))

/-- A state whose round can pass.  Every clause is an inequality among the state's
budgets, the oracle's rates, the suffix distribution's findability and the error budget —
no probability enters, and nothing here is a free parameter of the algorithm.  Reaching
such a state is what the computed ladder is for.

Only the uniform pool has to carry both labels, each on a `qmin` share of its mass; the gate
reads nothing else. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP qmin : ℝ) (nveto : ℕ) (B : State) : Prop :=
  ∃ (τ th tap γdec γscr γdirty gdirty Δ lcut f : ℝ) (n₁ : ℕ),
    0 < B.npref ∧ 2 ≤ B.k ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
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
    -- fraction of which flips, so each class's mean moves toward the other by up to `f`;
    -- the FNR reads the family with its seed, cut one count higher
    ∧ ((B.hi : ℝ) + 1 ≤ (B.k : ℝ) * (((1 - O.ηIn) * (1 - f)) - γdec))
    ∧ ((B.k : ℝ) * ((O.ηOut + (1 - O.ηOut) * f) + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * ((O.ηOut + (1 - O.ηOut) * f) + γdec) ≤ (B.hi : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * (((1 - O.ηIn) * (1 - f)) - γdec))
    -- each side of the cut holds `n₁` of the uniform pool's sample, and reads its class past
    -- the rate the cut splits at by `τ` more than a coin flip's deviation
    ∧ 0 ≤ qmin ∧ 0 < n₁ ∧ lcut < 1
    ∧ ((n₁ : ℝ) ≤ qmin / 2 * (B.npref : ℝ) - 3 * lcut * (B.npref : ℝ))
    ∧ B.lo + 1 ≤ B.k - 1 ∧ B.hi + 1 ≤ B.k - 1
    ∧ (∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
        (n : ℝ) * ((((B.hi : ℝ) + 1) / ((B.k - 1 : ℕ) : ℝ) + τ) + τ)
          ≤ (n : ℝ) * (1 - O.ηIn) - (1 - O.ηIn) * (2 * lcut * (B.npref : ℝ)))
    ∧ (∀ n : ℕ, n₁ ≤ n → n ≤ B.npref →
        (n : ℝ) * (((1 - ((B.lo : ℝ) + 1) / (((B.k - 1 : ℕ) : ℝ) + 1)) + τ) + τ)
          ≤ (n : ℝ) * (1 - O.ηOut) - (1 - O.ηOut) * (2 * lcut * (B.npref : ℝ)))
    ∧ (Real.exp (-2 * (n₁ : ℝ) * τ ^ 2) ≤ α)
    -- a side each of whose prefixes is of its class reads it past the rate the cut splits at,
    -- so the veto's test on it is valid
    ∧ (((B.hi : ℝ) + 1) / ((B.k - 1 : ℕ) : ℝ) ≤ 1 - O.ηIn)
    ∧ (O.ηOut ≤ ((B.lo : ℝ) + 1) / (((B.k - 1 : ℕ) : ℝ) + 1))
    -- the vote misfires at under half the cut budget, so the sample's count of misfires
    -- concentrates below the budget
    ∧ (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2) ≤ lcut / 2)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) lcut τ th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf qmin
            (((populations.card : ℝ) * Δ + gdirty) / f) nveto n₁ B
      ≤ δ / 2)

lemma mul_self_add_le_cube {x : ℝ} (hx : 0 ≤ x) : x * (x + 3) ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x * (x + 3) = x ^ 3 + 8 * x ^ 2 + 24 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma le_cube_of_nonneg {x : ℝ} (hx : 0 ≤ x) : x ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x = x ^ 3 + 9 * x ^ 2 + 26 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

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

/-- The gate's floor `n₁ = ⌊2·qmin·m/5⌋`: most of each class's share of the uniform pool's
sample, and large enough that any `L` the prefix count resolves at the scale `s·εcov/4` sits
under the side tail's exponent at `τ = s/8`. -/
lemma gate_floor {s εcov qmin L : ℝ} {m : ℕ} (hs : 0 < s) (hs1 : s ≤ 1 / 2) (hε : 0 < εcov)
    (hε1 : εcov ≤ 1) (hq : εcov ≤ s * qmin / 2) (hL : 0 ≤ L) (hm64 : 64 / εcov ≤ (m : ℝ))
    (hm : 64 * L / (εcov * (s * εcov / 4) ^ 2) ≤ (m : ℝ)) :
    256 ≤ qmin * (m : ℝ) ∧ qmin * (m : ℝ) / 3 ≤ (⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℝ)
      ∧ L ≤ 2 * (⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℝ) * (s / 8) ^ 2 := by
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have h64 : 64 ≤ εcov * (m : ℝ) := by rw [div_le_iff₀ hε] at hm64; linarith
  have hqpos : 0 < qmin := by
    by_contra h
    push_neg at h
    nlinarith
  have hqm : 0 ≤ qmin * (m : ℝ) := by positivity
  have h2 : 2 * (εcov * (m : ℝ)) ≤ s * (qmin * (m : ℝ)) := by
    nlinarith [mul_le_mul_of_nonneg_right hq hmR]
  have h3 : s * (qmin * (m : ℝ)) ≤ qmin * (m : ℝ) / 2 := by
    nlinarith [mul_le_mul_of_nonneg_right hs1 hqm]
  have ha : 256 ≤ qmin * (m : ℝ) := by linarith
  have hfl := Nat.lt_floor_add_one (2 * qmin * (m : ℝ) / 5)
  have hb : qmin * (m : ℝ) / 3 ≤ (⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℝ) := by linarith
  refine ⟨ha, hb, ?_⟩
  have hden : εcov * (s * εcov / 4) ^ 2 = εcov ^ 3 * s ^ 2 / 16 := by ring
  rw [hden, div_le_iff₀ (by positivity)] at hm
  have he2 : εcov ^ 2 * s ≤ 1 / 2 := by
    have : εcov ^ 2 ≤ 1 := by nlinarith
    nlinarith
  have hems : 0 ≤ εcov * (m : ℝ) * s := by positivity
  have hL2 : (m : ℝ) * (εcov ^ 3 * s ^ 2 / 16) ≤ εcov * (m : ℝ) * s / 32 := by
    have := mul_le_mul_of_nonneg_left he2 hems
    nlinarith
  have hL3 : εcov * (m : ℝ) * s / 48 ≤ 2 * (⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℝ) * (s / 8) ^ 2 := by
    have hs2 : 0 ≤ (s / 8) ^ 2 := sq_nonneg _
    have h4 := mul_le_mul_of_nonneg_right hb hs2
    have h5 : 2 * (εcov * (m : ℝ)) * s ≤ s * (qmin * (m : ℝ)) * s :=
      mul_le_mul_of_nonneg_right h2 hs.le
    nlinarith
  nlinarith

/-- The pool is deep enough to hold accept-preserving suffixes.  This is the one event the
ladder shares, so it is charged once against `δ/4` rather than per rung. -/
lemma solved_findability (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ}
    (hδ : 0 < δ) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 32 := by
  have ctap : Real.log (1 / (δ / 32)) / (2 * (pAP / 2) ^ 2)
      ≤ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℕ) : ℝ) := by
    have hmono : Real.log (1 / (δ / 32))
        ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
      refine Real.log_le_log (by positivity) ?_
      rw [one_div_div, div_le_div_iff₀ hδ hδ]
      nlinarith
    refine le_trans (div_le_div_of_nonneg_right hmono (by positivity)) ?_
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊
        ≤ poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
      rw [poolCount]; omega
    exact_mod_cast hle
  have h := tail_le_of_count (γ := pAP / 2) (c := 1) (by positivity)
    (by positivity : (0 : ℝ) < δ / 32) (by norm_num) ctap
  linarith

/-- The top rung's tails, each at `δ/(128·|populations|)`, and its collision terms. -/
lemma solved_tails (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (pAP / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) ^ 2
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
          * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
          * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
        * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (populations.card : ℝ) * (((populations.card : ℝ) + 3)
        * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) ^ 2 * ρ) ≤ δ
            / 64
    ∧ (populations.card : ℝ)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2 * ρsf)
        ≤ δ / 64 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hf : 0 < flipFrac η₀ := flipFrac_pos η₀ hsig
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
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
      / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
      ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)⌉₊ ≤
          m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cth : Real.log (1 / ε₀) / (2 * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2) ≤ (m : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ)
        / (2 * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2)⌉₊ ≤ m := by
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
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ :=
    tail_le_of_count (by positivity) hε₀ (by linarith) cdirty
  have tdirty' : (M : ℝ) * Real.exp (-2 * (m : ℝ)
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tdirty
    exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)
  have tth : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := cutBudget η₀ indecisionLimit εcov / 32) (c := 1) (by
      positivity) hε₀
      (by norm_num) cth
    linarith
  -- the collision terms, with the state's code kept opaque
  have hden : (0 : ℝ) < 64 * ((populations.card : ℝ) + 3) ^ 3
      * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) := by positivity
  have hcapval : collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit
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
    {indecisionLimit εcov δ α pAP ρ ρsf qmin : ℝ} {nveto : ℕ} (hη0 : 0 ≤ η₀)
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hq : εcov ≤ sig η₀ * qmin / 2)
    (hveto : 2 * (populations.card : ℝ) * nveto * εcov ≤ δ)
    (hpAP : 0 < pAP) (hind : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    (populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (sig η₀ / 8)
            (cutBudget η₀ indecisionLimit εcov / 32)
            (Real.exp (-2
              * (((solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).k
                  - 1 : ℕ) : ℝ)
              * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32)
            (pAP / 2) ρ ρsf qmin
            (((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
              + 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) / flipFrac η₀)
            nveto
            ⌊2 * qmin
              * ((solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).npref :
                  ℝ) / 5⌋₊
            (solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 2 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  obtain ⟨ttap, tscr, tdirty', tth, hcoll1, hcoll2⟩ :=
    solved_tails η₀ populations hsig hε hind hδ hpAP hcard hρ0 hρsf0 hρsmall hρsfsmall
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hBm : (solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).npref = m :=
    rfl
  have hBM : (solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).nsuff = M :=
    rfl
  have hBκ : ((solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).k - 1 : ℕ)
      = κ := by
    show κ + 1 - 1 = κ
    omega
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the gate's floor, and the tail it kills
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have heq : (2 : ℝ) / ε₀ = 256 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp; try ring
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  have hlog0 : 0 ≤ Real.log (256 * (populations.card : ℝ) / δ) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ]
    linarith
  have hmlog : 64 * Real.log (256 * (populations.card : ℝ) / δ)
      / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * Real.log (256 * (populations.card : ℝ) / δ)
        / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  obtain ⟨-, -, hLtail⟩ := gate_floor hs (by rw [sig]; linarith) hε hε1 hq hlog0 hsizeR hmlog
  have tgate : 2 * Real.exp (-2 * ((⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℕ) : ℝ) * (sig η₀ / 8) ^ 2)
      ≤ ε₀ := by
    refine tail_le_of_count (by positivity) hε₀ (by norm_num) ?_
    rw [heq, div_le_iff₀ (by positivity)]
    linarith
  have hqpos : 0 < qmin := by
    by_contra h
    push_neg at h
    nlinarith
  have tq : Real.exp (-2 * (m : ℝ) * (qmin / 2) ^ 2) ≤ ε₀ := by
    refine le_trans (exp_tail_anti hmR (by positivity) ?_) tth
    have := (cutBudget_le η₀ indecisionLimit εcov).1
    have hs1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    nlinarith
  -- the vote's misfire rate sits under half the cut budget, so both count tails are no worse
  -- than the threshold's
  have hEc : Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 :=
    famCount_tail_seed η₀ populations hsig hε hind
  have hE0 : 0 ≤ Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2) := Real.exp_nonneg _
  have tl : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tlc : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  -- the veto's draws, each a fresh prefix, cost the flip and misfire budget apiece
  have hFne : flipFrac η₀ ≠ 0 := ne_of_gt (flipFrac_pos η₀ hsig)
  have hheavy : ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
      + 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) / flipFrac η₀
      = cutBudget η₀ indecisionLimit εcov * (31 / 32) := by
    rw [flipBudget]
    field_simp
    ring
  have hcut8 := (cutBudget_le η₀ indecisionLimit εcov).1
  have hv0 : (0 : ℝ) ≤ (nveto : ℝ) := Nat.cast_nonneg _
  have hvε : (populations.card : ℝ) * ((nveto : ℝ) * (cutBudget η₀ indecisionLimit εcov * (31 / 32)
      + Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)))
      ≤ 3 * δ / 32 := by
    have h1 : cutBudget η₀ indecisionLimit εcov * (31 / 32)
        + Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)
        ≤ 3 * εcov / 16 := by linarith
    have h2 := mul_le_mul_of_nonneg_left h1 (mul_nonneg hcard.le hv0)
    nlinarith
  -- the veto's draws are few against the table, so their collisions are the table's
  have hvm : (nveto : ℝ) ≤ (m : ℝ) := by
    have hsizeR' : 64 / εcov ≤ (m : ℝ) := hsizeR
    rw [div_le_iff₀ hε] at hsizeR'
    have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
      have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
      exact_mod_cast h1
    nlinarith [mul_le_mul_of_nonneg_right hc1 (mul_nonneg hv0 hε.le)]
  have hmv : (2 * (m : ℝ) + nveto) ^ 2 ≤ 9 * (m : ℝ) ^ 2 := by nlinarith
  rw [roundFail, hBm, hBM, hBκ, hheavy]
  refine le_trans (mul_le_mul_of_nonneg_left
    (show _ ≤ 18 * ε₀ + (9 * ((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ
        + 2 * (M : ℝ) ^ 2 * ρsf)
        + (nveto : ℝ) * (cutBudget η₀ indecisionLimit εcov * (31 / 32)
          + Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
      from ?_) hcard.le) ?_
  · have tscr' : ((M : ℝ) + 2) ^ 2
        * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤
          ε₀ := tscr
    have hcol : (populations.card : ℝ) * (2 * (m : ℝ) + nveto) ^ 2 * ρ
        ≤ (populations.card : ℝ) * (9 * (m : ℝ) ^ 2) * ρ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hmv hcard.le) hρ0
    have hm2 : (0 : ℝ) ≤ (m : ℝ) ^ 2 * ρ := by positivity
    linarith [ttap, tscr', tdirty', tth, tl, tlc, tq, tgate]
  · have hexp : (populations.card : ℝ)
        * (18 * ε₀ + (9 * ((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ + 2 * (M : ℝ) ^ 2 * ρsf)
          + (nveto : ℝ) * (cutBudget η₀ indecisionLimit εcov * (31 / 32)
            + Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)))
        = 18 * ((populations.card : ℝ) * ε₀)
          + (9 * ((populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ))
            + 2 * ((populations.card : ℝ) * ((M : ℝ) ^ 2 * ρsf)))
          + (populations.card : ℝ) * ((nveto : ℝ) * (cutBudget η₀ indecisionLimit εcov * (31 / 32)
            + Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)))
          := by ring
    rw [hexp, hcε]
    linarith [hcoll1, hcoll2, hvε]

set_option maxHeartbeats 1000000 in
/-- The top rung carries its own share of the error budget, `δ/8`.  Validity's margins are all
wider than the round's, so each of its tails is under the round's own. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    stateFail η₀ populations indecisionLimit εcov ρ (solvedState η₀ bnd populations
      indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 8 := by
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcle := cutBudget_le η₀ indecisionLimit εcov
  obtain ⟨-, tscr, tdirty', tth, hcoll1, -⟩ :=
    solved_tails η₀ populations hsig hε hind hδ hpAP hcard hρ0 hρsf0 hρsmall hρsfsmall
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hBm : (solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).npref = m :=
    rfl
  have hBM : (solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit).nsuff = M :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hf := flipFrac_pos η₀ hsig
  have hs := sig_pos η₀ hsig
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  have hflipv := flipBudget_le_validFlip (indecisionLimit := indecisionLimit) (δ := δ)
    η₀ populations hsig hε.le hcard
  -- each of validity's margins is at least the round's that `solved_tails` killed
  have hmscr : screenMargin η₀ populations indecisionLimit εcov δ / 2
      ≤ validMargin η₀ populations εcov := by
    rw [screenMargin, validMargin]
    nlinarith [mul_le_mul_of_nonneg_right hflipv (sq_nonneg (sig η₀)),
      flipBudget_pos (δ := δ) η₀ populations hsig hε hind hcard, sq_nonneg (sig η₀)]
  have hmdirty : 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32
      ≤ εcov * validFrac η₀ / 8 := by
    nlinarith [mul_le_mul_of_nonneg_right hcle.1 hf.le,
      mul_le_mul_of_nonneg_left (flipFrac_le_validFrac η₀ hsig) hε.le]
  have tscr' : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2) ≤ ε₀ :=
    le_trans (mul_le_mul_of_nonneg_left (exp_tail_anti hmR (by positivity) hmscr)
      (by positivity)) tscr
  have tdirty'' : (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2) ≤ ε₀ :=
    le_trans (mul_le_mul_of_nonneg_left (exp_tail_anti hmR (by positivity) hmdirty)
      (Nat.cast_nonneg _)) tdirty'
  have t32 : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tcov : Real.exp (-2 * (m : ℝ) * (εcov / 16) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tund : Real.exp (-2 * (m : ℝ) * indecisionLimit ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  have hcoll1' : (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ)
      + (m : ℝ) ^ 2 * ρ) ≤ δ / 64 := by
    refine le_trans (mul_le_mul_of_nonneg_left ?_ hcard.le) hcoll1
    nlinarith [mul_nonneg (sq_nonneg (m : ℝ)) hρ0]
  rw [stateFail, hBm, hBM]
  have hsum : (((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
      + (Real.exp (-2 * (m : ℝ) * (εcov / 16) ^ 2)
        + (((m : ℝ) ^ 2 * ρ
            + ((M : ℝ) + 2) ^ 2
              * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
            + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
              + Real.exp (-2 * (m : ℝ) * indecisionLimit ^ 2))))))
      ≤ ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ) + (m : ℝ) ^ 2 * ρ) + 6 * ε₀ := by
    linarith [tcov, tscr', tdirty'', t32, tund]
  have hscale := mul_le_mul_of_nonneg_left hsum hcard.le
  have hexp : (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
        + (m : ℝ) ^ 2 * ρ) + 6 * ε₀)
      = (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ)
        + (m : ℝ) ^ 2 * ρ) + 6 * ((populations.card : ℝ) * ε₀) := by ring
  linarith

/-- The pool holds two accept-preserving draws in expectation. -/
lemma poolCount_found (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ} (hpAP : 0 < pAP) :
    (2 : ℝ) ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) * (pAP / 2) := by
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκ : (1 : ℝ) ≤ (κ : ℝ) := by exact_mod_cast famCount_pos η₀ populations εcov δ
  have hc : 2 * ((κ : ℝ) + 1) / pAP
      ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
    rw [poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊]
  have h := mul_le_mul_of_nonneg_right hc (by positivity : (0 : ℝ) ≤ pAP / 2)
  have heq : 2 * ((κ : ℝ) + 1) / pAP * (pAP / 2) = (κ : ℝ) + 1 := by field_simp
  linarith

/-- A tail a count `V` kills at `ε` is under `ε·V/m` at any count `m ≥ V`: past `V` it decays
faster than `V/m` does. -/
lemma tail_le_of_count_scaled {γ ε c V m : ℝ} (hγ : 0 < γ) (hε : 0 < ε) (hε6 : ε ≤ 1 / 6)
    (hc : 1 ≤ c) (hV0 : 0 < V) (hV : Real.log (c / ε) / (2 * γ ^ 2) ≤ V) (hVm : V ≤ m) :
    c * Real.exp (-2 * m * γ ^ 2) ≤ ε * V / m := by
  have hm0 : 0 < m := lt_of_lt_of_le hV0 hVm
  have ha : Real.log (c / ε) ≤ 2 * V * γ ^ 2 := by
    rw [div_le_iff₀ (by positivity)] at hV
    linarith
  have hlog1 : 1 ≤ Real.log (c / ε) := by
    have h6 : 6 ≤ c / ε := by rw [le_div_iff₀ hε]; linarith
    have h2 : Real.log ((2 : ℝ)⁻¹) ≤ (2 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
    have h3 : Real.log ((3 : ℝ)⁻¹) ≤ (3 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
    rw [Real.log_inv] at h2 h3
    have hl6 : Real.log 6 = Real.log 2 + Real.log 3 := by
      rw [show (6 : ℝ) = 2 * 3 by norm_num, Real.log_mul (by norm_num) (by norm_num)]
    have hmono : Real.log 6 ≤ Real.log (c / ε) := Real.log_le_log (by norm_num) h6
    linarith
  have h1 : c * Real.exp (-2 * V * γ ^ 2) ≤ ε := by
    have he : Real.exp (-2 * V * γ ^ 2) ≤ ε / c := by
      calc Real.exp (-2 * V * γ ^ 2) ≤ Real.exp (-Real.log (c / ε)) :=
            Real.exp_le_exp.2 (by linarith)
        _ = ε / c := by
            rw [Real.exp_neg, Real.exp_log (by positivity)]
            field_simp
    calc c * Real.exp (-2 * V * γ ^ 2) ≤ c * (ε / c) :=
          mul_le_mul_of_nonneg_left he (by linarith)
      _ = ε := by field_simp
  have h2 : Real.exp (-2 * (m - V) * γ ^ 2) ≤ V / m := by
    have hk : (m - V) / V ≤ 2 * (m - V) * γ ^ 2 := by
      rw [div_le_iff₀ hV0]
      nlinarith [sub_nonneg.2 hVm]
    have ht : m / V ≤ Real.exp ((m - V) / V) := by
      have h := Real.add_one_le_exp ((m - V) / V)
      have h' : (m - V) / V + 1 = m / V := by field_simp; ring
      linarith
    calc Real.exp (-2 * (m - V) * γ ^ 2) ≤ Real.exp (-((m - V) / V)) :=
          Real.exp_le_exp.2 (by linarith)
      _ = 1 / Real.exp ((m - V) / V) := by rw [Real.exp_neg, one_div]
      _ ≤ 1 / (m / V) := one_div_le_one_div_of_le (by positivity) ht
      _ = V / m := by field_simp
  have hsplit : c * Real.exp (-2 * m * γ ^ 2)
      = c * Real.exp (-2 * V * γ ^ 2) * Real.exp (-2 * (m - V) * γ ^ 2) := by
    rw [mul_assoc c, ← Real.exp_add]
    ring_nf
  rw [hsplit]
  calc c * Real.exp (-2 * V * γ ^ 2) * Real.exp (-2 * (m - V) * γ ^ 2) ≤ ε * (V / m) :=
        mul_le_mul h1 h2 (by positivity) hε.le
    _ = ε * V / m := by ring

set_option maxHeartbeats 1000000 in
/-- Every rung at or above `validCount` carries its share: its collision terms from the share
proportional to its count, its tails from the one inversely proportional. -/
theorem rung_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ) (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (i : ℕ) (hi : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hV : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit
      ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) :
    solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit
        (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i)
      ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set m : ℕ := N / 2 ^ i with hmdef
  have hB : solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit m
      ∈ schedule η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit := by
    rw [schedule]
    exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩
  have hVpos : 0 < V := by rw [hVdef, validCount]; omega
  have hVR : (0 : ℝ) < (V : ℝ) := by exact_mod_cast hVpos
  have hVm : (V : ℝ) ≤ (m : ℝ) := by exact_mod_cast hV
  have hmR : (0 : ℝ) < (m : ℝ) := lt_of_lt_of_le hVR hVm
  have hmN : (m : ℝ) ≤ (N : ℝ) := by exact_mod_cast Nat.div_le_self N (2 ^ i)
  have hNR : (0 : ℝ) < (N : ℝ) := lt_of_lt_of_le hmR hmN
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨(by show _ - _ - 1 ≤ _ + _; omega),
    by show 0 < m; exact_mod_cast hmR, poolCount_found η₀ populations hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov ρ
      (solvedStateAt η₀ bnd populations indecisionLimit εcov δ pAP crossLimit m)
    ≤ δ * (m : ℝ) / (8 * N) + if V ≤ m then δ * (V : ℝ) / (16 * m) else 0
  rw [if_pos hV, stateFail]
  show (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (m : ℝ) * (εcov / 16) ^ 2)
      + (((m : ℝ) ^ 2 * ρ
          + ((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
          + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
        + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + Real.exp (-2 * (m : ℝ) * indecisionLimit ^ 2))))))
    ≤ δ * (m : ℝ) / (8 * N) + δ * (V : ℝ) / (16 * m)
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  -- the collision terms, against the proportional share
  have hcoll : (populations.card : ℝ) * (((populations.card : ℝ) + 2) * (m : ℝ) ^ 2 * ρ)
      ≤ δ * (m : ℝ) / (8 * N) := by
    have hden : (0 : ℝ) < 64 * ((populations.card : ℝ) + 3) ^ 3
        * ((N : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) := by positivity
    have hro : ρ * (64 * ((populations.card : ℝ) + 3) ^ 3 * ((N : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1))
        ≤ δ := by
      have h := hρcap
      rw [collisionCap, ← hNdef, ← hMdef, le_div_iff₀ hden] at h
      exact h
    have hJ : (populations.card : ℝ) * ((populations.card : ℝ) + 2)
        ≤ ((populations.card : ℝ) + 3) ^ 3 := by
      nlinarith [mul_self_add_le_cube hcard.le]
    have hmm : (m : ℝ) ^ 2 * (8 * N) ≤ (m : ℝ) * ((N : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * 8 := by
      nlinarith [mul_le_mul_of_nonneg_left hmN hmR.le, sq_nonneg (M : ℝ)]
    rw [le_div_iff₀ (by positivity)]
    have hA : (0 : ℝ) ≤ ((populations.card : ℝ) + 3) ^ 3 := by positivity
    calc (populations.card : ℝ) * (((populations.card : ℝ) + 2) * (m : ℝ) ^ 2 * ρ) * (8 * N)
        = ((populations.card : ℝ) * ((populations.card : ℝ) + 2)) * ((m : ℝ) ^ 2 * (8 * N))
            * ρ := by ring
      _ ≤ ((populations.card : ℝ) + 3) ^ 3
            * ((m : ℝ) * ((N : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * 8) * ρ := by
          gcongr
      _ = (m : ℝ) / 8
            * (ρ * (64 * ((populations.card : ℝ) + 3) ^ 3 * ((N : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1))) := by
          ring
      _ ≤ (m : ℝ) / 8 * δ := by gcongr
      _ ≤ δ * (m : ℝ) := by linarith [mul_nonneg hmR.le hδ.le]
  -- the tails, against the inverse share
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  have hε₀3 : ε₀ ≤ 1 / 6 := by
    rw [hε₀def, div_le_iff₀ (by positivity)]
    nlinarith
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hf := validFrac_pos η₀ hsig
  have hvm : 0 < validMargin η₀ populations εcov := by
    rw [validMargin]
    have := validFlip_pos η₀ populations hsig hε hcard
    have := sig_pos η₀ hsig
    positivity
  set T1 : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ)
    / (2 * validMargin η₀ populations εcov ^ 2) with hT1
  set T2 : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
    / (2 * (εcov * validFrac η₀ / 8) ^ 2) with hT2
  set T3 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) / (2 * (εcov / 32) ^ 2) with hT3
  set T4 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) / (2 * indecisionLimit ^ 2) with hT4
  have hVsum : (V : ℝ) = (⌈T1⌉₊ : ℝ) + (⌈T2⌉₊ : ℝ) + (⌈T3⌉₊ : ℝ) + (⌈T4⌉₊ : ℝ) + 1 := by
    rw [hVdef, validCount, ← hMdef]
    push_cast
    ring
  have hcl : ∀ x : ℝ, (0 : ℝ) ≤ (⌈x⌉₊ : ℝ) := fun x => Nat.cast_nonneg _
  have hV1 : T1 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T2, hcl T3, hcl T4])
  have hV2 : T2 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T3, hcl T4])
  have hV3 : T3 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T2, hcl T4])
  have hV4 : T4 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T2, hcl T3])
  have hq1 : ((M : ℝ) + 2) ^ 2 / ε₀ = 128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ := by
    rw [hε₀def]; field_simp
  have hq2 : ((M : ℝ) + 1) / ε₀ = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
    rw [hε₀def]; field_simp
  have hq3 : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp
  have t1 := tail_le_of_count_scaled (c := ((M : ℝ) + 2) ^ 2) hvm hε₀ hε₀3 (by nlinarith) hVR
    (by rw [hq1]; exact hV1) hVm
  have t2 := tail_le_of_count_scaled (γ := εcov * validFrac η₀ / 8) (c := (M : ℝ) + 1)
    (by positivity) hε₀ hε₀3
    (by linarith) hVR (by rw [hq2]; exact hV2) hVm
  have t3 := tail_le_of_count_scaled (γ := εcov / 32) (c := 1) (by positivity) hε₀ hε₀3 le_rfl hVR
    (by rw [hq3]; exact hV3) hVm
  have t4 := tail_le_of_count_scaled (γ := indecisionLimit) (c := 1) hind hε₀ hε₀3 le_rfl hVR
    (by rw [hq3]; exact hV4) hVm
  have t2' : (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2)
      ≤ ε₀ * V / m :=
    le_trans (mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)) t2
  have tcov : Real.exp (-2 * (m : ℝ) * (εcov / 16) ^ 2) ≤ ε₀ * V / m :=
    le_trans (exp_tail_anti (a := εcov / 32) (b := εcov / 16) hmR.le (by positivity)
      (by linarith)) (by simpa using t3)
  have t3' : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2) ≤ ε₀ * V / m := by simpa using t3
  have t4' : Real.exp (-2 * (m : ℝ) * indecisionLimit ^ 2) ≤ ε₀ * V / m := by simpa using t4
  have htails : (populations.card : ℝ) * (6 * (ε₀ * V / m)) ≤ δ * (V : ℝ) / (16 * m) := by
    rw [hε₀def]
    field_simp
    nlinarith [mul_pos hδ hVR]
  have hsum : ((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
      + (Real.exp (-2 * (m : ℝ) * (εcov / 16) ^ 2)
        + (((m : ℝ) ^ 2 * ρ
            + ((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
            + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
              + Real.exp (-2 * (m : ℝ) * indecisionLimit ^ 2)))))
      ≤ ((populations.card : ℝ) + 2) * (m : ℝ) ^ 2 * ρ + 6 * (ε₀ * V / m) := by
    linarith [tcov, t1, t2', t3', t4']
  have hscale := mul_le_mul_of_nonneg_left hsum hcard.le
  nlinarith [hscale, hcoll, htails]

/-- Some rung the loop may stop at has fewer than twice `validCount` prefixes: the lowest rung
whose count is still at or above it, or any rung when the top is already below it. -/
theorem exists_small_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ) (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hne : (stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ).Nonempty) :
    ∃ B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ,
      (B.npref : ℝ) ≤ 2 * (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  have hVpos : 0 < V := by rw [hVdef, validCount]; omega
  by_cases hNV : N < V
  · obtain ⟨B, hB⟩ := hne
    refine ⟨B, hB, ?_⟩
    have hsched := Finset.mem_of_mem_filter _ hB
    rw [schedule] at hsched
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 hsched
    show ((N / 2 ^ j : ℕ) : ℝ) ≤ 2 * (V : ℝ)
    have : N / 2 ^ j ≤ V := le_trans (Nat.div_le_self _ _) hNV.le
    have : ((N / 2 ^ j : ℕ) : ℝ) ≤ (V : ℝ) := by exact_mod_cast this
    linarith [(Nat.cast_nonneg V : (0 : ℝ) ≤ V)]
  push_neg at hNV
  have hex : ∃ i, N / 2 ^ (i + 1) < V := by
    refine ⟨N, ?_⟩
    have hlt : N < 2 ^ (N + 1) :=
      lt_of_lt_of_le Nat.lt_two_pow_self (Nat.pow_le_pow_right (by norm_num) (by omega))
    rw [Nat.div_eq_of_lt hlt]
    exact hVpos
  set i := Nat.find hex with hidef
  have hi1 : N / 2 ^ (i + 1) < V := Nat.find_spec hex
  have hi0 : V ≤ N / 2 ^ i := by
    rcases Nat.eq_zero_or_pos i with h0 | hpos
    · rw [h0, pow_zero, Nat.div_one]; exact hNV
    · have h := Nat.find_min hex (show i - 1 < i by omega)
      push_neg at h
      rwa [show i - 1 + 1 = i by omega] at h
  have hhalf : N / 2 ^ i / 2 = N / 2 ^ (i + 1) := by
    rw [Nat.div_div_eq_div_mul, pow_succ]
  have hlt : N / 2 ^ i < 2 * V := by omega
  have hiL : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
    have hpow : 2 ^ i ≤ N := by
      have h1 : 1 ≤ N / 2 ^ i := le_trans hVpos hi0
      exact (Nat.one_le_div_iff (by positivity)).1 h1
    have := Nat.le_log_of_pow_le (by norm_num) hpow
    rw [ladderLen, ← hNdef]
    omega
  refine ⟨_, rung_mem_stoppable η₀ populations hη0 hsig hε hind hδ hδ1 hpAP hcard hρ0 hρcap i hiL
    hi0, ?_⟩
  show ((N / 2 ^ i : ℕ) : ℝ) ≤ 2 * (V : ℝ)
  exact_mod_cast hlt.le

set_option maxHeartbeats 4000000 in
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
every exponential — the share's own among them — then `α` at the gate's side tail
`exp(−2·n₁·τ²)`, with `n₁` a share of the uniform pool's rarer class, then the pool at `k/(pAP − t)`.  The collision masses enter as `m²ρ`, which is what
`collisionCap` bounds. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP qmin : ℝ) (nveto : ℕ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hbLo : voteSig η₀ ≤ bnd) (hbHi : bnd ≤ 1 - voteSig η₀)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hq : εcov ≤ sig η₀ * qmin / 2) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hveto : 2 * (populations.card : ℝ) * nveto * εcov ≤ δ)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfcap : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    ∃ B : State, B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ
      ∧ PassableAt O η₀ populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP qmin nveto B := by
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
  have hv := voteSlack_pos η₀ hη₀
  have hb := bandHalf_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀ hbLo
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκpos : 0 < κ := famCount_pos η₀ populations εcov δ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedState η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit with hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  -- the gate's floor
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast this
  have hs1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith [eta_nonneg O]
  have hlogα : 0 ≤ Real.log (1 / α) := Real.log_nonneg (by rw [le_div_iff₀ hαpos]; linarith)
  have hmα : 64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  obtain ⟨hqm, hn₁ge, hn₁α⟩ := gate_floor hs hs1 hεcov hε1 hq hlogα hsizeR hmα
  have hqpos : 0 < qmin := by
    by_contra h
    push_neg at h
    nlinarith
  have hcutq : cutBudget η₀ indecisionLimit εcov ≤ sig η₀ * qmin / 16 := by
    linarith [(cutBudget_le η₀ indecisionLimit εcov).1]
  have hseed : 8 ≤ (κ : ℝ) * voteSlack η₀ :=
    famCount_seed (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      η₀ populations εcov δ hη₀
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  have hvs : voteSig η₀ - flipFrac η₀ = voteSlack η₀ := by rw [flipFrac, voteSlack]; ring
  have hF0 := (flipFrac_pos η₀ hη₀).le
  have hFs : flipFrac η₀ ≤ voteSig η₀ := by
    rw [flipFrac]; linarith [voteSig_pos η₀ hη₀]
  obtain ⟨hlowShift, hhiShift⟩ := vote_shifts O hbOut hbIn hbLo hbHi hF0 hFs hκ0
  rw [hvs] at hlowShift hhiShift
  set c : ℕ := ⌈bnd * κ⌉₊ with hcdef
  have hcge : bnd * (κ : ℝ) ≤ (c : ℝ) := Nat.le_ceil _
  have hσ := voteSig_pos η₀ hη₀
  have hσs : voteSig η₀ = 7 * sig η₀ / 8 := rfl
  have hclt : (c : ℝ) < bnd * (κ : ℝ) + 1 :=
    Nat.ceil_lt_add_one (mul_nonneg (le_trans hσ.le hbLo) hκ0)
  have hbn0 : (0 : ℝ) ≤ (b : ℝ) := Nat.cast_nonneg _
  have hvs' : voteSlack η₀ = 3 * voteSig η₀ / 10 := rfl
  have hsκ : 80 / 3 ≤ voteSig η₀ * (κ : ℝ) := by rw [hvs'] at hseed; linarith
  have hb' : (b : ℝ) ≤ 3 * voteSig η₀ * (κ : ℝ) / 20 := by rw [hvs'] at hb; linarith
  have hκv : (0 : ℝ) ≤ (κ : ℝ) * voteSlack η₀ := mul_nonneg hκ0 hv.le
  -- the fields, named before the definitions are made opaque
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
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
  have hdS : decSlack η₀ populations indecisionLimit εcov δ crossLimit
      = voteSlack η₀ / 2 - 2 / ((κ : ℝ) + 1) := by rw [decSlack, hκdef]
  have hdS0 := decSlack_nonneg (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hdSle := decSlack_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hdSmul : (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit
      ≤ (κ : ℝ) * (voteSlack η₀ / 2) := mul_le_mul_of_nonneg_left hdSle hκ0
  have hκ1 : (0 : ℝ) ≤ (κ : ℝ) + 1 := by positivity
  obtain ⟨hlowShift1, hhiShift1⟩ := vote_shifts O hbOut hbIn hbLo hbHi hF0 hFs hκ1
  rw [hvs] at hlowShift1 hhiShift1
  have hdSk : ((κ : ℝ) + 1) * (2 / ((κ : ℝ) + 1)) = 2 := by field_simp
  -- the band's top, and its bottom, sit within `3s/10` of the centre, which is `s` inside `[0, 1]`
  have hhi1 : (B.hi : ℝ) + 1 ≤ (κ : ℝ) * (bnd + 9 * voteSig η₀ / 40) := by
    rw [hBhi]; push_cast; linarith [hclt, hb', hsκ]
  have hlo1 : (κ : ℝ) * bnd - 3 * voteSig η₀ * (κ : ℝ) / 20 ≤ (B.lo : ℝ) + 1 := by
    rw [hBlo, Nat.sub_sub, Nat.cast_sub hb1]; push_cast; linarith [hcge, hb']
  have hhiκ : B.hi + 1 ≤ κ := by
    have h : (B.hi : ℝ) + 1 ≤ (κ : ℝ) := by
      rw [hBhi]; push_cast; linarith [hclt, hb', hsκ, mul_le_mul_of_nonneg_right hbHi hκ0]
    exact_mod_cast h
  clear_value B b κ m c
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_⟩⟩,
    sig η₀ / 8, cutBudget η₀ indecisionLimit εcov / 32, pAP / 2,
    decSlack η₀ populations indecisionLimit εcov δ crossLimit,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32,
    flipBudget η₀ populations indecisionLimit εcov δ, cutBudget η₀ indecisionLimit εcov, flipFrac
      η₀, ⌊2 * qmin * (B.npref : ℝ) / 5⌋₊,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  -- in the schedule
  · rw [hBdef]
    exact solvedState_mem_schedule η₀ populations εcov δ α pAP
  -- Capped
  · rw [hBlo, hBhi]
    omega
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
    exact le_add_of_le_of_nonneg (solved_share η₀ populations hη₀ hεcov hindLim hδ hpAPPositive
      hcard hρ0 hρsf0 hρcap hρsfcap) (by split_ifs <;> positivity)
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · exact hindLim
  · exact hind1
  · exact hε1
  · exact div_nonneg hs.le (by norm_num)
  · linarith
  · linarith
  · exact hdS0
  · linarith [hγ.le]
  · linarith [hγ.le]
  · have := flipFrac_pos η₀ hη₀
    positivity
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
        + 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) / flipFrac η₀
        = cutBudget η₀ indecisionLimit εcov * (31 / 32) := by
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
      εcov δ α pAP crossLimit)
      hηle hη₀ hbOut hbIn hbLo hbHi hεcov hindLim hcard).2.2.1
  -- the pool holds a family
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    push_cast
    linarith
  -- the thresholds decide, and decide right
  · rw [hBhi, hBk, hdS]
    push_cast
    have e : ((κ : ℝ) + 1) * ((1 - O.ηIn) * (1 - flipFrac η₀)
          - (voteSlack η₀ / 2 - 2 / ((κ : ℝ) + 1)))
        = ((κ : ℝ) + 1) * ((1 - O.ηIn) * (1 - flipFrac η₀) - voteSlack η₀)
          + ((κ : ℝ) + 1) * voteSlack η₀ / 2 + ((κ : ℝ) + 1) * (2 / ((κ : ℝ) + 1)) := by ring
    rw [e, hdSk]
    linarith [hclt, hhiShift1, hb, hv, hbLo, hσ]
  · rw [hBlo, hBk, hdS, Nat.sub_sub, Nat.cast_sub hb1]
    push_cast
    have e : ((κ : ℝ) + 1) * (O.ηOut + (1 - O.ηOut) * flipFrac η₀
          + (voteSlack η₀ / 2 - 2 / ((κ : ℝ) + 1)))
        = ((κ : ℝ) + 1) * (O.ηOut + (1 - O.ηOut) * flipFrac η₀ + voteSlack η₀)
          - ((κ : ℝ) + 1) * voteSlack η₀ / 2 - ((κ : ℝ) + 1) * (2 / ((κ : ℝ) + 1)) := by ring
    rw [e, hdSk]
    linarith [hcge, hlowShift1, hb, hv, hbHi, hσ]
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega]
    push_cast
    have e : (κ : ℝ) * (O.ηOut + (1 - O.ηOut) * flipFrac η₀
          + decSlack η₀ populations indecisionLimit εcov δ crossLimit)
        = (κ : ℝ) * (O.ηOut + (1 - O.ηOut) * flipFrac η₀ + voteSlack η₀)
          - (κ : ℝ) * voteSlack η₀ + (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ
            crossLimit := by ring
    rw [e]
    linarith [hlowShift, hdSmul, hcge, hbn0, hκv]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.sub_sub, Nat.cast_sub hb1]
    push_cast
    have e : (κ : ℝ) * ((1 - O.ηIn) * (1 - flipFrac η₀)
          - decSlack η₀ populations indecisionLimit εcov δ crossLimit)
        = (κ : ℝ) * ((1 - O.ηIn) * (1 - flipFrac η₀) - voteSlack η₀)
          + (κ : ℝ) * voteSlack η₀ - (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ
            crossLimit := by ring
    rw [e]
    linarith [hhiShift, hdSmul, hclt, hbn0, hκv]
  -- the gate's two sides clear their thresholds
  · exact hqpos.le
  · rw [hBm]
    exact Nat.floor_pos.2 (by linarith)
  · linarith [(cutBudget_le η₀ indecisionLimit εcov).1]
  · rw [hBm]
    have hfl := Nat.floor_le (by positivity : (0 : ℝ) ≤ 2 * qmin * (m : ℝ) / 5)
    have h3 := mul_le_mul_of_nonneg_right hcutq hmR.le
    nlinarith
  · have : B.lo ≤ B.hi := by rw [hBlo, hBhi]; omega
    rw [hBkappa]
    omega
  · rw [hBkappa]
    exact hhiκ
  · intro n hn hnm
    rw [hBkappa, hBm]
    have hnR : ((⌊2 * qmin * (B.npref : ℝ) / 5⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    rw [hBm] at hnR
    have hmn : qmin * (m : ℝ) / 3 ≤ (n : ℝ) := le_trans hn₁ge hnR
    have hdiv : ((B.hi : ℝ) + 1) / (κ : ℝ) ≤ bnd + 9 * voteSig η₀ / 40 := by
      rw [div_le_iff₀ hκR]; linarith
    have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg _
    have h1 := mul_le_mul_of_nonneg_left hdiv hn0
    have h2 : 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) ≤ sig η₀ * (qmin * (m : ℝ)) / 8 := by
      linarith [mul_le_mul_of_nonneg_right hcutq hmR.le]
    have h3 : sig η₀ * (qmin * (m : ℝ)) ≤ sig η₀ * (3 * (n : ℝ)) :=
      mul_le_mul_of_nonneg_left (by linarith) hs.le
    have hsn : sig η₀ * (n : ℝ) ≤ (n : ℝ) / 2 := by
      nlinarith [mul_le_mul_of_nonneg_right (show sig η₀ ≤ 1 / 2 by rw [sig]; linarith) hn0]
    have hgap : 0 ≤ (n : ℝ) - 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) := by linarith
    have h6 := mul_le_mul_of_nonneg_right (show bnd + voteSig η₀ ≤ 1 - O.ηIn by linarith) hgap
    have h7 : (bnd + voteSig η₀) * (2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ))
        ≤ 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) :=
      mul_le_of_le_one_left (by positivity) (by linarith)
    rw [hσs] at h1 h6 h7
    linarith [h1, h2, h3, hsn, h6, h7, mul_nonneg hn0 hs.le]
  · intro n hn hnm
    rw [hBkappa, hBm]
    have hnR : ((⌊2 * qmin * (B.npref : ℝ) / 5⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    rw [hBm] at hnR
    have hmn : qmin * (m : ℝ) / 3 ≤ (n : ℝ) := le_trans hn₁ge hnR
    have hdiv : 1 - ((B.lo : ℝ) + 1) / ((κ : ℝ) + 1) ≤ (1 - bnd) + 3 * voteSig η₀ / 16 := by
      have : bnd - 3 * voteSig η₀ / 16 ≤ ((B.lo : ℝ) + 1) / ((κ : ℝ) + 1) := by
        rw [le_div_iff₀ (by positivity)]
        linarith [hlo1, hsκ, hbHi, hσ.le]
      linarith
    have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg _
    have h1 := mul_le_mul_of_nonneg_left hdiv hn0
    have h2 : 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) ≤ sig η₀ * (qmin * (m : ℝ)) / 8 := by
      linarith [mul_le_mul_of_nonneg_right hcutq hmR.le]
    have h3 : sig η₀ * (qmin * (m : ℝ)) ≤ sig η₀ * (3 * (n : ℝ)) :=
      mul_le_mul_of_nonneg_left (by linarith) hs.le
    have hsn : sig η₀ * (n : ℝ) ≤ (n : ℝ) / 2 := by
      nlinarith [mul_le_mul_of_nonneg_right (show sig η₀ ≤ 1 / 2 by rw [sig]; linarith) hn0]
    have hgap : 0 ≤ (n : ℝ) - 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) := by linarith
    have h6 := mul_le_mul_of_nonneg_right
      (show (1 - bnd) + voteSig η₀ ≤ 1 - O.ηOut by linarith) hgap
    have h7 : ((1 - bnd) + voteSig η₀) * (2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ))
        ≤ 2 * cutBudget η₀ indecisionLimit εcov * (m : ℝ) :=
      mul_le_of_le_one_left (by positivity) (by linarith)
    rw [hσs] at h1 h6 h7
    linarith [h1, h2, h3, hsn, h6, h7, mul_nonneg hn0 hs.le]
  · rw [hBm, show -2 * ((⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℕ) : ℝ) * (sig η₀ / 8) ^ 2
        = -(2 * ((⌊2 * qmin * (m : ℝ) / 5⌋₊ : ℕ) : ℝ) * (sig η₀ / 8) ^ 2) from by ring]
    exact exp_neg_le_of_log_le hαpos hn₁α
  -- the veto's tests are valid on sides of one class
  · rw [hBkappa, div_le_iff₀ hκR]
    linarith [hhi1,
      mul_le_mul_of_nonneg_left (show bnd + 9 * voteSig η₀ / 40 ≤ 1 - O.ηIn by linarith) hκ0]
  · rw [hBkappa, le_div_iff₀ (by positivity)]
    linarith [hlo1, hsκ, hbHi, hσ.le,
      mul_le_mul_of_nonneg_right (show O.ηOut ≤ bnd - 3 * voteSig η₀ / 16 by linarith) hκ1]
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail_seed η₀ populations hη₀ hεcov hindLim
  · rw [hBdef]
    exact solved_roundFail η₀ populations (le_trans hη0 hηle) hη₀ hεcov hε1 hδ hδ1 hq hveto
      hpAPPositive hindLim hind1 hcard hρ0 hρsf0 hρcap hρsfcap

open scoped Classical in
/-- Every population's own test passes, and still some population reads drifted. -/
noncomputable def driftFire (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (indecisionLimit α : ℝ) (nveto : ℕ) (B : State) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations, x ∈ retAt rule O populations uni indecisionLimit α nveto B j)
    ∧ ¬ noDrift O.mq populations uni B.lo B.hi α (clusterBy rule O.mq populations x B)
      B.npref nveto 0 x}

/-- A read comes back `1` at the rate its string's own class sets. -/
lemma measureReal_mq_one (O : Oracle μ S) (p : S) :
    μ.real {ω | O.mq p ω = 1} = O.rate p + (1 - 2 * O.rate p) * O.label p := by
  have hm : MeasurableSet {ω | O.mq p ω = 1} := (mq_meas O p) (measurableSet_singleton 1)
  have hind : O.mq p =ᵐ[μ] ({ω | O.mq p ω = 1}).indicator 1 := by
    filter_upwards [mq_bit O p] with ω hω
    rcases hω with h | h <;> simp [Set.indicator, h]
  rw [← mq_mean O p, integral_congr_ae hind, integral_indicator_one hm]

lemma measureReal_mq_one_in (O : Oracle μ S) {p : S} (hp : p ∈ O.L) :
    μ.real {ω | O.mq p ω = 1} = 1 - O.ηIn := by
  rw [measureReal_mq_one]
  simp [Oracle.rate, Oracle.label, hp]
  ring

lemma measureReal_mq_one_out (O : Oracle μ S) {p : S} (hp : p ∉ O.L) :
    μ.real {ω | O.mq p ω = 1} = O.ηOut := by
  rw [measureReal_mq_one]
  simp [Oracle.rate, Oracle.label, hp]

open scoped Classical in
/-- A condition decided by the reads on a finite set of strings is measurable. -/
lemma measurableSet_of_reads (O : Oracle μ S) (W : Finset S) (Ψ : Ω → Prop)
    (hΨ : ∀ ω ω', W.filter (fun w => O.mq w ω = 1) = W.filter (fun w => O.mq w ω' = 1) →
      (Ψ ω ↔ Ψ ω')) :
    MeasurableSet {ω | Ψ ω} := by
  classical
  have heq : {ω | Ψ ω} = {ω | (fun U : Finset S =>
      ∃ ω', W.filter (fun w => O.mq w ω' = 1) = U ∧ Ψ ω') (W.filter (fun w => O.mq w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq]
    exact ⟨fun h => ⟨ω, rfl, h⟩, fun ⟨ω', hU, h⟩ => (hΨ ω' ω hU).1 h⟩
  rw [heq]
  exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (A := W) (by simp)
    (fun U => ∃ ω', W.filter (fun w => O.mq w ω' = 1) = U ∧ Ψ ω'))

lemma measurableSet_image_cert (j : J) (m a : ℕ) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | (Finset.range m).image (fun i => certPrefix j (a + i) x) = Q} := by
  classical
  have hrw : ∀ x : Run Ω S J, (Finset.range m).image (fun i => certPrefix j (a + i) x)
      = (Finset.univ : Finset (Fin m)).image (fun i => (fun z : Fin m => certPrefix j (a + z.val) x) i) :=
    fun x => image_range_eq_image_univ m (fun i => certPrefix j (a + i) x)
  simp only [hrw]
  exact measurableSet_finData (fun z : Fin m => certPrefix j (a + z.val))
    (fun z => measurable_cert j _) (fun t => (Finset.univ : Finset (Fin m)).image t) Q

lemma clusterAt_card_le (O : Oracle μ S) (populations : Finset J) (B : State) (hk : 0 < B.k)
    (x : Run Ω S J) : (clusterBy rule O.mq populations x B).card ≤ B.k - 1 := by
  have h := clusterAt_card_add_one (rule := rule) O populations B x
  have h2 : (clusterFull rule O populations x B).card ≤ B.k :=
    rule.card_le _ _ _ _ _ B.k (poolOrder_injOn_screened O populations B x) hk
  omega

open scoped Classical in
lemma drifted_congr (mq : S → Ω → ℝ) (lo hi : ℕ) (L : ℝ) (F P : Finset S) {ω ω' : Ω}
    (hv : ∀ p ∈ P, voteCount mq F p ω = voteCount mq F p ω')
    (hm : ∀ p ∈ P, (mq p ω = 1 ↔ mq p ω' = 1)) :
    drifted mq lo hi L F P ω ↔ drifted mq lo hi L F P ω' := by
  have h1 : P.filter (fun p => hi < voteCount mq F p ω)
      = P.filter (fun p => hi < voteCount mq F p ω') :=
    Finset.filter_congr (fun p hp => by rw [hv p hp])
  have h2 : P.filter (fun p => voteCount mq F p ω ≤ lo)
      = P.filter (fun p => voteCount mq F p ω' ≤ lo) :=
    Finset.filter_congr (fun p hp => by rw [hv p hp])
  have h3 : P.filter (fun p => hi < voteCount mq F p ω ∧ mq p ω = 1)
      = P.filter (fun p => hi < voteCount mq F p ω' ∧ mq p ω' = 1) :=
    Finset.filter_congr (fun p hp => by rw [hv p hp, hm p hp])
  have h4 : P.filter (fun p => voteCount mq F p ω ≤ lo ∧ mq p ω = 1)
      = P.filter (fun p => voteCount mq F p ω' ≤ lo ∧ mq p ω' = 1) :=
    Finset.filter_congr (fun p hp => by rw [hv p hp, hm p hp])
  unfold drifted
  rw [h1, h2, h3, h4]

open scoped Classical in
lemma sidesHeld_congr (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) {ω ω' : Ω}
    (hv : ∀ p ∈ P, voteCount mq F p ω = voteCount mq F p ω') :
    sidesHeld mq lo hi F P ω = sidesHeld mq lo hi F P ω' := by
  have h1 : P.filter (fun p => hi < voteCount mq F p ω)
      = P.filter (fun p => hi < voteCount mq F p ω') :=
    Finset.filter_congr (fun p hp => by rw [hv p hp])
  have h2 : P.filter (fun p => voteCount mq F p ω ≤ lo)
      = P.filter (fun p => voteCount mq F p ω' ≤ lo) :=
    Finset.filter_congr (fun p hp => by rw [hv p hp])
  unfold sidesHeld
  rw [h1, h2]

open scoped Classical in
/-- The tests' own level.  At the uniform pool a certified side cannot also read drifted; at
the others every side the veto reads holds only prefixes of its class, so conditioned on the
sides the votes choose, each one's test fires at most its share of `α`. -/
theorem measureReal_driftFire_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (huni : uni ∈ populations)
    (indecisionLimit α : ℝ) (nveto : ℕ) (B : State) (hα0 : 0 ≤ α) (hα : α < 1 / 2)
    (hk : 2 ≤ B.k) (hhi : B.hi + 1 ≤ B.k - 1) (hlohi : B.lo + 1 ≤ B.k - 1)
    (hpA : ((B.hi : ℝ) + 1) / ((B.k - 1 : ℕ) : ℝ) ≤ 1 - O.ηIn)
    (hpR : O.ηOut ≤ ((B.lo : ℝ) + 1) / (((B.k - 1 : ℕ) : ℝ) + 1)) :
    (runMeasure μ D Dsf).real (driftFire rule O populations uni indecisionLimit α nveto B)
      ≤ α := by
  classical
  set κ : ℕ := B.k - 1 with hκ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast (show 0 < κ by omega)
  set pA : ℝ := ((B.hi : ℝ) + 1) / (κ : ℝ) with hpAdef
  set pR : ℝ := ((B.lo : ℝ) + 1) / ((κ : ℝ) + 1) with hpRdef
  have hpA0 : 0 ≤ pA := by positivity
  have hpA1 : pA ≤ 1 := by
    rw [hpAdef, div_le_one hκR]
    exact_mod_cast hhi
  have hpR0 : 0 ≤ pR := by positivity
  have hpR1 : pR ≤ 1 := by
    rw [hpRdef, div_le_one (by positivity)]
    have : ((B.lo : ℝ) + 1) ≤ (κ : ℝ) := by exact_mod_cast hlohi
    linarith
  set V : J → Run Ω S J → Finset S := fun j x => splitSample uni j B.npref nveto 0 x with hVdef
  have hVsub : ∀ j x, V j x ⊆ certOf j (2 * B.npref + nveto) x := by
    intro j x p hp
    simp only [hVdef, splitSample] at hp
    split_ifs at hp with hju
    · subst hju
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
      exact Finset.mem_image.2 ⟨B.npref + i, Finset.mem_range.2
        (by have := Finset.mem_range.1 hi; omega), rfl⟩
    · obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
      exact Finset.mem_image.2 ⟨B.npref + i, Finset.mem_range.2
        (by have := Finset.mem_range.1 hi; omega), rfl⟩
  have hVmeas : ∀ j Q, MeasurableSet {x : Run Ω S J | V j x = Q} := by
    intro j Q
    by_cases hju : j = uni
    · have h : ∀ x : Run Ω S J, V j x = (Finset.range (B.npref + 0)).image
          (fun i => certPrefix uni (B.npref + i) x) := by
        intro x
        simp only [hVdef, splitSample, if_pos hju, gateOf]
      simp only [h]
      exact measurableSet_image_cert uni _ B.npref Q
    · have h : ∀ x : Run Ω S J, V j x = (Finset.range nveto).image
          (fun i => certPrefix j (B.npref + i) x) := by
        intro x
        simp only [hVdef, splitSample, if_neg hju, vetoOf]
      simp only [h]
      exact measurableSet_image_cert j nveto B.npref Q
  -- the event priced: the family is of its round's size, the veto's sides are pure, and a
  -- population other than the uniform one reads drifted
  set Ψ : Finset S → (J → Finset S) → Ω → Prop := fun A₀ t ω =>
    A₀.card = κ
    ∧ (∀ j ∈ populations, j ≠ uni → ∀ p ∈ t j, cutCorrect O B.lo B.hi A₀ p ω)
    ∧ ∃ j ∈ populations, j ≠ uni ∧ drifted O.mq B.lo B.hi
      (α / ((∑ j' ∈ populations, sidesHeld O.mq B.lo B.hi A₀ (t j') ω : ℕ) : ℝ)) A₀ (t j) ω
    with hΨ
  set Fire : Set (Run Ω S J) := {x | Disjoint (prefixesAt populations B.npref x)
      (populations.biUnion (fun j => V j x))
    ∧ Ψ (clusterBy rule O.mq populations x B) (fun j => V j x) (oracleNoise x)} with hFire
  -- off the uniform pool, the drift is the veto's
  have hsub : driftFire rule O populations uni indecisionLimit α nveto B ⊆ Fire := by
    rintro x ⟨hall, hnd⟩
    set F := clusterBy rule O.mq populations x B with hF
    set ω := oracleNoise x with hω
    obtain ⟨hsize, -, hgate, -, -⟩ := hall uni huni
    have hcardF : F.card = κ := by
      have h1 : (clusterBy rule O.mq populations x B).card = κ := by
        have := clusterAt_card_le (rule := rule) O populations B (by omega) x
        omega
      exact h1
    refine ⟨?_, hcardF, ?_, ?_⟩
    · rw [Finset.disjoint_biUnion_right]
      intro j hj
      exact Finset.disjoint_of_subset_right (hVsub j x) (hall j hj).2.2.2.1
    · intro j hj hju p hp
      have hp' : p ∈ vetoOf j B.npref nveto x := by
        simpa [hVdef, splitSample, if_neg hju] using hp
      exact (hall j hj).2.2.2.2 hju p hp'
    · simp only [noDrift, not_forall, not_not, exists_prop] at hnd
      obtain ⟨j, hj, hdj⟩ := hnd
      refine ⟨j, hj, fun hju => ?_, hdj⟩
      rw [hju] at hdj
      obtain ⟨-, hcert⟩ := hgate rfl
      have hVu : splitSample uni uni B.npref nveto 0 x = gateOf uni B.npref 0 x := by
        simp [splitSample]
      -- a certified pool holds a side, so the level is at most `α`
      have hsidesN : 1 ≤ ∑ j' ∈ populations,
          sidesHeld O.mq B.lo B.hi F (splitSample uni j' B.npref nveto 0 x) ω := by
        refine le_trans ?_ (Finset.single_le_sum (f := fun j' =>
          sidesHeld O.mq B.lo B.hi F (splitSample uni j' B.npref nveto 0 x) ω)
          (fun _ _ => Nat.zero_le _) huni)
        show 1 ≤ sidesHeld O.mq B.lo B.hi F (splitSample uni uni B.npref nveto 0 x) ω
        rw [hVu]
        unfold sidesHeld
        rcases hcert.1 with h | h
        · rw [if_pos h]; omega
        · rw [if_pos h]; omega
      have hsides : (1 : ℝ) ≤ ((∑ j' ∈ populations,
          sidesHeld O.mq B.lo B.hi F (splitSample uni j' B.npref nveto 0 x) ω : ℕ) : ℝ) := by
        exact_mod_cast hsidesN
      have hL : α / ((∑ j' ∈ populations,
          sidesHeld O.mq B.lo B.hi F (splitSample uni j' B.npref nveto 0 x) ω : ℕ) : ℝ) ≤ α :=
        div_le_self hα0 hsides
      rw [hVu] at hdj
      have hpA' : ((B.hi : ℝ) + 1) / (F.card : ℝ) = pA := by rw [hcardF]
      have hpR' : ((B.lo : ℝ) + 1) / ((F.card : ℝ) + 1) = pR := by rw [hcardF]
      rcases hdj with ⟨hne, hc⟩ | ⟨hne, hc⟩
      · have hup := hcert.2.1 hne
        rw [hpA'] at hup hc
        exact not_lower_of_upper _ _ hpA0 hpA1 (Finset.card_le_card (fun p hp =>
          Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hp).1, (Finset.mem_filter.1 hp).2.1⟩))
          hα hL hup hc
      · have hlow := hcert.2.2 hne
        rw [hpR'] at hlow hc
        exact not_upper_of_lower _ _ hpR0 hpR1 (Finset.card_le_card (fun p hp =>
          Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hp).1, (Finset.mem_filter.1 hp).2.1⟩))
          hα hL hlow hc
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  -- the event is measurable: at fixed draws it is decided by the family and the reads
  have hFireMeas : MeasurableSet Fire := by
    have hR : ∀ (P C : Finset S) (wt : S → ℝ) (ord : S → ℕ) (t : J → Finset S), MeasurableSet (if
        (1 : S) ∈ C then
        {x : Run Ω S J | Disjoint P (populations.biUnion t)}
        ∩ oracleNoise ⁻¹' {ω : Ω | Ψ (clusterOf rule O wt ord B.sc B.scd P C B.k ω) t ω}
        else (∅ : Set (Run Ω S J))) := by
      intro P C wt ord t
      split_ifs with hone
      · refine MeasurableSet.inter (MeasurableSet.const _) (measurable_nz ?_)
        refine measurableSet_of_fam (T := C.powerset)
          (fun ω => Finset.mem_powerset.2 (clusterOf_subset O wt ord B.sc B.scd P C B.k ω hone))
          (fun A₀ => measurableSet_clusterOf O wt ord B.sc B.scd P C B.k hone A₀)
          (fun A₀ => {ω : Ω | Ψ A₀ t ω}) (fun A₀ => ?_)
        set Call : Finset S := populations.biUnion t with hCall
        refine measurableSet_of_reads O (Call ∪ Call.biUnion (fun p => A₀.image (fun v => p * v)))
          (fun ω => Ψ A₀ t ω) (fun ω ω' hU => ?_)
        have hmem : ∀ w ∈ Call ∪ Call.biUnion (fun p => A₀.image (fun v => p * v)),
            (O.mq w ω = 1 ↔ O.mq w ω' = 1) := by
          intro w hw
          have h1 : w ∈ (Call ∪ Call.biUnion (fun p => A₀.image (fun v => p * v))).filter
              (fun w => O.mq w ω = 1) ↔ O.mq w ω = 1 := by simp [hw]
          have h2 : w ∈ (Call ∪ Call.biUnion (fun p => A₀.image (fun v => p * v))).filter
              (fun w => O.mq w ω' = 1) ↔ O.mq w ω' = 1 := by simp [hw]
          rw [← h1, ← h2, hU]
        have hv : ∀ p ∈ Call, voteCount O.mq A₀ p ω = voteCount O.mq A₀ p ω' := by
          intro p hp
          unfold voteCount
          congr 1
          refine Finset.filter_congr (fun v hv => ?_)
          exact hmem _ (Finset.mem_union_right _ (Finset.mem_biUnion.2
            ⟨p, hp, Finset.mem_image.2 ⟨v, hv, rfl⟩⟩))
        have hm : ∀ p ∈ Call, (O.mq p ω = 1 ↔ O.mq p ω' = 1) :=
          fun p hp => hmem p (Finset.mem_union_left _ hp)
        have htj : ∀ j ∈ populations, t j ⊆ Call :=
          fun j hj p hp => Finset.mem_biUnion.2 ⟨j, hj, hp⟩
        have hsum : ∑ j' ∈ populations, sidesHeld O.mq B.lo B.hi A₀ (t j') ω
            = ∑ j' ∈ populations, sidesHeld O.mq B.lo B.hi A₀ (t j') ω' :=
          Finset.sum_congr rfl (fun j' hj' =>
            sidesHeld_congr O.mq B.lo B.hi A₀ (t j') (fun p hp => hv p (htj j' hj' hp)))
        have hcc : ∀ j ∈ populations, ∀ p ∈ t j,
            (cutCorrect O B.lo B.hi A₀ p ω ↔ cutCorrect O B.lo B.hi A₀ p ω') := by
          intro j hj p hp
          unfold cutCorrect
          rw [hv p (htj j hj hp)]
        simp only [hΨ, hsum]
        refine and_congr Iff.rfl (and_congr ?_ ?_)
        · exact ⟨fun h j hj hju p hp => (hcc j hj p hp).1 (h j hj hju p hp),
            fun h j hj hju p hp => (hcc j hj p hp).2 (h j hj hju p hp)⟩
        · refine exists_congr (fun j => ⟨fun ⟨hj, hju, hd⟩ => ⟨hj, hju, ?_⟩,
            fun ⟨hj, hju, hd⟩ => ⟨hj, hju, ?_⟩⟩)
          · exact (drifted_congr O.mq B.lo B.hi _ A₀ (t j)
              (fun p hp => hv p (htj j hj hp)) (fun p hp => hm p (htj j hj hp))).1 hd
          · exact (drifted_congr O.mq B.lo B.hi _ A₀ (t j)
              (fun p hp => hv p (htj j hj hp)) (fun p hp => hm p (htj j hj hp))).2 hd
      · exact MeasurableSet.empty
    have hrw : Fire = {x : Run Ω S J | x ∈ (fun P C wt ord t => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P (populations.biUnion t)}
          ∩ oracleNoise ⁻¹' {ω : Ω | Ψ (clusterOf rule O wt ord B.sc B.scd P C B.k ω) t ω}
          else (∅ : Set (Run Ω S J)))
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (prefixWeight populations B.npref x) (poolOrder B.nsuff x) (fun j => V j x)} := by
      ext x
      simp only [hFire, Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_inter_iff,
        Set.mem_preimage, ← clusterAt_eq_clusterOf O populations B x]
    rw [hrw]
    refine measurableSet_of_draws populations B (fun x j => V j x) (fun a => ?_) _ hR
    have h : {x : Run Ω S J | (fun j => V j x) = a} = ⋂ j : J, {x : Run Ω S J | V j x = a j} := by
      ext x; simp only [Set.mem_setOf_eq, Set.mem_iInter, funext_iff]
    rw [h]
    exact MeasurableSet.iInter (fun j => hVmeas j (a j))
  -- at fixed draws, condition on the sides the votes pick
  have hEnn : runMeasure μ D Dsf Fire ≤ ENNReal.ofReal α := by
    refine runMeasure_slice_le D Dsf _ hFireMeas _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hV' : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    set Vd : J → Finset S := fun j => if j = uni
      then (Finset.range (B.npref + 0)).image (fun i => d.2 uni (B.npref + i))
      else (Finset.range nveto).image (fun i => d.2 j (B.npref + i)) with hVd
    have hVdω : ∀ ω : Ω, ∀ j, V j ((ω, d) : Run Ω S J) = Vd j := by
      intro ω j
      by_cases h : j = uni
      · subst h
        simp [hVdef, hVd, splitSample, gateOf, certPrefix]
      · simp [hVdef, hVd, splitSample, vetoOf, certPrefix, h]
    set Call : Finset S := populations.biUnion Vd with hCall
    have hCallPre : ∀ p ∈ Call, p ∈ Pre := by
      intro p hp
      obtain ⟨j, hj, hpj⟩ := Finset.mem_biUnion.1 hp
      simp only [hVd] at hpj
      split_ifs at hpj with hju
      · obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hpj
        exact hdC uni huni _
      · obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hpj
        exact hdC j hj _
    set fam : Ω → Finset S :=
      fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
    by_cases hdisj : Disjoint Pd Call
    · set Q : Finset S := readSet Pd Cd ∪ readSet Call (Cd.erase 1) with hQ
      have hdisjQ : Disjoint (↑Call : Set S) (↑Q : Set S) :=
        disjoint_gateReads hflat Pd Cd Call hP hV' hCallPre hdisj
      have hfamC : ∀ ω, fam ω ⊆ Cd.erase 1 := fun ω v hv =>
        Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, clusterAt_subset O populations B _ hv⟩
      set sel : Ω → ℕ × (J → Finset S × Finset S) := fun ω =>
        ((fam ω).card, fun j => ((Vd j).filter (fun p => B.hi < voteCount O.mq (fam ω) p ω),
          (Vd j).filter (fun p => voteCount O.mq (fam ω) p ω ≤ B.lo))) with hsel
      set T : Finset (ℕ × (J → Finset S × Finset S)) := Finset.range (Cd.card + 1) ×ˢ
        Fintype.piFinset (fun _ : J => Call.powerset ×ˢ Call.powerset) with hT
      have hVdC : ∀ j ∈ populations, Vd j ⊆ Call := fun j hj p hp => Finset.mem_biUnion.2 ⟨j, hj, hp⟩
      -- only the populations' own samples are read; elsewhere the sides are empty
      set sel' : Ω → ℕ × (J → Finset S × Finset S) := fun ω =>
        ((sel ω).1, fun j => if j ∈ populations then (sel ω).2 j else (∅, ∅)) with hsel'
      have hselT : ∀ ω, sel' ω ∈ T := by
        intro ω
        refine Finset.mem_product.2 ⟨Finset.mem_range.2 (Nat.lt_succ_of_le
          (le_trans (Finset.card_le_card (hfamC ω)) (Finset.card_erase_le))), ?_⟩
        refine Fintype.mem_piFinset.2 (fun j => ?_)
        by_cases hj : j ∈ populations
        · simp only [hsel', if_pos hj, hsel]
          exact Finset.mem_product.2 ⟨Finset.mem_powerset.2
            ((Finset.filter_subset _ _).trans (hVdC j hj)),
            Finset.mem_powerset.2 ((Finset.filter_subset _ _).trans (hVdC j hj))⟩
        · simp only [hsel', if_neg hj]
          exact Finset.mem_product.2 ⟨Finset.empty_mem_powerset _, Finset.empty_mem_powerset _⟩
      have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ Call,
          voteCount O.mq (fam ω) p ω = voteCount O.mq (fam ω') p ω' := by
        intro ω ω' h p hp
        have hf : fam ω = fam ω' :=
          clusterAt_congr O populations B d (fun w hw => h w (Finset.mem_union_left _ hw))
        rw [← hf]
        exact voteCount_congr O _ p (fun v hv => by
          rw [mq_congr O (h _ (Finset.mem_union_right _ (mem_readSet hp (hfamC ω hv))))])
      have hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel' ω = sel' ω' := by
        intro ω ω' h
        have hf : fam ω = fam ω' :=
          clusterAt_congr O populations B d (fun w hw => h w (Finset.mem_union_left _ hw))
        have hvc' := hvc ω ω' h
        refine Prod.ext (by simp only [hsel', hsel, hf]) (funext (fun j => ?_))
        by_cases hj : j ∈ populations
        · simp only [hsel', hsel, if_pos hj]
          refine Prod.ext (Finset.filter_congr (fun p hp => ?_))
            (Finset.filter_congr (fun p hp => ?_))
          · rw [hvc' p (hVdC j hj hp)]
          · rw [hvc' p (hVdC j hj hp)]
        · simp only [hsel', if_neg hj]
      -- the test at a fixed split
      set Pr : ℕ × (J → Finset S × Finset S) → Finset S → Prop := fun t U =>
        t.1 = κ
        ∧ (∀ j ∈ populations, j ≠ uni → (∀ p ∈ (t.2 j).1, p ∈ O.L) ∧ (∀ p ∈ (t.2 j).2, p ∉ O.L))
        ∧ ∃ j ∈ populations, j ≠ uni ∧
          (((t.2 j).1.Nonempty ∧ binomCdfLe (t.2 j).1.card pA ((t.2 j).1 ∩ U).card
              ≤ α / ((∑ j' ∈ populations, ((if (t.2 j').1.Nonempty then 1 else 0)
                + if (t.2 j').2.Nonempty then 1 else 0) : ℕ) : ℝ))
            ∨ ((t.2 j).2.Nonempty ∧ binomSfGe (t.2 j).2.card pR ((t.2 j).2 ∩ U).card
              ≤ α / ((∑ j' ∈ populations, ((if (t.2 j').1.Nonempty then 1 else 0)
                + if (t.2 j').2.Nonempty then 1 else 0) : ℕ) : ℝ))) with hPr
      have hbad : ∀ t ∈ T, μ.real {ω | Pr t (Call.filter (fun p => O.mq p ω = 1))} ≤ α := by
        intro t ht
        have htC : ∀ j, (t.2 j).1 ⊆ Call ∧ (t.2 j).2 ⊆ Call := by
          intro j
          have h := Fintype.mem_piFinset.1 (Finset.mem_product.1 ht).2 j
          exact ⟨Finset.mem_powerset.1 (Finset.mem_product.1 h).1,
            Finset.mem_powerset.1 (Finset.mem_product.1 h).2⟩
        by_cases hpure : t.1 = κ ∧ ∀ j ∈ populations, j ≠ uni →
            (∀ p ∈ (t.2 j).1, p ∈ O.L) ∧ (∀ p ∈ (t.2 j).2, p ∉ O.L)
        · set N : ℝ := ((∑ j' ∈ populations, ((if (t.2 j').1.Nonempty then 1 else 0)
              + if (t.2 j').2.Nonempty then 1 else 0) : ℕ) : ℝ) with hN
          set L : ℝ := α / N with hLdef
          have hL0 : 0 ≤ L := div_nonneg hα0 (Nat.cast_nonneg _)
          -- each side's hits are its reads
          have hhits : ∀ (A : Finset S), A ⊆ Call → ∀ ω,
              (A ∩ Call.filter (fun p => O.mq p ω = 1)).card
                = (A.filter (fun p => O.mq p ω = 1)).card := by
            intro A hA ω
            congr 1
            ext p
            simp only [Finset.mem_inter, Finset.mem_filter]
            exact ⟨fun ⟨h1, _, h2⟩ => ⟨h1, h2⟩, fun ⟨h1, h2⟩ => ⟨h1, hA h1, h2⟩⟩
          set aA : J → Set Ω := fun j => {ω | (t.2 j).1.Nonempty
            ∧ binomCdfLe (t.2 j).1.card pA ((t.2 j).1.filter (fun p => O.mq p ω = 1)).card ≤ L}
            with haA
          set aR : J → Set Ω := fun j => {ω | (t.2 j).2.Nonempty
            ∧ binomSfGe (t.2 j).2.card pR ((t.2 j).2.filter (fun p => O.mq p ω = 1)).card ≤ L}
            with haR
          have hsubE : {ω | Pr t (Call.filter (fun p => O.mq p ω = 1))}
              ⊆ ⋃ j ∈ populations.erase uni, (aA j ∪ aR j) := by
            rintro ω ⟨-, -, j, hj, hju, hfire⟩
            refine Set.mem_biUnion (Finset.mem_erase.2 ⟨hju, hj⟩) ?_
            rcases hfire with ⟨hne, hc⟩ | ⟨hne, hc⟩
            · left
              refine ⟨hne, ?_⟩
              rw [← hhits _ (htC j).1 ω]
              exact hc
            · right
              refine ⟨hne, ?_⟩
              rw [← hhits _ (htC j).2 ω]
              exact hc
          have hA1 : ∀ j ∈ populations.erase uni, μ.real (aA j)
              ≤ (if (t.2 j).1.Nonempty then 1 else 0) * L := by
            intro j hj
            obtain ⟨hju, hj'⟩ := Finset.mem_erase.1 hj
            split_ifs with hne
            · refine le_trans (measureReal_mono (fun ω hω => hω.2) (measure_ne_top _ _)) ?_
              rw [one_mul]
              refine lowerTest_valid O _ hL0 hpA0 hpA1 (fun p hp => ?_)
              rw [measureReal_mq_one_in O ((hpure.2 j hj' hju).1 p hp)]
              rw [hpAdef]
              exact hpA
            · have : aA j = ∅ := by
                ext ω; simp only [haA, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
                exact fun h => hne h.1
              rw [this, measureReal_empty, zero_mul]
          have hR1 : ∀ j ∈ populations.erase uni, μ.real (aR j)
              ≤ (if (t.2 j).2.Nonempty then 1 else 0) * L := by
            intro j hj
            obtain ⟨hju, hj'⟩ := Finset.mem_erase.1 hj
            split_ifs with hne
            · refine le_trans (measureReal_mono (fun ω hω => hω.2) (measure_ne_top _ _)) ?_
              rw [one_mul]
              refine upperTest_valid O _ hL0 hpR0 hpR1 (fun p hp => ?_)
              rw [measureReal_mq_one_out O ((hpure.2 j hj' hju).2 p hp)]
              rw [hpRdef]
              exact hpR
            · have : aR j = ∅ := by
                ext ω; simp only [haR, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
                exact fun h => hne h.1
              rw [this, measureReal_empty, zero_mul]
          refine le_trans (measureReal_mono hsubE (measure_ne_top _ _))
            (le_trans (measureReal_biUnion_finset_le _ _) ?_)
          have hstep : ∑ j ∈ populations.erase uni, μ.real (aA j ∪ aR j)
              ≤ ∑ j ∈ populations.erase uni, ((((if (t.2 j).1.Nonempty then 1 else 0)
                + if (t.2 j).2.Nonempty then 1 else 0 : ℕ) : ℝ) * L) := by
            refine Finset.sum_le_sum (fun j hj => le_trans (measureReal_union_le _ _) ?_)
            have := hA1 j hj
            have := hR1 j hj
            push_cast
            split_ifs at * <;> linarith
          refine le_trans hstep ?_
          rw [← Finset.sum_mul]
          have hsub' : ∑ j ∈ populations.erase uni, ((((if (t.2 j).1.Nonempty then 1 else 0)
              + if (t.2 j).2.Nonempty then 1 else 0 : ℕ) : ℝ)) ≤ N := by
            rw [hN, Nat.cast_sum]
            exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.erase_subset _ _)
              (fun _ _ _ => Nat.cast_nonneg _)
          rcases (Nat.cast_nonneg _ : (0 : ℝ) ≤ N).eq_or_lt with hN0 | hNpos
          · have hz : ∑ j ∈ populations.erase uni, ((((if (t.2 j).1.Nonempty then 1 else 0)
                + if (t.2 j).2.Nonempty then 1 else 0 : ℕ) : ℝ)) = 0 :=
              le_antisymm (hN0 ▸ hsub') (Finset.sum_nonneg (fun _ _ => Nat.cast_nonneg _))
            rw [hz, zero_mul]
            exact hα0
          · calc (∑ j ∈ populations.erase uni, ((((if (t.2 j).1.Nonempty then 1 else 0)
                  + if (t.2 j).2.Nonempty then 1 else 0 : ℕ) : ℝ))) * L
                ≤ N * L := mul_le_mul_of_nonneg_right hsub' hL0
              _ = α := by rw [hLdef, mul_div_assoc']; exact mul_div_cancel_left₀ α hNpos.ne'
        · have hz : {ω | Pr t (Call.filter (fun p => O.mq p ω = 1))} = (∅ : Set Ω) := by
            ext ω
            simp only [Set.mem_empty_iff_false, iff_false]
            rintro ⟨h1, h2, -⟩
            exact hpure ⟨h1, h2⟩
          rw [hz, measureReal_empty]
          exact hα0
      have ht₀ : ((0 : ℕ), fun _ : J => ((∅ : Finset S), (∅ : Finset S))) ∈ T :=
        Finset.mem_product.2 ⟨Finset.mem_range.2 (Nat.succ_pos _), Fintype.mem_piFinset.2
          (fun _ => Finset.mem_product.2 ⟨Finset.empty_mem_powerset _, Finset.empty_mem_powerset _⟩)⟩
      have hmain := selection_read_bound O Call Q hdisjQ T _ ht₀ sel' hselT hcongr Pr α hα0 hbad
      -- the fired event at these draws is the test at the split the votes picked
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ Fire}
          ⊆ {ω | Pr (sel' ω) (Call.filter (fun p => O.mq p ω = 1))} := by
        rintro ω ⟨-, hcard, hpure, j, hj, hju, hdr⟩
        have hsidesEq : ∀ j' ∈ populations,
            sidesHeld O.mq B.lo B.hi (fam ω) (Vd j') ω
              = (if ((sel' ω).2 j').1.Nonempty then 1 else 0)
                + if ((sel' ω).2 j').2.Nonempty then 1 else 0 := by
          intro j' hj'
          simp only [hsel', hsel, if_pos hj', sidesHeld]
        refine ⟨hcard, fun j' hj' hju' => ⟨fun p hp => ?_, fun p hp => ?_⟩, j, hj, hju, ?_⟩
        · simp only [hsel', hsel, if_pos hj', Finset.mem_filter] at hp
          have := (hpure j' hj' hju' p hp.1).1 hp.2
          simp only [Oracle.label, Set.indicator_apply] at this
          by_contra hn
          rw [if_neg hn] at this
          norm_num at this
        · simp only [hsel', hsel, if_pos hj', Finset.mem_filter] at hp
          have := (hpure j' hj' hju' p hp.1).2 hp.2
          simp only [Oracle.label, Set.indicator_apply] at this
          intro hn
          rw [if_pos hn] at this
          norm_num at this
        · have hdr' : drifted O.mq B.lo B.hi
              (α / ((∑ j' ∈ populations, sidesHeld O.mq B.lo B.hi (fam ω) (Vd j') ω : ℕ) : ℝ))
              (fam ω) (Vd j) ω := hdr
          rw [Finset.sum_congr rfl hsidesEq] at hdr'
          have hcardR : ((fam ω).card : ℝ) = (κ : ℝ) := by exact_mod_cast hcard
          have hsetA : ((Vd j).filter (fun p => B.hi < voteCount O.mq (fam ω) p ω)
                ∩ Call.filter (fun p => O.mq p ω = 1))
              = (Vd j).filter (fun p => B.hi < voteCount O.mq (fam ω) p ω ∧ O.mq p ω = 1) := by
            ext p
            simp only [Finset.mem_inter, Finset.mem_filter]
            constructor
            · rintro ⟨⟨h1, h2⟩, -, h3⟩
              exact ⟨h1, h2, h3⟩
            · rintro ⟨h1, h2, h3⟩
              exact ⟨⟨h1, h2⟩, hVdC j hj h1, h3⟩
          have hsetR : ((Vd j).filter (fun p => voteCount O.mq (fam ω) p ω ≤ B.lo)
                ∩ Call.filter (fun p => O.mq p ω = 1))
              = (Vd j).filter (fun p => voteCount O.mq (fam ω) p ω ≤ B.lo ∧ O.mq p ω = 1) := by
            ext p
            simp only [Finset.mem_inter, Finset.mem_filter]
            constructor
            · rintro ⟨⟨h1, h2⟩, -, h3⟩
              exact ⟨h1, h2, h3⟩
            · rintro ⟨h1, h2, h3⟩
              exact ⟨⟨h1, h2⟩, hVdC j hj h1, h3⟩
          simp only [hsel', hsel, if_pos hj]
          rcases hdr' with ⟨hne, hc⟩ | ⟨hne, hc⟩
          · left
            refine ⟨hne, ?_⟩
            rw [hsetA, hpAdef, ← hcardR]
            exact hc
          · right
            refine ⟨hne, ?_⟩
            rw [hsetR, hpRdef, ← hcardR]
            exact hc
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ Fire} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hd, -⟩
        exact hdisj hd
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf Fire).toReal
      ≤ (ENNReal.ofReal α).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = α := ENNReal.toReal_ofReal hα0

set_option maxHeartbeats 1000000 in
/-- Part 2 — the loop terminates: except with probability `δ/2 + α`, some reachable state
passes both tests.  The `α` is the veto's: its tests refuse a sound family at up to their level.

Each growth step draws fresh suffixes, accept-preserving with probability `≥ pAP`; once the
pool holds enough of them and the prefix count is large enough, every population's vote is
decisive on all but `indecisionLimit` of its mass.  The per-step trigger is block-local, so
`geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under `δ/2`.

Here the gate has to pass, not merely be sound — the gate's power, the complement of the
`α` it spends — which is why `pAP > 0` is needed rather than merely useful.
`ACCEPT_PRESERVING_GIVE_UP = 20` caps the refusals in the code; the statement carries no
cap.

Only the uniform pool is gated, and it has to carry both labels, each on a `qmin` share of its
mass: the FNR asks the family to decide some prefix each way, and each side of the cut has to
hold enough of the sample to certify.  `εcov ≤ sig η₀ · qmin / 2` keeps the cut budget small
against that share, so each side is nearly pure.

`cutBudget` is held under `indecisionLimit/2` by its own definition, so the family size is
solved against a single misfire rate rather than the max of the round's two. -/
theorem loop_terminates {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α : ℝ) (ρ pAP δ qmin : ℝ)
    (hsig : O.η < 1 / 2) (huni : uni ∈ populations)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hbOut : O.ηOut + voteSig η₀ ≤ bnd)
    (hbIn : O.ηIn + voteSig η₀ ≤ 1 - bnd) (hb0 : voteSig η₀ ≤ bnd) (hb1 : bnd ≤ 1 - voteSig η₀)
    (hqL : qmin ≤ (D uni).real O.L) (hqN : qmin ≤ (D uni).real O.Lᶜ)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hq : εcov ≤ sig η₀ * qmin / 2) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hveto : 2 * (populations.card : ℝ) * nveto * εcov ≤ δ)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsf : collisionMass Dsf
        ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State //
        B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ},
      x ∉ retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val}
      ≤ δ / 2 + α := by
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  obtain ⟨B, hB, hpass⟩ := exists_passable O populations D Dsf indecisionLimit εcov α δ ρ
    (collisionMass Dsf) pAP qmin nveto hsig hpop hηle hη₀ hbOut hbIn hb0 hb1 hεcov hε1 hq hδ hδ1
    hveto hαpos hα
    hindLim
    hind1
    hpAPPositive
    hpAPBound hρ hρ0 le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hρcap hρsf
  obtain ⟨τ, th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, f, n₁, hmpos, hkpos, hindLim,
    hind1', hε1, hτ, hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hf0, hpAP0, hlcut, hlcl,
    hpAPBound, hρsf, hheavy, hscd, hscLow, hscHigh, hcount,
    hhiUp, hloUp, hhiLo, hloLo, hq0, hn₁pos, hlcut1, hn₁, hlohi, hhi, hga, hgr, hαn, hpA, hpR,
    hEhalf, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hlpos : 0 < l := by rw [hl]; linarith
  have hl2 : 2 * l = indecisionLimit := by rw [hl]; ring
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  -- the whole failure at the one state, population by population
  have hsub : {x : Run Ω S J | ∀ B' : {B : State //
          B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ},
        x ∉ retBy rule O.mq populations uni indecisionLimit α nveto B'.val.npref B'.val}
      ⊆ (⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt rule O populations uni indecisionLimit α nveto B j})
        ∪ driftFire rule O populations uni indecisionLimit α nveto B := by
    intro x hx
    by_cases hall : ∀ j ∈ populations,
        x ∈ retAt rule O populations uni indecisionLimit α nveto B j
    · refine Or.inr ⟨hall, fun hd => hx ⟨B, hB⟩ (mem_ret_of_retAt O populations huni
        indecisionLimit α B x hall hd)⟩
    · push_neg at hall
      obtain ⟨j, hj, hjx⟩ := hall
      exact Or.inl (Set.mem_biUnion hj hjx)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hfire := measureReal_driftFire_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf huni
    indecisionLimit α nveto B hαpos.le hα hkpos hhi hlohi hpA hpR
  refine le_trans (add_le_add (measureReal_biUnion_finset_le _ _) le_rfl) ?_
  have hper : ∀ j ∈ populations,
      (runMeasure μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt rule O populations uni indecisionLimit α nveto B j}
        ≤ roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) qmin
          (((populations.card : ℝ) * Δ + gdirty) / f) nveto n₁ B := by
    intro j hj
    have hstall := measureReal_stalled_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf B (by omega)
      j₀ hj₀ γscr pAP tap (collisionMass Dsf) ρ hγscr hpAP0 htap hpAPBound hscd hsig.le hscLow
      hcount hρsf hρsf0 (hρ j₀ hj₀) hρ0
    have hdirtyI := measureReal_dirtyMember_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj B
      hsig.le hmpos Δ γdirty gdirty ρ hΔ hγdirty hgdirty hρ0 (hρ j hj) hscd hscHigh
    have hnoclean := measureReal_noCleanRef_le D Dsf O populations B pAP tap
      (collisionMass Dsf) hpAP0 htap hpAPBound le_rfl hρsf0
      (by
        have hk2 : (2 : ℝ) ≤ (B.k : ℝ) := by exact_mod_cast hkpos
        linarith [hcount, hk2])
    have hdirty : (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + gdirty}
        ≤ ((B.npref : ℝ) ^ 2 * ρ
            + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
          + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
            + (B.nsuff : ℝ) ^ 2 * collisionMass Dsf) := by
      have hcov : {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
            flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + gdirty}
          ⊆ ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
                ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
              ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
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
    have hIn1 : O.ηIn ≤ 1 := by linarith [le_trans (le_max_left O.ηIn O.ηOut) hsig.le]
    have hOut1 : O.ηOut ≤ 1 := by linarith [le_trans (le_max_right O.ηIn O.ηOut) hsig.le]
    have hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E := by
      intro F p h1 hfl hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      have hcardI : ((insert 1 F).card : ℝ) = (B.k : ℝ) := by
        rw [Finset.card_insert_of_notMem h1, hcardF, hκ]
        have : 1 ≤ B.k := by omega
        push_cast [this]
        ring
      have hcle : (F.card : ℝ) ≤ ((insert 1 F).card : ℝ) := by
        exact_mod_cast Finset.card_le_card (Finset.subset_insert _ _)
      have hflI : flipCount O (insert 1 F) p ≤ ((insert 1 F).card : ℝ) * f :=
        le_trans (flipCount_insert_one O F p)
          (le_trans hfl (mul_le_mul_of_nonneg_right hcle hf0.le))
      refine le_trans (decided_whp O (insert 1 F) p B.lo (B.hi + 1) f γdec hγdec O.ηOut O.ηIn
        le_rfl le_rfl hOut1 hIn1 hflI ?_ ?_) ?_
      · rw [hcardI]; push_cast; exact hhiUp
      · rw [hcardI]; exact hloUp
      · rw [hE, hcardI]
        refine Real.exp_le_exp.2 ?_
        have hk : (κ : ℝ) ≤ (B.k : ℝ) := by rw [hκ]; exact_mod_cast Nat.sub_le _ _
        nlinarith [sq_nonneg γdec]
    have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E := by
      intro F p hfl hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      refine le_trans (cutCorrect_whp O F p B.lo B.hi f γdec O.ηOut O.ηIn le_rfl le_rfl hOut1
        hIn1 hfl hγdec ?_ ?_) ?_
      · rw [hcardF]; linarith [hhiLo]
      · rw [hcardF]; exact hloLo.le
      · rw [hE, hcardF]
    have hmain := measureReal_notRetAt_le (nveto := nveto) hflat O populations D Dsf hsupp hsuppSf j hj huni B
      hmpos hsig.le
      α τ l lcut f E ((populations.card : ℝ) * Δ + gdirty) ρ th qmin κ κ n₁ (by omega) le_rfl
      (Real.exp_nonneg _) (by linarith) (by rw [hl]; exact hlcl) hτ
      hρ hρ0
      (by positivity) hth hf0
      hheavy hq0 hqL hqN _ hstall _ hdirty hdec hcut hlcut1 hn₁ hn₁pos (by omega) hlohi hhi
      hga hgr hαn
    rw [hl2] at hmain
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  have hsum : ∑ j ∈ populations, (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations uni indecisionLimit α nveto B j}
      ≤ (populations.card : ℝ)
          * roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) qmin
            (((populations.card : ℝ) * Δ + gdirty) / f) nveto n₁ B := by
    refine le_trans (Finset.sum_le_sum hper) (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul]
  linarith [hsum, hbudget, hfire]

#print axioms validity_of_returned
#print axioms loop_terminates

end Loop

/-- `validity_of_returned` (whatever the loop returns is good, whenever it is returned) except
w.p. `δ/2`, and `loop_terminates` (it returns) except w.p. `δ/2 + α`, glued by
`sound_and_terminating`. -/
theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O rule populations uni D Dsf _ _ Pre Suf η₀ bnd indecisionLimit εcov α δ ρ
    pAP qmin crossLimit k a nveto
    hηle hη₀ hbOut hbIn hb0 hb1 huni hflat hsupp hsuppSf hρ hpAPPositive hpAPBound _hq0 hqL hqN hq
    hindLim hind1
    hαpos hα hεcov hε1 hδ _hbudget hρcap hρsf ha hveto
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ - α ≤ 0) measureReal_nonneg
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2
      * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 16 :=
    le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith)
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hcoll : (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2
      * collisionMass Dsf ≤ δ / 16 := by
    obtain ⟨-, -, -, -, -, h2⟩ := solved_tails η₀ populations hη₀ hεcov hindLim hδ hpAPPositive
      (by linarith) (le_trans (tsum_nonneg (fun a => sq_nonneg _))
        (hρ hpop.choose hpop.choose_spec)) hρsf0 hρcap hρsf
    have hnn : (0 : ℝ) ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2
        * collisionMass Dsf := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hcard1 hnn]
  have h := sound_and_terminating (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val ∩ FailAt rule O populations D indecisionLimit εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ bnd populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α nveto B.val.npref B.val) δ α
    (validity_of_returned hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α hindLim
      hsig hpop hηle hη₀ hbOut hbIn hb0 hb1 ρ (collisionMass Dsf) hρ le_rfl
      hρsf0 hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind hcoll)
    (loop_terminates hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α ρ pAP
      δ qmin hsig huni hηle hη₀ hbOut hbIn hb0 hb1 hqL hqN hεcov hε1 hq hδ hδ1 hveto hαpos hα
      hindLim
      hind1
      hpAPPositive
      hpAPBound hρ (le_trans (tsum_nonneg (fun a => sq_nonneg _))
        (hρ hpop.choose hpop.choose_spec)) hρcap hρsf)
  refine le_trans h (measureReal_mono ?_ (measure_ne_top _ _))
  intro x hx
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not] at hx ⊢
  obtain ⟨⟨B₀, hex⟩, hall⟩ := hx
  refine ⟨⟨B₀, by rwa [certSize_of_mem_schedule ha (mem_schedule_of_mem_stoppable B₀.property)]⟩,
    fun B hB j hj => ?_⟩
  rw [certSize_of_mem_schedule ha (mem_schedule_of_mem_stoppable B.property)] at hB
  obtain ⟨hcov, hund⟩ := hall B hB j hj
  refine ⟨le_trans hcov (measureReal_mono (fun p hp => cutCorrect_insert_one O _ _ _
    (one_not_mem_clusterAt O populations B.val x) p _ hp) (measure_ne_top _ _)), hund⟩

#print axioms clustering_correct

end OrthoDFA
