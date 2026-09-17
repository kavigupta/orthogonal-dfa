import OrthoDFA.Distributional
import OrthoDFA.Model
import OrthoDFA.BinomTail
import OrthoDFA.Grouping
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The adaptive clustering loop: the proof

`ClusteringCorrect`, in `OrthoDFA.Model`, is what is claimed; this file is how it is
reached.  Two parts, composed by `sound_and_terminating`:

* `validity_of_returned` — whatever is returned is valid, whenever it is returned;
* `loop_terminates` — the loop returns at some round;

each except w.p. `δ/2`.

They meet at `PassableAt`, the arithmetic a round has to satisfy for both of its tests to
pass, and `exists_passable` shows the computed schedule reaches such a state.  That
arithmetic is solved in order: the miscut budget `lcut` below the indecision limit, the flip
budget `Δ` below it over the family size, the screen's two margins below `Δ(1−2η)²`, then the
prefix count large enough for every exponential — including the share's own condition, which
mentions `log` of the prefix count and is closed by `log x ≤ 2√x` — then `α` at the gate's
tail, then the pool at `k/(pAP − t)`.

The union bound over `stoppable` is a finite sum: the ladder has `log₂(prefCount) + 1` rungs
each carrying `δ/(2·L)`, so no summable weight over all budgets is needed and no state has to
be encoded as a number.  `vote_mem_grid` is what lets the state be a `Budget` at all — a
threshold enters every event only through the count it cuts at.
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

/-- Binomial CDF: `P[Bin(N,p) ≤ j]`. -/
noncomputable def binomCdf (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (j + 1), (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

lemma mq_meas (O : Oracle μ S) (p : S) : Measurable (mq O p) := by
  show Measurable (fun ω => O.label p + (1 - 2 * O.label p) * O.noise p ω)
  exact measurable_const.add (measurable_const.mul (O.noise_meas p))

lemma mq_indep (O : Oracle μ S) : iIndepFun (fun p : S => mq O p) μ :=
  O.noise_indep.comp (fun p x => O.label p + (1 - 2 * O.label p) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma mq_bit (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, mq O p ω = 0 ∨ mq O p ω = 1 := by
  filter_upwards [O.noise_bit p] with ω hω
  rcases O.label_bit p with hl | hl <;> rcases hω with hn | hn <;>
    norm_num [mq, hl, hn]

lemma mq_icc (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, mq O p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [mq_bit O p] with ω hω
  rcases hω with h | h <;> rw [h] <;> norm_num

/-- `E[mq p] = η + (1−2η)·ℓ(p)`: a truly-accepting prefix reads accepting with probability
`1 − η`, a truly-rejecting one with probability `η`. -/
lemma mq_mean (O : Oracle μ S) (p : S) : μ[mq O p] = O.η + (1 - 2 * O.η) * O.label p := by
  have : μ[mq O p] = ∫ ω, (O.label p + (1 - 2 * O.label p) * O.noise p ω) ∂μ := rfl
  rw [this, integral_add (integrable_const _) ((O.noise_int p).const_mul _), integral_const,
    integral_const_mul, O.noise_mean p]
  simp only [measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
  ring

lemma eta_nonneg (O : Oracle μ S) : 0 ≤ O.η := by
  rw [← O.noise_mean (1 : S)]
  refine integral_nonneg_of_ae ?_
  filter_upwards [O.noise_icc (1 : S)] with ω hω
  exact hω.1

lemma mq_integrable (O : Oracle μ S) (w : S) : Integrable (mq O w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (mq_meas O w).aemeasurable (mq_icc O w)

open scoped Classical in
/-- The gate's hit count *is* the sum of the column's reads: they are `0/1`. -/
lemma hits_eq_sum (O : Oracle μ S) (A : Finset S) :
    ∀ᵐ ω ∂μ, ((A.filter (fun p => mq O p ω = 1)).card : ℝ) = ∑ p ∈ A, mq O p ω := by
  filter_upwards [(ae_ball_iff A.countable_toSet).2 (fun p _ => mq_bit O p)] with ω hω
  rw [← Finset.sum_filter_add_sum_filter_not A (fun p => mq O p ω = 1)]
  have h1 : ∑ p ∈ A.filter (fun p => mq O p ω = 1), mq O p ω
      = ((A.filter (fun p => mq O p ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun p hp => (Finset.mem_filter.mp hp).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ p ∈ A.filter (fun p => ¬ (mq O p ω = 1)), mq O p ω = 0 := by
    refine Finset.sum_eq_zero (fun p hp => ?_)
    obtain ⟨hpA, hne⟩ := Finset.mem_filter.mp hp
    rcases hω p hpA with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

/-! ## The run space

`runMeasure` is a concrete space with a concrete measure, so the marginals the concentration
arguments consume (`map_firstDraws`, `map_drawBlock`) are lemmas rather than hypotheses.

On the deduplication gap `OrthoDFA.Model` records: do not be tempted to close it by modelling
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

/-! The whole blocks of a run, as opposed to `suffixDraw`/`prefixDraw`/`certPrefix`, which pick out one draw.
The marginal arguments slice the product at these boundaries. -/

/-- The draw block: the suffix stream paired with every population's prefix stream. -/
def draws (x : Run Ω S J) : (ℕ → S) × (J → ℕ → S) := x.2.1

/-- The prefix streams. -/
def prefixStreams (x : Run Ω S J) : J → ℕ → S := x.2.1.2

/-- The certification stream. -/
def certStream (x : Run Ω S J) : J × ℕ → S := x.2.2

@[fun_prop]
lemma measurable_draws : Measurable (draws : Run Ω S J → (ℕ → S) × (J → ℕ → S)) :=
  measurable_fst.comp measurable_snd

@[fun_prop]
lemma measurable_prfs : Measurable (prefixStreams : Run Ω S J → J → ℕ → S) :=
  measurable_snd.comp (measurable_fst.comp measurable_snd)

@[fun_prop]
lemma measurable_certs : Measurable (certStream : Run Ω S J → J × ℕ → S) :=
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
noncomputable def solvedBudget (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : Budget :=
  solvedBudgetAt O populations εcov δ pAP (prefCount O populations εcov δ α pAP)

lemma ladderLen_pos (O : Oracle μ S) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < ladderLen O populations εcov δ α pAP := by
  rw [ladderLen]; omega

lemma solvedBudget_mem_schedule (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) :
    solvedBudget O populations εcov δ α pAP
      ∈ schedule O populations εcov δ α pAP := by
  rw [schedule, solvedBudget]
  refine Finset.mem_image.2 ⟨0, Finset.mem_range.2 (ladderLen_pos _ _ _ _ _ _), ?_⟩
  norm_num

lemma schedule_card_le (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) :
    (schedule O populations εcov δ α pAP).card
      ≤ ladderLen O populations εcov δ α pAP := by
  rw [schedule]
  exact le_trans Finset.card_image_le (by simp)

lemma sig_pos (O : Oracle μ S) (hsig : O.η < 1 / 2) : 0 < sig O := by
  rw [sig]; linarith

lemma cutBudget_pos {εcov : ℝ} (hε : 0 < εcov) : 0 < cutBudget εcov := by
  rw [cutBudget]; positivity

lemma famCount_pos (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) :
    0 < famCount O populations εcov δ := by
  rw [famCount]; omega

lemma flipBudget_pos (O : Oracle μ S) (populations : Finset J) {εcov δ : ℝ}
    (hε : 0 < εcov) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < flipBudget O populations εcov δ := by
  rw [flipBudget]
  refine div_pos (cutBudget_pos hε) (mul_pos (mul_pos (by norm_num) hcard) ?_)
  exact_mod_cast famCount_pos O populations εcov δ

lemma screenMargin_pos (O : Oracle μ S) (populations : Finset J) {εcov δ : ℝ}
    (hsig : O.η < 1 / 2) (hε : 0 < εcov) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < screenMargin O populations εcov δ := by
  rw [screenMargin]
  exact mul_pos (flipBudget_pos O populations hε hcard) (pow_pos (sig_pos O hsig) 2)

lemma one_mem_poolAt (M : ℕ) (x : Run Ω S J) : (1 : S) ∈ poolAt M x :=
  Finset.mem_insert_self _ _

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, mq O (p * v) ω) / F.card

open scoped Classical in
/-- Votes live on a grid.  Every membership query is `0` or `1`, so a family of `k`
suffixes votes in `{0, 1/k, …, 1}`.

This is what collapses the union over boundaries.  Every comparison the algorithm makes
against a real-valued threshold — the cluster centre's `cn/cd`, the gate's `lo` and `hi` —
comes down to which grid cell that threshold sits in, an integer in `{0, …, k+1}`, so
`Budget` can carry the counts instead. -/
lemma vote_mem_grid (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ∃ j : ℕ, j ≤ F.card ∧ vote O F p ω = (j : ℝ) / F.card := by
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  refine ⟨(F.filter (fun v => mq O (p * v) ω = 1)).card,
    Finset.card_le_card (Finset.filter_subset _ _), ?_⟩
  have hsum : ∑ v ∈ F, mq O (p * v) ω
      = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
    rw [← Finset.sum_filter_add_sum_filter_not F (fun v => mq O (p * v) ω = 1)]
    have h1 : ∑ v ∈ F.filter (fun v => mq O (p * v) ω = 1), mq O (p * v) ω
        = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
      rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.mp hv).2), Finset.sum_const,
        nsmul_eq_mul, mul_one]
    have h0 : ∑ v ∈ F.filter (fun v => ¬ (mq O (p * v) ω = 1)), mq O (p * v) ω = 0 := by
      refine Finset.sum_eq_zero (fun v hv => ?_)
      obtain ⟨hvF, hne⟩ := Finset.mem_filter.mp hv
      rcases hω v hvF with h | h
      · exact h
      · exact absurd h hne
    rw [h1, h0, add_zero]
  show (∑ v ∈ F, mq O (p * v) ω) / F.card = _
  rw [hsum]

/-- The cluster never drifts off the seed.  `identify_cluster_around` stops the moment
`ε` would leave, so every family the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround O cn cd P cands ω k := by
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
    screened O sc scd P cands ω ⊆ cands :=
  Finset.filter_subset _ _

lemma screenCount_one (O : Oracle μ S) (P : Finset S) (ω : Ω) : screenCount O P 1 ω = 0 := by
  classical
  unfold screenCount
  rw [Finset.card_eq_zero]
  exact Finset.filter_eq_empty_iff.2 (fun p _ => by simp [mul_one])

open scoped Classical in
lemma one_mem_screened (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (ω : Ω)
    (hone : (1 : S) ∈ cands) : (1 : S) ∈ screened O sc scd P cands ω := by
  classical
  refine Finset.mem_filter.2 ⟨hone, ?_⟩
  show scd * screenCount O P 1 ω ≤ sc * P.card
  rw [screenCount_one]
  simpa using Nat.zero_le _

lemma screenedAt_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : screenedAt O populations B x ⊆ poolAt B.M x :=
  screened_subset _ _ _ _ _ _

open scoped Classical in
/-- The seed always survives: it is the reference, so its disagreement count is zero. -/
lemma one_mem_screenedAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : (1 : S) ∈ screenedAt O populations B x :=
  one_mem_screened O B.sc B.scd _ _ _ (one_mem_poolAt B.M x)

/-- The seed's column is read at a different string from the split.  The gate counts
`mq p`, the oracle at `p`; the split reads `p · v` for the family members `v`.  With `ε`
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
    mq O w ω = mq O w ω' := by simp [mq, h]

lemma voteCount_congr (O : Oracle μ S) (F : Finset S) (p : S) {ω ω' : Ω}
    (h : ∀ v ∈ F, (mq O (p * v) ω = 1 ↔ mq O (p * v) ω' = 1)) :
    voteCount O F p ω = voteCount O F p ω' := by
  classical
  unfold voteCount
  exact congrArg Finset.card (Finset.filter_congr (fun v hv => h v hv))

lemma hammingLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) {P cands : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    hammingLoss O F cn cd P ω v = hammingLoss O F cn cd P ω' v := by
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
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterLoss O F cn cd P cands ω = clusterLoss O F cn cd P cands ω' := by
  classical
  funext v
  unfold clusterLoss
  split_ifs with hv
  · exact hammingLoss_congr O F cn cd hF hv h
  · rfl

lemma lloydStep_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ) {F : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    lloydStep O cn cd P cands ω k F = lloydStep O cn cd P cands ω' k F := by
  classical
  unfold lloydStep
  rw [clusterLoss_congr O F cn cd P cands hF h]

lemma clusterLoss_nonneg (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : 0 ≤ clusterLoss O F cn cd P cands ω v := by
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
    clusterLoss O {(1 : S)} cn cd P cands ω 1 = 0 := by
  classical
  unfold clusterLoss
  rw [if_pos hone, hammingLoss]
  have hempty : P.filter (fun p => ¬ ((mq O (p * 1) ω = 1)
      ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p ω)) = ∅ := by
    refine Finset.filter_eq_empty_iff.2 (fun p _ => ?_)
    simp only [Classical.not_not, Finset.card_singleton, mul_one, mul_comm]
    unfold voteCount
    by_cases h : mq O p ω = 1
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq O (p * v) ω = 1) = {(1 : S)} := by
        refine Finset.filter_eq_self.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_singleton, mul_one]
      exact ⟨fun _ => hcd, fun _ => h⟩
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq O (p * v) ω = 1) = ∅ := by
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
    (lloydStep O cn cd P cands ω k {(1 : S)}).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  have hguard : ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss O {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss O {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss O {(1 : S)} cn cd P cands ω v
        ≤ clusterLoss O {(1 : S)} cn cd P cands ω w := by
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
    (hF : F.card = k) : (lloydStep O cn cd P cands ω k F).card = k := by
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
    (clusterAround O cn cd P cands ω k).card = k := by
  classical
  unfold clusterAround
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card = k →
      ((lloydStep O cn cd P cands ω k)^[n] F).card = k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (lloydStep_card_keep O cn cd P cands ω k hone hk hkpos hF)
  rw [show k * P.card + 1 = (k * P.card) + 1 from rfl, Function.iterate_succ_apply]
  exact hiter _ _ (lloydStep_seed_card O hcd P cands ω k hone hk hkpos)

lemma lloydStep_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) {F : Finset S} (hF : F ⊆ cands) :
    lloydStep O cn cd P cands ω k F ⊆ cands := by
  classical
  unfold lloydStep
  split_ifs
  · exact Finset.insert_subset hone
      (le_trans (leastLossSubset_subset' _ _ _) (Finset.erase_subset _ _))
  · exact hF

lemma lloydIterate_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands → (lloydStep O cn cd P cands ω k)^[n] F ⊆ cands := by
  intro n
  induction n with
  | zero => intro F hF; rw [Function.iterate_zero_apply]; exact hF
  | succ n ih =>
      intro F hF
      rw [Function.iterate_succ_apply]
      exact ih _ (lloydStep_subset O cn cd P cands ω k hone hF)

lemma lloydIterate_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands →
      (lloydStep O cn cd P cands ω k)^[n] F = (lloydStep O cn cd P cands ω' k)^[n] F := by
  intro n
  induction n with
  | zero => intro F _; rw [Function.iterate_zero_apply, Function.iterate_zero_apply]
  | succ n ih =>
      intro F hF
      calc (lloydStep O cn cd P cands ω k)^[n + 1] F
          = (lloydStep O cn cd P cands ω k)^[n] (lloydStep O cn cd P cands ω k F) :=
            Function.iterate_succ_apply _ _ _
        _ = (lloydStep O cn cd P cands ω k)^[n] (lloydStep O cn cd P cands ω' k F) :=
            congrArg (fun z => (lloydStep O cn cd P cands ω k)^[n] z)
              (lloydStep_congr O cn cd P cands k hF h)
        _ = (lloydStep O cn cd P cands ω' k)^[n] (lloydStep O cn cd P cands ω' k F) :=
            ih _ (lloydStep_subset O cn cd P cands ω' k hone hF)
        _ = (lloydStep O cn cd P cands ω' k)^[n + 1] F :=
            (Function.iterate_succ_apply _ _ _).symm

/-- The cluster reads only `readSet`.  Two noise draws agreeing at `p · v` for every
representative prefix and candidate suffix give the same family — so neither the family nor
any vote cast with it is decided by the oracle's bit at a bare prefix, which is the bit the
gate scores. -/
lemma clusterAround_congr_mq (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterAround O cn cd P cands ω k = clusterAround O cn cd P cands ω' k :=
  lloydIterate_congr O cn cd P cands k hone h _ _ (by simpa using hone)

/-- The screen reads only `readSet`.  Its two reads at a prefix are `p · v` and
`p · ε = p`, and the seed is a candidate, so both are already there. -/
lemma screenCount_congr (O : Oracle μ S) {P cands : Finset S} (hone : (1 : S) ∈ cands)
    {v : S} (hv : v ∈ cands) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    screenCount O P v ω = screenCount O P v ω' := by
  classical
  unfold screenCount
  refine congrArg Finset.card (Finset.filter_congr (fun p hp => ?_))
  have h1 := h _ (mem_readSet hp hv)
  have h0 : (mq O p ω = 1 ↔ mq O p ω' = 1) := by
    have := h _ (mem_readSet hp hone)
    rwa [mul_one] at this
  exact not_congr (iff_congr h1 h0)

lemma screenedAt_congr (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    screenedAt O populations B (ω, d) = screenedAt O populations B (ω', d) := by
  classical
  unfold screenedAt
  refine Finset.filter_congr (fun v hv => ?_)
  show B.scd * screenCount O (prefixesAt populations B.m ((ω, d) : Run Ω S J)) v ω
      ≤ B.sc * (prefixesAt populations B.m ((ω, d) : Run Ω S J)).card
    ↔ B.scd * screenCount O (prefixesAt populations B.m ((ω', d) : Run Ω S J)) v ω'
      ≤ B.sc * (prefixesAt populations B.m ((ω', d) : Run Ω S J)).card
  rw [screenCount_congr O (one_mem_poolAt _ _) hv h]
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
    (hone : (1 : S) ∈ cands) : clusterAround O cn cd P cands ω k ⊆ cands := by
  classical
  unfold clusterAround
  exact lloydIterate_subset O cn cd P cands ω k hone _ _ (by simpa using hone)

lemma clusterAt_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : clusterAt O populations x B ⊆ poolAt B.M x :=
  fun v hv => screenedAt_subset O populations B x
    (clusterAround_subset O B.cn B.cd _ _ (oracleNoise x) B.k (one_mem_screenedAt O populations B x) hv)

/-- The family is decided by the bits on `readSet` — the screen's reads and the
clustering's alike. -/
lemma clusterAt_congr (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    clusterAt O populations (ω, d) B = clusterAt O populations (ω', d) B := by
  classical
  have hbit : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), (mq O w ω = 1 ↔ mq O w ω' = 1) :=
    fun w hw => by rw [mq_congr O (h w hw)]
  have hcands : screenedAt O populations B (ω, d) = screenedAt O populations B (ω', d) :=
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
    voteCount O F p ω ≤ voteCount O (F.erase 1) p ω + 1 := by
  classical
  unfold voteCount
  have hsub : F.filter (fun v => mq O (p * v) ω = 1)
      ⊆ insert (1 : S) ((F.erase 1).filter (fun v => mq O (p * v) ω = 1)) := by
    intro v hv
    obtain ⟨hvF, hvm⟩ := Finset.mem_filter.1 hv
    by_cases h1 : v = 1
    · rw [h1]
      exact Finset.mem_insert_self _ _
    · exact Finset.mem_insert_of_mem (Finset.mem_filter.2 ⟨Finset.mem_erase.2 ⟨h1, hvF⟩, hvm⟩)
  exact le_trans (Finset.card_le_card hsub) (Finset.card_insert_le _ _)

open scoped Classical in
/-- The prefixes the gate's family accepts at population `j`. -/
noncomputable def sideAcc (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  (certOf j B.m x).filter
    (fun p => B.hi - 1 < voteCount O ((clusterAt O populations x B).erase 1) p (oracleNoise x))

open scoped Classical in
/-- The prefixes it rejects. -/
noncomputable def sideRej (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  (certOf j B.m x).filter
    (fun p => voteCount O ((clusterAt O populations x B).erase 1) p (oracleNoise x) ≤ B.lo)

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
    MeasurableSet[noiseAlg O T] {ω | mq O w ω = 1} := by
  have hpre : {ω | mq O w ω = 1}
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
    MeasurableSet[noiseAlg O T] {ω | A.filter (fun p => mq O p ω = 1) = U} :=
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
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun p => mq O p ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun p => mq O p ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun p => mq O p ω = 1) = U} := by
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
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun v => mq O (r v) ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun v => mq O (r v) ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun v => mq O (r v) ω = 1) = U} := by
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
/-- A worst case survives the scoring rule being chosen elsewhere.

`selection_side_bound` lets the *set* being scored be chosen by `Q`; this lets anything be,
so long as it is chosen by `Q`.  The read set is fixed at `C`, and the selection is an
arbitrary value the block `Q` determines — for the gate it is the pair (accept side, decided
set), which no single `Finset S` records but which the votes fix all the same.

Conditioning is free for the same reason as before: the selection is `Q`-measurable, the
score is `C`-measurable, and `C` and `Q` are disjoint. -/
theorem selection_read_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    {β : Type*} [DecidableEq β] (T : Finset β) (t₀ : β) (ht₀ : t₀ ∈ T)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω')
    (P : β → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ t ∈ T, μ.real {ω | P t (C.filter (fun p => mq O p ω = 1))} ≤ E) :
    μ.real {ω | P (sel ω) (C.filter (fun p => mq O p ω = 1))} ≤ E := by
  classical
  set Bad : β → Set Ω :=
    fun t => {ω | P t (C.filter (fun p => mq O p ω = 1))} with hBaddef
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
  have hsub : {ω | P (sel ω) (C.filter (fun p => mq O p ω = 1))}
      ⊆ {ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (sel' ω)
      rw [hsel'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | P (sel ω) (C.filter (fun p => mq O p ω = 1))}
      ≤ μ.real ({ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (sel' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (sel' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

open scoped Classical in
/-- A worst case survives the side being chosen elsewhere.

`hbad` bounds the score's failure for each *fixed* side; the conclusion bounds it for the
side the run actually produces.  What makes that free is that the side is decided by the
block `Q` while the score reads the strings `r '' C`, and those are disjoint.  The gate
takes `r = id` (it scores prefixes); the vote at a population prefix `p` takes
`r v = p * v`. -/
theorem selection_side_bound (O : Oracle μ S) (C Q : Finset S) (r : S → S)
    (hdisj : Disjoint (↑(C.image r) : Set S) (↑Q : Set S))
    (T : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ C)
    (side : Ω → Finset S) (hside : ∀ ω, side ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (P : Finset S → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, μ.real {ω | P A₀ (A₀.filter (fun v => mq O (r v) ω = 1))} ≤ E) :
    μ.real {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))} ≤ E := by
  classical
  set Bad : Finset S → Set Ω :=
    fun A₀ => {ω | P A₀ (A₀.filter (fun v => mq O (r v) ω = 1))} with hBaddef
  set side' : Ω → Finset S := fun ω => if ω ∈ noiseClean O Q then side ω else t₀ with hside'def
  have hmeasBad : ∀ A₀, MeasurableSet (Bad A₀) := fun A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) r (by simp) _)
  have hmeasBadC : ∀ A₀ ∈ T, MeasurableSet[noiseAlg O ↑(C.image r)] (Bad A₀) :=
    fun A₀ hA₀ => measurableSet_filter_pred_map O r
      (fun v hv => Finset.mem_image_of_mem r (hTC A₀ hA₀ hv)) _
  have hsel' : ∀ ω, side' ω ∈ T := by
    intro ω
    rw [hside'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simpa [hc] using hside ω
    · simpa [hc] using ht₀
  have hsplit : ∀ A₀, {ω | side' ω = A₀}
      = ({ω | side ω = A₀} ∩ noiseClean O Q) ∪ (if A₀ = t₀ then (noiseClean O Q)ᶜ else ∅) := by
    intro A₀
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : A₀ = t₀ <;>
      simp [hside'def, hc, he, Set.mem_setOf_eq, eq_comm (a := t₀)]
  have hmeasSelQ : ∀ A₀, MeasurableSet[noiseAlg O ↑Q] {ω | side' ω = A₀} := by
    intro A₀
    rw [hsplit A₀]
    refine MeasurableSet.union (measurableSet_side_clean O Q side hcongr A₀) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ A₀, MeasurableSet {ω | side' ω = A₀} := fun A₀ =>
    noiseAlg_le O ↑Q _ (hmeasSelQ A₀)
  have hindep : ∀ A₀ ∈ T,
      μ.real ({ω | side' ω = A₀} ∩ Bad A₀) = μ.real {ω | side' ω = A₀} * μ.real (Bad A₀) := by
    intro A₀ hA₀
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ A₀)
      (hmeasBadC A₀ hA₀)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) T side' hsel' hmeasSel Bad hmeasBad E
    hindep (fun A₀ hA₀ => hbad A₀ hA₀) hE
  have hsub : {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))}
      ⊆ {ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (side' ω)
      rw [hside'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))}
      ≤ μ.real ({ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (side' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (side' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

open scoped Classical in
/-- The label sum over a block splits into its truly-rejecting and truly-accepting parts. -/
lemma sum_label_eq (O : Oracle μ S) (A : Finset S) :
    ∑ p ∈ A, O.label p = ((A.card : ℝ) - ((A.filter (fun p => O.label p = 0)).card : ℝ)) := by
  classical
  rw [← Finset.sum_filter_add_sum_filter_not A (fun p => O.label p = 0)]
  have h0 : ∑ p ∈ A.filter (fun p => O.label p = 0), O.label p = 0 :=
    Finset.sum_eq_zero (fun p hp => (Finset.mem_filter.1 hp).2)
  have hone : ∀ p ∈ A.filter (fun p => ¬ (O.label p = 0)), O.label p = 1 := by
    intro p hp
    rcases O.label_bit p with hl | hl
    · exact absurd hl (Finset.mem_filter.1 hp).2
    · exact hl
  have h1 : ∑ p ∈ A.filter (fun p => ¬ (O.label p = 0)), O.label p
      = ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ) := by
    rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
  rw [h0, h1, zero_add]
  have hc := Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = 0)
  have : ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
      = (A.card : ℝ) - ((A.filter (fun p => O.label p = 0)).card : ℝ) := by
    have : ((A.filter (fun p => O.label p = 0)).card : ℝ)
        + ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ) = (A.card : ℝ) := by
      exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) hc
    linarith
  rw [this]

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
    (Measure.infinitePi fun z : J × ℕ => D z.1).real
        {c | (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
          ≤ (m : ℝ) * (εcov - t)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J × ℕ → S) → ℝ := fun i c => ind (c (j, i)) with hXdef
  have hcoord : iIndepFun (fun (z : J × ℕ) (c : J × ℕ → S) => c z) ν :=
    iIndepFun_infinitePi (fun _ => measurable_id)
  have hinj : Function.Injective (fun i : ℕ => (j, i)) := fun a b hab => (Prod.mk.inj hab).2
  have hindep : iIndepFun X ν :=
    (hcoord.precomp hinj).comp (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c (j, i) ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J × ℕ → S => c (j, i)) ν (D j) :=
      measurePreserving_eval_infinitePi _ (j, i)
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.range m).card : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumLower_le X (Finset.range m) εcov t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J × ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c (j, i) ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c
        = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c (j, i) ∈ W)), X i c = 0 :=
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
    (Measure.infinitePi fun z : J × ℕ => D z.1).real
        {c | (m : ℝ) * (q + t)
          ≤ (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J × ℕ → S) → ℝ := fun i c => ind (c (j, i)) with hXdef
  have hcoord : iIndepFun (fun (z : J × ℕ) (c : J × ℕ → S) => c z) ν :=
    iIndepFun_infinitePi (fun _ => measurable_id)
  have hinj : Function.Injective (fun i : ℕ => (j, i)) := fun a b hab => (Prod.mk.inj hab).2
  have hindep : iIndepFun X ν :=
    (hcoord.precomp hinj).comp (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c (j, i) ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J × ℕ → S => c (j, i)) ν (D j) :=
      measurePreserving_eval_infinitePi _ (j, i)
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ∑ i ∈ Finset.range m, ν[X i] ≤ ((Finset.range m).card : ℝ) * q := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumUpper_le X (Finset.range m) q t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J × ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c (j, i) ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c
        = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c (j, i) ∈ W)), X i c = 0 :=
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
complement where the cut rejects.  Its mean is `1 − η` where the cut is right and `η` where
it is wrong. -/
noncomputable def agreeVar (O : Oracle μ S) (A : Finset S) (p : S) (ω : Ω) : ℝ :=
  if p ∈ A then mq O p ω else 1 - mq O p ω

lemma agreeVar_meas (O : Oracle μ S) (A : Finset S) (p : S) :
    AEMeasurable (agreeVar O A p) μ := by
  classical
  by_cases h : p ∈ A
  · have hfun : agreeVar O A p = mq O p := by funext ω; simp [agreeVar, h]
    rw [hfun]
    exact (mq_meas O p).aemeasurable
  · have hfun : agreeVar O A p = fun ω => 1 - mq O p ω := by funext ω; simp [agreeVar, h]
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
    ∀ᵐ ω ∂μ, ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
      = ∑ p ∈ Dset, agreeVar O A p ω := by
  classical
  filter_upwards [hits_eq_sum O A, hits_eq_sum O (Dset \ A)] with ω hA hR
  have hsplit : ∑ p ∈ Dset \ A, agreeVar O A p ω + ∑ p ∈ A, agreeVar O A p ω
      = ∑ p ∈ Dset, agreeVar O A p ω := Finset.sum_sdiff hAD
  have hAY : ∑ p ∈ A, agreeVar O A p ω = ∑ p ∈ A, mq O p ω :=
    Finset.sum_congr rfl (fun p hp => by simp [agreeVar, hp])
  have hRY : ∑ p ∈ Dset \ A, agreeVar O A p ω
      = ((Dset \ A).card : ℝ) - ∑ p ∈ Dset \ A, mq O p ω := by
    rw [Finset.sum_congr rfl (fun p hp => by
      simp [agreeVar, (Finset.mem_sdiff.1 hp).2] :
        ∀ p ∈ Dset \ A, agreeVar O A p ω = 1 - mq O p ω)]
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one]
  have hAcap : A.filter (fun p => mq O p ω = 1)
      = A ∩ Dset.filter (fun p => mq O p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqA, hq1⟩ := Finset.mem_filter.1 hq
      exact Finset.mem_inter.2 ⟨hqA, Finset.mem_filter.2 ⟨hAD hqA, hq1⟩⟩
    · intro hq
      obtain ⟨hqA, hqD⟩ := Finset.mem_inter.1 hq
      exact Finset.mem_filter.2 ⟨hqA, (Finset.mem_filter.1 hqD).2⟩
  have hRsdiff : (Dset \ A).filter (fun p => ¬ mq O p ω = 1)
      = (Dset \ A) \ Dset.filter (fun p => mq O p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_filter.1 hq
      exact Finset.mem_sdiff.2 ⟨hqR, fun hc => hne (Finset.mem_filter.1 hc).2⟩
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_sdiff.1 hq
      refine Finset.mem_filter.2 ⟨hqR, fun hc => hne ?_⟩
      exact Finset.mem_filter.2 ⟨(Finset.mem_sdiff.1 hqR).1, hc⟩
  have hRcount : ((Dset \ A).card : ℝ) - ∑ p ∈ Dset \ A, mq O p ω
      = ((((Dset \ A) \ Dset.filter (fun p => mq O p ω = 1)).card : ℕ) : ℝ) := by
    rw [← hR, ← hRsdiff]
    have hc : (((Dset \ A).filter (fun p => mq O p ω = 1)).card : ℝ)
        + (((Dset \ A).filter (fun p => ¬ mq O p ω = 1)).card : ℝ)
        = ((Dset \ A).card : ℝ) := by
      exact_mod_cast Finset.card_filter_add_card_filter_not (s := Dset \ A)
        (fun p => mq O p ω = 1)
    linarith
  rw [← hsplit, hAY, hRY, hRcount, ← hA, hAcap, agreeOf]
  push_cast
  ring

open scoped Classical in
/-- The agreement statistic's mean: `1 − η` per decided prefix, less `(1 − 2η)` for each
one the cut gets wrong. -/
lemma agree_mean_eq (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset) :
    ∑ p ∈ Dset, μ[agreeVar O A p]
      = (Dset.card : ℝ) * (1 - O.η)
        - (1 - 2 * O.η) * ((miscutOf O A Dset : ℕ) : ℝ) := by
  classical
  have hsplit : ∑ p ∈ Dset \ A, μ[agreeVar O A p] + ∑ p ∈ A, μ[agreeVar O A p]
      = ∑ p ∈ Dset, μ[agreeVar O A p] := Finset.sum_sdiff hAD
  have hA' : ∑ p ∈ A, μ[agreeVar O A p]
      = ∑ p ∈ A, (O.η + (1 - 2 * O.η) * O.label p) := by
    refine Finset.sum_congr rfl (fun p hp => ?_)
    have hfun : agreeVar O A p = mq O p := by funext ω; simp [agreeVar, hp]
    rw [hfun, mq_mean O p]
  have hR' : ∑ p ∈ Dset \ A, μ[agreeVar O A p]
      = ∑ p ∈ Dset \ A, (1 - (O.η + (1 - 2 * O.η) * O.label p)) := by
    refine Finset.sum_congr rfl (fun p hp => ?_)
    have hnot : p ∉ A := (Finset.mem_sdiff.1 hp).2
    have hfun : agreeVar O A p = fun ω => 1 - mq O p ω := by funext ω; simp [agreeVar, hnot]
    rw [hfun, integral_sub (integrable_const 1) (mq_integrable O p), integral_const,
      mq_mean O p]
    simp only [measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
  have hlabA : ∑ p ∈ A, O.label p
      = ((A.card : ℝ) - ((A.filter (fun p => O.label p = 0)).card : ℝ)) := sum_label_eq O A
  have hlabR : ∑ p ∈ Dset \ A, O.label p
      = (((Dset \ A).card : ℝ)
          - (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)) :=
    sum_label_eq O (Dset \ A)
  have hmisA : ((A.filter (fun p => ¬ (O.label p = 1))).card : ℝ)
      = ((A.filter (fun p => O.label p = 0)).card : ℝ) := by
    refine congrArg (fun n : ℕ => (n : ℝ)) (congrArg Finset.card (Finset.filter_congr ?_))
    intro p _
    rcases O.label_bit p with h | h <;> simp [h]
  have hmisR : (((Dset \ A).filter (fun p => ¬ (O.label p = 0))).card : ℝ)
      = ((Dset \ A).card : ℝ)
        - (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ) := by
    have := Finset.card_filter_add_card_filter_not (s := Dset \ A) (fun p => O.label p = 0)
    have hc : (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
        + (((Dset \ A).filter (fun p => ¬ (O.label p = 0))).card : ℝ)
        = ((Dset \ A).card : ℝ) := by exact_mod_cast this
    linarith
  have hcards : ((Dset \ A).card : ℝ) + (A.card : ℝ) = (Dset.card : ℝ) := by
    have := Finset.card_sdiff_add_card_eq_card hAD
    exact_mod_cast this
  rw [← hsplit, hA', hR', Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul,
    ← Finset.mul_sum, hlabA, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
    Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum, hlabR,
    miscutOf]
  push_cast
  rw [hmisA, hmisR, ← hcards]
  ring

/-- A cut that is mostly right reads as agreeing often enough.  Unlike the two
side-wise tests this replaces, the denominator is the whole decided set, so the bound does
not degrade when one side of the cut is small. -/
lemma agree_sound_of_wrong (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset)
    (θ τ w : ℝ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hw : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w)
    (hθ : (Dset.card : ℝ) * (θ + τ)
      ≤ (Dset.card : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w) :
    μ.real {ω | ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
        ≤ (Dset.card : ℝ) * θ}
      ≤ Real.exp (-2 * (Dset.card : ℝ) * τ ^ 2) := by
  classical
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
  have hmean : (Dset.card : ℝ) * (θ + τ) ≤ ∑ p ∈ Dset, μ[agreeVar O A p] := by
    rw [agree_mean_eq O A Dset hAD]
    nlinarith [hw, hθ]
  have hsum := sumLower_le (fun p => agreeVar O A p) Dset (θ + τ) τ
    (agreeVar_meas O A) (agreeVar_indep O A) (agreeVar_icc O A) hmean hτ
  have hsub : {ω | ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
        ≤ (Dset.card : ℝ) * θ}
      ≤ᵐ[μ] {ω | ∑ p ∈ Dset, agreeVar O A p ω ≤ (Dset.card : ℝ) * ((θ + τ) - τ)} := by
    filter_upwards [agreeOf_eq_sum O A Dset hAD] with ω hω hmem
    change ∑ p ∈ Dset, agreeVar O A p ω ≤ (Dset.card : ℝ) * ((θ + τ) - τ)
    have hrw : (Dset.card : ℝ) * ((θ + τ) - τ) = (Dset.card : ℝ) * θ := by ring
    rw [hrw, ← hω]
    exact hmem
  exact le_trans (ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)) hsum

/-- A cut that is badly wrong does not read as agreeing.  The upper tail: with `w`
prefixes mis-cut the statistic sits at `n(1 − η) − w(1 − 2η)`, so it clears a threshold
`τ` above that only with the Hoeffding probability. -/
lemma agree_sound_of_right (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset)
    (θ τ w : ℝ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hw : w ≤ ((miscutOf O A Dset : ℕ) : ℝ))
    (hθ : (Dset.card : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w
      ≤ (Dset.card : ℝ) * (θ - τ)) :
    μ.real {ω | (Dset.card : ℝ) * θ
        ≤ ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
      ≤ Real.exp (-2 * (Dset.card : ℝ) * τ ^ 2) := by
  classical
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
  have hmean : ∑ p ∈ Dset, μ[agreeVar O A p] ≤ (Dset.card : ℝ) * (θ - τ) := by
    rw [agree_mean_eq O A Dset hAD]
    nlinarith [hw, hθ]
  have hsum := wrongDecisive_le (fun p => agreeVar O A p) Dset (θ - τ) τ
    (agreeVar_meas O A) (agreeVar_indep O A) (agreeVar_icc O A) hmean hτ
  have hsub : {ω | (Dset.card : ℝ) * θ
        ≤ ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
      ≤ᵐ[μ] {ω | (Dset.card : ℝ) * ((θ - τ) + τ) ≤ ∑ p ∈ Dset, agreeVar O A p ω} := by
    filter_upwards [agreeOf_eq_sum O A Dset hAD] with ω hω hmem
    change (Dset.card : ℝ) * ((θ - τ) + τ) ≤ ∑ p ∈ Dset, agreeVar O A p ω
    have hrw : (Dset.card : ℝ) * ((θ - τ) + τ) = (Dset.card : ℝ) * θ := by ring
    rw [hrw, ← hω]
    exact hmem
  exact le_trans (ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)) hsum

open scoped Classical in
/-- The agreement count only reads the decided prefixes, so it does not matter whether the
hit set is taken over the sample or over the decided part of it. -/
lemma agreeOf_filter_of_subset (O : Oracle μ S) (A Dset C : Finset S) (hAD : A ⊆ Dset)
    (hDC : Dset ⊆ C) (ω : Ω) :
    agreeOf A Dset (C.filter (fun p => mq O p ω = 1))
      = agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) := by
  classical
  have h1 : A ∩ C.filter (fun p => mq O p ω = 1)
      = A ∩ Dset.filter (fun p => mq O p ω = 1) := by
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
  have h2 : (Dset \ A) \ C.filter (fun p => mq O p ω = 1)
      = (Dset \ A) \ Dset.filter (fun p => mq O p ω = 1) := by
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
      (n : ℝ) * (θ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w) :
    μ.real {ω | n₀ ≤ (cutSides O lo hi (fam ω) C ω).2.card
        ∧ ((miscutOf O (cutSides O lo hi (fam ω) C ω).1
              (cutSides O lo hi (fam ω) C ω).2 : ℕ) : ℝ) ≤ w
        ∧ ((agreeOf (cutSides O lo hi (fam ω) C ω).1 (cutSides O lo hi (fam ω) C ω).2
              (C.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
            ≤ ((cutSides O lo hi (fam ω) C ω).2.card : ℝ) * θ}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ C,
      voteCount O (fam ω) p ω = voteCount O (fam ω') p ω' := by
    intro ω ω' h p hp
    rw [← hcongr ω ω' h]
    exact voteCount_congr O _ p (fun v hv => by rw [mq_congr O (h _ (hQ ω p hp v hv))])
  have hselcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') →
      cutSides O lo hi (fam ω) C ω = cutSides O lo hi (fam ω') C ω' := by
    intro ω ω' h
    rw [cutSides, cutSides]
    exact Prod.ext (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
      (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
  set Pr : (Finset S × Finset S) → Finset S → Prop := fun t U =>
    t.1 ⊆ t.2 ∧ n₀ ≤ t.2.card
      ∧ ((miscutOf O t.1 t.2 : ℕ) : ℝ) ≤ w
      ∧ ((agreeOf t.1 t.2 U : ℕ) : ℝ) ≤ (t.2.card : ℝ) * θ with hPr
  have hsel : ∀ ω, cutSides O lo hi (fam ω) C ω ∈ C.powerset ×ˢ C.powerset := fun ω =>
    Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
  have hAD : ∀ ω, (cutSides O lo hi (fam ω) C ω).1 ⊆ (cutSides O lo hi (fam ω) C ω).2 := by
    intro ω q hq
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hq).1, Or.inl (Finset.mem_filter.1 hq).2⟩
  have hbad : ∀ t ∈ C.powerset ×ˢ C.powerset,
      μ.real {ω | Pr t (C.filter (fun p => mq O p ω = 1))}
        ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
    rintro ⟨A, Dset⟩ hmem
    by_cases hADt : A ⊆ Dset
    · by_cases hn : n₀ ≤ Dset.card
      · by_cases hwc : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w
        · have hDC : Dset ⊆ C := Finset.mem_powerset.1 (Finset.mem_product.1 hmem).2
          have hDcard : Dset.card ≤ C.card := Finset.card_le_card hDC
          have hsub : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))}
              ⊆ {ω | ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
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
        · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
            ext ω; simp [hPr, hwc]
          rw [hz]; simpa using Real.exp_nonneg _
      · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
          ext ω; simp [hPr, hn]
        rw [hz]; simpa using Real.exp_nonneg _
    · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
        ext ω; simp [hPr, hADt]
      rw [hz]; simpa using Real.exp_nonneg _
  have hmain := selection_read_bound O C Q hdisj (C.powerset ×ˢ C.powerset) (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset C, Finset.empty_mem_powerset C⟩)
    (fun ω => cutSides O lo hi (fam ω) C ω) hsel hselcongr Pr _ (Real.exp_nonneg _) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  exact ⟨hAD ω, hω.1, hω.2.1, hω.2.2⟩

open scoped Classical in
/-- The mirror, for soundness.  A cut that the sample shows is badly wrong reads as
agreeing often enough to pass only with the Hoeffding probability. -/
theorem gate_agree_bound_right (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (fam : Ω → Finset S)
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (θ τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w ≤ (n : ℝ) * (θ - τ)) :
    μ.real {ω | n₀ ≤ (cutSides O lo hi (fam ω) C ω).2.card
        ∧ w ≤ ((miscutOf O (cutSides O lo hi (fam ω) C ω).1
              (cutSides O lo hi (fam ω) C ω).2 : ℕ) : ℝ)
        ∧ ((cutSides O lo hi (fam ω) C ω).2.card : ℝ) * θ
            ≤ ((agreeOf (cutSides O lo hi (fam ω) C ω).1 (cutSides O lo hi (fam ω) C ω).2
              (C.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ C,
      voteCount O (fam ω) p ω = voteCount O (fam ω') p ω' := by
    intro ω ω' h p hp
    rw [← hcongr ω ω' h]
    exact voteCount_congr O _ p (fun v hv => by rw [mq_congr O (h _ (hQ ω p hp v hv))])
  have hselcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') →
      cutSides O lo hi (fam ω) C ω = cutSides O lo hi (fam ω') C ω' := by
    intro ω ω' h
    rw [cutSides, cutSides]
    exact Prod.ext (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
      (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
  set Pr : (Finset S × Finset S) → Finset S → Prop := fun t U =>
    t.1 ⊆ t.2 ∧ n₀ ≤ t.2.card
      ∧ w ≤ ((miscutOf O t.1 t.2 : ℕ) : ℝ)
      ∧ (t.2.card : ℝ) * θ ≤ ((agreeOf t.1 t.2 U : ℕ) : ℝ) with hPr
  have hsel : ∀ ω, cutSides O lo hi (fam ω) C ω ∈ C.powerset ×ˢ C.powerset := fun ω =>
    Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
  have hAD : ∀ ω, (cutSides O lo hi (fam ω) C ω).1 ⊆ (cutSides O lo hi (fam ω) C ω).2 := by
    intro ω q hq
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hq).1, Or.inl (Finset.mem_filter.1 hq).2⟩
  have hbad : ∀ t ∈ C.powerset ×ˢ C.powerset,
      μ.real {ω | Pr t (C.filter (fun p => mq O p ω = 1))}
        ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
    rintro ⟨A, Dset⟩ hmem
    by_cases hADt : A ⊆ Dset
    · by_cases hn : n₀ ≤ Dset.card
      · by_cases hwc : w ≤ ((miscutOf O A Dset : ℕ) : ℝ)
        · have hDC : Dset ⊆ C := Finset.mem_powerset.1 (Finset.mem_product.1 hmem).2
          have hDcard : Dset.card ≤ C.card := Finset.card_le_card hDC
          have hsub : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))}
              ⊆ {ω | (Dset.card : ℝ) * θ
                    ≤ ((agreeOf A Dset (Dset.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)} := by
            intro ω hω
            have h4 := hω.2.2.2
            rwa [agreeOf_filter_of_subset O A Dset C hADt hDC ω] at h4
          refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
          refine le_trans (agree_sound_of_right O A Dset hADt θ τ w hτ hsig hwc
            (hθ Dset.card hn hDcard)) ?_
          refine Real.exp_le_exp.2 ?_
          have hc : (n₀ : ℝ) ≤ (Dset.card : ℝ) := by exact_mod_cast hn
          nlinarith [sq_nonneg τ]
        · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
            ext ω; simp [hPr, hwc]
          rw [hz]; simpa using Real.exp_nonneg _
      · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
          ext ω; simp [hPr, hn]
        rw [hz]; simpa using Real.exp_nonneg _
    · have hz : {ω | Pr (A, Dset) (C.filter (fun p => mq O p ω = 1))} = (∅ : Set Ω) := by
        ext ω; simp [hPr, hADt]
      rw [hz]; simpa using Real.exp_nonneg _
  have hmain := selection_read_bound O C Q hdisj (C.powerset ×ˢ C.powerset) (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset C, Finset.empty_mem_powerset C⟩)
    (fun ω => cutSides O lo hi (fam ω) C ω) hsel hselcongr Pr _ (Real.exp_nonneg _) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  exact ⟨hAD ω, hω.1, hω.2.1, hω.2.2⟩

open scoped Classical in
/-- Both of the gate's wrong-counts are charged to the same mis-cut set.  A prefix the
cut accepts but the oracle rejects, and one the cut rejects but the oracle accepts, are each
a prefix where the cut is wrong — and the two kinds are disjoint. -/
lemma miscutOf_le_cutWrong (O : Oracle μ S) (lo hi : ℕ) (F C : Finset S) (ω : Ω) :
    miscutOf O (cutSides O lo hi F C ω).1 (cutSides O lo hi F C ω).2
      ≤ (C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω)).card := by
  classical
  have hsubA : ((cutSides O lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω) := by
    intro q hq
    obtain ⟨hqA, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqA
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.1 hv)⟩
  have hsubR : (((cutSides O lo hi F C ω).2 \ (cutSides O lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0)))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω) := by
    intro q hq
    obtain ⟨hqR, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqD, hqA⟩ := Finset.mem_sdiff.1 hqR
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqD
    have hnot : ¬ (hi - 1 < voteCount O F q ω) := fun h =>
      hqA (Finset.mem_filter.2 ⟨hqC, h⟩)
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.2 (hv.resolve_left hnot))⟩
  have hdj : Disjoint (((cutSides O lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1)))
      (((cutSides O lo hi F C ω).2 \ (cutSides O lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0))) := by
    refine Finset.disjoint_left.2 (fun q hq hq' => ?_)
    exact (Finset.mem_sdiff.1 (Finset.mem_filter.1 hq').1).2 (Finset.mem_filter.1 hq).1
  rw [miscutOf, ← Finset.card_union_of_disjoint hdj]
  exact Finset.card_le_card (Finset.union_subset hsubA hsubR)

open scoped Classical in
/-- The mis-cut prefixes all lie in the decided set. -/
lemma miscutOf_le_card (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset) :
    miscutOf O A Dset ≤ Dset.card := by
  classical
  have hdj : Disjoint (A.filter (fun p => ¬ (O.label p = 1)))
      ((Dset \ A).filter (fun p => ¬ (O.label p = 0))) := by
    refine Finset.disjoint_left.2 (fun q hq hq' => ?_)
    exact (Finset.mem_sdiff.1 (Finset.mem_filter.1 hq').1).2 (Finset.mem_filter.1 hq).1
  rw [miscutOf, ← Finset.card_union_of_disjoint hdj]
  refine Finset.card_le_card (Finset.union_subset ?_ ?_)
  · exact fun q hq => hAD (Finset.mem_filter.1 hq).1
  · exact fun q hq => (Finset.mem_sdiff.1 (Finset.mem_filter.1 hq).1).1

/-- The family at a reachable state is invalid: on some population its cut is wrong on
more than an `εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (εcov : ℝ)
    (B : Budget) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)}}

/-- A wrong prefix sits on the accept side with label `0`, or the reject side with label
`1`. -/
lemma wrong_mem_side (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω)
    (h : ¬ cutCorrect O lo hi F p ω) :
    (hi < voteCount O F p ω ∧ O.label p = 0) ∨ (voteCount O F p ω ≤ lo ∧ O.label p = 1) := by
  unfold cutCorrect at h
  rw [not_and_or] at h
  rcases h with h | h
  · rw [Classical.not_imp] at h
    refine Or.inl ⟨h.1, ?_⟩
    rcases O.label_bit p with hl | hl
    · exact hl
    · exact absurd hl h.2
  · rw [Classical.not_imp] at h
    refine Or.inr ⟨h.1, ?_⟩
    rcases O.label_bit p with hl | hl
    · exact absurd hl h.2
    · exact hl

open scoped Classical in


lemma voteCount_mono (O : Oracle μ S) {F F' : Finset S} (h : F ⊆ F') (p : S) (ω : Ω) :
    voteCount O F p ω ≤ voteCount O F' p ω := by
  classical
  exact Finset.card_le_card (Finset.filter_subset_filter _ h)

open scoped Classical in
/-- The gate sees every prefix the cut gets wrong.  A wrong prefix is decided against
its label, and with the accept side shifted by the seed's own vote it lands on the side the
gate scores.  Without the shift the gate would be blind to exactly the prefixes the seed's
own misread pushed over the line. -/
lemma wrong_mem_gate_side (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (hhi : 1 ≤ B.hi) (x : Run Ω S J) {p : S} (hp : p ∈ certOf j B.m x)
    (h : ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)) :
    (p ∈ sideAcc O populations j B x ∧ O.label p = 0)
      ∨ (p ∈ sideRej O populations j B x ∧ O.label p = 1) := by
  classical
  rcases wrong_mem_side O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x) h with ⟨hlt, hl⟩ | ⟨hle, hl⟩
  · refine Or.inl ⟨Finset.mem_filter.2 ⟨hp, ?_⟩, hl⟩
    have hstep := voteCount_le_erase_succ O (clusterAt O populations x B) p (oracleNoise x)
    omega
  · refine Or.inr ⟨Finset.mem_filter.2 ⟨hp, ?_⟩, hl⟩
    exact le_trans (voteCount_mono O (Finset.erase_subset _ _) p (oracleNoise x)) hle

open scoped Classical in
/-- Every wrong certification prefix is counted against one of the gate's two sides. -/
lemma card_cert_wrong_le (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (hhi : 1 ≤ B.hi) (x : Run Ω S J) :
    ((certOf j B.m x).filter
        (fun p => ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x))).card
      ≤ ((sideAcc O populations j B x).filter (fun p => O.label p = 0)).card
        + ((sideRej O populations j B x).filter (fun p => ¬ (O.label p = 0))).card := by
  classical
  refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
  intro p hp
  obtain ⟨hpC, hbad⟩ := Finset.mem_filter.1 hp
  rcases wrong_mem_gate_side O populations j B hhi x hpC hbad with ⟨hs, hl⟩ | ⟨hs, hl⟩
  · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hs, hl⟩)
  · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hs, ?_⟩)
    rw [hl]; norm_num

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

/-- The lower-tail counterpart for `binomCdf`. -/
theorem binomCdf_le (n j : ℕ) (θ τ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (h : (j : ℝ) ≤ (n : ℝ) * (θ - τ)) :
    binomCdf n θ j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  classical
  set p : unitInterval := ⟨θ, hθ0, hθ1⟩ with hp
  have hbin : binomCdf n θ j
      = (ProbabilityTheory.binomial n p).real ↑(Finset.range (j + 1)) := by
    rw [binomial_real_finset]
    rfl
  rw [hbin]
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (binomial_real_le_le n p τ hτ)
  intro i hi
  simp only [Finset.coe_range, Set.mem_Iio] at hi
  have : (i : ℝ) ≤ (j : ℝ) := by exact_mod_cast Nat.lt_succ_iff.1 hi
  exact le_trans this h

/-- The two tails together cover everything: `P[X ≥ j] + P[X ≤ j] ≥ 1`. -/
lemma one_le_binomSfGe_add_binomCdf (n j : ℕ) (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    1 ≤ binomSfGe n θ j + binomCdf n θ j := by
  classical
  set f : ℕ → ℝ := fun i => (n.choose i : ℝ) * θ ^ i * (1 - θ) ^ (n - i) with hf
  have hf0 : ∀ i, 0 ≤ f i := fun i => by
    have h1 : (0 : ℝ) ≤ 1 - θ := by linarith
    positivity
  have htotal : ∑ i ∈ Finset.range (n + 1), f i = 1 := by
    have hpow := add_pow θ (1 - θ) n
    rw [show θ + (1 - θ) = 1 from by ring, one_pow] at hpow
    rw [hpow, hf]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  have hsub : Finset.range (n + 1) ⊆ Finset.Icc j n ∪ Finset.range (j + 1) := by
    intro i hi
    rcases le_or_gt j i with hij | hij
    · exact Finset.mem_union_left _ (Finset.mem_Icc.2 ⟨hij, by
        simpa using Nat.lt_succ_iff.1 (Finset.mem_range.1 hi)⟩)
    · exact Finset.mem_union_right _ (Finset.mem_range.2 (Nat.lt_succ_of_lt hij))
  have hunion : ∑ i ∈ Finset.Icc j n ∪ Finset.range (j + 1), f i
      ≤ ∑ i ∈ Finset.Icc j n, f i + ∑ i ∈ Finset.range (j + 1), f i := by
    have hsum := Finset.sum_union_inter (s₁ := Finset.Icc j n) (s₂ := Finset.range (j + 1))
      (f := f)
    have hinter : 0 ≤ ∑ i ∈ Finset.Icc j n ∩ Finset.range (j + 1), f i :=
      Finset.sum_nonneg (fun i _ => hf0 i)
    linarith
  calc (1 : ℝ) = ∑ i ∈ Finset.range (n + 1), f i := htotal.symm
    _ ≤ ∑ i ∈ Finset.Icc j n ∪ Finset.range (j + 1), f i :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub (fun i _ _ => hf0 i)
    _ ≤ ∑ i ∈ Finset.Icc j n, f i + ∑ i ∈ Finset.range (j + 1), f i := hunion
    _ = binomSfGe n θ j + binomCdf n θ j := rfl

/-- A small upper tail puts the count above the mean, up to the Hoeffding slack: if
`P[X ≥ j] ≤ α` and `α` leaves room for the lower tail at `τ`, then `j` clears `n(θ − τ)`.

This replaces the exact median bound (Kaas–Buhrman) with what the two Hoeffding tails
already give; the price is the slack `τ`, which the gate's margin absorbs. -/
theorem lt_of_binomSfGe_le (n j : ℕ) (θ τ α : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (hα : α + Real.exp (-2 * (n : ℝ) * τ ^ 2) < 1) (h : binomSfGe n θ j ≤ α) :
    (n : ℝ) * (θ - τ) < j := by
  by_contra hc
  push_neg at hc
  have hcdf := binomCdf_le n j θ τ hθ0 hθ1 hτ hc
  have hone := one_le_binomSfGe_add_binomCdf n j θ hθ0 hθ1
  linarith

/-- The gate's rates are probabilities, with no premise beyond the coverage being a
fraction: `gateAcc` runs from `1 − η` down to `½` as `εcov` runs from `0` to `1`. -/
lemma gateAcc_mem (O : Oracle μ S) {εcov : ℝ} (h0 : 0 ≤ εcov) (h1 : εcov ≤ 1)
    (hsig : O.η ≤ 1 / 2) : 0 ≤ gateAcc O εcov ∧ gateAcc O εcov ≤ 1 := by
  have hη0 := eta_nonneg O
  unfold gateAcc
  constructor <;> nlinarith [mul_le_of_le_one_right (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h1,
    mul_nonneg (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h0]

/-- Markov on a per-prefix failure count.  Both gates ask for a *fraction* of the
certification sample, not for every prefix to behave, so a per-prefix bound `E` only has to
beat the limit `l`: the cost is `E / l`, with no union over the sample.

`G` is the event on which the per-prefix failures are covered by the sets `Bad` — in use it
is where the family is light enough for the per-prefix bound to apply at all. -/
theorem count_frac_le (C : Finset S) (Bad : S → Set Ω) (Pr : Ω → S → Prop)
    [inst : ∀ ω, DecidablePred (fun p => Pr ω p)] (G : Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l)
    (hCpos : 0 < C.card) (hper : ∀ p ∈ C, μ.real (Bad p) ≤ E)
    (hsub : ∀ ω ∈ G, ∀ p ∈ C, Pr ω p → ω ∈ Bad p) :
    μ.real (G ∩ {ω | l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ)})
      ≤ E / l := by
  classical
  set Y : Ω → ℝ := fun ω => ∑ p ∈ C, (Bad p).indicator (fun _ => (1 : ℝ)) ω with hY
  have hYnn : 0 ≤ᵐ[μ] Y := by
    filter_upwards with ω
    exact Finset.sum_nonneg (fun p _ => Set.indicator_nonneg (fun _ _ => zero_le_one) ω)
  have hYint : Integrable Y μ :=
    integrable_finset_sum C (fun p _ => (integrable_const (1 : ℝ)).indicator (hmeas p))
  have hYmean : ∫ ω, Y ω ∂μ ≤ (C.card : ℝ) * E := by
    rw [hY, integral_finset_sum _ (fun p _ => (integrable_const (1 : ℝ)).indicator (hmeas p))]
    calc ∑ p ∈ C, ∫ ω, (Bad p).indicator (fun _ => (1 : ℝ)) ω ∂μ
        = ∑ p ∈ C, μ.real (Bad p) :=
          Finset.sum_congr rfl (fun p _ => integral_indicator_one (hmeas p))
      _ ≤ ∑ _p ∈ C, E := Finset.sum_le_sum hper
      _ = (C.card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hsub' : G ∩ {ω | l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ)}
      ⊆ {ω | l * (C.card : ℝ) ≤ Y ω} := by
    rintro ω ⟨hG, hlt⟩
    have hlt' : l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ) := hlt
    refine le_trans (le_of_lt hlt') ?_
    show ((C.filter (fun p => Pr ω p)).card : ℝ)
      ≤ ∑ p ∈ C, (Bad p).indicator (fun _ => (1 : ℝ)) ω
    rw [← Finset.sum_filter_add_sum_filter_not C (fun p => Pr ω p)]
    have h1 : ∀ p ∈ C.filter (fun p => Pr ω p),
        (Bad p).indicator (fun _ => (1 : ℝ)) ω = 1 := by
      intro p hp
      obtain ⟨hpC, hpr⟩ := Finset.mem_filter.1 hp
      simp [Set.indicator_of_mem (hsub ω hG p hpC hpr)]
    have h2 : 0 ≤ ∑ p ∈ C.filter (fun p => ¬ Pr ω p),
        (Bad p).indicator (fun _ => (1 : ℝ)) ω :=
      Finset.sum_nonneg (fun p _ => Set.indicator_nonneg (fun _ _ => zero_le_one) ω)
    rw [Finset.sum_congr rfl h1, Finset.sum_const, nsmul_eq_mul, mul_one]
    linarith
  have hCR : (0 : ℝ) < (C.card : ℝ) := by exact_mod_cast hCpos
  have hstep : (l * (C.card : ℝ)) * μ.real {ω | l * (C.card : ℝ) ≤ Y ω} ≤ (C.card : ℝ) * E :=
    le_trans (mul_meas_ge_le_integral_of_nonneg hYnn hYint (l * (C.card : ℝ))) hYmean
  have hfin := measureReal_mono hsub' (measure_ne_top (μ := μ) _)
  have hkey : (C.card : ℝ) * (l * μ.real {ω | l * (C.card : ℝ) ≤ Y ω}) ≤ (C.card : ℝ) * E := by
    calc (C.card : ℝ) * (l * μ.real {ω | l * (C.card : ℝ) ≤ Y ω})
        = (l * (C.card : ℝ)) * μ.real {ω | l * (C.card : ℝ) ≤ Y ω} := by ring
      _ ≤ (C.card : ℝ) * E := hstep
  have hdiv := le_of_mul_le_mul_left hkey hCR
  rw [le_div_iff₀ hl]
  nlinarith [hfin, hdiv, hl.le]

open scoped Classical in
/-- A mostly-correct cut on sides that carry prefixes is admitted.  The fractional form:
the cut has to be right on all but `w` of the certification sample, not on all of it, and
`count_frac_le` is what supplies that `w`.  Both sides draw their wrong-member budget from
the same count, since a side member carrying the wrong label *is* a mis-cut prefix. -/
theorem admitted_whp (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (εcov α τ w wi : ℝ) (n₀ nlo : ℕ) (fam : Ω → Finset S)
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hsig : O.η ≤ 1 / 2)
    (hnloC : (nlo : ℝ) ≤ (C.card : ℝ) - wi)
    (hga : ∀ n : ℕ, nlo ≤ n → n ≤ C.card →
      (n : ℝ) * (gateAcc O εcov + τ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w)
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ w)
        ∧ (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ wi)
        ∧ ¬ admitted O lo hi n₀ εcov α (fam ω) C ω}
      ≤ 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2) := by
  classical
  have hsub : {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
        ≤ w) ∧ (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ wi)
        ∧ ¬ admitted O lo hi n₀ εcov α (fam ω) C ω}
      ⊆ {ω | nlo ≤ (cutSides O lo hi (fam ω) C ω).2.card
          ∧ ((miscutOf O (cutSides O lo hi (fam ω) C ω).1
                (cutSides O lo hi (fam ω) C ω).2 : ℕ) : ℝ) ≤ w
          ∧ ((agreeOf (cutSides O lo hi (fam ω) C ω).1 (cutSides O lo hi (fam ω) C ω).2
                (C.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)
              ≤ ((cutSides O lo hi (fam ω) C ω).2.card : ℝ) * (gateAcc O εcov + τ)} := by
    rintro ω ⟨hw, hwi, hadm⟩
    have hdec : (nlo : ℝ) ≤ ((cutSides O lo hi (fam ω) C ω).2.card : ℝ) := by
      have hcompl := Finset.card_filter_add_card_filter_not (s := C)
        (fun p => decided O lo (hi - 1) (fam ω) p ω)
      have hc : (((C.filter (fun p => decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
          + (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
          = (C.card : ℝ) := by exact_mod_cast hcompl
      have heq : (cutSides O lo hi (fam ω) C ω).2
          = C.filter (fun p => decided O lo (hi - 1) (fam ω) p ω) := by
        show C.filter (fun p => hi - 1 < voteCount O (fam ω) p ω
              ∨ voteCount O (fam ω) p ω ≤ lo)
            = C.filter (fun p => decided O lo (hi - 1) (fam ω) p ω)
        exact Finset.filter_congr (fun p _ => Iff.rfl)
      rw [heq]
      linarith
    have hdecN : nlo ≤ (cutSides O lo hi (fam ω) C ω).2.card := by exact_mod_cast hdec
    rw [admitted] at hadm
    push_neg at hadm
    obtain ⟨hn, hbin⟩ := hadm
    refine ⟨hdecN, ?_, ?_⟩
    · refine le_trans ?_ hw
      exact_mod_cast miscutOf_le_cutWrong O lo hi (fam ω) C ω
    · by_contra hc
      push_neg at hc
      refine absurd ?_ (not_le.2 hbin)
      have hcount : ((agreeCount O lo hi (fam ω) C ω).2 : ℝ)
          * (gateAcc O εcov + τ) ≤ ((agreeCount O lo hi (fam ω) C ω).1 : ℝ) := le_of_lt hc
      refine le_trans (binomSfGe_le _ _ _ τ (gateAcc_mem O hε0 hε1 hsig).1
        (gateAcc_mem O hε0 hε1 hsig).2 hτ hcount) ?_
      refine le_trans (Real.exp_le_exp.2 ?_) hα
      have hnR : (n₀ : ℝ) ≤ ((agreeCount O lo hi (fam ω) C ω).2 : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have hmain := gate_agree_bound O C Q hdisj lo hi fam hQ hcongr (gateAcc O εcov + τ) τ w nlo
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

lemma stoppable_card_le (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP ρ : ℝ) :
    (stoppable O populations εcov δ α pAP ρ).card
      ≤ ladderLen O populations εcov δ α pAP := by
  classical
  exact le_trans (Finset.card_filter_le _ _) (schedule_card_le O populations εcov δ α pAP)

lemma capped_of_mem_stoppable {O : Oracle μ S} {populations : Finset J}
    {εcov δ α pAP ρ : ℝ} {B : Budget}
    (hB : B ∈ stoppable O populations εcov δ α pAP ρ) :
    Capped O populations εcov δ ρ (ladderLen O populations εcov δ α pAP) B := by
  classical
  exact (Finset.mem_filter.1 hB).2

open scoped Classical in
/-- The first step's loss at one prefix, in the exact form `hammingLoss` uses. -/
noncomputable def seedLoss (O : Oracle μ S) (cn cd : ℕ) (v p : S) (ω : Ω) : ℝ :=
  if ((mq O (p * v) ω = 1) ↔ cn * ({(1 : S)} : Finset S).card
      < cd * voteCount O {(1 : S)} p ω) then 0 else 1

open scoped Classical in

lemma seedLoss_icc (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards with ω
  unfold seedLoss
  split_ifs <;> norm_num

lemma measurable_voteCount (O : Oracle μ S) (F : Finset S) (p : S) :
    Measurable (fun ω => voteCount O F p ω) := by
  classical
  have hfun : (fun ω => voteCount O F p ω)
      = fun ω => ∑ v ∈ F, (if mq O (p * v) ω = 1 then 1 else 0) := by
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
column: `cn < cd` makes `cn·1 < cd·voteCount {ε}` say exactly `mq p = 1`. -/
lemma seedLoss_eq_disagree (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω
      = mq O (p * v) ω + mq O p ω - 2 * (mq O (p * v) ω * mq O p ω) := by
  classical
  filter_upwards [mq_bit O (p * v), mq_bit O p] with ω h1 h0
  have hvc : voteCount O {(1 : S)} p ω = if mq O p ω = 1 then 1 else 0 := by
    unfold voteCount
    rcases h0 with h | h <;> simp [voteCount, Finset.filter_singleton, mul_one, h]
  unfold seedLoss
  rw [hvc]
  rcases h1 with h1 | h1 <;> rcases h0 with h0 | h0 <;>
    · rw [h1, h0]
      norm_num [hcd, Nat.not_lt.2 (Nat.zero_le cn)]

lemma mq_mul_integrable (O : Oracle μ S) (w w' : S) :
    Integrable (fun ω => mq O w ω * mq O w' ω) μ := by
  refine MeasureTheory.Integrable.of_mem_Icc 0 1
    (((mq_meas O w).mul (mq_meas O w')).aemeasurable) ?_
  filter_upwards [mq_icc O w, mq_icc O w'] with ω h1 h0
  rw [Set.mem_Icc] at h1 h0 ⊢
  exact ⟨mul_nonneg h1.1 h0.1, by nlinarith [h1.1, h1.2, h0.1, h0.2]⟩


/-- The first step's mean separates by the square of the signal.  Two noisy reads are
compared, so an accept-preserving candidate disagrees at `2η(1−η)` and one that flips at `p`
at `1 − 2η(1−η)`; the difference is `(1−2η)²`.  This is exactly the statistic
`_screen_cohort` tests against, and the square is why its power is weaker than a comparison
against the truth would be. -/
lemma seedLoss_mean (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) (hv : p * v ≠ p) :
    μ[seedLoss O cn cd v p] = 2 * O.η * (1 - O.η) + O.flip v p * (1 - 2 * O.η) ^ 2 := by
  have hprod : μ[fun ω => mq O (p * v) ω * mq O p ω] = μ[mq O (p * v)] * μ[mq O p] :=
    ProbabilityTheory.IndepFun.integral_mul_eq_mul_integral
      ((mq_indep O).indepFun hv) (mq_meas O _).aestronglyMeasurable
      (mq_meas O _).aestronglyMeasurable
  have hsplit : μ[seedLoss O cn cd v p]
      = μ[mq O (p * v)] + μ[mq O p] - 2 * (μ[mq O (p * v)] * μ[mq O p]) := by
    rw [integral_congr_ae (seedLoss_eq_disagree O hcd v p),
      integral_sub (f := fun ω => mq O (p * v) ω + mq O p ω)
        (g := fun ω => 2 * (mq O (p * v) ω * mq O p ω))
        ((mq_integrable O _).add (mq_integrable O _))
        ((mq_mul_integrable O (p * v) p).const_mul 2),
      integral_add (mq_integrable O _) (mq_integrable O _), integral_const_mul, hprod]
  rw [hsplit, mq_mean, mq_mean]
  show _ = 2 * O.η * (1 - O.η)
    + (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p) * (1 - 2 * O.η) ^ 2
  ring

open scoped Classical in
/-- The screen's statistic is the summed `seedLoss`, exactly rather than almost everywhere:
`voteCount` over the singleton `{ε}` is a card, so it is a bit by construction. -/
lemma screenCount_eq_sum (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P : Finset S) (v : S)
    (ω : Ω) : ((screenCount O P v ω : ℝ)) = ∑ p ∈ P, seedLoss O cn cd v p ω := by
  classical
  have hcond : ∀ p : S, (cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p ω)
      ↔ (mq O p ω = 1) := by
    intro p
    have hvc : voteCount O {(1 : S)} p ω = if mq O p ω = 1 then 1 else 0 := by
      unfold voteCount
      by_cases h : mq O p ω = 1 <;> simp [Finset.filter_singleton, mul_one, h]
    rw [hvc]
    by_cases h : mq O p ω = 1 <;> simp [h, hcd]
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
    have hA : MeasurableSet[noiseAlg O T] {ω | mq O (p.val * v) ω = 1} :=
      measurableSet_mq_eq_one O (by simp [hT])
    have hC : MeasurableSet[noiseAlg O T] {ω | cn * ({(1 : S)} : Finset S).card
        < cd * voteCount O {(1 : S)} p.val ω} :=
      measurableSet_filter_pred_map O (T := T) (A := {(1 : S)}) (fun w => p.val * w)
        (by intro w hw; simp only [Finset.mem_singleton] at hw; simp [hw, hT])
        (fun U => cn * ({(1 : S)} : Finset S).card < cd * U.card)
    have hQ : MeasurableSet[noiseAlg O T] {ω | (mq O (p.val * v) ω = 1)
        ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p.val ω} := by
      have hrw : {ω | (mq O (p.val * v) ω = 1)
          ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p.val ω}
          = ({ω | mq O (p.val * v) ω = 1} ∩ {ω | cn * ({(1 : S)} : Finset S).card
              < cd * voteCount O {(1 : S)} p.val ω})
            ∪ ({ω | mq O (p.val * v) ω = 1}ᶜ ∩ {ω | cn * ({(1 : S)} : Finset S).card
              < cd * voteCount O {(1 : S)} p.val ω}ᶜ) := by
        ext ω
        by_cases h1 : mq O (p.val * v) ω = 1 <;>
          by_cases h2 : cn * ({(1 : S)} : Finset S).card
            < cd * voteCount O {(1 : S)} p.val ω <;>
          simp [h1, h2]
      rw [hrw]
      exact ((hA.inter hC).union (hA.compl.inter hC.compl))
    unfold seedLoss
    exact Measurable.ite hQ measurable_const measurable_const

/-! ### The first Lloyd step

Its centre is `{ε}`, so its loss is `seedLoss`, whose mean separates a candidate that never
flips from one that flips on a `Δ` fraction by `Δ(1−2η)²` — the screen's statistic.  The
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

Part 1 takes the gate instead, which measures the conclusion on draws the family was never
selected from.  What the clustering still has to deliver is only that the family is clean
(`measureReal_dirtyMember_le`, off the screen) and has the round's size. -/

/-- An accept-preserving candidate passes the screen.  Its disagreement with the seed's
column has mean exactly `2η(1−η)` — two noisy reads of the same bit — so a cutoff `γ` above
that is cleared except in the upper tail.  This is `screen_tail`'s mirror, and it is what
keeps the candidate pool from emptying. -/
theorem screen_pass {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (γ : ℝ) (sc scd : ℕ) (hγ : 0 ≤ γ) (hscd : 0 < scd)
    (hclean : ∀ p ∈ P, O.flip v p = 0)
    (hsc : (scd : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (sc : ℝ)) :
    μ.real {ω | ¬ (scd * screenCount O P v ω ≤ sc * P.card)}
      ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  set b : ℝ := 2 * O.η * (1 - O.η) with hb
  have hcard : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) = (P.card : ℝ) := by
    simp [Finset.card_univ]
  have hmean : ∑ i : {p // p ∈ P}, μ[seedLoss O cn cd v i.val]
      ≤ ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) * b := by
    rw [Finset.sum_congr rfl
      (fun i _ => seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv)),
      Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    simp only [nsmul_eq_mul, Fintype.card_coe]
    have hzero : ∑ i : {p // p ∈ P}, O.flip v i.val * (1 - 2 * O.η) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun i _ => by rw [hclean i.val i.property]; ring)
    rw [hzero, hb]
    simp
  have htail := sumUpper_le (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    (Finset.univ : Finset {p // p ∈ P}) b γ
    (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v)
    (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  rw [hcard] at htail
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) htail
  intro ω hω
  show (P.card : ℝ) * (b + γ) ≤ ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
  have hsum : ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
      = ∑ p ∈ P, seedLoss O cn cd v p ω :=
    Finset.sum_attach P (fun p => seedLoss O cn cd v p ω)
  rw [hsum, ← screenCount_eq_sum O hcd P v ω]
  have hscdR : (0 : ℝ) < (scd : ℝ) := by exact_mod_cast hscd
  have hωR : (sc : ℝ) * (P.card : ℝ)
      < (scd : ℝ) * ((screenCount O P v ω : ℕ) : ℝ) := by
    exact_mod_cast not_le.1 hω
  have hPc : (0 : ℝ) ≤ (P.card : ℝ) := Nat.cast_nonneg _
  nlinarith [hsc, hωR]

open scoped Classical in
/-- A badly-flipping candidate rarely passes the screen.  Its disagreement with the
seed's column has mean `2η(1−η) + φ(1−2η)²`, so a cutoff `γ` below that is cleared only in
the lower tail. -/
theorem screen_tail {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (Δ γ : ℝ) (sc scd : ℕ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2) (hscd : 0 < scd)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p)
    (hsc : (sc : ℝ) ≤ (scd : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    μ.real {ω | scd * screenCount O P v ω ≤ sc * P.card}
      ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  set b : ℝ := 2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2 with hb
  have hcard : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) = (P.card : ℝ) := by
    simp [Finset.card_univ]
  have hmean : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) * b
      ≤ ∑ i : {p // p ∈ P}, μ[seedLoss O cn cd v i.val] := by
    rw [Finset.sum_congr rfl
      (fun i _ => seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv)),
      Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    simp only [nsmul_eq_mul, Fintype.card_coe]
    rw [← Finset.sum_mul, hb]
    have hattach : ∑ i : {p // p ∈ P}, O.flip v i.val = ∑ p ∈ P, O.flip v p :=
      Finset.sum_attach P (fun p => O.flip v p)
    rw [hattach]
    have hsq : (0 : ℝ) ≤ (1 - 2 * O.η) ^ 2 := sq_nonneg _
    nlinarith [hflip]
  have htail := sumLower_le (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    (Finset.univ : Finset {p // p ∈ P}) b γ
    (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v)
    (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  rw [hcard] at htail
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) htail
  intro ω hω
  show ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω ≤ (P.card : ℝ) * (b - γ)
  have hsum : ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
      = ((screenCount O P v ω : ℝ)) := by
    rw [screenCount_eq_sum O hcd P v ω]
    exact (Finset.sum_attach P (fun p => seedLoss O cn cd v p ω)).symm ▸ rfl
  rw [hsum]
  have hscdR : (0 : ℝ) < (scd : ℝ) := by exact_mod_cast hscd
  have hωR : (scd : ℝ) * ((screenCount O P v ω : ℕ) : ℝ) ≤ (sc : ℝ) * (P.card : ℝ) := by
    exact_mod_cast hω
  have hPc : (0 : ℝ) ≤ (P.card : ℝ) := Nat.cast_nonneg _
  nlinarith [hsc, hωR]

/-! ### The iterate, and what actually bounds its flips

The first step's centre is the seed's own column, which the screen already controls.
Every step after that ranks candidates against the *current* centre's majority vote, and
that ranking is by agreement with the centre's drift rather than by flipping little: writing
`Dset` for the centre's error set, a candidate scores mean `η·#P + (1−2η)·#(Φ_v Δ Dset)`, so
one that flips exactly `Dset` scores zero.  Chasing the bound through the majority vote gives
`d' ≤ (2 / c) · d` with `c = (s + eps) / (2 * s)`, about `3` at the usual settings — no
contraction.

None of that matters, because the ranking is not what bounds the flips: the screen is.
A suffix that fails it never becomes a fully observed column and so is never a clustering
candidate at all, and the iterate can only choose among what is left. -/

/-- The family flips no more than the screened pool does. -/
theorem clusterAt_flip_bound (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) (Δ : ℝ)
    (hscreen : ∀ v ∈ screenedAt O populations B x,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p)) :
    ∀ w ∈ clusterAt O populations x B,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip w p) :=
  fun w hw => hscreen w (clusterAround_subset O B.cn B.cd _ _ (oracleNoise x) B.k
    (one_mem_screenedAt O populations B x) hw)

/-! ### From flip mass to a correct cut

The band is what turns "few members flip" into "the cut is right", and it is far more
generous than a member count suggests.  A rejecting prefix is accepted only when the vote
clears `hi ≈ k(½ + eps)`, and members preserving at `p` contribute only through noise, so
with `f` members flipping the count is at most `f + Bin(k − f, η)`.  In the mean that
needs

    f · 2s > k · (s + eps)

— two *thirds* of the family at `s = 0.3, eps = 0.1`, not the `eps · k` a naive reading of
the margin suggests.  Markov over the members' flip masses then gives a misclassified mass
of about `1.5 · Δ` rather than `Δ / eps`, which is a factor of six. -/

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

This is what turns per-member flip mass into misclassified mass: the band puts
`c = (s + eps) / (2 * s)`, so the price is a constant near `3 / 2` and not the `1 / eps` a
naive reading of the band charges. -/
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
  ∑ v ∈ F, mq O (p * v) ω

lemma voteCount_eq_voteSum (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ((voteCount O F p ω : ℝ)) = voteSum O F p ω := by
  classical
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  unfold voteCount voteSum
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => mq O (p * v) ω = 1)]
  have h1 : ∑ v ∈ F.filter (fun v => mq O (p * v) ω = 1), mq O (p * v) ω
      = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.1 hv).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ v ∈ F.filter (fun v => ¬ (mq O (p * v) ω = 1)), mq O (p * v) ω = 0 := by
    refine Finset.sum_eq_zero (fun v hv => ?_)
    obtain ⟨hvF, hne⟩ := Finset.mem_filter.1 hv
    rcases hω v hvF with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

lemma voteSum_meanSum (O : Oracle μ S) (F : Finset S) (p : S) :
    ∑ v ∈ F, μ[mq O (p * v)]
      = (F.card : ℝ) * O.η + (1 - 2 * O.η) * ∑ v ∈ F, O.label (p * v) := by
  rw [Finset.sum_congr rfl (fun v _ => mq_mean O (p * v)), Finset.sum_add_distrib,
    Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum]

lemma mq_indep_shift (O : Oracle μ S) (p : S) :
    iIndepFun (fun v : S => mq O (p * v)) μ :=
  (mq_indep O).precomp (mul_right_injective p)

/-- Upper tail on a rejecting prefix.  A flip fraction of `f` lifts the vote's mean only
to `η + 2 * s * f`. -/
theorem voteSum_upper (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ : ℝ) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hlab : ∀ v ∈ F, O.label (p * v) = O.flip v p := by
    intro v _
    show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
    rw [hp]; ring
  have hmean : ∑ v ∈ F, μ[mq O (p * v)] ≤ (F.card : ℝ) * (O.η + (1 - 2 * O.η) * f) := by
    rw [voteSum_meanSum, Finset.sum_congr rfl hlab, ← flipCount_eq_sum]
    have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    nlinarith [hf]
  exact sumUpper_le (fun v : S => mq O (p * v)) F (O.η + (1 - 2 * O.η) * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- Lower tail on an accepting prefix.  Mirror of `voteSum_upper`: the vote's mean only
falls to `1 - η - 2 * s * f`. -/
theorem voteSum_lower (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ : ℝ) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hlab : ∀ v ∈ F, O.label (p * v) = 1 - O.flip v p := by
    intro v _
    show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
    rw [hp]; ring
  have hmean : (F.card : ℝ) * (O.η + (1 - 2 * O.η) * (1 - f)) ≤ ∑ v ∈ F, μ[mq O (p * v)] := by
    rw [voteSum_meanSum, Finset.sum_congr rfl hlab, Finset.sum_sub_distrib, Finset.sum_const,
      nsmul_eq_mul, mul_one, ← flipCount_eq_sum]
    have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    nlinarith [hf]
  exact sumLower_le (fun v : S => mq O (p * v)) F (O.η + (1 - 2 * O.η) * (1 - f)) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- The cut survives the family being chosen by the clustering.  At a population prefix
whose query strings the clustering never read, the family is decided by bits independent of
the ones the vote reads, so the fixed-family bound carries over.

`good` is there because the fixed-family bound holds only for families this prefix is light
for: a prefix a lot of the family flips is charged to `flipCount_mass_le` instead. -/
theorem cutCorrect_selected_whp (O : Oracle μ S) (cands Q : Finset S) (p : S) (lo hi : ℕ)
    (hdisj : Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑Q : Set S))
    (T good : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, A₀ ∈ good → μ.real {ω | ¬ cutCorrect O lo hi A₀ p ω} ≤ E) :
    μ.real {ω | fam ω ∈ good ∧ ¬ cutCorrect O lo hi (fam ω) p ω} ≤ E := by
  classical
  refine selection_side_bound O cands Q (fun v => p * v) hdisj T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ U => A₀ ∈ good ∧ ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0)))
    E hE ?_
  intro A₀ hA₀
  by_cases hg : A₀ ∈ good
  · refine le_trans (le_of_eq ?_) (hbad A₀ hA₀ hg)
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, cutCorrect, voteCount, hg, true_and]
  · rw [show {ω | A₀ ∈ good ∧ ¬ ((hi < (A₀.filter (fun v => mq O (p * v) ω = 1)).card
      → O.label p = 1) ∧ ((A₀.filter (fun v => mq O (p * v) ω = 1)).card ≤ lo
      → O.label p = 0))} = (∅ : Set Ω) by ext ω; simp [hg]]
    simpa using hE

open scoped Classical in
/-- Decisiveness survives the family being chosen by the clustering, by the same
argument as `cutCorrect_selected_whp`: both are predicates of the vote's count, and the
vote's reads are ones the clustering never made. -/
theorem decided_selected_whp (O : Oracle μ S) (cands Q : Finset S) (p : S) (lo ha : ℕ)
    (hdisj : Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑Q : Set S))
    (T good : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, A₀ ∈ good → μ.real {ω | ¬ decided O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | fam ω ∈ good ∧ ¬ decided O lo ha (fam ω) p ω} ≤ E := by
  classical
  refine selection_side_bound O cands Q (fun v => p * v) hdisj T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ U => A₀ ∈ good ∧ ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo)) E hE ?_
  intro A₀ hA₀
  by_cases hg : A₀ ∈ good
  · refine le_trans (le_of_eq ?_) (hbad A₀ hA₀ hg)
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, decided, voteCount, hg, true_and]
  · rw [show {ω | A₀ ∈ good
      ∧ ¬ (ha < (A₀.filter (fun v => mq O (p * v) ω = 1)).card
        ∨ (A₀.filter (fun v => mq O (p * v) ω = 1)).card ≤ lo)} = (∅ : Set Ω) by
      ext ω; simp [hg]]
    simpa using hE

open scoped Classical in

lemma measureReal_le_of_ae_imp {A B : Set Ω} (h : ∀ᵐ ω ∂μ, ω ∈ A → ω ∈ B) :
    μ.real A ≤ μ.real B :=
  ENNReal.toReal_mono (measure_ne_top μ B) (measure_mono_ae h)

/-- The cut is correct at a prefix the family barely flips.  Only the side the prefix
actually sits on can fail, so one tail — not two — pays for it. -/
theorem cutCorrect_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : (lo : ℝ) < (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)) :
    μ.real {ω | ¬ cutCorrect O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hacc : ¬ (hi < voteCount O F p ω → O.label p = 1) := by
      intro hacc
      exact hbad ⟨hacc, fun _ => hp⟩
    have hgt : hi < voteCount O F p ω := by
      by_contra hc
      exact hacc (fun h => absurd h hc)
    have : (hi : ℝ) < (voteCount O F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]; linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hrej : ¬ (voteCount O F p ω ≤ lo → O.label p = 0) := by
      intro hrej
      exact hbad ⟨fun _ => hp, hrej⟩
    have hle : voteCount O F p ω ≤ lo := by
      by_contra hc
      exact hrej (fun h => absurd h hc)
    have : (voteCount O F p ω : ℝ) ≤ (lo : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)
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
    Measure ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) :=
  (((Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))).prod
    (Measure.infinitePi fun z : J × ℕ => D z.1)

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (drawMeasure D Dsf) := by
  unfold drawMeasure; infer_instance

lemma runMeasure_eq_prod (D : J → Measure S) (Dsf : Measure S) :
    runMeasure μ D Dsf = μ.prod (drawMeasure D Dsf) := rfl

/-- One table coordinate has the population's own law. -/
lemma map_drawCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i)
        (drawMeasure D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i)
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
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S))
            | ¬ (j ∈ populations → d.1.2 j i ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i) ⁻¹' Preᶜ := by
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
      (Measure.infinitePi fun z : J × ℕ => D z.1)
        {c | ((y.1, (y.2, c)) : Run Ω S J) ∈ A} ≤ E) :
    runMeasure μ D Dsf A ≤ E := by
  set νsq : Measure ((ℕ → S) × (J → ℕ → S)) :=
    (Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) with hνsq
  set νc : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνc
  have hmap : Measure.map (MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
      × (J × ℕ → S) ≃ᵐ Ω × (((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)))
      ((μ.prod νsq).prod νc) = runMeasure μ D Dsf :=
    (measurePreserving_prodAssoc μ νsq νc).map_eq
  have hpre : runMeasure μ D Dsf A = ((μ.prod νsq).prod νc)
      ((MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
        × (J × ℕ → S) ≃ᵐ Run Ω S J) ⁻¹' A) := by
    rw [← hmap, Measure.map_apply (MeasurableEquiv.prodAssoc).measurable hA]
  rw [hpre, Measure.prod_apply ((MeasurableEquiv.prodAssoc).measurable hA)]
  calc ∫⁻ y, νc (Prod.mk y ⁻¹' (MeasurableEquiv.prodAssoc ⁻¹' A)) ∂(μ.prod νsq)
      ≤ ∫⁻ _, E ∂(μ.prod νsq) := lintegral_mono (fun y => h y)
    _ = E := by simp

/-- The certification stream's own law. -/
lemma map_certStream (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] :
    Measure.map (certStream : Run Ω S J → _) (runMeasure μ D Dsf)
      = Measure.infinitePi fun z : J × ℕ => D z.1 := by
  rw [show (certStream : Run Ω S J → _) = Prod.snd ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_snd measurable_snd, runMeasure, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_snd_prod]
  simp

/-- One certification coordinate has the population's own law. -/
lemma map_certCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i))
        (drawMeasure D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i))
      = (fun c : J × ℕ → S => c (j, i)) ∘ Prod.snd := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) measurable_snd, drawMeasure, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact (measurePreserving_eval_infinitePi (fun z : J × ℕ => D z.1) (j, i)).map_eq

/-- The certification prefixes land in the flat set too, for the same reason the table's
do. -/
lemma ae_cert_mem_Pre (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Pre : Set S)
    (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ᵐ d ∂(drawMeasure D Dsf), ∀ j ∈ populations, ∀ i : ℕ, d.2 (j, i) ∈ Pre := by
  have hmeasPre : MeasurableSet (Preᶜ : Set S) := (Set.to_countable _).measurableSet
  have hcoord : ∀ z : J × ℕ, ∀ᵐ d ∂(drawMeasure D Dsf),
      z.1 ∈ populations → d.2 (z.1, z.2) ∈ Pre := by
    rintro ⟨j, i⟩
    by_cases hj : j ∈ populations
    · have hz : drawMeasure D Dsf {d | ¬ (j ∈ populations → d.2 (j, i) ∈ Pre)} = 0 := by
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S))
            | ¬ (j ∈ populations → d.2 (j, i) ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i)) ⁻¹' Preᶜ := by
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
      = (fun c : J × ℕ → S => (fun i : Fin m => c (j, i.val)))
        ∘ (certStream : Run Ω S J → _) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop), map_certStream D Dsf]
  refine (Measure.pi_eq (μ := fun _ : Fin m => D j) fun t ht => ?_).symm
  have hpre : (fun c : J × ℕ → S => (fun i : Fin m => c (j, i.val))) ⁻¹' Set.univ.pi t
      = Set.pi ↑((Finset.range m).image (fun i => (j, i)))
        (fun z => if h : z.2 < m then t ⟨z.2, h⟩ else Set.univ) := by
    ext c
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_image,
      Finset.coe_range, Set.mem_image, Set.mem_Iio]
    constructor
    · rintro h z ⟨a, ha, rfl⟩
      rw [dif_pos ha]
      exact h ⟨a, ha⟩
    · intro h i
      have := h (j, i.val) ⟨i.val, i.isLt, rfl⟩
      simpa [dif_pos i.isLt] using this
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre,
    Measure.infinitePi_pi]
  · rw [Finset.prod_image (fun a _ b _ h => (Prod.mk.inj h).2), ← Fin.prod_univ_eq_prod_range]
    exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
  · intro z _
    split_ifs with h
    exacts [ht ⟨z.2, h⟩, .univ]

/-- A table prefix and a certification prefix are drawn from independent streams, so their
joint law is the product — which is what lets `cross_collision_le` price a collision between
the two. -/
lemma map_prefCertPair (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (i i' : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
        (drawMeasure D Dsf) = (D j').prod (D j) := by
  have hf : Measurable (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i) := by fun_prop
  have hg : Measurable (fun c : J × ℕ → S => c (j, i')) := by fun_prop
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
  have hmapg : Measure.map (fun c : J × ℕ → S => c (j, i'))
      (Measure.infinitePi fun z : J × ℕ => D z.1) = D j :=
    (measurePreserving_eval_infinitePi (fun z : J × ℕ => D z.1) (j, i')).map_eq
  rw [show (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
      = Prod.map (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
        (fun c : J × ℕ → S => c (j, i')) from rfl,
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
lemma measurableSet_of_run_data (populations : Finset J) (B : Budget)
    (R : Finset S → Finset S → Set (Run Ω S J)) (hR : ∀ P C, MeasurableSet (R P C)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)}
      = ⋃ z : Finset S × Finset S, (({x : Run Ω S J | prefixesAt populations B.m x = z.1}
          ∩ {x : Run Ω S J | poolAt B.M x = z.2}) ∩ R z.1 z.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨(prefixesAt populations B.m x, poolAt B.M x), ⟨rfl, rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C⟩, ⟨hP, hC⟩, hx⟩
    simp only at hP hC
    rw [hP, hC]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    ((measurableSet_prefixesAt populations B.m z.1).inter
      (measurableSet_poolAt B.M z.2)).inter (hR z.1 z.2))

lemma measureReal_le_one' (Dj : Measure S) [IsProbabilityMeasure Dj] (A : Set S) :
    Dj.real A ≤ 1 := by
  have h := measureReal_mono (μ := Dj) (Set.subset_univ A) (by finiteness)
  simpa using h

lemma measureReal_le_one_of_prob {X : Type*} [MeasurableSpace X] (ν : Measure X)
    [IsProbabilityMeasure ν] (A : Set X) : ν.real A ≤ 1 := by
  have h := measureReal_mono (μ := ν) (Set.subset_univ A) (by finiteness)
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

open scoped Classical in
theorem decided_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (γ : ℝ) (hγ : 0 ≤ γ)
    (hclean : flipCount O F p = 0)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * ((1 - O.η) - γ))
    (hlo : (F.card : ℝ) * (O.η + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hf : flipCount O F p ≤ (F.card : ℝ) * 0 := by rw [hclean, mul_zero]
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp 0 γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * 0) + γ) ≤ voteSum O F p ω
    rw [← heq, mul_zero, add_zero]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp 0 γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - 0)) - γ)
    rw [← heq]
    have : O.η + (1 - 2 * O.η) * (1 - 0) = 1 - O.η := by ring
    rw [this]
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
      = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
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

open scoped Classical in


lemma measurableSet_cutCorrect (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0))))

lemma measurableSet_decided (O : Oracle μ S) (lo ha : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet {ω | ¬ decided O lo ha A₀ p ω} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo)))

open scoped Classical in
/-- The indecision event at one prefix, for the family the run produces. -/
lemma measurableSet_indecisive {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (O : Oracle μ S) (lo ha : ℕ) (good : Finset (Finset S)) (p : S) :
    MeasurableSet {ω | fam ω ∈ good ∧ ¬ decided O lo ha (fam ω) p ω} := by
  classical
  refine measurableSet_of_fam hfam hfamMeas
    (fun A₀ => {ω | A₀ ∈ good ∧ ¬ decided O lo ha A₀ p ω}) (fun A₀ => ?_)
  by_cases hg : A₀ ∈ good
  · simpa [hg] using measurableSet_decided O lo ha A₀ p
  · simpa [hg] using MeasurableSet.empty

open scoped Classical in
/-- The mis-cut event at one prefix, for the family the run produces. -/
lemma measurableSet_miscut {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (O : Oracle μ S) (lo ha : ℕ) (good : Finset (Finset S)) (p : S) :
    MeasurableSet {ω | fam ω ∈ good ∧ ¬ cutCorrect O lo ha (fam ω) p ω} := by
  classical
  refine measurableSet_of_fam hfam hfamMeas
    (fun A₀ => {ω | A₀ ∈ good ∧ ¬ cutCorrect O lo ha A₀ p ω}) (fun A₀ => ?_)
  by_cases hg : A₀ ∈ good
  · simpa [hg] using measurableSet_cutCorrect O lo ha A₀ p
  · simpa [hg] using MeasurableSet.empty

open scoped Classical in
/-- The indecision rate on the certification sample is below the limit, off an `E / l`
set.  `count_frac_le` fed by the per-prefix bound, with no union over the sample. -/
theorem indecision_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hCpos : 0 < C.card)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ decided O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ E / l := by
  classical
  have hrw : {ω | l * (C.card : ℝ)
      < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)}
      = (Set.univ : Set Ω) ∩ {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)} := by
    rw [Set.univ_inter]
  rw [hrw]
  refine count_frac_le (μ := μ) C (fun p => {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω})
    (fun ω p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω) Set.univ
    (fun p => measurableSet_indecisive hfam hfamMeas O lo ha (good p) p) E l hE hl hCpos ?_ ?_
  · intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact decided_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  · intro ω _ p _ hpr
    exact hpr

open scoped Classical in
/-- The same for the rate at which the cut is wrong. -/
theorem miscut_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hCpos : 0 < C.card)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ cutCorrect O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ E / l := by
  classical
  have hrw : {ω | l * (C.card : ℝ)
      < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      = (Set.univ : Set Ω) ∩ {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)} := by
    rw [Set.univ_inter]
  rw [hrw]
  refine count_frac_le (μ := μ) C (fun p => {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω})
    (fun ω p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω) Set.univ
    (fun p => measurableSet_miscut hfam hfamMeas O lo ha (good p) p) E l hE hl hCpos ?_ ?_
  · intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact cutCorrect_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  · intro ω _ p _ hpr
    exact hpr

open scoped Classical in
/-- At a fixed table, the state returns.  Each failure count splits into the prefixes
the family is light for — bounded by `count_frac_le` — and the heavy ones, whose number is
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
    (lo hi : ℕ) (εcov α τ l lcut : ℝ) (n₀ nlo : ℕ)
    (hnloC : (nlo : ℝ) ≤ (C.card : ℝ) - 2 * l * (C.card : ℝ))
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (E : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hlcut : 0 < lcut) (hlcl : lcut ≤ l)
    (hCpos : 0 < C.card)
    (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O lo (hi - 1) A₀ p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo (hi - 1) A₀ p ω} ≤ E)
    (hga : ∀ n : ℕ, nlo ≤ n → n ≤ C.card →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * lcut * (C.card : ℝ)))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ ((((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
            ∧ admitted O lo hi n₀ εcov α (fam ω) C ω)}
      ≤ E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) := by
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
      ∧ ¬ ((((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
              ≤ 2 * l * (C.card : ℝ))
          ∧ admitted O lo hi n₀ εcov α (fam ω) C ω)}
      ⊆ {ω | l * (C.card : ℝ)
            < ((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)}
        ∪ ({ω | lcut * (C.card : ℝ)
              < ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)}
          ∪ {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * lcut * (C.card : ℝ))
              ∧ (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
              ∧ ¬ admitted O lo hi n₀ εcov α (fam ω) C ω}) := by
    rintro ω ⟨hheavy, hbad⟩
    by_cases hindL : ((C.filter (fun p => fam ω ∈ good p
        ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ l * (C.card : ℝ)
    · by_cases hmisL : ((C.filter (fun p => fam ω ∈ good p
          ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ lcut * (C.card : ℝ)
      · have hind : (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
            ≤ 2 * l * (C.card : ℝ) := by
          have := hsplit ω (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)
          have hc : (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
              ≤ ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
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
        have hadm : ¬ admitted O lo hi n₀ εcov α (fam ω) C ω := fun h => hbad ⟨hind, h⟩
        exact Or.inr (Or.inr ⟨hmis, hind, hadm⟩)
      · exact Or.inr (Or.inl (not_le.1 hmisL))
    · exact Or.inl (not_le.1 hindL)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · exact indecision_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
      fam hfam hfamMeas hcongr E l hE hl hCpos hdec
  · refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
    · exact miscut_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
        fam hfam hfamMeas hcongr E lcut hE hlcut hCpos hcut
    · exact admitted_whp O C Q hdisjQ lo hi εcov α τ (2 * lcut * (C.card : ℝ))
        (2 * l * (C.card : ℝ)) n₀ nlo
        fam hQ (fun ω ω' h => hcongr ω ω' (fun w hw => h w (hQsup hw)))
        hτ hε0 hε1 hsig hnloC hga hα








lemma measurableSet_screenRate (O : Oracle μ S) (P : Finset S) (v : S) (sc scd : ℕ) :
    MeasurableSet[noiseAlg O Set.univ]
      {ω | scd * screenCount O P v ω ≤ sc * P.card} := by
  have hPr : ∀ p ∈ P, MeasurableSet[noiseAlg O Set.univ]
      {ω | ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1))} := by
    intro p _
    have h1 := measurableSet_mq_eq_one O (T := Set.univ) (w := p * v) (Set.mem_univ _)
    have h0 := measurableSet_mq_eq_one O (T := Set.univ) (w := p) (Set.mem_univ _)
    have hiff : {ω | (mq O (p * v) ω = 1) ↔ (mq O p ω = 1)}
        = ({ω | mq O (p * v) ω = 1} ∩ {ω | mq O p ω = 1})
          ∪ ({ω | mq O (p * v) ω = 1}ᶜ ∩ {ω | mq O p ω = 1}ᶜ) := by
      ext ω
      by_cases ha : mq O (p * v) ω = 1 <;> by_cases hb : mq O p ω = 1 <;> simp [ha, hb]
    have : MeasurableSet[noiseAlg O Set.univ] {ω | (mq O (p * v) ω = 1) ↔ (mq O p ω = 1)} := by
      rw [hiff]
      exact ((h1.inter h0).union (h1.compl.inter h0.compl))
    exact this.compl
  exact measurableSet_filter_pred' O (fun p ω => ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1)))
      hPr (fun U => scd * U.card ≤ sc * P.card)

lemma measurableSet_screenRate' (O : Oracle μ S) (P : Finset S) (v : S) (sc scd : ℕ) :
    MeasurableSet {ω | scd * screenCount O P v ω ≤ sc * P.card} :=
  noiseAlg_le O Set.univ _ (measurableSet_screenRate O P v sc scd)


/-- `clusterAt` with the draws fixed: the family is a function of the noise alone. -/
noncomputable def clusterOf (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (ω : Ω) : Finset S :=
  clusterAround O cn cd P (screened O sc scd P cands ω) ω k

lemma clusterAt_eq_clusterOf (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) :
    clusterAt O populations x B
      = clusterOf O B.cn B.cd B.sc B.scd (prefixesAt populations B.m x) (poolAt B.M x) B.k (oracleNoise x) :=
  rfl

lemma clusterOf_subset (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) : clusterOf O cn cd sc scd P cands k ω ⊆ cands :=
  fun v hv => screened_subset O sc scd P cands ω
    (clusterAround_subset O cn cd P _ ω k (one_mem_screened O sc scd P cands ω hone) hv)

lemma clusterOf_congr_mq (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterOf O cn cd sc scd P cands k ω = clusterOf O cn cd sc scd P cands k ω' := by
  classical
  have hscr : screened O sc scd P cands ω = screened O sc scd P cands ω' :=
    Finset.filter_congr (fun v hv => by rw [screenCount_congr O hone hv hbit])
  have hsub : ∀ w ∈ readSet P (screened O sc scd P cands ω'), (mq O w ω = 1 ↔ mq O w ω' = 1) := by
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
    (fun w => mq O w ω' = 1) = U ∧ clusterOf O cn cd sc scd P cands k ω' = A₀ with hPred
  have hcov : {ω | clusterOf O cn cd sc scd P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => mq O w ω = 1))} := by
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

open scoped Classical in




lemma measurable_badMassReal (O : Oracle μ S) (Dj : Measure S) (lo hi : ℕ) (A₀ : Finset S) :
    Measurable (fun ω => Dj.real {p | ¬ cutCorrect O lo hi A₀ p ω}) :=
  ENNReal.measurable_toReal.comp (measurable_badMass Dj
    (fun p => {ω | ¬ cutCorrect O lo hi A₀ p ω}) (fun p => measurableSet_cutCorrect O lo hi A₀ p))

open scoped Classical in

/-- The runs at one state whose screen lets through a candidate the table says flips more
than `Δ` of it. -/
noncomputable def screenBad (O : Oracle μ S) (populations : Finset J) (B : Budget) (Δ : ℝ) :
    Set (Run Ω S J) :=
  {x | B.m ≤ (prefixesAt populations B.m x).card
    ∧ ¬ ∀ v ∈ screenedAt O populations B x,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p)}

open scoped Classical in
lemma measurableSet_screenBad (O : Oracle μ S) (populations : Finset J) (B : Budget) (Δ : ℝ) :
    MeasurableSet (screenBad O populations B Δ) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet
      {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
    intro P C
    by_cases hm : B.m ≤ P.card
    swap
    · have : {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} = (∅ : Set (Run Ω S J)) := by
        ext x; simp [hm]
      rw [this]; exact MeasurableSet.empty
    have hsimp : {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
        = {x : Run Ω S J | ∃ v ∈ C, B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
      ext x; simp [hm]
    rw [hsimp]
    have hcov : {x : Run Ω S J | ∃ v ∈ C, B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
        = ⋃ v ∈ C.filter (fun v => Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p),
            oracleNoise ⁻¹' {ω : Ω | B.scd * screenCount O P v ω ≤ B.sc * P.card} := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_preimage, Finset.mem_coe,
        Finset.mem_filter, exists_prop]
      exact ⟨fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h2⟩, h1⟩,
        fun ⟨v, ⟨hv, h2⟩, h1⟩ => ⟨v, hv, h1, h2⟩⟩
    rw [hcov]
    exact Finset.measurableSet_biUnion _
      (fun v _ => measurable_nz (measurableSet_screenRate' O P v B.sc B.scd))
  have hrw : screenBad O populations B Δ
      = {x : Run Ω S J | x ∈ (fun P C => {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C,
          B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card
            ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p})
        (prefixesAt populations B.m x) (poolAt B.M x)} := by
    ext x
    simp only [screenBad, screened, screenedAt, Set.mem_setOf_eq, not_forall, Finset.mem_filter,
      not_not, exists_prop, and_congr_right_iff]
    intro _
    exact ⟨fun ⟨v, hv, h⟩ => ⟨v, hv.1, hv.2, h⟩, fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h1⟩, h2⟩⟩
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

/-- The screen rarely lets a badly-flipping candidate through.  One `screen_tail` per
pool member, at a fixed table: the flip counts are deterministic once the draws are, so this
is a plain union bound and costs `M + 1`. -/
theorem measureReal_screenBad_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.m) (Δ γ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    (runMeasure μ D Dsf).real (screenBad O populations B Δ)
      ≤ ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runMeasure μ D Dsf (screenBad O populations B Δ) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_screenBad O populations B Δ) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    by_cases hm : B.m ≤ Pd.card
    · have hPpos : 0 < Pd.card := lt_of_lt_of_le hmpos hm
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ}
          ⊆ ⋃ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
              {ω : Ω | B.scd * screenCount O Pd v ω ≤ B.sc * Pd.card} := by
        rintro ω ⟨-, hbad⟩
        simp only [not_forall, not_not] at hbad
        obtain ⟨v, hv, hflip⟩ := hbad
        obtain ⟨hvC, hvs⟩ := Finset.mem_filter.1 hv
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hvC, hflip⟩) hvs
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
          μ {ω : Ω | B.scd * screenCount O Pd v ω ≤ B.sc * Pd.card}
            ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * γ ^ 2)) := by
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
        have hcardR : (B.m : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        have htail := screen_tail hflat O hcd hP v hv1 Δ γ B.sc B.scd hγ hsig hscd hflip hsc
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail (Real.exp_le_exp.2 ?_))
        nlinarith [sq_nonneg γ]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef, ENNReal.ofReal_mul (by positivity),
        ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_one,
        ENNReal.ofReal_natCast]
      refine mul_le_mul' ?_ le_rfl
      have hCard : Cd.card ≤ B.M + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by simpa using le_trans Finset.card_image_le (by simp : (Finset.range B.M).card ≤ B.M))
      have hfc : (Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p)).card
          ≤ B.M + 1 := le_trans (Finset.card_filter_le _ _) hCard
      exact_mod_cast Nat.cast_le.2 hfc
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
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
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
/-- The pool holds accept-preserving suffixes.  Findability says a draw is
accept-preserving with probability at least `pAP`, so `M` draws hold about `pAP·M` of them. -/
theorem measureReal_apShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S) (M : ℕ)
    (pAP t : ℝ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}) :
    (runMeasure μ D Dsf).real {x : Run Ω S J |
        (((Finset.univ : Finset (Fin M)).filter
          (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)}
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
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
noncomputable def screenFail (O : Oracle μ S) (populations : Finset J) (B : Budget) :
    Set (Run Ω S J) :=
  {x | B.m ≤ (prefixesAt populations B.m x).card
    ∧ ¬ ∀ v ∈ poolAt B.M x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
      B.scd * screenCount O (prefixesAt populations B.m x) v (oracleNoise x) ≤ B.sc * (prefixesAt populations B.m x).card}

open scoped Classical in
lemma measurableSet_screenFail (O : Oracle μ S) (populations : Finset J) (B : Budget) :
    MeasurableSet (screenFail O populations B) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if B.m ≤ P.card then
      {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card} else ∅) := by
    intro P C
    split_ifs with hm
    · have hset : {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
          B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card}
          = oracleNoise ⁻¹' (⋃ v ∈ C.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
            {ω : Ω | ¬ (B.scd * screenCount O P v ω ≤ B.sc * P.card)}) := by
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
        (fun v _ => (measurableSet_screenRate' O P v B.sc B.scd).compl))
    · exact MeasurableSet.empty
  have hrw : screenFail O populations B
      = {x : Run Ω S J | x ∈ (fun P C => if B.m ≤ P.card then
          {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
            B.scd * screenCount O P v (oracleNoise x) ≤ B.sc * P.card} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)} := by
    ext x
    simp only [screenFail, Set.mem_setOf_eq]
    by_cases hm : B.m ≤ (prefixesAt populations B.m x).card
    · rw [if_pos hm]
      exact ⟨fun h => h.2, fun h => ⟨hm, h⟩⟩
    · rw [if_neg hm]
      exact ⟨fun h => absurd h.1 hm, fun h => absurd h (Set.notMem_empty x)⟩
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

open scoped Classical in
/-- The screen keeps the accept-preserving candidates.  One `screen_pass` per pool
member; the cutoff sits above the clean rate `2η(1−η)` by `γ`. -/
theorem measureReal_screenFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (γ : ℝ) (hγ : 0 ≤ γ) (hscd : 0 < B.scd)
    (hsc : (B.scd : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ)) :
    (runMeasure μ D Dsf).real (screenFail O populations B)
      ≤ ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runMeasure μ D Dsf (screenFail O populations B) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_screenFail O populations B) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    have hPle : Pd.card ≤ populations.card * B.m := by
      rw [hPd]
      refine le_trans Finset.card_biUnion_le ?_
      calc ∑ j ∈ populations, ((Finset.range B.m).image (fun i => d.1.2 j i)).card
          ≤ ∑ _j ∈ populations, B.m :=
            Finset.sum_le_sum (fun j _ => le_trans Finset.card_image_le (by simp))
        _ = populations.card * B.m := by rw [Finset.sum_const, smul_eq_mul]
    by_cases hm : B.m ≤ Pd.card
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenFail O populations B}
          ⊆ ⋃ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
              {ω : Ω | ¬ (B.scd * screenCount O Pd v ω ≤ B.sc * Pd.card)} := by
        rintro ω ⟨-, hbad⟩
        simp only [not_forall] at hbad
        obtain ⟨v, hv, hv1, hap, hfail⟩ := hbad
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hv, hv1, hap⟩) hfail
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
          μ {ω : Ω | ¬ (B.scd * screenCount O Pd v ω ≤ B.sc * Pd.card)}
            ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * γ ^ 2)) := by
        intro v hv
        obtain ⟨-, hv1, hap⟩ := Finset.mem_filter.1 hv
        have hclean : ∀ p ∈ Pd, O.flip v p = 0 := by
          intro p _
          show O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p = 0
          rw [hap p]
          rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
        have htail := screen_pass hflat O hcd hP v hv1 γ B.sc B.scd hγ hscd hclean hsc
        have hcardR : (B.m : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail (Real.exp_le_exp.2 ?_))
        nlinarith [sq_nonneg γ]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef, ENNReal.ofReal_mul (by positivity),
        ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_one,
        ENNReal.ofReal_natCast]
      refine mul_le_mul' ?_ le_rfl
      have hCard : Cd.card ≤ B.M + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by simpa using le_trans Finset.card_image_le (by simp : (Finset.range B.M).card ≤ B.M))
      have hfc : (Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p)).card
          ≤ B.M + 1 := le_trans (Finset.card_filter_le _ _) hCard
      exact_mod_cast Nat.cast_le.2 hfc
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
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd)
    (hsc : (B.scd : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.M : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.M => suffixDraw i.val x)} with hE1
  set E2 : Set (Run Ω S J) := {x | (((Finset.univ : Finset (Fin B.M)).filter
    (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ)
      ≤ (B.M : ℝ) * (pAP - t)} with hE2
  set E3 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => prefixDraw j₀ i.val x)} with hE3
  set E4 : Set (Run Ω S J) := screenFail O populations B with hE4
  have hsub : {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ⊆ E1 ∪ (E2 ∪ (E3 ∪ E4)) := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, not_or] at hcon
    obtain ⟨h1, h2, h3, h4⟩ := hcon
    have hinj : Function.Injective (fun i : Fin B.M => suffixDraw i.val x) := by
      by_contra h; exact h1 h
    have hapcount : (B.M : ℝ) * (pAP - t)
        < (((Finset.univ : Finset (Fin B.M)).filter
          (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ) := not_le.1 h2
    have hmP : B.m ≤ (prefixesAt populations B.m x).card := by
      have hinjP : Function.Injective (fun i : Fin B.m => prefixDraw j₀ i.val x) := by
        by_contra h; exact h3 h
      have hinjOn : Set.InjOn (fun i => prefixDraw j₀ i x) ↑(Finset.range B.m) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.m => prefixDraw j₀ i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.m => prefixDraw j₀ i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j₀ B.m x).card = B.m := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.m = (prefixesOf j₀ B.m x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.m x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j₀, hj₀, hq⟩)
    have hscreen : ∀ v ∈ poolAt B.M x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        B.scd * screenCount O (prefixesAt populations B.m x) v (oracleNoise x)
          ≤ B.sc * (prefixesAt populations B.m x).card := by
      by_contra h
      exact h4 ⟨hmP, h⟩
    -- the accept-preserving draws, as distinct strings, all survive the screen
    set I : Finset (Fin B.M) := (Finset.univ : Finset (Fin B.M)).filter
      (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p) with hI
    have hImg : I.image (fun i => suffixDraw i.val x) ⊆ screenedAt O populations B x := by
      intro v hv
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
      have hpool : suffixDraw i.val x ∈ poolAt B.M x :=
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
    have : B.k ≤ (screenedAt O populations B x).card := by
      have hcast : ((I.image (fun i => suffixDraw i.val x)).card : ℝ)
          ≤ ((screenedAt O populations B x).card : ℝ) := by
        exact_mod_cast Finset.card_le_card hImg
      have : (B.k : ℝ) ≤ ((screenedAt O populations B x).card : ℝ) := le_trans hk hcast
      exact_mod_cast this
    exact hx this
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ≤ (runMeasure μ D Dsf).real (E1 ∪ (E2 ∪ (E3 ∪ E4))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runMeasure μ D Dsf).real E1 + ((runMeasure μ D Dsf).real E2
        + ((runMeasure μ D Dsf).real E3 + (runMeasure μ D Dsf).real E4)) := by
        have h34 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 E4
        have h234 := measureReal_union_le (μ := runMeasure μ D Dsf) E2 (E3 ∪ E4)
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) E1 (E2 ∪ (E3 ∪ E4))
        linarith
    _ ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) := by
        gcongr
        · exact suffix_not_injective_le D Dsf B.M ρsf hρsf hρsf0
        · exact measureReal_apShort_le D Dsf O B.M pAP t hpAP0 ht hpAPBound
        · exact prefix_not_injective_le D Dsf j₀ B.m ρ hρj hρ0
        · exact measureReal_screenFail_le hflat O populations D Dsf hsupp B hcd γ hγ hscd hsc

open scoped Classical in
/-- Every member the clustering keeps is clean.  Three things have to go right: the
population's draws distinct, the screen holding, and the drawn prefixes not understating a
candidate's flip mass.  The Lloyd ranking does not appear — the family is a subset of what
the screen left, so it inherits the bound. -/
theorem measureReal_dirtyMember_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : Budget) (hcd : B.cn < B.cd) (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.m)
    (Δ γ g ρ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hρ0 : 0 ≤ ρ)
    (hρD : collisionMass (D j) ≤ ρ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
        flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + g}
      ≤ (B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2)
        + (B.M : ℝ) * Real.exp (-2 * (B.m : ℝ) * g ^ 2) := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + g with hΔp
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => prefixDraw j i.val x)} with hE1
  set E2 : Set (Run Ω S J) := screenBad O populations B Δ with hE2
  set E3 : Set (Run Ω S J) := {x | ¬ ∀ v ∈ poolAt B.M x,
    (∑ i : Fin B.m, O.flip v (prefixDraw j i.val x) ≤ (B.m : ℝ) * (Δp - g)) →
      flipMass O (D j) v ≤ Δp} with hE3
  have hsub : {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
      flipMass O (D j) v ≤ Δp} ⊆ (E1 ∪ E2) ∪ E3 := by
    intro x hx
    by_contra hnot
    simp only [Set.mem_union, not_or] at hnot
    obtain ⟨⟨h1, h2⟩, h3⟩ := hnot
    have hinjP : Function.Injective (fun i : Fin B.m => prefixDraw j i.val x) := by
      by_contra h; exact h1 h
    have hcardPre : B.m ≤ (prefixesAt populations B.m x).card := by
      have hinjOn : Set.InjOn (fun i => prefixDraw j i x) ↑(Finset.range B.m) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.m => prefixDraw j i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.m => prefixDraw j i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j B.m x).card = B.m := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.m = (prefixesOf j B.m x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.m x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j, hj, hq⟩)
    -- the screen held, so nothing it left flips much of the table
    have hscreen : ∀ v ∈ screenedAt O populations B x,
        ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
          ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p) := by
      by_contra h
      exact h2 ⟨hcardPre, h⟩
    have hrank := clusterAt_flip_bound O populations B x Δ hscreen
    obtain ⟨v, hv, hbad⟩ : ∃ v ∈ clusterAt O populations x B, ¬ (flipMass O (D j) v ≤ Δp) := by
      by_contra h
      exact hx (fun v hv => by
        by_contra hc
        exact h ⟨v, hv, hc⟩)
    have hvpool : v ∈ poolAt B.M x := clusterAt_subset O populations B x hv
    have hcount : ∑ i : Fin B.m, O.flip v (prefixDraw j i.val x) ≤ (B.m : ℝ) * (Δp - g) := by
      rw [sum_eq_sum_prefixesOf O j B.m x v hinjP]
      have hle1 := sum_prefixesOf_le_prefixesAt O populations j hj B.m x v
      have hle2 := not_le.1 (hrank v hv)
      have hle3 : ((prefixesAt populations B.m x).card : ℝ)
          ≤ (populations.card : ℝ) * (B.m : ℝ) := by
        exact_mod_cast card_prefixesAt_le populations B.m x
      have hΔ0 : (0 : ℝ) ≤ Δ := le_of_lt hΔ
      have hrw : (B.m : ℝ) * (Δp - g) = (populations.card : ℝ) * Δ * (B.m : ℝ) := by
        rw [hΔp]; ring
      rw [hrw]
      nlinarith
    exact hbad (by
      by_contra h
      exact h3 (by
        simp only [hE3, Set.mem_setOf_eq, not_forall]
        exact ⟨v, hvpool, hcount, h⟩))
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
        flipMass O (D j) v ≤ Δp}
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ E3) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2) + (runMeasure μ D Dsf).real E3 := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h123 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) E3
        linarith
    _ ≤ _ := by
        gcongr
        · exact prefix_not_injective_le D Dsf j B.m ρ hρD hρ0
        · exact measureReal_screenBad_le hflat O populations D Dsf hsupp B hcd hsig hmpos Δ γ
            hΔ hγ hscd hsc
        · exact pool_flipMass_le D Dsf O j B.m B.M Δp g hg (by positivity)

lemma measurableSet_cutCorrect' (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0)))

open scoped Classical in
/-- Events about the table, the pool and the certification draws are measurable: all three
take countably many values. -/
lemma measurableSet_of_run_data_cert (populations : Finset J) (j : J) (B : Budget)
    (R : Finset S → Finset S → (Fin B.m → S) → Set (Run Ω S J))
    (hR : ∀ P C t, MeasurableSet (R P C t)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)
      (fun i : Fin B.m => certPrefix j i.val x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => certPrefix j i.val x)}
      = ⋃ z : Finset S × Finset S × (Fin B.m → S),
          ((({x : Run Ω S J | prefixesAt populations B.m x = z.1}
            ∩ {x : Run Ω S J | poolAt B.M x = z.2.1})
            ∩ ⋂ i : Fin B.m, {x : Run Ω S J | certPrefix j i.val x = z.2.2 i})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_iInter]
    refine ⟨fun h => ⟨(prefixesAt populations B.m x, poolAt B.M x,
      fun i : Fin B.m => certPrefix j i.val x), ⟨⟨rfl, rfl⟩, fun _ => rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, t⟩, ⟨⟨hP, hC⟩, ht⟩, hx⟩
    simp only at hP hC ht
    rw [hP, hC, show (fun i : Fin B.m => certPrefix j i.val x) = t from funext ht]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.m z.1).inter
      (measurableSet_poolAt B.M z.2.1)).inter
        (MeasurableSet.iInter (fun i : Fin B.m =>
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
    (B : Budget) (εcov t : ℝ) : Set (Run Ω S J) :=
  {x | Function.Injective (fun i : Fin B.m => certPrefix j i.val x)
    ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)}
    ∧ (((certOf j B.m x).filter (fun p =>
        ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x))).card : ℝ)
      ≤ (B.m : ℝ) * (εcov - t)}

open scoped Classical in
lemma measurableSet_hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : Budget) (εcov t : ℝ) :
    MeasurableSet (hitShort O populations Dj j B εcov t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Function.Injective tt
        ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
            (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
        ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card
              : ℝ)
          ≤ (B.m : ℝ) * (εcov - t)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · by_cases hinj : Function.Injective tt
      · have hω : MeasurableSet {ω : Ω |
            εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω}
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω)).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω | εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p ω}
              ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ) ≤ (B.m : ℝ) * (εcov - t)})
            (fun A₀ => MeasurableSet.inter
              (measurableSet_le measurable_const (measurable_badMassReal O Dj B.lo B.hi A₀))
              (noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
                (fun p ω => ¬ cutCorrect O B.lo B.hi A₀ p ω)
                (fun p _ => measurableSet_cutCorrect' O B.lo B.hi A₀ p)
                (fun U => ((U.card : ℝ) ≤ (B.m : ℝ) * (εcov - t))))))
        have hset : {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)}
            = oracleNoise ⁻¹' {ω : Ω |
              εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω)).card : ℝ)
                ≤ (B.m : ℝ) * (εcov - t)} := by
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
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), hitShort,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

lemma measurableSet_voteCount_gt (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | n < voteCount O F p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => n < V.card)

lemma measurableSet_voteCount_le (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | voteCount O F p ω ≤ n} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => V.card ≤ n)

open scoped Classical in
lemma measurableSet_sideOf_gt (O : Oracle μ S) (A F : Finset S) (n : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => n < voteCount O F p ω) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => n < voteCount O F p ω)
    (fun p _ => measurableSet_voteCount_gt O F n p) (fun V => V = U))

open scoped Classical in
lemma measurableSet_decOf (O : Oracle μ S) (A F : Finset S) (lo hi : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => hi - 1 < voteCount O F p ω
        ∨ voteCount O F p ω ≤ lo) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => hi - 1 < voteCount O F p ω ∨ voteCount O F p ω ≤ lo)
    (fun p _ => (measurableSet_voteCount_gt O F (hi - 1) p).union
      (measurableSet_voteCount_le O F lo p)) (fun V => V = U))

open scoped Classical in
/-- The cut's two sides at a state, for the run's own family and certification sample. -/
noncomputable def cutSidesAt (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S × Finset S :=
  cutSides O B.lo B.hi ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)

open scoped Classical in
/-- The gate admits a cut the sample shows is badly wrong.  One event now, not one per
side: the decided set is big enough to test, at least `w` of its prefixes are mis-cut, and
the agreement statistic still clears `θ`. -/
noncomputable def gateBadAgree (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (θ w : ℝ) (n₀ : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
    ∧ n₀ ≤ (cutSidesAt O populations j B x).2.card
    ∧ w ≤ ((miscutOf O (cutSidesAt O populations j B x).1
        (cutSidesAt O populations j B x).2 : ℕ) : ℝ)
    ∧ ((cutSidesAt O populations j B x).2.card : ℝ) * θ
        ≤ ((agreeOf (cutSidesAt O populations j B x).1 (cutSidesAt O populations j B x).2
            ((certOf j B.m x).filter (fun p => mq O p (oracleNoise x) = 1)) : ℕ) : ℝ)}

open scoped Classical in
/-- At a fixed family and sample, the bad-agreement event is measurable: the cut's sides are
a fibre of the votes and the statistic is a fibre of the reads. -/
lemma measurableSet_badAgreeFixed (O : Oracle μ S) (lo hi : ℕ) (F A : Finset S)
    (θ w : ℝ) (n₀ : ℕ) :
    MeasurableSet {ω | n₀ ≤ (cutSides O lo hi F A ω).2.card
      ∧ w ≤ ((miscutOf O (cutSides O lo hi F A ω).1 (cutSides O lo hi F A ω).2 : ℕ) : ℝ)
      ∧ ((cutSides O lo hi F A ω).2.card : ℝ) * θ
          ≤ ((agreeOf (cutSides O lo hi F A ω).1 (cutSides O lo hi F A ω).2
              (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)} := by
  classical
  have hcover : {ω | n₀ ≤ (cutSides O lo hi F A ω).2.card
        ∧ w ≤ ((miscutOf O (cutSides O lo hi F A ω).1 (cutSides O lo hi F A ω).2 : ℕ) : ℝ)
        ∧ ((cutSides O lo hi F A ω).2.card : ℝ) * θ
            ≤ ((agreeOf (cutSides O lo hi F A ω).1 (cutSides O lo hi F A ω).2
                (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
      = ⋃ t ∈ A.powerset ×ˢ A.powerset,
          ({ω | cutSides O lo hi F A ω = t}
            ∩ {ω | n₀ ≤ t.2.card
                ∧ w ≤ ((miscutOf O t.1 t.2 : ℕ) : ℝ)
                ∧ (t.2.card : ℝ) * θ
                    ≤ ((agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe,
      exists_prop]
    constructor
    · intro h
      refine ⟨cutSides O lo hi F A ω, ?_, rfl, h⟩
      exact Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
        Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
    · rintro ⟨t, -, hEq, hb⟩
      rw [hEq]
      exact hb
  rw [hcover]
  refine Finset.measurableSet_biUnion _ (fun t _ => MeasurableSet.inter ?_ ?_)
  · have hpair : {ω | cutSides O lo hi F A ω = t}
        = {ω | A.filter (fun p => hi - 1 < voteCount O F p ω) = t.1}
          ∩ {ω | A.filter (fun p => hi - 1 < voteCount O F p ω
                ∨ voteCount O F p ω ≤ lo) = t.2} := by
      ext ω
      constructor
      · intro h
        exact ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
      · rintro ⟨h1, h2⟩
        exact Prod.ext h1 h2
    rw [hpair]
    exact (measurableSet_sideOf_gt O A F (hi - 1) t.1).inter
      (measurableSet_decOf O A F lo hi t.2)
  · by_cases h : n₀ ≤ t.2.card ∧ w ≤ ((miscutOf O t.1 t.2 : ℕ) : ℝ)
    · have hrw : {ω : Ω | n₀ ≤ t.2.card
          ∧ w ≤ ((miscutOf O t.1 t.2 : ℕ) : ℝ)
          ∧ (t.2.card : ℝ) * θ
              ≤ ((agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
          = {ω : Ω | (t.2.card : ℝ) * θ
              ≤ ((agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)} := by
        ext ω; simp [h.1, h.2]
      rw [hrw]
      exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp)
        (fun V => (t.2.card : ℝ) * θ ≤ ((agreeOf t.1 t.2 V : ℕ) : ℝ)))
    · have hrw : {ω : Ω | n₀ ≤ t.2.card
          ∧ w ≤ ((miscutOf O t.1 t.2 : ℕ) : ℝ)
          ∧ (t.2.card : ℝ) * θ
              ≤ ((agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, -⟩
        exact h ⟨h1, h2⟩
      rw [hrw]
      exact MeasurableSet.empty

open scoped Classical in
lemma measurableSet_gateBadAgree (O : Oracle μ S) (populations : Finset J) (j : J)
    (B : Budget) (θ w : ℝ) (n₀ : ℕ) :
    MeasurableSet (gateBadAgree O populations j B θ w n₀) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
        ∧ oracleNoise x ∈ {ω : Ω |
            n₀ ≤ (cutSides O B.lo B.hi
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card
          ∧ w ≤ ((miscutOf O
              (cutSides O B.lo B.hi
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
              (cutSides O B.lo B.hi
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2 : ℕ) : ℝ)
          ∧ ((cutSides O B.lo B.hi
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card : ℝ) * θ
              ≤ ((agreeOf
                  (cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
                  (cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).2
                  (((Finset.univ : Finset (Fin B.m)).image tt).filter
                    (fun p => mq O p ω = 1)) : ℕ) : ℝ)}} else ∅) := by
    intro P C tt
    split_ifs with hone
    · have hω : MeasurableSet {ω : Ω |
          n₀ ≤ (cutSides O B.lo B.hi
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
              ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card
        ∧ w ≤ ((miscutOf O
            (cutSides O B.lo B.hi
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
              ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
            (cutSides O B.lo B.hi
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
              ((Finset.univ : Finset (Fin B.m)).image tt) ω).2 : ℕ) : ℝ)
        ∧ ((cutSides O B.lo B.hi
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
              ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card : ℝ) * θ
            ≤ ((agreeOf
                (cutSides O B.lo B.hi
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                  ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
                (cutSides O B.lo B.hi
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                  ((Finset.univ : Finset (Fin B.m)).image tt) ω).2
                (((Finset.univ : Finset (Fin B.m)).image tt).filter
                  (fun p => mq O p ω = 1)) : ℕ) : ℝ)} :=
        measurableSet_of_fam (T := C.powerset)
          (fun ω => Finset.mem_powerset.2
            (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
          (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
          (fun A₀ => {ω : Ω |
            n₀ ≤ (cutSides O B.lo B.hi (A₀.erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card
          ∧ w ≤ ((miscutOf O
              (cutSides O B.lo B.hi (A₀.erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
              (cutSides O B.lo B.hi (A₀.erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2 : ℕ) : ℝ)
          ∧ ((cutSides O B.lo B.hi (A₀.erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card : ℝ) * θ
              ≤ ((agreeOf
                  (cutSides O B.lo B.hi (A₀.erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
                  (cutSides O B.lo B.hi (A₀.erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).2
                  (((Finset.univ : Finset (Fin B.m)).image tt).filter
                    (fun p => mq O p ω = 1)) : ℕ) : ℝ)})
          (fun A₀ => measurableSet_badAgreeFixed O B.lo B.hi (A₀.erase 1)
            ((Finset.univ : Finset (Fin B.m)).image tt) θ w n₀)
      have hconst : MeasurableSet {x : Run Ω S J |
          Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)} := by
        by_cases hdj : Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
        · have he : {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)}
              = Set.univ := by ext x; simp [hdj]
          rw [he]; exact MeasurableSet.univ
        · have he : {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)}
              = ∅ := by ext x; simp [hdj]
          rw [he]; exact MeasurableSet.empty
      exact hconst.inter (measurable_nz hω)
    · exact MeasurableSet.empty
  have hrw : gateBadAgree O populations j B θ w n₀
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
            ∧ oracleNoise x ∈ {ω : Ω |
                n₀ ≤ (cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card
              ∧ w ≤ ((miscutOf O
                  (cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
                  (cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).2 : ℕ) : ℝ)
              ∧ ((cutSides O B.lo B.hi
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) ω).2.card : ℝ) * θ
                  ≤ ((agreeOf
                      (cutSides O B.lo B.hi
                        ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                        ((Finset.univ : Finset (Fin B.m)).image tt) ω).1
                      (cutSides O B.lo B.hi
                        ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1)
                        ((Finset.univ : Finset (Fin B.m)).image tt) ω).2
                      (((Finset.univ : Finset (Fin B.m)).image tt).filter
                        (fun p => mq O p ω = 1)) : ℕ) : ℝ)}} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), gateBadAgree, cutSidesAt,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

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
/-- The gate does not admit a cut the sample shows is badly wrong.  The run-level form
of `gate_agree_bound_right`: slice by the draws, then condition on the votes. -/
theorem measureReal_gateBadAgree_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (θ τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ B.m →
      (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w ≤ (n : ℝ) * (θ - τ)) :
    (runMeasure μ D Dsf).real (gateBadAgree O populations j B θ w n₀)
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) with hEdef
  have hEnn : runMeasure μ D Dsf (gateBadAgree O populations j B θ w n₀)
      ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_gateBadAgree O populations j B θ w n₀) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.m).image (fun i => d.2 (j, i)) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    have hAcard : Ad.card ≤ B.m := by
      rw [hAd]
      exact le_trans Finset.card_image_le (by simp)
    by_cases hdisj : Disjoint Pd Ad
    · set fam : Ω → Finset S :=
        fun ω => (clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1 with hfamdef
      have hmain := gate_agree_bound_right O Ad (readSet Pd Cd ∪ readSet Ad (Cd.erase 1))
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj) B.lo B.hi fam
        (fun ω p hp v hv => Finset.mem_union_right _ (mem_readSet hp
          (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
            clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2⟩)))
        (fun ω ω' h => congrArg (fun t : Finset S => t.erase 1)
          (clusterAt_congr O populations B d
            (fun q hq => h q (Finset.mem_union_left _ hq))))
        θ τ w n₀ hτ hsig
        (fun n hn hnc => hθ n hn (le_trans hnc hAcard))
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadAgree O populations j B θ w n₀}
          ⊆ {ω : Ω | n₀ ≤ (cutSides O B.lo B.hi (fam ω) Ad ω).2.card
              ∧ w ≤ ((miscutOf O (cutSides O B.lo B.hi (fam ω) Ad ω).1
                    (cutSides O B.lo B.hi (fam ω) Ad ω).2 : ℕ) : ℝ)
              ∧ ((cutSides O B.lo B.hi (fam ω) Ad ω).2.card : ℝ) * θ
                  ≤ ((agreeOf (cutSides O B.lo B.hi (fam ω) Ad ω).1
                      (cutSides O B.lo B.hi (fam ω) Ad ω).2
                      (Ad.filter (fun p => mq O p ω = 1)) : ℕ) : ℝ)} := by
        rintro ω ⟨-, h1, h2, h3⟩
        exact ⟨h1, h2, h3⟩
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadAgree O populations j B θ w n₀}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hdisj h1
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (gateBadAgree O populations j B θ w n₀)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
/-- A cut wrong on the population is wrong on the sample.  The family is a function of
the noise and the table, so the certification draws are independent of it and plain
Hoeffding applies. -/
theorem measureReal_hitShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : Budget) (εcov t : ℝ) (hε : 0 ≤ εcov) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (hitShort O populations (D j) j B εcov t)
      ≤ Real.exp (-2 * (B.m : ℝ) * t ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (B.m : ℝ) * t ^ 2) with hEdef
  have hEnn : runMeasure μ D Dsf (hitShort O populations (D j) j B εcov t) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_hitShort O populations (D j) j B εcov t) _ ?_
    intro y
    set F : Finset S := clusterAt O populations ((y.1, (y.2, fun _ => (1 : S))) : Run Ω S J) B
      with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J × ℕ → S,
        clusterAt O populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : εcov ≤ (D j).real W
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ hitShort O populations (D j) j B εcov t}
          ⊆ {c | (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
            ≤ (B.m : ℝ) * (εcov - t)} := by
        rintro c ⟨hinj, -, hcount⟩
        show (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
          ≤ (B.m : ℝ) * (εcov - t)
        calc (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
            = (((certOf j B.m ((y.1, (y.2, c)) : Run Ω S J)).filter
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1)).card : ℝ) :=
              congrArg (fun n : ℕ => (n : ℝ)) (card_filter_certOf j B.m
                ((y.1, (y.2, c)) : Run Ω S J)
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1) _ _ hinj)
          _ ≤ (B.m : ℝ) * (εcov - t) := hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.infinitePi fun z : J × ℕ => D z.1) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [hEdef]
      have hcert := cert_hits_wrongSet D j B.m W εcov t hε ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ hitShort O populations (D j) j B εcov t} = (∅ : Set (J × ℕ → S)) := by
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
    (j : J) (B : Budget) (f q t : ℝ) : Set (Run Ω S J) :=
  {x | Dj.real {p | ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
        ≤ (((clusterAt O populations x B).erase 1).card : ℝ) * f)} ≤ q
    ∧ (B.m : ℝ) * (q + t)
        ≤ (((certOf j B.m x).filter (fun p =>
          ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
            ≤ (((clusterAt O populations x B).erase 1).card : ℝ) * f))).card : ℝ)}

open scoped Classical in
lemma measurableSet_heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : Budget) (f q t : ℝ) :
    MeasurableSet (heavyHits O populations Dj j B f q t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
            ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
          ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ) * f)} ≤ q
        ∧ (B.m : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
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
          ∧ (B.m : ℝ) * (q + t)
              ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)})
        (fun A₀ => ?_))
      by_cases hcond : Dj.real {p | ¬ (flipCount O (A₀.erase 1) p
            ≤ ((A₀.erase 1).card : ℝ) * f)} ≤ q
          ∧ (B.m : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
              ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)
      · simpa [hcond] using MeasurableSet.univ
      · simpa [hcond] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : heavyHits O populations Dj j B f q t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
              ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ) * f)} ≤ q
            ∧ (B.m : ℝ) * (q + t)
                ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ (flipCount O ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                    ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ)
                      * f))).card : ℝ)} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), heavyHits,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The flip-heavy prefixes are a `q` fraction of the population, so the sample sees at most
`m(q + t)` of them. -/
theorem measureReal_heavyHits_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : Budget) (f q t : ℝ) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (heavyHits O populations (D j) j B f q t)
      ≤ Real.exp (-2 * (B.m : ℝ) * t ^ 2) := by
  classical
  have hEnn : runMeasure μ D Dsf (heavyHits O populations (D j) j B f q t)
      ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * t ^ 2)) := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_heavyHits O populations (D j) j B f q t) _ ?_
    intro y
    set F : Finset S :=
      (clusterAt O populations ((y.1, (y.2, fun _ => (1 : S))) : Run Ω S J) B).erase 1 with hF
    set W : Set S := {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} with hW
    have hFeq : ∀ c : J × ℕ → S,
        (clusterAt O populations ((y.1, (y.2, c)) : Run Ω S J) B).erase 1 = F := fun c => rfl
    by_cases hmass : (D j).real W ≤ q
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ heavyHits O populations (D j) j B f q t}
          ⊆ {c | (B.m : ℝ) * (q + t)
            ≤ (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)} := by
        rintro c ⟨-, hcount⟩
        refine le_trans hcount ?_
        exact_mod_cast card_filter_certOf_le j B.m ((y.1, (y.2, c)) : Run Ω S J)
          (fun p => p ∈ W) _ _
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.infinitePi fun z : J × ℕ => D z.1) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_upper D j B.m W q t ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ heavyHits O populations (D j) j B f q t} = (∅ : Set (J × ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (heavyHits O populations (D j) j B f q t)).toReal
      ≤ (ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * t ^ 2))).toReal :=
        ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = Real.exp (-2 * (B.m : ℝ) * t ^ 2) := ENNReal.toReal_ofReal (Real.exp_nonneg _)

lemma measurableSet_decided' (O : Oracle μ S) (lo ha : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ decided O lo ha A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo))

open scoped Classical in
lemma measurableSet_indecisionCount (O : Oracle μ S) (lo ha : ℕ) (A₀ A : Finset S) (r : ℝ) :
    MeasurableSet {ω | (((A.filter (fun p => ¬ decided O lo ha A₀ p ω)).card : ℝ) ≤ r)} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => ¬ decided O lo ha A₀ p ω)
    (fun p _ => measurableSet_decided' O lo ha A₀ p) (fun U => ((U.card : ℝ) ≤ r)))

open scoped Classical in
/-- The gate's verdict is measurable: each side is a fibre of the votes, and the count it
scores is a fibre of the bits at the certification prefixes. -/
lemma measurableSet_admittedFixed (O : Oracle μ S) (lo hi n₀ : ℕ) (F A : Finset S)
    (εcov α : ℝ) : MeasurableSet {ω | admitted O lo hi n₀ εcov α F A ω} := by
  classical
  have hcover : {ω | admitted O lo hi n₀ εcov α F A ω}
      = ⋃ t ∈ A.powerset ×ˢ A.powerset,
          ({ω | cutSides O lo hi F A ω = t}
            ∩ {ω | n₀ ≤ t.2.card →
                binomSfGe t.2.card (gateAcc O εcov)
                  (agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1))) ≤ α}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe,
      exists_prop]
    constructor
    · intro h
      refine ⟨cutSides O lo hi F A ω, ?_, rfl, ?_⟩
      · exact Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
          Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
      · simpa [admitted, agreeCount] using h
    · rintro ⟨t, -, hEq, hb⟩
      simpa [admitted, agreeCount, hEq] using hb
  rw [hcover]
  refine Finset.measurableSet_biUnion _ (fun t _ => MeasurableSet.inter ?_ ?_)
  · have hpair : {ω | cutSides O lo hi F A ω = t}
        = {ω | A.filter (fun p => hi - 1 < voteCount O F p ω) = t.1}
          ∩ {ω | A.filter (fun p => hi - 1 < voteCount O F p ω
                ∨ voteCount O F p ω ≤ lo) = t.2} := by
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
          binomSfGe t.2.card (gateAcc O εcov)
            (agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1))) ≤ α}
          = {ω : Ω | binomSfGe t.2.card (gateAcc O εcov)
            (agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1))) ≤ α} := by
        ext ω; simp [h]
      rw [hrw]
      exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp)
        (fun V => binomSfGe t.2.card (gateAcc O εcov) (agreeOf t.1 t.2 V) ≤ α))
    · have hrw : {ω : Ω | n₀ ≤ t.2.card →
          binomSfGe t.2.card (gateAcc O εcov)
            (agreeOf t.1 t.2 (A.filter (fun p => mq O p ω = 1))) ≤ α}
          = (Set.univ : Set Ω) := by
        ext ω; simp [h]
      rw [hrw]
      exact MeasurableSet.univ

/-- The family is usable at `p`: no member flips it, and the family is neither too
small for the thresholds nor larger than the round allows. -/
def famGood (O : Oracle μ S) (kmin kmax : ℕ) (F : Finset S) (p : S) : Prop :=
  flipCount O F p ≤ (F.card : ℝ) * 0 ∧ kmin ≤ F.card ∧ F.card ≤ kmax

open scoped Classical in
/-- The draws were good and the state still did not return.  Everything the state needs
of its draws — the certification prefixes fresh and nonempty, and the family light on all
but an `l` fraction of them — holds, and the round's two tests still fail.
This is the event `ret_at_whp` prices; the rest of the lift charges the draws. -/
noncomputable def retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov α l lcut : ℝ) (n₀ kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
    ∧ (certOf j B.m x).card = B.m
    ∧ 0 < (certOf j B.m x).card
    ∧ (((certOf j B.m x).filter (fun p =>
        ¬ famGood O kmin kmax ((clusterAt O populations x B).erase 1) p)).card : ℝ)
        ≤ lcut * ((certOf j B.m x).card : ℝ)
    ∧ ¬ ((((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
            ((clusterAt O populations x B).erase 1) p (oracleNoise x))).card : ℝ)
          ≤ 2 * l * ((certOf j B.m x).card : ℝ)
        ∧ admitted O B.lo B.hi B.gmin εcov α ((clusterAt O populations x B).erase 1)
            (certOf j B.m x) (oracleNoise x))}

open scoped Classical in
lemma measurableSet_retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov α l lcut : ℝ) (n₀ kmin kmax : ℕ) :
    MeasurableSet (retMiss O populations j B εcov α l lcut n₀ kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
        ∧ ((Finset.univ : Finset (Fin B.m)).image tt).card = B.m
        ∧ 0 < ((Finset.univ : Finset (Fin B.m)).image tt).card
        ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
            ¬ famGood O kmin kmax
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
            ≤ lcut * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
        ∧ ¬ (((((Finset.univ : Finset (Fin B.m)).image tt).filter
                (fun p => ¬ decided O B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
              ≤ 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ∧ admitted O B.lo B.hi B.gmin εcov α
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) (oracleNoise x))} else ∅) := by
    intro P C tt
    split_ifs with hone
    · set A : Finset S := (Finset.univ : Finset (Fin B.m)).image tt with hA
      by_cases hdraw : Disjoint P A ∧ A.card = B.m ∧ 0 < A.card
      · have hω : MeasurableSet {ω : Ω |
            ((A.filter (fun p => ¬ famGood O kmin kmax
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p)).card : ℝ)
              ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi B.gmin εcov α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) A ω)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω : Ω |
              ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O B.lo B.hi B.gmin εcov α (A₀.erase 1) A ω)})
            (fun A₀ => ?_)
          by_cases hlight : ((A.filter (fun p =>
              ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ) ≤ lcut * (A.card : ℝ)
          · have hrw : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O B.lo B.hi B.gmin εcov α (A₀.erase 1) A ω)}
                = ({ω : Ω | ((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                        ≤ 2 * l * (A.card : ℝ)}
                  ∩ {ω : Ω | admitted O B.lo B.hi B.gmin εcov α (A₀.erase 1) A ω})ᶜ := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_inter_iff, hlight, true_and]
            rw [hrw]
            exact ((measurableSet_indecisionCount O B.lo (B.hi - 1) (A₀.erase 1) A
              (2 * l * (A.card : ℝ))).inter
                (measurableSet_admittedFixed O B.lo B.hi B.gmin (A₀.erase 1) A εcov α)).compl
          · have hz : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O B.lo B.hi B.gmin εcov α (A₀.erase 1) A ω)} = (∅ : Set Ω) := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
              rintro ⟨h, -⟩
              exact hlight h
            rw [hz]
            exact MeasurableSet.empty
        have hset : {x : Run Ω S J | Disjoint P A ∧ A.card = B.m ∧ 0 < A.card
            ∧ ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi B.gmin εcov α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) A (oracleNoise x))}
            = oracleNoise ⁻¹' {ω : Ω |
              ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p)).card : ℝ)
                ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O B.lo B.hi B.gmin εcov α
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) A ω)} := by
          ext x
          constructor
          · rintro ⟨-, -, -, hl, hn⟩
            exact ⟨hl, hn⟩
          · rintro ⟨hl, hn⟩
            exact ⟨hdraw.1, hdraw.2.1, hdraw.2.2, hl, hn⟩
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | Disjoint P A ∧ A.card = B.m ∧ 0 < A.card
            ∧ ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi B.gmin εcov α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) A (oracleNoise x))}
            = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro ⟨h1, h2, h3, -⟩
          exact hdraw ⟨h1, h2, h3⟩
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : retMiss O populations j B εcov α l lcut n₀ kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
            ∧ ((Finset.univ : Finset (Fin B.m)).image tt).card = B.m
            ∧ 0 < ((Finset.univ : Finset (Fin B.m)).image tt).card
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ famGood O kmin kmax
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                ≤ lcut * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ∧ ¬ (((((Finset.univ : Finset (Fin B.m)).image tt).filter
                    (fun p => ¬ decided O B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                ∧ admitted O B.lo B.hi B.gmin εcov α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) (oracleNoise x))} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), retMiss,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    constructor
    · rintro ⟨h1, h2, h3, h4, h5⟩
      exact ⟨h1, h2, h3, h4, h5⟩
    · rintro ⟨h1, h2, h3, h4, h5⟩
      exact ⟨h1, h2, h3, h4, h5⟩
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
lemma measurableSet_clusterOf_erase (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S)
    (k : ℕ) (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | (clusterOf O cn cd sc scd P cands k ω).erase 1 = A₀} := by
  classical
  have hrw : {ω | (clusterOf O cn cd sc scd P cands k ω).erase 1 = A₀}
      = {ω | ω ∈ (fun A₁ : Finset S =>
        if A₁.erase 1 = A₀ then (Set.univ : Set Ω) else ∅) (clusterOf O cn cd sc scd P cands k ω)} := by
    ext ω
    by_cases h : (clusterOf O cn cd sc scd P cands k ω).erase 1 = A₀ <;> simp [h]
  rw [hrw]
  refine measurableSet_of_fam (T := cands.powerset)
    (fun ω => Finset.mem_powerset.2 (clusterOf_subset O cn cd sc scd P cands k ω hone))
    (fun A₁ => measurableSet_clusterOf O cn cd sc scd P cands k hone A₁)
    (fun A₁ => if A₁.erase 1 = A₀ then Set.univ else ∅) (fun A₁ => ?_)
  split_ifs
  · exact MeasurableSet.univ
  · exact MeasurableSet.empty

open scoped Classical in
/-- The lift of `ret_at_whp`.  At every table the state's own round fails only as often
as the two fractional counts and the two gate tails allow; the draws are charged elsewhere. -/
theorem measureReal_retMiss_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (εcov α τ l lcut E : ℝ) (kmin kmax nlo : ℕ)
    (hE : 0 ≤ E) (hl : 0 < l) (hlcut : 0 < lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1)
    (hl1 : 2 * l ≤ 1) (hnloB : (nlo : ℝ) ≤ (1 - 2 * l) * (B.m : ℝ))
    (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, nlo ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * lcut * (c : ℝ)))
    (hα : Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real (retMiss O populations j B εcov α l lcut B.gmin kmin kmax)
      ≤ E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) := by
  classical
  set R : ℝ := E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) with hR
  have hR0 : 0 ≤ R := by
    rw [hR]
    have : (0 : ℝ) ≤ E / l := div_nonneg hE hl.le
    have hcut0 : (0 : ℝ) ≤ E / lcut := div_nonneg hE hlcut.le
    have h2 : (0 : ℝ) ≤ Real.exp (-2 * (nlo : ℝ) * τ ^ 2) := Real.exp_nonneg _
    linarith
  have hEnn : runMeasure μ D Dsf (retMiss O populations j B εcov α l lcut B.gmin kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_retMiss O populations j B εcov α l lcut B.gmin kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.m).image (fun i => d.2 (j, i)) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    have hAcard : Ad.card ≤ B.m := by
      rw [hAd]
      exact le_trans Finset.card_image_le (by simp)
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.m ∧ 0 < Ad.card
    · obtain ⟨hdisj, hAdm, hCpos⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => (clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1 with hfamdef
      set T : Finset (Finset S) := Cd.powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * 0 ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2)
      have hmain := ret_at_whp hflat O Pd Cd Ad hP hA hdisj
        (readSet Pd Cd ∪ readSet Ad (Cd.erase 1)) Finset.subset_union_left
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj)
        B.lo B.hi εcov α τ l lcut B.gmin nlo
        (by
          have hcard : (Ad.card : ℝ) = (B.m : ℝ) := by exact_mod_cast hAdm
          rw [hcard]
          nlinarith [hnloB])
        T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => Finset.mem_powerset.1 ht) fam hfamT
        (fun A₀ => measurableSet_clusterOf_erase O B.cn B.cd B.sc B.scd Pd Cd B.k
          (Finset.mem_insert_self 1 _) A₀)
        (fun ω ω' h => congrArg (fun t : Finset S => t.erase 1)
          (clusterAt_congr O populations B d h))
        (fun ω p hp v hv => Finset.mem_union_right _ (mem_readSet hp
          (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
            clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2⟩)))
        E hE hl hlcut hlcl hCpos hτ hε0 hε1 hsig
        (fun p _ A₀ _ hgp => hdec A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun n hn hnc => hga n Ad.card hn hnc hAcard)
        hα
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ retMiss O populations j B εcov α l lcut B.gmin kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (Ad.card : ℝ)
            ∧ ¬ ((((Ad.filter (fun p =>
                    ¬ decided O B.lo (B.hi - 1) (fam ω) p ω)).card : ℝ)
                  ≤ 2 * l * (Ad.card : ℝ))
              ∧ admitted O B.lo B.hi B.gmin εcov α (fam ω) Ad ω)} := by
        rintro ω ⟨-, -, -, hlight, hbad⟩
        refine ⟨le_trans (le_of_eq ?_) hlight, hbad⟩
        refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
        intro p _
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
        exact Iff.rfl
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ retMiss O populations j B εcov α l lcut B.gmin kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        exact hdraw ⟨h1, h2, h3⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (retMiss O populations j B εcov α l lcut B.gmin kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

open scoped Classical in
/-- The round's own test at one population: the FNR count and the gate. -/
noncomputable def retAt (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) (j : J) : Set (Run Ω S J) :=
  {x | (((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
          ((clusterAt O populations x B).erase 1) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * ((certOf j B.m x).card : ℝ)
    ∧ admitted O B.lo B.hi B.gmin εcov α ((clusterAt O populations x B).erase 1)
        (certOf j B.m x) (oracleNoise x)}

lemma mem_ret_iff (O : Oracle μ S) (populations : Finset J) (indecisionLimit εcov α : ℝ)
    (B : Budget) (x : Run Ω S J) :
    x ∈ ret O populations indecisionLimit εcov α B
      ↔ ∀ j ∈ populations, x ∈ retAt O populations indecisionLimit εcov α B j := by
  constructor
  · rintro ⟨h1, h2⟩ j hj
    exact ⟨h1 j hj, h2 j hj⟩
  · intro h
    exact ⟨fun j hj => (h j hj).1, fun j hj => (h j hj).2⟩

open scoped Classical in
/-- The runs where the clustering stalls on the seed, or overshoots the round's size.  The
gate refuses a stalled family through whichever side the population populates, so this is the liveness
half's obligation, not the round's. -/
noncomputable def stalled (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | ¬ (kmin ≤ ((clusterAt O populations x B).erase 1).card
      ∧ ((clusterAt O populations x B).erase 1).card ≤ kmax)}

open scoped Classical in
/-- A pool of `k` screened candidates is a family of `k`.  The seed is one of them and
the ranking keeps it, so the family the gate sees has `k − 1` members besides the seed. -/
lemma stalled_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (hcd : B.cn < B.cd) (hkpos : 0 < B.k) :
    stalled O populations B (B.k - 1) (B.k - 1)
      ⊆ {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)} := by
  intro x hx
  by_contra hk
  simp only [Set.mem_setOf_eq, Classical.not_not] at hk
  have hcard : (clusterAt O populations x B).card = B.k :=
    clusterAround_card O hcd (prefixesAt populations B.m x) (screenedAt O populations B x)
      (oracleNoise x) B.k (one_mem_screenedAt O populations B x) hk hkpos
  have hone : (1 : S) ∈ clusterAt O populations x B :=
    one_mem_clusterAround O B.cn B.cd _ _ (oracleNoise x) B.k
  have herase : ((clusterAt O populations x B).erase 1).card = B.k - 1 := by
    rw [Finset.card_erase_of_mem hone, hcard]
  exact hx ⟨le_of_eq herase.symm, le_of_eq herase⟩

open scoped Classical in
/-- The liveness half's obligation, discharged: the clustering has a family of the round's
own size except on the events `measureReal_smallScreen_le` prices. -/
theorem measureReal_stalled_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd)
    (hsc : (B.scd : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.M : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real (stalled O populations B (B.k - 1) (B.k - 1))
      ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hcd hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations D Dsf hsupp B hcd j₀ hj₀ γ pAP t ρsf ρ
      hγ hpAP0 ht hpAPBound hscd hsc hcount hρsf hρsf0 hρj hρ0)

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- The round returns at one population.  Everything outside `retMiss` is a fact about
the draws: the certification prefixes repeat, or meet the table, or under-represent a class,
or the family is dirty and the sample sees it.  The cluster's own size is the liveness
half's business and is carried as `Estall`. -/
theorem measureReal_notRetAt_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (hmpos : 0 < B.m) (hsig : O.η ≤ 1 / 2)
    (εcov α τ l lcut E Δp ρ th : ℝ) (kmin kmax nlo : ℕ)
    (hE : 0 ≤ E) (hl : 0 < l) (hlcut : 0 < lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1)
    (hl1 : 2 * l ≤ 1) (hnloB : (nlo : ℝ) ≤ (1 - 2 * l) * (B.m : ℝ))
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th)
    (hheavy : Δp * (kmax : ℝ) + th ≤ lcut)
    (Estall : ℝ) (hstall : (runMeasure μ D Dsf).real (stalled O populations B kmin kmax) ≤ Estall)
    (Edirty : ℝ) (hdirty : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, nlo ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * lcut * (c : ℝ)))
    (hα : Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
      ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
          + (E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => certPrefix j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.m x) (certOf j B.m x)} with hE2
  set E5 : Set (Run Ω S J) := stalled O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits O populations (D j) j B 0 (Δp * (kmax : ℝ)) th with hE7
  set E8 : Set (Run Ω S J) := retMiss O populations j B εcov α l lcut B.gmin kmin kmax with hE8
  have hsub : {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
      ⊆ (E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8))) := by
    intro x hx
    by_cases h1 : Function.Injective (fun i : Fin B.m => certPrefix j i.val x)
    · by_cases h2 : Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
      · by_cases h5 : kmin ≤ ((clusterAt O populations x B).erase 1).card
            ∧ ((clusterAt O populations x B).erase 1).card ≤ kmax
        · by_cases h6 : ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp
          · -- the certification sample is good and the family is clean
            have hcard : ((certOf j B.m x).card : ℝ) = (B.m : ℝ) := by
              have hinjOn : Set.InjOn (fun i => certPrefix j i x) ↑(Finset.range B.m) := by
                intro a ha b hb hab
                have := h1 (show (fun i : Fin B.m => certPrefix j i.val x)
                    ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
                  = (fun i : Fin B.m => certPrefix j i.val x)
                    ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
                simpa using congrArg Fin.val this
              unfold certOf
              rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
            have hCpos : 0 < (certOf j B.m x).card := by
              have : (0 : ℝ) < ((certOf j B.m x).card : ℝ) := by
                rw [hcard]; exact_mod_cast hmpos
              exact_mod_cast this
            have hmass : (D j).real {p | ¬ (flipCount O
                  ((clusterAt O populations x B).erase 1) p
                ≤ ((((clusterAt O populations x B).erase 1).card : ℝ)) * 0)}
                ≤ Δp * (kmax : ℝ) := by
              set F : Finset S := (clusterAt O populations x B).erase 1 with hFdef
              rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
              · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * 0)}
                    = (∅ : Set S) := by
                  ext p
                  simp only [Set.mem_empty_iff_false, iff_false, not_not]
                  have : F = ∅ := Finset.card_eq_zero.1 hF0
                  simp [flipCount, this]
                rw [hz]
                simp only [measureReal_empty]
                positivity
              · have hsetle : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * 0)}
                    ⊆ {p | (1 / (F.card : ℝ)) * (F.card : ℝ) ≤ flipCount O F p} := by
                  intro p hp
                  have hFc : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFpos
                  have : (0 : ℝ) < flipCount O F p := by
                    have := not_le.1 hp
                    simpa using this
                  have hone : (1 / (F.card : ℝ)) * (F.card : ℝ) = 1 := by
                    field_simp
                  rw [Set.mem_setOf_eq, hone]
                  unfold flipCount at this ⊢
                  exact_mod_cast this
                have hFc : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFpos
                refine le_trans (measureReal_mono hsetle (measure_ne_top _ _)) ?_
                refine le_trans (flipCount_mass_le O (D j) F Δp (1 / (F.card : ℝ))
                  (by positivity) hFpos (fun v hv => h6 v (Finset.mem_erase.1 hv).2)) ?_
                have hdiv : Δp / (1 / (F.card : ℝ)) = Δp * (F.card : ℝ) := by
                  field_simp
                rw [hdiv]
                have hkm : (F.card : ℝ) ≤ (kmax : ℝ) := by exact_mod_cast h5.2
                nlinarith
            by_cases h7 : (B.m : ℝ) * (Δp * (kmax : ℝ) + th)
                ≤ (((certOf j B.m x).filter (fun p =>
                  ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
                    ≤ ((((clusterAt O populations x B).erase 1).card : ℝ)) * 0))).card : ℝ)
            · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩)))
            · refine Or.inr (Or.inr (Or.inr (Or.inr ⟨h2, ?_, hCpos, ?_, ?_⟩)))
              · exact_mod_cast hcard
              · have hlt := not_le.1 h7
                have hle : (((certOf j B.m x).filter (fun p =>
                    ¬ famGood O kmin kmax ((clusterAt O populations x B).erase 1) p)).card : ℝ)
                    = (((certOf j B.m x).filter (fun p =>
                      ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
                        ≤ ((((clusterAt O populations x B).erase 1).card : ℝ))
                          * 0))).card : ℝ) := by
                  refine congrArg (fun t : Finset S => (t.card : ℝ))
                    (Finset.filter_congr (fun p _ => ?_))
                  unfold famGood
                  simp only [h5.1, h5.2, and_true, true_and]
                rw [hle, hcard]
                have hm0 : (0 : ℝ) ≤ (B.m : ℝ) := Nat.cast_nonneg _
                nlinarith
              · intro hgood
                exact hx hgood
          · exact Or.inr (Or.inr (Or.inl h6))
        · exact Or.inr (Or.inl h5)
      · exact Or.inl (Or.inr h2)
    · exact Or.inl (Or.inl h1)
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
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
    _ ≤ ((B.m : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.m : ℝ) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
          + (E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.m ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.m ρ hρ (hρ j hj) hρ0
        · exact measureReal_heavyHits_le D Dsf O populations j B 0 (Δp * (kmax : ℝ)) th hth
        · exact measureReal_retMiss_le hflat O populations D Dsf hsupp j hj B εcov α τ l lcut E
            kmin kmax nlo hE hl hlcut hlcl hτ hε0 hε1 hl1 hnloB hsig hdec hcut hga hα
    _ = ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
          + (E / l + (E / lcut + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by ring

open scoped Classical in
/-- The sample's wrong prefixes are charged to the gate's own two sides.
`wrong_mem_gate_side` absorbs the off-by-one between the cut the guarantee speaks about —
the full cluster — and the one the gate scores, which drops the seed. -/
lemma certWrong_le_miscutAt (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (hlohi : B.lo < B.hi) (x : Run Ω S J) :
    ((certOf j B.m x).filter
        (fun p => ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x))).card
      ≤ miscutOf O (cutSidesAt O populations j B x).1
          (cutSidesAt O populations j B x).2 := by
  classical
  have hhi : 1 ≤ B.hi := by omega
  have hA : (cutSidesAt O populations j B x).1 = sideAcc O populations j B x := rfl
  have hR : (cutSidesAt O populations j B x).2 \ (cutSidesAt O populations j B x).1
      = sideRej O populations j B x := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqD, hqA⟩ := Finset.mem_sdiff.1 hq
      obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqD
      have hnot : ¬ (B.hi - 1
          < voteCount O ((clusterAt O populations x B).erase 1) q (oracleNoise x)) :=
        fun h => hqA (Finset.mem_filter.2 ⟨hqC, h⟩)
      exact Finset.mem_filter.2 ⟨hqC, hv.resolve_left hnot⟩
    · intro hq
      obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hq
      refine Finset.mem_sdiff.2 ⟨Finset.mem_filter.2 ⟨hqC, Or.inr hv⟩, ?_⟩
      intro hc
      have hgt := (Finset.mem_filter.1 hc).2
      omega
  have hlab : (sideAcc O populations j B x).filter (fun p => ¬ (O.label p = 1))
      = (sideAcc O populations j B x).filter (fun p => O.label p = 0) := by
    refine Finset.filter_congr (fun p _ => ?_)
    rcases O.label_bit p with h | h <;> simp [h]
  rw [miscutOf, hR, hA, hlab]
  exact card_cert_wrong_le O populations j B hhi x

open scoped Classical in
/-- Part 1 at one state and one population.  A family the gate admits is right on all
but `εcov` of the population, except on four events: the certification draws repeat, they
meet the table, the sample misses the wrong set, or the gate admits a cut the sample shows
is badly wrong.

One gate event, not one per side.  A mis-cut prefix is decided — `cutCorrect` holds
vacuously on the indecisive band — so the `3εcov/4` of the wrong mass the sample delivers
all sits inside the decided set, and the decided set is what the agreement statistic is
measured against.  There is no side too small to charge a tail to, so no `β` to spend and
no case split. -/
theorem measureReal_admitFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (hlohi : B.lo < B.hi) (εcov α ρ : ℝ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hα : α < 1 / 2)
    (hsig : O.η ≤ 1 / 2) (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hαgate : α + Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
      * ((1 - 2 * O.η) * εcov / 32) ^ 2) < 1)
    (hgmin : (B.gmin : ℝ) ≤ εcov / 32 * (B.m : ℝ)) :
    (runMeasure μ D Dsf).real ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
          ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
        ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
            {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)})})
      ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
          + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
              * ((1 - 2 * O.η) * εcov / 32) ^ 2)) := by
  classical
  have hη := eta_nonneg O
  have hmR : (0 : ℝ) ≤ (B.m : ℝ) := Nat.cast_nonneg _
  set q : ℝ := (1 - 2 * O.η) * εcov / 32 with hqdef
  have hq0 : (0 : ℝ) ≤ q := by rw [hqdef]; nlinarith
  have hq2 : q ^ 2 ≤ 1 / 1024 := by
    have h1 : (0 : ℝ) ≤ (1 - 2 * O.η) * εcov := by nlinarith
    have h2 : (1 - 2 * O.η) * εcov ≤ 1 := by nlinarith
    rw [hqdef]
    nlinarith [h1, h2]
  set τg : ℝ := 2 * q with hτgdef
  have hτg0 : (0 : ℝ) ≤ τg := by rw [hτgdef]; linarith
  set Wr : ℝ := 3 * εcov / 4 * (B.m : ℝ) with hWrdef
  have hWr0 : (0 : ℝ) ≤ Wr := by rw [hWrdef]; positivity
  have hbase : (0 : ℝ) ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ := by positivity
  have hcov : (0 : ℝ) ≤ Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2) := Real.exp_nonneg _
  by_cases hdeg : εcov * (B.m : ℝ) ≤ 64
  · refine le_trans (measureReal_le_one_of_prob (runMeasure μ D Dsf) _) ?_
    have hxle : 2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2 ≤ 1 / 2 := by
      have hprod : (εcov * (B.m : ℝ)) * q ^ 2 ≤ 64 * (1 / 1024) :=
        mul_le_mul hdeg hq2 (sq_nonneg q) (by norm_num)
      nlinarith [hprod]
    have hexp : (1 : ℝ) / 2 ≤ Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2) := by
      have h := Real.add_one_le_exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2)
      linarith
    linarith
  · push_neg at hdeg
    set n₀ : ℕ := ⌊Wr⌋₊ with hn₀def
    have hn₀le : (n₀ : ℝ) ≤ Wr := Nat.floor_le hWr0
    have hn₀ge : Wr - 1 ≤ (n₀ : ℝ) := by
      have := Nat.lt_floor_add_one Wr
      linarith
    set E1 : Set (Run Ω S J) :=
      {x | ¬ Function.Injective (fun i : Fin B.m => certPrefix j i.val x)} with hE1
    set E2 : Set (Run Ω S J) :=
      {x | ¬ Disjoint (prefixesAt populations B.m x) (certOf j B.m x)} with hE2
    set E3 : Set (Run Ω S J) := hitShort O populations (D j) j B εcov (εcov / 4) with hE3
    set E4 : Set (Run Ω S J) :=
      gateBadAgree O populations j B (gateAcc O εcov - τg) Wr n₀ with hE4
    have hsub : ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
          ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
        ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
            {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)})})
        ⊆ (E1 ∪ E2) ∪ (E3 ∪ E4) := by
      rintro x ⟨hadm, hfail⟩
      by_cases hinj : Function.Injective (fun i : Fin B.m => certPrefix j i.val x)
      · by_cases hdisj : Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
        · have hpop : εcov ≤ (D j).real
              {p | ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)} := by
            rw [show {p : S | ¬ cutCorrect O B.lo B.hi
                  (clusterAt O populations x B) p (oracleNoise x)}
                = {p : S | cutCorrect O B.lo B.hi
                  (clusterAt O populations x B) p (oracleNoise x)}ᶜ from rfl,
              measureReal_compl (measurableSet_of_countable _), measureReal_def, measure_univ,
              ENNReal.toReal_one]
            push_neg at hfail
            simp only [Set.mem_setOf_eq] at hfail
            linarith
          by_cases hshort : (((certOf j B.m x).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - εcov / 4)
          · exact Or.inr (Or.inl ⟨hinj, hpop, hshort⟩)
          · push_neg at hshort
            have hWle : Wr ≤ (((certOf j B.m x).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x))).card : ℝ) := by
              refine le_trans (le_of_eq ?_) (le_of_lt hshort)
              rw [hWrdef]; ring
            have hmis : Wr ≤ ((miscutOf O (cutSidesAt O populations j B x).1
                (cutSidesAt O populations j B x).2 : ℕ) : ℝ) := by
              refine le_trans hWle ?_
              exact_mod_cast certWrong_le_miscutAt O populations j B hlohi x
            have hAD : (cutSidesAt O populations j B x).1
                ⊆ (cutSidesAt O populations j B x).2 := by
              intro r hr
              exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hr).1,
                Or.inl (Finset.mem_filter.1 hr).2⟩
            have hdec : Wr ≤ ((cutSidesAt O populations j B x).2.card : ℝ) := by
              refine le_trans hmis ?_
              exact_mod_cast miscutOf_le_card O _ _ hAD
            have hgminle : B.gmin ≤ (cutSidesAt O populations j B x).2.card := by
              have : (B.gmin : ℝ) ≤ ((cutSidesAt O populations j B x).2.card : ℝ) := by
                refine le_trans hgmin (le_trans ?_ hdec)
                rw [hWrdef]; nlinarith
              exact_mod_cast this
            have hαside : α + Real.exp (-2
                * ((cutSidesAt O populations j B x).2.card : ℝ) * τg ^ 2) < 1 := by
              have hmono : Real.exp (-2
                    * ((cutSidesAt O populations j B x).2.card : ℝ) * τg ^ 2)
                  ≤ Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2) := by
                refine Real.exp_le_exp.2 ?_
                rw [hτgdef]
                nlinarith [hdec, hq0, sq_nonneg q]
              linarith [hαgate]
            refine Or.inr (Or.inr ⟨hdisj, ?_, hmis, ?_⟩)
            · have : (n₀ : ℝ) ≤ ((cutSidesAt O populations j B x).2.card : ℝ) :=
                le_trans hn₀le hdec
              exact_mod_cast this
            · refine le_of_lt (lt_of_binomSfGe_le _ _ (gateAcc O εcov) τg α
                (gateAcc_mem O hε0 hε1 hsig).1 (gateAcc_mem O hε0 hε1 hsig).2 hτg0
                hαside (hadm ?_))
              exact hgminle
        · exact Or.inl (Or.inr hdisj)
      · exact Or.inl (Or.inl hinj)
    calc (runMeasure μ D Dsf).real ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
          ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
        ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
            {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)})})
        ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ (E3 ∪ E4)) :=
          measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
          + ((runMeasure μ D Dsf).real E3 + (runMeasure μ D Dsf).real E4) := by
          have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
          have h34 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 E4
          have hall := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) (E3 ∪ E4)
          linarith
      _ ≤ ((B.m : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.m : ℝ) ^ 2 * ρ)
          + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
            + Real.exp (-2 * (n₀ : ℝ) * τg ^ 2)) := by
          gcongr
          · exact cert_not_injective_le D Dsf j B.m ρ (hρ j hj) hρ0
          · exact prefix_cert_disjoint_le D Dsf populations j B.m ρ hρ (hρ j hj) hρ0
          · exact measureReal_hitShort_le D Dsf O populations j B εcov (εcov / 4) hε0
              (by positivity)
          · exact measureReal_gateBadAgree_le hflat O populations D Dsf hsupp j hj B
              (gateAcc O εcov - τg) τg Wr n₀ hτg0 hsig
              (by
                intro n hn hnm
                have hnR : (n₀ : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
                have hWn : Wr - 1 ≤ (n : ℝ) := le_trans hn₀ge hnR
                have hnm' : (n : ℝ) ≤ (B.m : ℝ) := by exact_mod_cast hnm
                rw [hτgdef, gateAcc, hqdef]
                nlinarith [hWn, hnm', hmR])
      _ ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
          + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
            + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2)) := by
          have hmono : Real.exp (-2 * (n₀ : ℝ) * τg ^ 2)
              ≤ Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2) := by
            refine Real.exp_le_exp.2 ?_
            rw [hτgdef]
            nlinarith [hn₀ge, hq0, sq_nonneg q, hmR]
          have hpos : (0 : ℝ) ≤ Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * q ^ 2) :=
            Real.exp_nonneg _
          nlinarith [hmono, hpos]

/-- Part 1, reduced to one state.  The ladder is finite, so the union over the states
the loop may stop at is a finite sum: at most `L` rungs, each carrying `δ/(2·L)`.  No
summable weight over all budgets is needed, and so no encoding of a budget as a number.

What remains of Part 1 is `hper`: at one rung, a family that passes both gates is valid on
every population except with probability `δ/(2·L)`. -/
theorem validity_of_ladder (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ : ℝ) (hδ : 0 ≤ δ) (s : Finset Budget) (L : ℕ) (hL : 0 < L)
    (hcard : s.card ≤ L)
    (hper : ∀ B ∈ s, (runMeasure μ D Dsf).real
      (ret O populations indecisionLimit εcov α B ∩ FailAt O populations D εcov B)
        ≤ δ / (2 * L)) :
    (runMeasure μ D Dsf).real (⋃ B : {B : Budget // B ∈ s},
        ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  classical
  have hLR : (0 : ℝ) < (L : ℕ) := by exact_mod_cast hL
  have hcardR : ((s.card : ℕ) : ℝ) ≤ (L : ℕ) := by exact_mod_cast hcard
  rw [Set.iUnion_subtype]
  calc (runMeasure μ D Dsf).real (⋃ B, ⋃ (_ : B ∈ s),
        ret O populations indecisionLimit εcov α B ∩ FailAt O populations D εcov B)
      ≤ ∑ B ∈ s, (runMeasure μ D Dsf).real
          (ret O populations indecisionLimit εcov α B ∩ FailAt O populations D εcov B) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _B ∈ s, δ / (2 * (L : ℕ)) := Finset.sum_le_sum hper
    _ = (s.card : ℝ) * (δ / (2 * (L : ℕ))) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (L : ℝ) * (δ / (2 * (L : ℕ))) := by
        refine mul_le_mul_of_nonneg_right hcardR (by positivity)
    _ = δ / 2 := by field_simp

/-- Part 1 — whatever is returned is valid, whenever it is returned.

Except with probability `δ/2`, no reachable state is both returned and invalid.  So the loop
may grow and stop however it likes: neither its schedule nor its stopping rule has to be
modelled or itself proved correct.

The intersection with `ret` is load-bearing.  Validity at every state, returned or not, is a
strictly stronger claim and a false one: at a state whose candidate pool has outgrown the
prefixes, the clustering really can produce a drifted family.  The algorithm does not return
it, which is what the gate is for.

The gate certifies exactly what `cutCorrect` asks for.  A returned family has passed
`admitted`, which tests on the seed's column — where membership of `p · ε` is membership of
`p` — that the cut reads as its own class; a mis-cut prefix reads at `η` rather than `1 − η`,
so `W` of them drag the agreement count down by `W(1 − 2η)`, off the tail `admitted` needs.
Reads at distinct prefixes are independent and `certOf` is a `Finset`, so the null is
honest. -/
theorem per_state_le (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (Pre : Set S) (hflat : Flat Pre) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : α < 1 / 2)
    (L : ℕ) (hL : 0 < L) (B : Budget) (hB : Capped O populations εcov δ ρ L B) :
    (runMeasure μ D Dsf).real (ret O populations indecisionLimit εcov α B
      ∩ FailAt O populations D εcov B) ≤ δ / (2 * L) := by
  classical
  have hρ0 : 0 ≤ ρ := by
    obtain ⟨j₀, hj₀⟩ := hpop
    exact le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ j₀ hj₀)
  have hLR : (1 : ℝ) ≤ (L : ℕ) := by exact_mod_cast hL
  have hαgate : α + Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
        * ((1 - 2 * O.η) * εcov / 32) ^ 2) < 1 := by
      have hsh := hB.share
      have hpopone : (1 : ℝ) ≤ (populations.card : ℝ) := by
        exact_mod_cast Finset.card_pos.2 hpop
      set E : ℝ := Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
        * ((1 - 2 * O.η) * εcov / 32) ^ 2) with hEdef
      have hE0 : 0 < E := Real.exp_pos _
      have hT1 : (0 : ℝ) ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ := by
        positivity
      have hT2 : (0 : ℝ) ≤ Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2) :=
        (Real.exp_nonneg _)
      have hstate : 2 * E ≤ stateFail O populations εcov ρ B := by
        rw [stateFail]
        nlinarith [hE0.le]
      have hdiv : δ / (2 * (L : ℕ)) ≤ 1 / 2 := by
        rw [div_le_iff₀ (by positivity : (0 : ℝ) < 2 * ((L : ℕ) : ℝ))]
        nlinarith
      linarith
  by_cases hε1 : εcov ≤ 1
  · have hsub : ret O populations indecisionLimit εcov α B
          ∩ FailAt O populations D εcov B
          ⊆ ⋃ j ∈ populations, ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
              ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
            ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (oracleNoise x)})})
          := by
        rintro x ⟨⟨-, hadm⟩, hfail⟩
        simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
        obtain ⟨j, hj, hfj⟩ := hfail
        exact Set.mem_biUnion hj ⟨hadm j hj, hfj⟩
    calc (runMeasure μ D Dsf).real (ret O populations indecisionLimit εcov α B
            ∩ FailAt O populations D εcov B)
          ≤ (runMeasure μ D Dsf).real (⋃ j ∈ populations,
              ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
                ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
              ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                  {p | cutCorrect O B.lo B.hi
                    (clusterAt O populations x B) p (oracleNoise x)})})) :=
            measureReal_mono hsub (measure_ne_top _ _)
        _ ≤ ∑ j ∈ populations, (runMeasure μ D Dsf).real
              ({x : Run Ω S J | admitted O B.lo B.hi B.gmin εcov α
                ((clusterAt O populations x B).erase 1) (certOf j B.m x) (oracleNoise x)}
              ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                  {p | cutCorrect O B.lo B.hi
                    (clusterAt O populations x B) p (oracleNoise x)})}) :=
            measureReal_biUnion_finset_le _ _
        _ ≤ ∑ _j ∈ populations, (((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
              + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
                + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
                    * ((1 - 2 * O.η) * εcov / 32) ^ 2))) :=
            Finset.sum_le_sum (fun j hj => measureReal_admitFail_le hflat O populations D Dsf
              hsupp j hj B hB.lohi εcov α ρ hεcov.le hε1 hα hsig.le hρ hρ0
              hαgate hB.gfloor)
        _ = stateFail O populations εcov ρ B := by
            rw [Finset.sum_const, nsmul_eq_mul]; rfl
        _ ≤ δ / (2 * (L : ℝ)) := hB.share
  · have hempty : FailAt O populations D εcov B = (∅ : Set (Run Ω S J)) := by
        ext x
        simp only [FailAt, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_not]
        intro j _
        exact le_trans (by linarith [not_le.1 hε1]) measureReal_nonneg
    rw [hempty, Set.inter_empty]
    simpa using (by positivity : (0 : ℝ) ≤ δ / (2 * (L : ℝ)))


/-! ### How `per_state_le` gets its bound

At a state `B`, suppose the family `F = clusterAt O populations x B` passes the gates and
yet some population `j` has `(D j) {p | ¬ cutCorrect …} > εcov`.  Write `W` for that
wrong-set.  `W` depends on `ω` and on the pool and table draws, but not on the
certification draws, which are fresh.  `measureReal_admitFail_le` is the three steps:

1. *The sample sees the wrongness* (`measureReal_hitShort_le`).  Conditionally on `ω` and
   the table draws, `W` is a fixed set and `certPrefix j 0 … certPrefix j (m-1)` are i.i.d. from `D j`,
   so at least `εcov·m/32` of them land in `W` except with probability
   `exp(−2m(εcov/4)²)`.  Plain Hoeffding over the draws; no noise enters, which is why the
   fresh stream matters twice over.  `certWrong_le_miscutAt` is what turns "landed in `W`"
   into a lower bound on the *miscut* count the gate's statistic is dragged by — the gate
   scores `F.erase 1`, so the wrong prefixes have to be routed onto that cut's sides.

2. *The draws are distinct* (`pi_not_injective_le`).  Reads at a repeated string are the
   same bit, so step 3 needs the certification prefixes distinct from each other and from
   the table prefixes.  That costs `(#populations + 1)·m²ρ`, which is what capping `ρ` at
   `collisionCap` pays for.

3. *The gate cannot pass on a wrong cut* (`measureReal_gateBadAgree_le`).  A miscut prefix
   reads as its cut's class with probability `η` rather than `1 − η`, so by
   `agree_mean_eq` the agreement count's mean is `n(1 − η) − W(1 − 2η)`; `agree_sound_of_wrong`
   is the lower tail that says it does not climb back to the null `gateAcc`.

   The cut is `ω`-dependent, which is what `selection_read_bound` resolves: the pair
   (accept side, decided set) is determined by the reads at `p · v` for `v ∈ F.erase 1`,
   all distinct from `p` by flatness, so conditioning on the votes fixes the cut while
   leaving the seed column's law alone. -/

theorem validity_of_returned (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (Pre : Set S) (hflat : Flat Pre) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : α < 1 / 2)
    (pAP : ℝ) :
    (runMeasure μ D Dsf).real (⋃ B : {B : Budget // B ∈ stoppable O populations εcov δ α pAP ρ},
        ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  classical
  refine validity_of_ladder O populations D Dsf indecisionLimit εcov α δ hδ.le _
    (ladderLen O populations εcov δ α pAP) (ladderLen_pos _ _ _ _ _ _)
    (stoppable_card_le O populations εcov δ α pAP ρ) (fun B hB => ?_)
  exact per_state_le O populations D Dsf indecisionLimit εcov α hsig hpop Pre hflat hsupp
    ρ hρ hεcov δ hδ hδ1 hα _ (ladderLen_pos _ _ _ _ _ _) B (capped_of_mem_stoppable hB)

/-- What one round at one population can cost: the draws, the family's size and cleanliness,
the sample's two class counts and its heavy fraction, and the round's own two tests. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut τ th E γscr γdirty gdirty tap ρ ρsf : ℝ) (n₀ : ℕ) (B : Budget) : ℝ :=
  ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
    + ((((B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * tap ^ 2)
          + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γscr ^ 2))))
        + (((B.m : ℝ) ^ 2 * ρ + (((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γdirty ^ 2)
            + (B.M : ℝ) * Real.exp (-2 * (B.m : ℝ) * gdirty ^ 2)))
          + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
            + (E / l + (E / lcut + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)))))))

/-- A state whose round can pass.  Every clause is an inequality among the state's
budgets, the oracle's rates, the suffix distribution's findability and the error budget —
no probability enters, and nothing here is a free parameter of the algorithm.  Reaching
such a state is what the computed ladder is for.

A population that never rejects is fine: the gate skips a sample below `gmin`, and
`admitted` is an implication, so a skipped test is passed over rather than failed.  Nothing
here asks a population to carry both labels. -/
def PassableAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (B : Budget) : Prop :=
  ∃ (τ th tap γdec γscr γdirty gdirty Δ lcut : ℝ),
    0 < B.m ∧ 0 < B.k ∧ B.cn < B.cd ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ εcov ≤ 1
    ∧ 0 ≤ τ ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γdec ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 ≤ pAP ∧ 0 < lcut ∧ lcut ≤ indecisionLimit / 2
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification sample holds few heavy prefixes
    ∧ (((populations.card : ℝ) * Δ + gdirty) * ((B.k - 1 : ℕ) : ℝ) + th ≤ lcut)
    -- the screen's rate sits above the clean one and below the dirty one
    ∧ 0 < B.scd
    ∧ ((B.scd : ℝ) * (2 * O.η * (1 - O.η) + γscr) ≤ (B.sc : ℝ))
    ∧ ((B.sc : ℝ)
        ≤ (B.scd : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γdirty))
    -- the pool holds a family
    ∧ ((B.k : ℝ) ≤ (B.M : ℝ) * (pAP - tap))
    -- the thresholds decide, and decide right, on a clean family of the round's size
    ∧ (((B.hi - 1 : ℕ) : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η) - γdec))
    ∧ (((B.k - 1 : ℕ) : ℝ) * (O.η + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * (O.η + γdec) ≤ ((B.hi - 1 : ℕ) : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η) - γdec))
    -- the gate's margin clears its null on every decided count the FNR test allows
    ∧ (∀ n c : ℕ, ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ ≤ n → n ≤ c → c ≤ B.m →
        (n : ℝ) * (gateAcc O εcov + τ + τ)
          ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * lcut * (c : ℝ)))
    ∧ (Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) lcut τ th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf
            ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ B
      ≤ δ / 2)

lemma mul_self_add_le_cube {x : ℝ} (hx : 0 ≤ x) : x * (x + 3) ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x * (x + 3) = x ^ 3 + 8 * x ^ 2 + 24 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma le_cube_of_nonneg {x : ℝ} (hx : 0 ≤ x) : x ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x = x ^ 3 + 9 * x ^ 2 + 26 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma le_mul_of_one_le_right' {x s c : ℝ} (hxs : x ≤ s) (hs : 0 ≤ s) (hc : 1 ≤ c) :
    x ≤ s * c := by nlinarith

lemma one_le_capScale (O : Oracle μ S) (populations : Finset J) (εcov δ α pAP : ℝ) :
    1 ≤ capScale O populations εcov δ α pAP := by
  rw [capScale]
  exact_mod_cast ladderLen_pos O populations εcov δ α pAP

/-- `√(m+2) ≤ √m + 2`. -/
lemma sqrt_add_two_le {m : ℝ} (hm : 0 ≤ m) : Real.sqrt (m + 2) ≤ Real.sqrt m + 2 := by
  have h : m + 2 ≤ (Real.sqrt m + 2) ^ 2 := by
    have hsq := Real.sq_sqrt hm
    nlinarith [Real.sqrt_nonneg m]
  calc Real.sqrt (m + 2) ≤ Real.sqrt ((Real.sqrt m + 2) ^ 2) := Real.sqrt_le_sqrt h
    _ = Real.sqrt m + 2 := Real.sqrt_sq (by positivity)

/-- The count that clears its own logarithm.  `(a + C)²` meets `a² + C·√m`, which is
what turns a bound mentioning `log` of the count into a closed form. -/
lemma sqrt_step {a C m : ℝ} (ha : 0 ≤ a) (hC : 0 ≤ C) (hm : (a + C) ^ 2 ≤ m) :
    a ^ 2 + C * Real.sqrt m ≤ m := by
  have hm0 : 0 ≤ m := le_trans (sq_nonneg _) hm
  have h1 : a + C ≤ Real.sqrt m := by
    have h := Real.sqrt_le_sqrt hm
    rwa [Real.sqrt_sq (by positivity)] at h
  have h2 : Real.sqrt m ^ 2 = m := Real.sq_sqrt hm0
  nlinarith [Real.sqrt_nonneg m]

/-- To put `exp (-a)` under `ε` it is enough that `a` clears `log (1/ε)`. -/
lemma exp_neg_le_of_log_le {a ε : ℝ} (hε : 0 < ε) (h : Real.log (1 / ε) ≤ a) :
    Real.exp (-a) ≤ ε := by
  have h1 : Real.exp (-a) ≤ Real.exp (-Real.log (1 / ε)) := Real.exp_le_exp.2 (by linarith)
  have h2 : Real.exp (-Real.log (1 / ε)) = ε := by
    rw [← Real.log_inv, one_div, inv_inv, Real.exp_log hε]
  linarith [h1, h2.le, h2.ge]

/-- `log x ≤ 2√x`: sublinear enough that a count required to exceed its own logarithm has a
closed-form solution. -/
lemma log_le_two_sqrt {x : ℝ} (hx : 0 < x) : Real.log x ≤ 2 * Real.sqrt x := by
  have hs : (0 : ℝ) < Real.sqrt x := Real.sqrt_pos.2 hx
  have h := Real.log_le_sub_one_of_pos hs
  have hlog : Real.log x = 2 * Real.log (Real.sqrt x) := by
    rw [Real.log_sqrt hx.le]
    ring
  rw [hlog]
  linarith

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
lemma solved_alpha (O : Oracle μ S) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : O.η < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hα : 0 < α)
    (hα1 : α < 1 / 2)
    (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-2 * ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
      * (sig O * εcov / 4) ^ 2) ≤ α := by
  have hs : 0 < sig O := sig_pos O hsig
  have hτ : (0 : ℝ) < sig O * εcov / 4 := by positivity
  have hτ2 : (0 : ℝ) < (sig O * εcov / 4) ^ 2 := pow_pos hτ 2
  set m : ℕ := prefCount O populations εcov δ α pAP with hmdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the prefix count clears the gate's tail
  have hmlog : 64 * Real.log (1 / α) / (εcov * (sig O * εcov / 4) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * Real.log (1 / α) / (εcov * (sig O * εcov / 4) ^ 2)⌉₊ ≤ m := by
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
      ≤ ((solvedBudget O populations εcov δ α pAP).gmin : ℝ) := by
    show εcov * (m : ℝ) / 64 ≤ (⌊εcov * (m : ℝ) / 32⌋₊ : ℝ)
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    linarith
  rw [show (-2 * ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
        * (sig O * εcov / 4) ^ 2)
      = -(2 * ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
        * (sig O * εcov / 4) ^ 2) from by ring]
  refine exp_neg_le_of_log_le hα ?_
  have hlog0 : 0 ≤ Real.log (1 / α) := Real.log_nonneg (by rw [le_div_iff₀ hα]; linarith)
  rw [div_le_iff₀ (by positivity)] at hmlog
  nlinarith [mul_le_mul_of_nonneg_right hgmin hτ2.le, hmR, hlog0]

set_option maxHeartbeats 1000000 in
/-- The round's failure, over every population, fits the error budget. -/
lemma solved_roundFail (O : Oracle μ S) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : O.η < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hα : 0 < α)
    (hα1 : α < 1 / 2)
    (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hind : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcutlim : cutBudget εcov ≤ indecisionLimit / 2)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap O populations εcov δ α pAP)
    (hρsfsmall : ρsf ≤ collisionCap O populations εcov δ α pAP) :
    (populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) (cutBudget εcov) (sig O * εcov / 4)
            (cutBudget εcov / 4)
            (Real.exp (-2 * (((solvedBudget O populations εcov δ α pAP).k - 1 : ℕ) : ℝ)
              * (sig O / 2) ^ 2))
            (screenMargin O populations εcov δ) (screenMargin O populations εcov δ)
            ((populations.card : ℝ) * flipBudget O populations εcov δ) (pAP / 2) ρ ρsf
            ⌊(1 - indecisionLimit) * ((solvedBudget O populations εcov δ α pAP).m : ℝ)⌋₊
            (solvedBudget O populations εcov δ α pAP)
      ≤ δ / 2 := by
  have hs : 0 < sig O := sig_pos O hsig
  have hcut : 0 < cutBudget εcov := cutBudget_pos hε
  have hΔ : 0 < flipBudget O populations εcov δ := flipBudget_pos O populations hε hcard
  have hγ : 0 < screenMargin O populations εcov δ :=
    screenMargin_pos O populations hsig hε hcard
  have hτ : (0 : ℝ) < sig O * εcov / 4 := by positivity
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  set ε₀ : ℝ := δ / (32 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount O populations εcov δ α pAP with hmdef
  set M : ℕ := poolCount O populations εcov δ pAP with hMdef
  set κ : ℕ := famCount O populations εcov δ with hκdef
  have hBm : (solvedBudget O populations εcov δ α pAP).m = m := rfl
  have hBM : (solvedBudget O populations εcov δ α pAP).M = M := rfl
  have hBκ : ((solvedBudget O populations εcov δ α pAP).k - 1 : ℕ) = κ := by
    show κ + 1 - 1 = κ
    omega
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- each count clears the logarithm its tail asks for
  have ctap : Real.log (1 / ε₀) / (2 * (pAP / 2) ^ 2) ≤ (M : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 32 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (32 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊ ≤ M := by
      rw [hMdef, poolCount]; omega
    exact_mod_cast hle
  have cscr : Real.log (((M : ℝ) + 1) / ε₀)
      / (2 * screenMargin O populations εcov δ ^ 2) ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 32 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * screenMargin O populations εcov δ ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cdirty : Real.log (((M : ℝ) + 1) / ε₀)
      / (2 * ((populations.card : ℝ) * flipBudget O populations εcov δ) ^ 2) ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 32 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * ((populations.card : ℝ) * flipBudget O populations εcov δ) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cth : Real.log (1 / ε₀) / (2 * (cutBudget εcov / 4) ^ 2) ≤ (m : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 32 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (32 * (populations.card : ℝ) / δ)
        / (2 * (cutBudget εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have cfam : Real.log ((1 / cutBudget εcov) / ε₀) / (2 * (sig O / 2) ^ 2) ≤ (κ : ℝ) := by
    have heq : (1 / cutBudget εcov) / ε₀
        = 32 * (populations.card : ℝ) / (cutBudget εcov * δ) := by
      rw [hε₀def]; field_simp; try ring
    have hsq : 2 * (sig O / 2) ^ 2 = sig O ^ 2 / 2 := by ring
    rw [heq, hsq]
    refine le_trans (le_of_eq ?_) (le_trans (Nat.le_ceil
      (2 * Real.log (32 * (populations.card : ℝ) / (cutBudget εcov * δ)) / sig O ^ 2)) ?_)
    · field_simp
    · have hle : ⌈2 * Real.log (32 * (populations.card : ℝ) / (cutBudget εcov * δ))
          / sig O ^ 2⌉₊ ≤ κ := by
        rw [hκdef, famCount]; omega
      exact_mod_cast hle
  -- the tails those counts kill
  have ttap : Real.exp (-2 * (M : ℝ) * (pAP / 2) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := pAP / 2) (c := 1) (by positivity) hε₀ (by norm_num) ctap
    linarith
  have tscr : ((M : ℝ) + 1) * Real.exp (-2 * (m : ℝ) * screenMargin O populations εcov δ ^ 2)
      ≤ ε₀ := tail_le_of_count hγ hε₀ (by linarith) cscr
  have tdirty : ((M : ℝ) + 1) * Real.exp (-2 * (m : ℝ)
      * ((populations.card : ℝ) * flipBudget O populations εcov δ) ^ 2) ≤ ε₀ :=
    tail_le_of_count (by positivity) hε₀ (by linarith) cdirty
  have tth : Real.exp (-2 * (m : ℝ) * (cutBudget εcov / 4) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := cutBudget εcov / 4) (c := 1) (by positivity) hε₀
      (by norm_num) cth
    linarith
  have tfam : Real.exp (-2 * (κ : ℝ) * (sig O / 2) ^ 2) / cutBudget εcov ≤ ε₀ := by
    have h := tail_le_of_count (γ := sig O / 2) (c := 1 / cutBudget εcov) (by positivity) hε₀
      (by positivity) cfam
    rw [div_mul_eq_mul_div, one_mul] at h
    linarith
  -- the gate's floor, and the tail it kills
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have hsize : 2 ≤ εcov * (m : ℝ) / 32 := by
    rw [div_le_iff₀ hε] at hsizeR
    linarith
  have hgmin : εcov * (m : ℝ) / 64
      ≤ ((solvedBudget O populations εcov δ α pAP).gmin : ℝ) := by
    show εcov * (m : ℝ) / 64 ≤ (⌊εcov * (m : ℝ) / 32⌋₊ : ℝ)
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    linarith
  have cgate : Real.log (2 / ε₀) / (2 * (sig O * εcov / 4) ^ 2)
      ≤ ((solvedBudget O populations εcov δ α pAP).gmin : ℝ) := by
    have heq : (2 : ℝ) / ε₀ = 64 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    have hmlog : 64 * Real.log (64 * (populations.card : ℝ) / δ)
        / (εcov * (sig O * εcov / 4) ^ 2) ≤ (m : ℝ) := by
      refine le_trans (Nat.le_ceil _) ?_
      have hle : ⌈64 * Real.log (64 * (populations.card : ℝ) / δ)
          / (εcov * (sig O * εcov / 4) ^ 2)⌉₊ ≤ m := by
        rw [hmdef, prefCount]; omega
      exact_mod_cast hle
    have hlog0 : 0 ≤ Real.log (64 * (populations.card : ℝ) / δ) := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ]
      nlinarith
    rw [div_le_iff₀ (by positivity)] at hmlog ⊢
    linarith [mul_le_mul_of_nonneg_right hgmin
      (by positivity : (0 : ℝ) ≤ 2 * (sig O * εcov / 4) ^ 2)]
  have tgate : 2 * Real.exp (-2
      * ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
      * (sig O * εcov / 4) ^ 2) ≤ ε₀ :=
    tail_le_of_count hτ hε₀ (by norm_num) cgate
  -- the gate's tail, now floored at the decided count rather than at `gmin`
  have hfloorge : ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
      ≤ ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ) := by
    have hg : (solvedBudget O populations εcov δ α pAP).gmin = ⌊εcov * (m : ℝ) / 32⌋₊ := rfl
    rw [hg]
    have hmono : ⌊εcov * (m : ℝ) / 32⌋₊ ≤ ⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ :=
      Nat.floor_mono (by nlinarith [hmR])
    exact_mod_cast hmono
  have tgate' : 2 * Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
      * (sig O * εcov / 4) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tgate
    have hexp : Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
          * (sig O * εcov / 4) ^ 2)
        ≤ Real.exp (-2 * ((solvedBudget O populations εcov δ α pAP).gmin : ℝ)
          * (sig O * εcov / 4) ^ 2) := by
      refine Real.exp_le_exp.2 ?_
      nlinarith [hfloorge, sq_nonneg (sig O * εcov / 4)]
    linarith
  -- the collision terms, with the state's code kept opaque
  set c2 : ℝ := capScale O populations εcov δ α pAP with hc2def
  have hc21 : (1 : ℝ) ≤ c2 := by
    rw [hc2def]
    exact one_le_capScale O populations εcov δ α pAP
  have hden : (0 : ℝ) < 64 * ((populations.card : ℝ) + 3) ^ 3
      * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 := by
    have h1 : (0 : ℝ) < ((populations.card : ℝ) + 3) ^ 3 := by positivity
    have h2 : (0 : ℝ) < (m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1 := by positivity
    exact mul_pos (mul_pos (mul_pos (by norm_num) h1) h2) (by linarith)
  have hcapval : collisionCap O populations εcov δ α pAP
      = δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) := by
    rw [collisionCap, ← hmdef, ← hMdef, ← hc2def]
  have hcoll1 : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ)
      ≤ δ / 64 := by
    have hstep : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2)
        ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 := by
      have h1 : (populations.card : ℝ) * ((populations.card : ℝ) + 3)
          ≤ ((populations.card : ℝ) + 3) ^ 3 := mul_self_add_le_cube hcard.le
      have h2 : (m : ℝ) ^ 2 ≤ ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 :=
        le_mul_of_one_le_right' (by nlinarith [sq_nonneg (M : ℝ)]) (by positivity) hc21
      calc (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2)
          = ((populations.card : ℝ) * ((populations.card : ℝ) + 3)) * (m : ℝ) ^ 2 := by ring
        _ ≤ (((populations.card : ℝ) + 3) ^ 3)
              * (((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) :=
            mul_le_mul h1 h2 (by positivity) (by positivity)
        _ = ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 := by ring
    have hρ' : ρ ≤ δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) := by rwa [hcapval] at hρsmall
    rw [le_div_iff₀ hden] at hρ'
    have h3 : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2) * ρ
        ≤ (((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) * ρ :=
      mul_le_mul_of_nonneg_right hstep hρ0
    linarith
  have hcoll2 : (populations.card : ℝ) * ((M : ℝ) ^ 2 * ρsf) ≤ δ / 64 := by
    have hstep : (populations.card : ℝ) * (M : ℝ) ^ 2
        ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 := by
      have h1 : (populations.card : ℝ) ≤ ((populations.card : ℝ) + 3) ^ 3 :=
        le_cube_of_nonneg hcard.le
      have h2 : (M : ℝ) ^ 2 ≤ ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 :=
        le_mul_of_one_le_right' (by nlinarith [sq_nonneg (m : ℝ)]) (by positivity) hc21
      calc (populations.card : ℝ) * (M : ℝ) ^ 2
          ≤ (((populations.card : ℝ) + 3) ^ 3)
              * (((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) :=
            mul_le_mul h1 h2 (by positivity) (by positivity)
        _ = ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2 := by ring
    have hρ' : ρsf ≤ δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) := by rwa [hcapval] at hρsfsmall
    rw [le_div_iff₀ hden] at hρ'
    have h3 : (populations.card : ℝ) * (M : ℝ) ^ 2 * ρsf
        ≤ (((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * c2) * ρsf :=
      mul_le_mul_of_nonneg_right hstep hρsf0
    linarith
  -- the indecision budget is at least the cut budget, so its tail is no worse
  have hEl : Real.exp (-2 * (κ : ℝ) * (sig O / 2) ^ 2) / (indecisionLimit / 2)
      ≤ Real.exp (-2 * (κ : ℝ) * (sig O / 2) ^ 2) / cutBudget εcov :=
    div_le_div_of_nonneg_left (Real.exp_nonneg _) hcut hcutlim
  have tdirty' : (M : ℝ) * Real.exp (-2 * (m : ℝ)
      * ((populations.card : ℝ) * flipBudget O populations εcov δ) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tdirty
    exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)
  have hcε : (populations.card : ℝ) * ε₀ = δ / 32 := by
    rw [hε₀def]
    field_simp
  rw [roundFail, hBm, hBM, hBκ]
  refine le_trans (mul_le_mul_of_nonneg_left
    (c := 9 * ε₀ + (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ + (M : ℝ) ^ 2 * ρsf))
    ?_ hcard.le) ?_
  · linarith [ttap, tscr, tdirty', tth, tfam, tgate', hEl]
  · have hexp : (populations.card : ℝ)
        * (9 * ε₀ + (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ + (M : ℝ) ^ 2 * ρsf))
        = 9 * ((populations.card : ℝ) * ε₀)
          + ((populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ)
            + (populations.card : ℝ) * ((M : ℝ) ^ 2 * ρsf)) := by ring
    rw [hexp, hcε]
    linarith

/-- A count in closed form that clears `A + log (m + 2)` at rate `r`.  The logarithm is
under the square root, so one step of `log x ≤ 2√x` closes the loop. -/
lemma share_count_spec {r A m : ℝ} (hr : 0 < r) (hA : 0 ≤ A)
    (hm : (Real.sqrt ((A + 4) / r) + 2 / r) ^ 2 ≤ m) :
    A + Real.log (m + 2) ≤ r * m := by
  have hr0 : r ≠ 0 := ne_of_gt hr
  have h0 : 0 ≤ m := le_trans (sq_nonneg _) hm
  have hlog : Real.log (m + 2) ≤ 2 * Real.sqrt m + 4 := by
    have h1 := log_le_two_sqrt (by linarith : (0 : ℝ) < m + 2)
    have h2 := sqrt_add_two_le h0
    linarith
  have hstep := sqrt_step (Real.sqrt_nonneg ((A + 4) / r))
    (by positivity : (0 : ℝ) ≤ 2 / r) hm
  rw [Real.sq_sqrt (by positivity : (0 : ℝ) ≤ (A + 4) / r)] at hstep
  have hLHS : (A + 4) / r + 2 / r * Real.sqrt m = (A + 4 + 2 * Real.sqrt m) / r := by
    field_simp
  rw [hLHS, div_le_iff₀ hr] at hstep
  have hcomm : m * r = r * m := mul_comm m r
  linarith

set_option maxHeartbeats 1000000 in
/-- The state carries its own share of the error budget. -/
lemma solved_share (O : Oracle μ S) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hsig : O.η < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hα : 0 < α) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρsmall : ρ ≤ collisionCap O populations εcov δ α pAP) :
    stateFail O populations εcov ρ (solvedBudget O populations εcov δ α pAP)
      ≤ δ / (2 * (ladderLen O populations εcov δ α pAP : ℕ)) := by
  classical
  have hη0 : 0 ≤ O.η := eta_nonneg O
  have hs : 0 < sig O := sig_pos O hsig
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h : 1 ≤ populations.card := by
      by_contra hc
      have : populations.card = 0 := by omega
      rw [this] at hcard; norm_num at hcard
    exact_mod_cast h
  set m : ℕ := prefCount O populations εcov δ α pAP with hmdef
  set M : ℕ := poolCount O populations εcov δ pAP with hMdef
  set L : ℕ := ladderLen O populations εcov δ α pAP with hLdef
  have hLpos : 0 < L := by rw [hLdef]; exact ladderLen_pos _ _ _ _ _ _
  have hL1 : (1 : ℝ) ≤ (L : ℝ) := by exact_mod_cast hLpos
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hr : 0 < shareRate O εcov := by rw [shareRate]; positivity
  -- the ladder is shorter than the prefix count it is built from
  have hLm : (L : ℝ) ≤ (m : ℝ) + 2 := by
    have h : L ≤ m + 2 := by
      rw [hLdef, ladderLen, ← hmdef]
      have := Nat.log_le_self 2 m
      omega
    exact_mod_cast h
  -- the prefix count clears the share's condition
  have hshare : Real.log (16 * (populations.card : ℝ) / δ) + Real.log ((m : ℝ) + 2)
      ≤ shareRate O εcov * (m : ℝ) := by
    refine share_count_spec hr (Real.log_nonneg ?_) ?_
    · rw [le_div_iff₀ hδ]; nlinarith
    · have hle : shareCount O populations εcov δ ≤ m := by
        rw [hmdef, prefCount]; omega
      have h1 : ((shareCount O populations εcov δ : ℕ) : ℝ) ≤ (m : ℝ) := by exact_mod_cast hle
      refine le_trans ?_ h1
      rw [shareCount]
      exact Nat.le_ceil _
  set q : ℝ := δ / ((populations.card : ℝ) * (L : ℝ)) with hqdef
  have hq0 : 0 < q := by rw [hqdef]; positivity
  -- the share's own tail
  have hexp : Real.exp (-(shareRate O εcov * (m : ℝ))) ≤ q / 16 := by
    refine exp_neg_le_of_log_le (by positivity) ?_
    have hrw : (1 : ℝ) / (q / 16) = (16 * (populations.card : ℝ) / δ) * (L : ℝ) := by
      rw [hqdef]; field_simp
    rw [hrw, Real.log_mul (by positivity) (by positivity)]
    have hlogL : Real.log (L : ℝ) ≤ Real.log ((m : ℝ) + 2) :=
      Real.log_le_log (by linarith) hLm
    linarith
  -- the gate's tail, at the slower of the two rates
  have hT3 : 2 * Real.exp (-2 * (εcov / 32 * (m : ℝ)) * ((1 - 2 * O.η) * εcov / 32) ^ 2)
      ≤ q / 8 := by
    have heq : -2 * (εcov / 32 * (m : ℝ)) * ((1 - 2 * O.η) * εcov / 32) ^ 2
        = -(shareRate O εcov * (m : ℝ)) := by
      simp only [shareRate, sig]; ring
    rw [heq]
    linarith
  -- the coverage tail, at the faster one
  have hT2 : Real.exp (-2 * (m : ℝ) * (εcov / 4) ^ 2) ≤ q / 16 := by
    refine le_trans (Real.exp_le_exp.2 ?_) hexp
    have hle : shareRate O εcov ≤ 2 * (εcov / 4) ^ 2 := by
      simp only [shareRate, sig]
      have hu2 : (1 / 2 - O.η) ^ 2 ≤ 1 / 4 := by nlinarith
      have key : εcov * (1 / 2 - O.η) ^ 2 ≤ 512 := by nlinarith
      have key2 := mul_le_mul_of_nonneg_left key (sq_nonneg εcov)
      nlinarith [key2]
    nlinarith
  -- the collisions, against the allowance
  have hT1 : ((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ ≤ q / 64 := by
    have hden : (0 : ℝ) < 64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * (L : ℝ) := by positivity
    have hcap : collisionCap O populations εcov δ α pAP
        = δ / (64 * ((populations.card : ℝ) + 3) ^ 3
          * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * (L : ℝ)) := by
      rw [collisionCap, capScale, ← hmdef, ← hMdef, ← hLdef]
    rw [hcap] at hρsmall
    rw [hqdef, div_div,
      le_div_iff₀ (by positivity : (0 : ℝ) < (populations.card : ℝ) * (L : ℝ) * 64)]
    have hρden : ρ * (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) * (L : ℝ)) ≤ δ := by
      rw [← le_div_iff₀ hden]
      exact hρsmall
    have hc0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
    have h1 : (populations.card : ℝ) * ((populations.card : ℝ) + 1)
        ≤ ((populations.card : ℝ) + 3) ^ 3 := by
      have e : ((populations.card : ℝ) + 3) ^ 3
          = (populations.card : ℝ) ^ 3 + 9 * (populations.card : ℝ) ^ 2
            + 27 * (populations.card : ℝ) + 27 := by ring
      rw [e]
      nlinarith [hc0, sq_nonneg ((populations.card : ℝ)), pow_nonneg hc0 3]
    have h2 : (m : ℝ) ^ 2 ≤ (m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1 := by
      nlinarith [sq_nonneg ((M : ℕ) : ℝ)]
    have h3 : (populations.card : ℝ) * ((populations.card : ℝ) + 1) * (m : ℝ) ^ 2
        ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) :=
      mul_le_mul h1 h2 (by positivity) (by positivity)
    have h4 := mul_le_mul_of_nonneg_right h3 (by positivity : (0 : ℝ) ≤ 64 * (L : ℝ))
    have h5 := mul_le_mul_of_nonneg_left h4 hρ0
    linarith
  -- assemble
  have hBm : (solvedBudget O populations εcov δ α pAP).m = m := rfl
  rw [stateFail, hBm]
  have hcardne : (populations.card : ℝ) ≠ 0 := by linarith
  have hLne : (L : ℝ) ≠ 0 := by linarith
  have hgoal : δ / (2 * (L : ℕ)) = (populations.card : ℝ) * (q / 2) := by
    rw [hqdef]; field_simp
  rw [hgoal]
  refine mul_le_mul_of_nonneg_left ?_ (by linarith)
  linarith

set_option maxHeartbeats 1000000 in
/-- The computed ladder holds a state whose round can pass.

`PassableAt` is arithmetic: every clause is an inequality among the state's budgets, the
oracle's rates, the suffix distribution's findability and the error budget.  The witness is
`solvedBudget`, the ladder's top rung, and every clause is discharged from the closed forms
that define it — no reachability is assumed.

The order the constants come out in: the miscut budget `lcut = cutBudget εcov` under the
indecision limit (`hcutlim`), the flip budget
`Δ = flipBudget` under `lcut` over the family size, the screen's two margins at
`screenMargin` (so the rate window is non-empty), then the prefix count large enough for
every exponential — including the share's own condition, which mentions the ladder's length
and hence `log` of the prefix count — then `α` at the gate's own tail `exp(−2·gmin·τ²)`,
then the pool at `k/(pAP − t)`.  The collision masses enter as `m²ρ`, which is what
`collisionCap` bounds. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcutlim : cutBudget εcov ≤ indecisionLimit / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap O populations εcov δ α pAP)
    (hρsfcap : ρsf ≤ collisionCap O populations εcov δ α pAP) :
    ∃ B : Budget, B ∈ stoppable O populations εcov δ α pAP ρ
      ∧ PassableAt O populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP B := by
  classical
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hη0 : 0 ≤ O.η := eta_nonneg O
  have hs : 0 < sig O := sig_pos O hsig
  have hsval : sig O = 1 / 2 - O.η := rfl
  have hcut : 0 < cutBudget εcov := cutBudget_pos hεcov
  have hΔ : 0 < flipBudget O populations εcov δ := flipBudget_pos O populations hεcov hcard
  have hγ : 0 < screenMargin O populations εcov δ :=
    screenMargin_pos O populations hsig hεcov hcard
  set κ : ℕ := famCount O populations εcov δ with hκdef
  have hκpos : 0 < κ := famCount_pos O populations εcov δ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  set m : ℕ := prefCount O populations εcov δ α pAP with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : Budget := solvedBudget O populations εcov δ α pAP with hBdef
  -- the fields, as the definitions give them
  have hBm : B.m = m := rfl
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
  have hκs : 1 ≤ (κ : ℝ) * sig O := by
    have h1 : (1 : ℝ) / sig O ≤ (κ : ℝ) := by
      refine le_trans (Nat.le_ceil _) ?_
      have hle : ⌈1 / sig O⌉₊ ≤ κ := by rw [hκdef, famCount]; omega
      exact_mod_cast hle
    rw [div_le_iff₀ hs] at h1
    linarith
  set x : ℝ := (κ : ℝ) * (1 / 2 - sig O / 2) with hxdef
  have hxpos : 0 < x := by
    rw [hxdef, hsval]
    nlinarith
  have hceil1 : 1 ≤ ⌈x⌉₊ := Nat.one_le_ceil_iff.2 hxpos
  have hceilx : (⌈x⌉₊ : ℝ) ≤ x + 1 := le_of_lt (Nat.ceil_lt_add_one hxpos.le)
  have hgap : (κ : ℝ) * ((1 - O.η) - sig O / 2) - x = (κ : ℝ) * sig O := by
    rw [hxdef, hsval]; ring
  have hmid : (κ : ℝ) * (O.η + sig O / 2) = x := by
    rw [hxdef, hsval]; ring
  -- the fields, named before the definitions are made opaque
  have hBM : B.M = poolCount O populations εcov δ pAP := rfl
  have hBcn : B.cn = 1 := rfl
  have hBcd : B.cd = 2 := rfl
  have hBlo : B.lo = ⌈x⌉₊ - 1 := rfl
  have hBhi : B.hi = ⌈x⌉₊ + 1 := rfl
  have hBscd : B.scd = ⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 : ℕ) : ℝ)
      * (2 * O.η * (1 - O.η) + screenMargin O populations εcov δ)⌉₊ := rfl
  have hMceil : 2 * ((κ : ℝ) + 1) / pAP ≤ ((B.M : ℕ) : ℝ) := by
    rw [hBM, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (16 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊]
  clear_value B κ m x
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_⟩⟩,
    sig O * εcov / 4, cutBudget εcov / 4, pAP / 2,
    sig O / 2, screenMargin O populations εcov δ, screenMargin O populations εcov δ,
    (populations.card : ℝ) * flipBudget O populations εcov δ,
    flipBudget O populations εcov δ, cutBudget εcov,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  -- in the schedule
  · rw [hBdef]
    exact solvedBudget_mem_schedule O populations εcov δ α pAP
  -- Capped
  · rw [hBlo, hBhi]
    omega
  · rw [hBm]
    refine le_trans hgminLe (le_of_eq ?_)
    ring
  · rw [hBdef]
    exact solved_share O populations hsig hεcov hε1 hδ hδ1 hαpos hpAPPositive hcard hρ0
      hρcap
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
  · linarith
  · exact hγ.le
  · exact hγ.le
  · exact mul_nonneg hcard.le hΔ.le
  · exact hΔ
  · linarith
  · exact hcut
  · exact hcutlim
  · exact hpAPBound
  · exact hρsf
  -- the family's flips fit the cut budget
  · rw [hBkappa]
    clear hBsc hBscd hBlo hBhi hBM hMceil hBcn hBcd hBdef hBk hBgmin hBm
      hρcap hρsfcap hρ hρsf hpAPBound
    have hid : ((populations.card : ℝ) * flipBudget O populations εcov δ
        + (populations.card : ℝ) * flipBudget O populations εcov δ) * (κ : ℝ)
        = cutBudget εcov / 4 := by
      rw [flipBudget, ← hκdef]
      field_simp
      ring
    rw [hid]
    linarith
  -- the screen's rate
  · rw [hBscd]
    omega
  · rw [hBsc, hBscd]
    exact Nat.le_ceil _
  · rw [hBsc, hBscd]
    clear hBsc hBscd hBlo hBhi hBM hMceil hBcn hBcd hBdef hBk hBkappa hBgmin hBm
      hceilx hmid hgap hκs hgminLe hgminGe hsize hsizeR
      hρcap hρsfcap hρ hρsf hpAPBound
    have hsq : flipBudget O populations εcov δ * (1 - 2 * O.η) ^ 2
        = 4 * screenMargin O populations εcov δ := by
      rw [screenMargin, hsval]
      ring
    rw [hsq]
    have hone : (1 : ℝ) ≤ 2 * ((⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 : ℕ) : ℝ)
        * screenMargin O populations εcov δ := by
      have hcl : (1 : ℝ) / (2 * screenMargin O populations εcov δ)
          ≤ ((⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 : ℕ) : ℝ) := by
        refine le_trans (Nat.le_ceil _) ?_
        push_cast
        linarith
      rw [div_le_iff₀ (by positivity)] at hcl
      linarith
    have hnn : (0 : ℝ) ≤ ((⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 : ℕ) : ℝ)
        * (2 * O.η * (1 - O.η) + screenMargin O populations εcov δ) := by
      have h2η : (0 : ℝ) ≤ 2 * O.η * (1 - O.η) := by nlinarith
      exact mul_nonneg (Nat.cast_nonneg _) (by linarith [hγ.le])
    have hsc := le_of_lt (Nat.ceil_lt_add_one hnn)
    linarith
  -- the pool holds a family
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    push_cast
    linarith
  -- the thresholds decide, and decide right
  · rw [hBhi, hBk, show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega,
      show (κ + 1 - 1 : ℕ) = κ from by omega]
    linarith [hceilx, hgap, hκs]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.cast_sub hceil1, hmid]
    push_cast
    linarith [Nat.le_ceil x]
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega,
      show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega, hmid]
    exact Nat.le_ceil x
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.cast_sub hceil1]
    push_cast
    linarith [hceilx, hgap, hκs]
  -- the gate's two sides clear their thresholds
  · intro n c hn₀ hnc hcm
    have hcR : (c : ℝ) ≤ (m : ℝ) := by rw [← hBm]; exact_mod_cast hcm
    have hnR : ((⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast hn₀
    have hfloor : (1 - indecisionLimit) * (m : ℝ) - 1
        ≤ ((⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ : ℕ) : ℝ) := by
      rw [hBm]
      linarith [Nat.lt_floor_add_one ((1 - indecisionLimit) * (m : ℝ))]
    have hlow : (1 - indecisionLimit) * (m : ℝ) - 1 ≤ (n : ℝ) := le_trans hfloor hnR
    have hm64 : (64 : ℝ) ≤ (m : ℝ) := by
      have h1 : (64 : ℝ) / εcov ≤ (m : ℝ) := hsizeR
      rw [div_le_iff₀ hεcov] at h1
      nlinarith
    clear hBsc hBscd hBlo hBhi hBM hMceil hBcn hBcd hBdef hBk hBkappa hBgmin hBm
      hceilx hmid hgap hκs hgminLe hgminGe hnR hfloor hsize hsizeR
      hρcap hρsfcap hρ hρsf hpAPBound
    have key : 4 * cutBudget εcov * (c : ℝ) ≤ εcov * (n : ℝ) / 2 := by
      rw [cutBudget]
      have hcn : (c : ℝ) ≤ 8 * (n : ℝ) := by nlinarith [hcR, hlow, hm64, hind1]
      nlinarith [hcn, hεcov.le]
    have hs' : (0 : ℝ) ≤ 1 / 2 - O.η := by linarith
    rw [gateAcc]
    simp only [sig]
    linarith [mul_le_mul_of_nonneg_left key hs']
  · rw [hBdef]
    exact solved_alpha O populations hsig hεcov hε1 hδ hαpos hα hpAPPositive hcard
  · rw [hBdef]
    exact solved_roundFail O populations hsig hεcov hε1 hδ hαpos hα hδ1 hpAPPositive
      hindLim hind1 hcutlim hcard hρ0 hρsf0 hρcap hρsfcap

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

`hcutlim : cutBudget εcov ≤ indecisionLimit/2` is the one relation left between the
algorithm's tolerances: the miscut budget is the binding one of the round's two Markov
thresholds, so the family size is solved against a single rate rather than their max. -/
theorem loop_terminates {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (indecisionLimit εcov α : ℝ) (ρ pAP δ : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcutlim : cutBudget εcov ≤ indecisionLimit / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap O populations εcov δ α pAP)
    (hρsf : collisionMass Dsf ≤ collisionCap O populations εcov δ α pAP) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : Budget // B ∈ stoppable O populations εcov δ α pAP ρ},
      x ∉ ret O populations indecisionLimit εcov α B.val} ≤ δ / 2 := by
  classical
  obtain ⟨B, hB, hpass⟩ := exists_passable O populations D Dsf indecisionLimit εcov α δ ρ
    (collisionMass Dsf) pAP hsig hpop hεcov hε1 hδ hδ1 hαpos hα hindLim hind1 hcutlim
    hpAPPositive
    hpAPBound hρ hρ0 le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hρcap hρsf
  obtain ⟨τ, th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, hmpos, hkpos, hcd, hindLim,
    hind1', hε1, hτ, hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hpAP0, hlcut, hlcl,
    hpAPBound, hρsf, hheavy, hscd, hscLow, hscHigh, hcount,
    hhiUp, hloUp, hhiLo, hloLo, hga, hα, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hlpos : 0 < l := by rw [hl]; linarith
  have hl2 : 2 * l = indecisionLimit := by rw [hl]; ring
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  -- the whole failure at the one state, population by population
  have hsub : {x : Run Ω S J | ∀ B' : {B : Budget // B ∈ stoppable O populations εcov δ α pAP ρ},
        x ∉ ret O populations indecisionLimit εcov α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_setOf_eq, not_and] at hc
    refine hx ⟨B, hB⟩ ((mem_ret_iff O populations indecisionLimit εcov α B x).2 (fun j hj => ?_))
    by_contra hcj
    exact hc j hj hcj
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ j ∈ populations,
      (runMeasure μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j}
        ≤ roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ B := by
    intro j hj
    have hstall := measureReal_stalled_le hflat O populations D Dsf hsupp B hcd hkpos j₀ hj₀
      γscr pAP tap (collisionMass Dsf) ρ hγscr hpAP0 htap hpAPBound hscd hscLow hcount hρsf hρsf0
      (hρ j₀ hj₀) hρ0
    have hdirty := measureReal_dirtyMember_le hflat O populations D Dsf hsupp j hj B hcd hsig.le
      hmpos Δ γdirty gdirty ρ hΔ hγdirty hgdirty hρ0 (hρ j hj) hscd hscHigh
    have hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hf hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      have hclean : flipCount O F p = 0 := by
        have h0 : (0 : ℝ) ≤ flipCount O F p := Nat.cast_nonneg _
        have := hf
        rw [mul_zero] at this
        linarith
      refine le_trans (decided_whp O F p B.lo (B.hi - 1) γdec hγdec hclean ?_ ?_) ?_
      · rw [hcardF]; exact hhiUp
      · rw [hcardF]; exact hloUp
      · rw [hE, hcardF]
    have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hf hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      refine le_trans (cutCorrect_whp O F p B.lo (B.hi - 1) 0 γdec hf hγdec ?_ ?_) ?_
      · rw [hcardF]
        have : (κ : ℝ) * ((O.η + (1 - 2 * O.η) * 0) + γdec) = (κ : ℝ) * (O.η + γdec) := by ring
        rw [this]
        exact hhiLo
      · rw [hcardF]
        have : (κ : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - 0)) - γdec)
            = (κ : ℝ) * ((1 - O.η) - γdec) := by ring
        rw [this]
        exact hloLo
      · rw [hE, hcardF]
    have hl1' : 2 * l ≤ 1 := by rw [hl]; linarith [hind1']
    have hnloB' : ((⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ : ℕ) : ℝ)
        ≤ (1 - 2 * l) * (B.m : ℝ) := by
      have hmnn : (0 : ℝ) ≤ (B.m : ℝ) := Nat.cast_nonneg _
      have hnn : (0 : ℝ) ≤ (1 - indecisionLimit) * (B.m : ℝ) := by nlinarith [hind1']
      have hfl := Nat.floor_le hnn
      rw [hl]
      linarith [hfl]
    have hmain := measureReal_notRetAt_le hflat O populations D Dsf hsupp j hj B hmpos hsig.le
      εcov α τ l lcut E ((populations.card : ℝ) * Δ + gdirty) ρ th κ κ
      ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊
      (Real.exp_nonneg _) hlpos hlcut (by rw [hl]; exact hlcl) hτ hεcov.le hε1
      hl1' hnloB'
      hρ hρ0
      (by positivity) hth
      hheavy _ hstall _ hdirty hdec hcut hga hα
    rw [hl2] at hmain
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  calc ∑ j ∈ populations, (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j}
      ≤ ∑ _j ∈ populations,
          roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ B :=
        Finset.sum_le_sum hper
    _ = (populations.card : ℝ)
          * roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) ⌊(1 - indecisionLimit) * (B.m : ℝ)⌋₊ B := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ δ / 2 := hbudget

#print axioms validity_of_returned
#print axioms loop_terminates

end Loop

end OrthoDFA
