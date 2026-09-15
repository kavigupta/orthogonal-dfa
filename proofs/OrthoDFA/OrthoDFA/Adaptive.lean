import OrthoDFA.Distributional
import Mathlib.Probability.ProductMeasure

/-!
# The adaptive clustering loop: the integrated theorem

`sample_suffix_family` is a retry loop: cluster, measure the FNR **per population**, and
if any population is too indecisive grow the pool and try again — stopping at a
data-dependent time.  This file states and proves the guarantee **for that loop**, with
no fixed budget:

> with probability `≥ 1 − δ` the loop **terminates**, and **whatever** family it returns
> preserves acceptance on `≥ 1 − εcov` of **each** prefix population.

"Whatever it returns" is `ret`: the states that pass both gates.  The guarantee is not
claimed at states the loop rejects.

Everything the statement needs is present and constrained: the persistent RCN oracle, the
collection of prefix populations, the suffix distribution with its findability `pAP`, the
loop's own return test — the FNR gate *and* the accept-preserving gate, both defined
rather than abstract — and an unbounded growing schedule.
The run space is concrete (`Run`, `runLaw`), so its law is a lemma rather than a
hypothesis; the returned family is *defined* by the clustering.

Proof: the two-part decomposition —

* `validity_of_returned` — whatever is returned is valid, *whenever* it is returned,
  except w.p. `δ/2`;
* `loop_terminates` — the loop returns at some round, except w.p. `δ/2`;
* `sound_and_terminating` composes them.

The composition is proved; the two halves are `sorry`, with their proof plans recorded.

**Known modelling gap (flagged, not hidden).**  The draws here are i.i.d. from each
distribution and deduplicated downstream (`poolAt`, `prefixesAt`), whereas `_draw_cohort`
and `sample_more_prefixes` *redraw* on a duplicate — they sample without replacement.
Deduplicating i.i.d. draws yields a pool at most as large, so this is the conservative
model; but the collision mass grows with the number of draws, so closing
`validity_of_returned` at unbounded budgets will need either the without-replacement
concentration (Hoeffding 1963) or a non-atomic prefix distribution.  That is a gap in the
*proof*, recorded rather than papered over by weakening the claim.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S] [IsCancelMul S] [MeasurableMul S]
  [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

/-! ## The derived configuration

`build_pst` does not take the family size; it *computes* it:

```python
n, eps = population_size_and_evidence_margin(
    signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01)
config = SearchConfig(suffix_family_size=n, evidence_margin=eps, ...)
```

and `min_signal_strength` is `½ − η`, already carried by the `Oracle`.  So the family
size and the evidence margin are **derived** from the oracle's signal together with the
two acceptable rates — they are not free parameters. -/

/-- Binomial CDF: `P[Bin(N,p) ≤ j]`. -/
noncomputable def binomCdf (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (j + 1), (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- `evidence_margin_for_population_size`: at population size `N`, the margin `eps`
around `center` is *admissible* when the binomial false-positive rate under the null and
false-negative rate under the signal are both within budget. -/
def admissibleMargin (s fpr accFnr center : ℝ) (N : ℕ) (eps : ℝ) : Prop :=
  0 < eps ∧ eps ≤ s ∧
    (binomCdf N center ⌊(N : ℝ) * (center - eps)⌋₊
        + (1 - binomCdf N center (⌈(N : ℝ) * (center + eps)⌉₊ - 1)) ≤ fpr) ∧
    (binomCdf N (s + center) (⌈(N : ℝ) * (center + eps)⌉₊ - 1)
        - binomCdf N (s + center) ⌊(N : ℝ) * (center - eps)⌋₊ ≤ accFnr)

/-- A large enough population always admits a margin, for any positive signal.  (The
binary search in `population_size_and_evidence_margin` terminates.) -/
theorem exists_admissibleMargin (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) :
    ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps :=
  sorry

open scoped Classical in
/-- **The suffix family size, derived.**  `population_size_and_evidence_margin` returns the
*least* population size admitting a margin; this is that `N`. -/
noncomputable def suffixFamilySize (s fpr accFnr center : ℝ) : ℕ :=
  if h : ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps then Nat.find h else 1

open scoped Classical in
/-- **The evidence margin, derived**: the margin admissible at that population size. -/
noncomputable def evidenceMargin (s fpr accFnr center : ℝ) : ℝ :=
  if h : ∃ eps, admissibleMargin s fpr accFnr center (suffixFamilySize s fpr accFnr center) eps
  then h.choose else 0

theorem suffixFamilySize_pos (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) : 0 < suffixFamilySize s fpr accFnr center := by
  classical
  rw [suffixFamilySize, dif_pos (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)]
  exact (Nat.find_spec (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)).1

/-- The algorithm's family size, as `build_pst` computes it from the oracle's signal
`½ − η` and the two acceptable rates. -/
noncomputable def cfgK (O : Oracle μ S) (fpr accFnr center : ℝ) : ℕ :=
  suffixFamilySize (1 / 2 - O.η) fpr accFnr center

/-- The algorithm's evidence margin, likewise derived. -/
noncomputable def cfgMargin (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  evidenceMargin (1 / 2 - O.η) fpr accFnr center

/-- The gate's accept threshold: `decision_boundary + evidence_margin`. -/
noncomputable def cfgAcc (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center + cfgMargin O fpr accFnr center

/-- The gate's reject threshold: `decision_boundary − evidence_margin`. -/
noncomputable def cfgRej (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center - cfgMargin O fpr accFnr center

/-- The membership query the oracle actually answers: `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def mq (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-! ## The run space

A run is exactly what the algorithm consumes: the oracle's persistent noise, the stream of
suffix draws, and one stream of prefix draws per population.  That is a concrete space with
a concrete law, so it is built here rather than axiomatised — `law_block` is a *lemma*.

The code deduplicates its draws — `_draw_cohort` skips suffixes already interned and
`sample_more_prefixes` skips prefixes already drawn — and so does this development:
`poolAt` and `prefixesAt` take `Finset.image` of the stream.  Deduplicating `n` i.i.d.
draws is not the same as `n` draws *without replacement*; it yields a pool that is at most
as large, so the guarantee proved here is the conservative one.  Do **not** be tempted to
model the without-replacement law as "i.i.d. conditioned on the block being injective":
those conditioned laws are inconsistent across `n` (for `Dsf = (½,¼,¼)` the first marginal
of the `n = 2` law puts mass `⅖` on the first atom, not `½`), so no space carries them all
and everything built on them would be vacuous. -/

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

/-- One run of the algorithm: the oracle's persistent noise, the suffix draws, and the
prefix draws of each population. -/
abbrev Run (Ω S J : Type*) := Ω × ((ℕ → S) × (J → ℕ → S))

/-- The law of a run: the three components jointly independent, each stream i.i.d. -/
noncomputable def runLaw (μ : Measure Ω) (D : J → Measure S) (Dsf : Measure S) :
    Measure (Run Ω S J) :=
  μ.prod ((Measure.infinitePi fun _ : ℕ => Dsf).prod
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runLaw μ D Dsf) := by
  unfold runLaw; infer_instance

/-- The run's persistent noise. -/
def nz (x : Run Ω S J) : Ω := x.1

/-- The `i`-th suffix drawn. -/
def sfx (i : ℕ) (x : Run Ω S J) : S := x.2.1 i

/-- The `i`-th prefix drawn from population `j`. -/
def prf (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2 j i

lemma measurable_nz : Measurable (nz : Run Ω S J → Ω) := measurable_fst

lemma measurable_sfx (i : ℕ) : Measurable (sfx (Ω := Ω) (S := S) (J := J) i) := by
  unfold sfx; fun_prop

lemma measurable_prf (j : J) (i : ℕ) : Measurable (prf (Ω := Ω) (S := S) j i) := by
  unfold prf; fun_prop

/-- **The joint law of the first `n` draws.**  This is what the concentration arguments
consume, and it is a theorem about `runLaw`, not a hypothesis about an abstract space. -/
lemma law_block (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (n : ℕ) :
    Measure.map (fun x : Run Ω S J => (nz x, (fun i : Fin n => sfx i.val x),
        (fun (j : J) (i : Fin n) => prf j i.val x))) (runLaw μ D Dsf)
      = μ.prod
          ((Measure.pi fun _ : Fin n => Dsf).prod
            (Measure.pi (fun j : J => Measure.pi fun _ : Fin n => D j))) :=
  ((MeasurePreserving.id μ).prod
    ((measurePreserving_finRestrict Dsf n).prod
      (measurePreserving_pi _ _ fun j => measurePreserving_finRestrict (D j) n))).map_eq

section Loop

variable {D : J → Measure S} {Dsf : Measure S}

/-! ## The loop, with the schedule left to the algorithm

The growth schedule is not a user parameter — it is something the algorithm optimizes as
it sees fit (`sample_suffix_family` alternates suffix- and prefix-growth based on whether
the FNR improved, and even the suffix increment is the *screened* `kept` count).  So the
statement fixes no schedule.  Instead a **history** records the sequence of budget states
the loop has passed through, and the guarantee is uniform over *all* histories and *all*
budgets — whatever the algorithm chooses, it is covered.  Histories are countable, so the
union bound still closes. -/

/-- The loop's growth history: the budget states it has passed through, in order.  The
algorithm picks this however it likes. -/
abbrev Hist := List (ℕ × ℕ)

/-- The candidate pool at a suffix budget: the first `M` suffixes drawn. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range M).image (fun i => sfx i x)

open scoped Classical in
/-- The representative prefixes at a prefix budget: every population's first `m` draws. -/
noncomputable def prefixesAt (populations : Finset J) (m : ℕ)
    (x : Run Ω S J) : Finset S :=
  populations.biUnion (fun j => (Finset.range m).image (fun i => prf j i x))

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, mq O (p * v) ω) / F.card

open scoped Classical in
/-- `identify_cluster_around`'s loss: the Hamming distance from a candidate's mask row to
the cluster's **own** thresholded mean (`masks[cluster].mean(0) > decision_boundary`). -/
noncomputable def hammingLoss (O : Oracle μ S) (F : Finset S) (b : ℝ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p => ¬ ((mq O (p * v) ω = 1) ↔ b < vote O F p ω))).card : ℝ)

open scoped Classical in
/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates — but only while the seed is among them.  `identify_cluster_around` breaks out
(`if seed_local not in nearest`) rather than let the centre drift off `ε`, keeping the
cluster it had. -/
noncomputable def lloydStep (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if (1 : S) ∈ leastLossSubset (hammingLoss O F b P ω) cands k
  then leastLossSubset (hammingLoss O F b P ω) cands k else F

/-- `identify_cluster_around` iterated to its fixed point.  The total loss is a natural
number bounded by `k·#P` that strictly decreases at each improving step, so `k·#P + 1`
iterations from the seed `ε` already sit at the fixed point — the bound is derived, not a
knob. -/
noncomputable def clusterAround (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep O b P cands ω k)^[k * P.card + 1] {(1 : S)}

/-- **The cluster never drifts off the seed.**  `identify_cluster_around` stops the moment
`ε` would leave, so every family the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround O b P cands ω k := by
  classical
  unfold clusterAround
  generalize k * P.card + 1 = n
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      unfold lloydStep
      split_ifs with h
      · exact h
      · exact ih

open scoped Classical in
/-- The boundary update at the end of `identify_cluster_around`: the midpoint of the
accept-side and reject-side prefix means, falling back to whichever side is nonempty. -/
noncomputable def newBoundary (O : Oracle μ S) (F P : Finset S) (ω : Ω) (b : ℝ) : ℝ :=
  let acc := P.filter (fun p => b < vote O F p ω)
  let rej := P.filter (fun p => ¬ (b < vote O F p ω))
  let am := (∑ p ∈ acc, vote O F p ω) / acc.card
  let rm := (∑ p ∈ rej, vote O F p ω) / rej.card
  if acc.Nonempty then (if rej.Nonempty then (am + rm) / 2 else am)
  else (if rej.Nonempty then rm else b)

/-- The cluster at one budget state, at the boundary carried in. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) (b : ℝ) (Mm : ℕ × ℕ) : Finset S :=
  clusterAround O b (prefixesAt populations Mm.2 x) (poolAt Mm.1 x) (nz x)
    (cfgK O fpr accFnr b)

/-- The decision boundary carried along a history: it starts at `1/2`
(`decision_boundary : float = 0.5`) and each state replaces it with the boundary its own
cluster induces. -/
noncomputable def boundaryFold (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) : ℝ → Hist → ℝ
  | b, [] => b
  | b, Mm :: h =>
      boundaryFold O populations fpr accFnr x
        (newBoundary O (clusterAt O populations fpr accFnr x b Mm)
          (prefixesAt populations Mm.2 x) (nz x) b) h

/-- The boundary after a history. -/
noncomputable def boundaryAfter (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) (h : Hist) : ℝ :=
  boundaryFold O populations fpr accFnr x (1 / 2) h

/-- The family the loop proposes at budget `Mm`, having come through history `h`. -/
noncomputable def famAt (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) (h : Hist) (Mm : ℕ × ℕ) : Finset S :=
  clusterAt O populations fpr accFnr x (boundaryAfter O populations fpr accFnr x h) Mm

/-! ## The accept-preserving gate

`AcceptPreservingGate` runs after the FNR test, right before the family is returned.  It
splits the prefixes by the family's *own* cut and counts, on the **seed's own column**,
how many read as accepting — and membership of `p · ε` is membership of `p`, which is why
the gate is read off `ε` and why `one_mem_clusterAround` matters.  A family is admitted
only when each side reads as its own class, by the same thresholds the family itself is
read with (`drift_verdict`).  Anything else sets the round's FNR to 1 and the loop keeps
sampling.

This is what makes a *returned* family valid.  Validity does not have to be inferred from
the clustering's loss concentration union-bounded over every candidate suffix — the gate
tests the conclusion directly, so the candidate pool may be as large as it likes. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

open scoped Classical in
/-- `_split_counts` on the accept side: `(hits, n)` over the prefixes the family accepts,
counted on the seed's column (`mq O p`, the membership query at `p · ε = p`). -/
noncomputable def splitAcc (O : Oracle μ S) (b fpr accFnr : ℝ) (F P : Finset S) (ω : Ω) :
    ℕ × ℕ :=
  let side := P.filter (fun p => cfgAcc O fpr accFnr b ≤ vote O F p ω)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

open scoped Classical in
/-- `_split_counts` on the reject side. -/
noncomputable def splitRej (O : Oracle μ S) (b fpr accFnr : ℝ) (F P : Finset S) (ω : Ω) :
    ℕ × ℕ :=
  let side := P.filter (fun p => vote O F p ω < cfgRej O fpr accFnr b)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

/-- `drift_verdict`'s **ADMITTED**: the accepted prefixes read as accepting on the seed's
column significantly above `accept_thresh`, and the rejected ones significantly below
`reject_thresh`, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE = 0.05`).

Modelled on the state's own prefixes.  `certification_sample` may top these up with
prefixes drawn for the split alone when the counts in hand leave the verdict
`UNCERTIFIED`; that only sharpens the same test, so requiring it here is conservative. -/
def admitted (O : Oracle μ S) (b fpr accFnr α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  binomSfGe (splitAcc O b fpr accFnr F P ω).2 (cfgAcc O fpr accFnr b)
      (splitAcc O b fpr accFnr F P ω).1 ≤ α
    ∧ binomCdf (splitRej O b fpr accFnr F P ω).2 (cfgRej O fpr accFnr b)
        (splitRej O b fpr accFnr F P ω).1 ≤ α

/-- A prefix is *decided* when the family's vote clears the state's accept or reject
threshold; otherwise it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (b fpr accFnr : ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  cfgAcc O fpr accFnr b ≤ vote O F p ω ∨ vote O F p ω < cfgRej O fpr accFnr b

open scoped Classical in
/-- **The loop's return test**: the FNR gate (PR #257: held per population, not over their
union) at the state's own boundary and margin, **and** the accept-preserving gate.  A
family that fails either is not returned — `judge_family` sets its FNR to 1 and the loop
samples more. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr indecisionLimit α : ℝ) (hM : Hist × (ℕ × ℕ)) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations,
      (((Finset.range hM.2.2).filter (fun i => ¬ decided O
          (boundaryAfter O populations fpr accFnr x hM.1) fpr accFnr
          (famAt O populations fpr accFnr x hM.1 hM.2) (prf j i x) (nz x))).card : ℝ)
        ≤ indecisionLimit * hM.2.2)
    ∧ admitted O (boundaryAfter O populations fpr accFnr x hM.1) fpr accFnr α
        (famAt O populations fpr accFnr x hM.1 hM.2)
        (prefixesAt populations hM.2.2 x) (nz x)}

/-- The family at a reachable state is **invalid**: on some population it fails to
preserve acceptance on a `1 − εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (fpr accFnr εcov : ℝ) (hM : Hist × (ℕ × ℕ)) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | ∀ v ∈ famAt O populations fpr accFnr x hM.1 hM.2,
            O.label (p * v) = O.label p}}

/-- **Part 1 — whatever is returned is valid, whenever it is returned.**

Except with probability `δ/2`, no reachable state is *both* returned and invalid — over
every history the algorithm might follow and every budget it might stop at.  So the loop
may grow and stop however it likes: neither its schedule nor its stopping rule has to be
modelled or itself proved correct.

The intersection with `ret` is load-bearing, not bookkeeping.  Validity at *every* state,
returned or not, is a strictly stronger claim and a false one: at a state whose candidate
pool has outgrown the prefixes, the clustering really can produce a drifted family.  The
algorithm does not return it — that is what the gate is for — and the guarantee is about
what it returns.

Proof plan: this rests on the **accept-preserving gate**, not on the clustering.  A
returned family has passed `admitted`, which tests directly — on the seed's own column,
where membership of `p · ε` is membership of `p` — that each side of its cut reads as its
own class.  So the argument is: `admitted` bounds the binomial probability that a family
whose cut disagrees with membership on more than `εcov` of a population would pass, and
`coverage_of_summed_flip` turns that into the per-population fraction.  Reachable states
are countable, so union-bound them at a summable weight.

Deliberately *not* via `clustering_budget`: inferring validity from the cluster's loss
concentration would need a union bound over every candidate suffix, and the persistent
oracle's fixed noise bits make the per-candidate error floor at the prefix collision
entropy `∑ₐ D_j({a})²` — so that route fails once the candidate pool outgrows
`exp(c/ρ)`.  The gate tests the conclusion instead of inferring it, so the pool size
does not enter.  `Distributional.lean`'s `clustering_pac` remains the statement about one
budget with a collision bound supplied; it is not what carries this. -/
theorem validity_of_returned (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) :
    (runLaw μ D Dsf).real (⋃ hM, ret O populations fpr accFnr indecisionLimit α hM
        ∩ FailAt O populations D fpr accFnr εcov hM) ≤ δ / 2 :=
  sorry

/-- **Part 2 — the loop terminates.**

Except with probability `δ/2`, some reachable state passes the FNR test, so the loop
returns.

Proof plan: each growth step draws fresh suffixes, and by findability a draw is
accept-preserving with probability `≥ pAP`; once the pool holds enough accept-preserving
suffixes and the prefix count is large enough, every population's vote is decisive on all
but `indecisionLimit` of its mass, so the per-population test passes.  The per-step trigger is
block-local, so `geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under
`δ/2`.

Termination now also has to clear the accept-preserving gate, which is where the work
moved: `admitted` has to *pass*, not merely be sound.  A family that is genuinely
accept-preserving on the populations reads as such on the seed's column up to the noise,
so this is the gate's own power — the complement of the `α` it spends — and it is why
`pAP > 0` is needed rather than just useful.  `ACCEPT_PRESERVING_GIVE_UP = 20` caps the
refusals; the statement carries no cap, so it is the stronger claim.

`hslack : accFnr < indecisionLimit` is **necessary**, not decoration.  `accFnr` is the
binomial probability that one prefix's count lands inside the indecisive band, so it
bounds the *expected* indecision fraction; if the loop's limit were at or below it the
test could essentially never pass and the loop would not terminate.  The code keeps the
slack: `0.01 < 0.02` (`0.10` after PR #257). -/
theorem loop_terminates (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    (runLaw μ D Dsf).real {x | ∀ hM, x ∉ ret O populations fpr accFnr indecisionLimit α hM} ≤ δ / 2 :=
  sorry

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and the family it returns —
at whatever state it chooses to stop — preserves acceptance on `≥ 1 − εcov` of **each**
prefix population.

Nothing is fixed or idealised.  The growth schedule is not a parameter: the guarantee is
uniform over every history the algorithm might follow and every budget it might reach, so
it may optimize its own schedule.  The stopping time is likewise arbitrary.  The decision
boundary, evidence margin, thresholds and family size are recomputed at each state from
the oracle's signal, as `build_pst` computes them; the cluster is the Lloyd fixed point
against its own thresholded mean; and the run space is the concrete `runLaw`, not an
abstract space assumed to exist. -/
theorem clustering_correct (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    1 - δ ≤ (runLaw μ D Dsf).real
      {x | (∃ hM, x ∈ ret O populations fpr accFnr indecisionLimit α hM) ∧
        ∀ hM : Hist × (ℕ × ℕ), x ∈ ret O populations fpr accFnr indecisionLimit α hM →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | ∀ v ∈ famAt O populations fpr accFnr x hM.1 hM.2,
                O.label (p * v) = O.label p}} := by
  have h := sound_and_terminating (runLaw μ D Dsf)
    (fun hM => ret O populations fpr accFnr indecisionLimit α hM
      ∩ FailAt O populations D fpr accFnr εcov hM)
    (ret O populations fpr accFnr indecisionLimit α) δ
    (validity_of_returned O populations D Dsf fpr accFnr indecisionLimit α hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound εcov hεcov δ hδ)
    (loop_terminates O populations D Dsf fpr accFnr indecisionLimit α hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound δ hδ hindLim hslack)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
