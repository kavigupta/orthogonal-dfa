import OrthoDFA.Clustering

/-!
# The schedule, and what it is solved from

`Model` says what the algorithm does and what it promises.  This is how a budget is spent to
get there: counts solved backwards from the tails they have to clear, the ladder of states the
loop climbs, and `ClusteringCorrect`, which names them.

`ClusteringGuarantee` names none of it, so auditing the claim never brings you here.  Proving
it does -- `clustering_guarantee` supplies these as the witnesses it hides.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runMeasure μ D Dsf) := by
  unfold runMeasure; infer_instance

deriving instance DecidableEq for State

instance : Countable State :=
  Function.Injective.countable
    (f := fun b => (b.nsuff, b.npref, b.k, b.cn, b.cd, b.lo, b.hi, b.sc, b.scd, b.gmin))
    (by rintro ⟨⟩ ⟨⟩ h; simp_all)

/-- `s = 1/2 − η`. -/
noncomputable def sig (η : ℝ) : ℝ := 1 / 2 - η

/-- The scale the counts are resolved against: the smallest rate the round has to clear.

Stated in the rates themselves rather than in `cutBudget`, which is one particular way of
taking their minimum.  What a reader checks is that the budget is polynomial in the rates
asked for; which divisors the proof chose to hold each of them at is the proof's business. -/
noncomputable def budgetScale (η indecisionLimit εcov : ℝ) : ℝ :=
  min εcov (min (sig η) indecisionLimit)

/-- The log factor the counts share.  Every tail is a deviation bound at one of the failure
probabilities or rates in play, over a number of draws that is itself polynomial in them, so
a log of their reciprocals covers all of it up to a constant. -/
noncomputable def budgetLog (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) : ℝ :=
  Real.log (((populations.card : ℝ) + 2)
    / (δ * α * pAP * budgetScale η indecisionLimit εcov))

/-- What a budget of `k` buys.

`sig` enters at the sixth power because the screen's tail binds: its margin is a rate times
`sig³`, and a deviation bound squares the margin it is given.  The population count enters
squared because that margin is divided by it.

The scale enters at the *third* power, not the second: the coverage tails divide by `εcov`
once before squaring a margin that already carries `εcov`, so they are cubic in it, and a
cap quadratic in the scale is exceeded as `εcov → 0`. -/
noncomputable def budgetCap (populations : Finset J)
    (η indecisionLimit εcov δ α pAP k : ℝ) : ℝ :=
  k * (populations.card : ℝ) ^ 2 * budgetLog populations η indecisionLimit εcov δ α pAP
    / (sig η ^ 6 * budgetScale η indecisionLimit εcov ^ 3)

/-- What fraction of the family the vote absorbs flipping.  A flip moves a read by a whole
bit rather than by `2s`, so it costs the vote `(1 − η)·f`, and what is left is `voteSlack`:

    (1 − η) · flipFrac η + voteSlack η = sig η

exactly, at `η` as well as below it.  Nothing else fixes the split; scaling with `s` is what
keeps both sides positive up to `η = 1/2`. -/
noncomputable def flipFrac (η : ℝ) : ℝ := 7 * sig η / (10 * (1 - η))

/-- How far into the margin left by `flipFrac` the vote's count is read. -/
noncomputable def voteSlack (η : ℝ) : ℝ := 3 * sig η / 10

/-- What the gate's margin can absorb, so what a round charges wrongly-cut prefixes at.  A
right cut earns the gate `s` over a coin flip per prefix, and a wrong one can cost it a whole
read when the rates are lopsided, so the budget is held under `s` as well as `εcov`.

The three are held at their own divisors and not a common one: `εcov/16` is what the round's
own budget leaves once the cut, the coverage tail and the threshold tail have taken their
shares, `s/64` is what the gate needs to clear a coin flip, and `indecisionLimit/2` is what
the FNR gate tolerates.  None answers another's constraint, so the tightest divisor is not
the safer choice for all three. -/
noncomputable def cutBudget (η indecisionLimit εcov : ℝ) : ℝ :=
  min (εcov / 16) (min (sig η / 64) (indecisionLimit / 2))

/-- A family's vote fails at `exp (-κ·voteSlack²/2)`: a clean vote sits `s` from the centre,
a flipping `flipFrac` of the family spends all but `voteSlack` of that, and the count is read
the rest of the way in.  That rate only has to sit under half the cut budget: the misfires on
the certification sample are independent given the family, so their count concentrates below
the budget.

The size is even, and the `+ 1` is inside the doubling, because `flipFrac` spends its side of
the margin exactly: the thresholds sit at `κ/2`, and an odd `κ` would round that up past what
`voteSlack` has left to pay with. -/
noncomputable def famCount (η : ℝ) (_populations : Finset J) (indecisionLimit εcov _δ : ℝ) : ℕ :=
  2 * (⌈Real.log (2 / cutBudget η indecisionLimit εcov) / (4 * voteSlack η ^ 2)⌉₊ + 1)

/-- What one family member may flip.  The vote absorbs a `flipFrac` fraction of the family
flipping, so Markov charges the cut budget at that fraction and not at the family's size. -/
noncomputable def flipBudget (η : ℝ) (populations : Finset J) (indecisionLimit εcov _δ : ℝ) : ℝ :=
  cutBudget η indecisionLimit εcov * flipFrac η / (3 * (populations.card : ℝ))

/-- The screen's margin, at the flip budget.

The separation is `(1 − 2η)² = 4·sig η²`, and the test spends four one-sided deviations on it
— the seed's and the candidate's, each against the measured floor — so the factor here is
capped below `2`.  `sc/scd` then has to resolve finer than `a/(4 − 2a)`, which is the `15/2`
in `solvedStateAt`; the two constants move together. -/
noncomputable def screenMargin (η : ℝ) (populations : Finset J) (indecisionLimit εcov δ : ℝ) : ℝ :=
  flipBudget η populations indecisionLimit εcov δ * sig η ^ 2 * (15 / 8)

/-- Enough suffixes that a family of `k` fits inside the findable fraction. -/
noncomputable def poolCount (η : ℝ) (populations : Finset J) (indecisionLimit εcov δ pAP : ℝ) : ℕ :=
  ⌈2 * ((famCount η populations indecisionLimit εcov δ : ℝ) + 1) / pAP⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊

/-- The counts the round's tails ask for, summed so each is met. -/
noncomputable def prefCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : ℕ :=
  ⌈Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2)⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (cutBudget η indecisionLimit εcov / 4) ^
      2)⌉₊
    + ⌈64 * Real.log (1 / α) / (εcov * (sig η * εcov / 4) ^ 2)⌉₊
    + ⌈64 * Real.log (256 * (populations.card : ℝ) / δ)
        / (εcov * (sig η * εcov / 4) ^ 2)⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ)
        / (2 * ((populations.card : ℝ) * flipBudget η populations indecisionLimit εcov δ) ^ 2)⌉₊
    + ⌈64 / εcov⌉₊
    + 1

/-- The state at a given prefix count: every other field read off the condition it has to
meet. -/
noncomputable def solvedStateAt (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP : ℝ) (mi : ℕ) : State where
  nsuff := poolCount η populations indecisionLimit εcov δ pAP
  npref := mi
  k := famCount η populations indecisionLimit εcov δ + 1
  cn := 1
  cd := 2
  lo := ⌈(famCount η populations indecisionLimit εcov δ : ℝ) / 2⌉₊ - 1
  hi := ⌈(famCount η populations indecisionLimit εcov δ : ℝ) / 2⌉₊ + 1
  sc := ⌈((⌈15 / (2 * screenMargin η populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ)
    * screenMargin η populations indecisionLimit εcov δ⌉₊
  scd := ⌈15 / (2 * screenMargin η populations indecisionLimit εcov δ)⌉₊ + 1
  gmin := ⌊εcov * (mi : ℝ) / 32⌋₊

/-- How many times the loop runs the gate: the prefix count halves down to one.  There is no
other state the loop can return at. -/
noncomputable def ladderLen (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : ℕ :=
  Nat.log 2 (prefCount η populations indecisionLimit εcov δ α pAP) + 1

/-- The states the loop runs the gate at: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : Finset State :=
  (Finset.range (ladderLen η populations indecisionLimit εcov δ α pAP)).image
    (fun i => solvedStateAt η populations indecisionLimit εcov δ pAP
      (prefCount η populations indecisionLimit εcov δ α pAP / 2 ^ i))

/-- What one tested state may cost, summed over the populations: the certification draws
repeating or meeting the table, the sample missing the wrong set, a family member flipping more
than the screen allows, the sample holding too many prefixes the family flips, and the family's
vote misfiring on too many of the rest — and then, once, the pool's draws colliding.

The pool's *findability* is not here: that event does not mention the prefixes, so it is the
same at every rung and is charged once rather than per rung. -/
noncomputable def stateFail (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov δ ρ ρsf : ℝ)
    (B : State) : ℝ :=
  (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
      + (((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2
            * Real.exp (-2 * (B.npref : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ /
              2) ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ)
            * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2))
        + (Real.exp (-2 * (B.npref : ℝ) * (cutBudget η₀ indecisionLimit εcov / 4) ^ 2)
          + Real.exp (-2 * (B.npref : ℝ) * (cutBudget η₀ indecisionLimit εcov / 2) ^ 2)))))
    + (B.nsuff : ℝ) ^ 2 * ρsf

/-- A state can be stopped at when its thresholds are in order and it has drawn enough
prefixes to carry its share of the error budget.

At a handful of prefixes neither the screen nor the certification sample can see a dirty
family, so the guarantee cannot cover a stop there. -/
structure Capped (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov δ ρ ρsf pAP : ℝ)
    (N : ℕ) (B : State) : Prop where
  /-- Reject strictly below accept, so the two gate sides are disjoint. -/
  lohi : B.lo < B.hi
  /-- The centre's boundary is a proper fraction. -/
  bdry : B.cn < B.cd
  /-- There are prefixes to read the rate off. -/
  mpos : 0 < B.npref
  /-- Two accept-preserving draws are expected in the pool, so one of them is not the seed
  and the floor has a clean candidate to sit at. -/
  found : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP / 2)
  /-- The rungs divide `δ/4` between them in proportion to their prefix counts, which halve
  down the ladder and so sum to at most `2N`; the other `δ/4` of the validity half pays for
  the pool's findability, once.

  Proportional and not uniform: a rung's tails are exponential in its own count, so a
  `δ/(4·L)` split would make every count clear `log L`, and `L` is read off the top count. -/
  share : stateFail η₀ populations indecisionLimit εcov δ ρ ρsf B ≤ δ * (B.npref : ℝ) / (8 * N)

open scoped Classical in

/-- The rungs of the ladder that carry their share.  A `Finset`, so the union bound over it is
a finite sum and no summable weight over all budgets is needed. -/
noncomputable def stoppable (η₀ : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP ρ ρsf : ℝ) : Finset State :=
  (schedule η₀ populations indecisionLimit εcov δ α pAP).filter
    (Capped η₀ populations indecisionLimit εcov δ ρ ρsf pAP (prefCount η₀ populations
      indecisionLimit εcov δ α pAP))

/-- How much collision mass the populations may carry: the round pays `m²ρ` for prefix
collisions, so the mass is capped against the prefix count and the state's own share. -/
noncomputable def collisionCap (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : ℝ :=
  δ / (64 * ((populations.card : ℝ) + 3) ^ 3
    * ((prefCount η populations indecisionLimit εcov δ α pAP : ℝ) ^ 2
      + (poolCount η populations indecisionLimit εcov δ pAP : ℝ) ^ 2 + 1))

/-- The E-L\* clustering algorithm is PAC-correct: with probability `≥ 1 − δ` the loop
terminates, and the family it returns — at whatever state it stops — cuts `≥ 1 − εcov` of
each prefix population the way the noiseless oracle does.

The algorithm is not told the noise rate, only an upper bound `η₀` on it — `η₀` is what
`min_signal_strength` gives `build_pst`, and every computed field of `State` is solved from
it, never from `O.η`.  Nothing asks the bound to be tight: the screen reads its cutoff off
`screenBase`, a quantity it measures, and the gate is held to a coin flip, so over-estimating
the noise only costs prefixes.

The hypotheses, in the order they appear: both of the oracle's noise rates are at most `η₀`,
which has signal; there is a population to certify; the populations are supported on a `Flat` prefix
set; their collision mass is at most `ρ` and `pAP` of the suffix measure is
accept-preserving; `indecisionLimit`, `α`, `εcov` and `δ` are in range; and `ρ` and `Dsf`'s
collision mass fit `collisionCap`.

No hypothesis is a parameter of the algorithm: `State` is computed (`solvedStateAt` along
`schedule`), and the guarantee is uniform over the rungs that carry their share, so the loop
may stop wherever on the ladder it likes. -/
def ClusteringCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre : Set S) (η₀ indecisionLimit εcov α δ ρ pAP k : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  populations.Nonempty →
  Flat Pre →
  (∀ j ∈ populations, D j Preᶜ = 0) →
  (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
  0 < pAP →
  pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  -- The budget the schedule is built from costs no more than `k` buys: a count that is
  -- logarithmic in the failure probability and polynomial in the rates it resolves.
  -- Derivable rather than assumed -- `prefCount_le_poly` discharges it -- and carried
  -- here so the statement says what the algorithm costs and not only that it works.
  (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ)
    ≤ budgetCap populations η₀ indecisionLimit εcov δ α pAP k →
  ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP →
  collisionMass Dsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP →
  1 - δ ≤ (runMeasure μ D Dsf).real
    {x | (∃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ
      (collisionMass Dsf)},
        x ∈ ret O.mq populations indecisionLimit α B.val) ∧
      ∀ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ
        (collisionMass Dsf)},
        x ∈ ret O.mq populations indecisionLimit α B.val →
        ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
              (clusterAt O.mq populations x B.val) p (oracleNoise x)}}

end OrthoDFA
