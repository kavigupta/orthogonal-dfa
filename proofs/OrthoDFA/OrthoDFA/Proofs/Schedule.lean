import OrthoDFA.Proofs.Clusterer
import OrthoDFA.Proofs.GateCharge
import OrthoDFA.Proofs.Coins

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
variable {rule : Clusterer S}
variable {J : Type*} [Fintype J]
variable {hsz : ℕ}

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runMeasure μ D Dsf) := by
  unfold runMeasure; infer_instance

deriving instance DecidableEq for State

instance : Countable State :=
  Function.Injective.countable
    (f := fun b => (b.nsuff, b.npref, b.k, b.lo, b.hi, b.sc, b.scd))
    (by rintro ⟨⟩ ⟨⟩ h; simp_all)

/-- `s = 1/2 − η`. -/
noncomputable def sig (η : ℝ) : ℝ := 1 / 2 - η

/-- The scale the counts are resolved against: the smallest rate the round has to clear.

Stated in the rates themselves rather than in `cutBudget`, which is one particular way of
taking their minimum.  What a reader checks is that the budget is polynomial in the rates
asked for; which divisors the proof chose to hold each of them at is the proof's business. -/
noncomputable def budgetScale (η indecisionLimit εcov : ℝ) : ℝ :=
  min (min εcov (1 / 2) * gateA η ^ 2) indecisionLimit

/-- What fraction of the family the vote absorbs flipping, at any rates under `η`.  A flip moves
a member's mean by the gap between the classes, at most `1 − η`, and what is left is
`voteSlack`:

    (1 − η) · flipFrac η + voteSlack η = sig η

Only termination needs this: the gate certifies the miscut itself, but a family it can pass
has to be found among members the screen admits flipping up to `flipBudget`.

An oracle with a narrower gap absorbs proportionally more flips, and its screen sees each one
proportionally less; `screenMargin` is solved at that pairing, not at this fraction. -/
noncomputable def flipFrac (η : ℝ) : ℝ := 3 * sig η / (5 * (1 - η))

/-- How far into the margin left by `flipFrac` the vote's count is read. -/
noncomputable def voteSlack (η : ℝ) : ℝ := 2 * sig η / 5

/-- The share of a population a round lets its family cut wrongly.  `εcov/8` and `s/10` hold the
vote's misfire rate and the screen's cutoff, `indecisionLimit/2` is what the FNR test tolerates,
and the last term is what the gate's charge leaves room for: a clean family is wrong on at most
`13/8` of this, and the gate charges each wrong prefix `1/gateKappa`, so the charge takes at most
`4/5` of the gate's limit. -/
noncomputable def cutBudget (η indecisionLimit εcov : ℝ) : ℝ :=
  min (εcov / 8) (min (sig η / 10) (min (indecisionLimit / 2)
    (32 * gateKappa η * min εcov (1 / 2) / 65)))

/-- What the gate's limit leaves a clean family once its charge and its sampler's redraws are
paid: the room its deviations must fit in, per prefix. -/
noncomputable def gateRoom (η indecisionLimit εcov : ℝ) : ℝ :=
  min εcov (1 / 2) - 13 * cutBudget η indecisionLimit εcov / (8 * gateKappa η)
    - cutBudget η indecisionLimit εcov / 128

/-- Hoeffding's size for the family, which `famCount` never exceeds.  Its vote fails at
`exp (-κ·voteSlack²/2)`: a clean vote sits `s` from the centre, a flipping `flipFrac` of the
family spends all but `voteSlack` of that, and the band's far edge takes half of the rest.  That
rate only has to sit under half the cut budget: the misfires on the certification sample are
independent given the family, so their count concentrates below the budget.

The band is about `κ·voteSlack` wide and a crossing vote travels all of it, so its tail is
`exp (-2κ·voteSlack²)`, which the second term puts under `crossLimit`.  Each tail asks for its
own size, and the family takes the larger.  The third pays for the seed: the family
returned is one member larger than the band is centred for, which its vote pays for out of the
slack, at `decSlack`.

The size is even, and the `+ 1` is inside the doubling, because `flipFrac` spends its side of
the margin exactly: the band is centred at `κ/2`, and an odd `κ` would round that up past what
`voteSlack` has left to pay with. -/
noncomputable def famHoeff (η : ℝ) (_populations : Finset J)
    (indecisionLimit εcov _δ crossLimit : ℝ) : ℕ :=
  2 * (max ⌈Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2⌉₊
      ⌈Real.log (1 / crossLimit) / (4 * voteSlack η ^ 2)⌉₊
    + ⌈1 / voteSlack η⌉₊ + 1)

/-- How far `famHoeff`'s vote, seed included, clears its cut at `lo` and `hi + 1`: the band's
`voteSlack/2`, less the half read the extra member costs spread over all of them. -/
noncomputable def decSlack (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ crossLimit : ℝ) : ℝ :=
  voteSlack η / 2
    - 1 / (2 * ((famHoeff η populations indecisionLimit εcov δ crossLimit : ℝ) + 1))

/-- How many counts the band of a family of `κ` reaches either side of its centre. -/
noncomputable def bandAt (η : ℝ) (κ : ℕ) : ℕ := ⌊(κ : ℝ) * voteSlack η / 2⌋₊

/-- The family of `κ` rejects at or below this count. -/
noncomputable def loAt (η : ℝ) (κ : ℕ) : ℕ := ⌈(κ : ℝ) / 2⌉₊ - bandAt η κ - 1

/-- The family of `κ` accepts above this count. -/
noncomputable def hiAt (η : ℝ) (κ : ℕ) : ℕ := ⌈(κ : ℝ) / 2⌉₊ + bandAt η κ

/-- The vote of a family of `κ` with its seed, at a mean rate of `1/2 + voteSlack`, failing to
clear `hi + 1`. -/
noncomputable def misfireTail (η : ℝ) (κ : ℕ) : ℝ :=
  1 - binomSfGe (κ + 1) (1 / 2 + voteSlack η) (hiAt η κ + 2)

/-- A vote over `κ` reads whose mean is `hi` landing at or below `lo`. -/
noncomputable def crossTail (η : ℝ) (κ : ℕ) : ℝ :=
  1 - binomSfGe κ ((hiAt η κ : ℝ) / κ) (loAt η κ + 1)

/-- A vote over `κ` reads whose mean is the band's centre landing at or below `lo`. -/
noncomputable def halfTail (η : ℝ) (κ : ℕ) : ℝ :=
  1 - binomSfGe κ (1 / 2) (loAt η κ + 1)

/-- A family of `κ` the exact tails allow.  The tails are binomials, which bound a vote's reads
whatever their rates (`BitSumTails`).  The band spans at least five counts, which that bound asks
of it, and the family is at least 128 strong, which the pool's size bounds lean on.

`halfTail` is for the loop's band check, which reads the band from its centre at the fourth root
of `crossLimit`. -/
def FamilyFits (η cut crossLimit : ℝ) (κ : ℕ) : Prop :=
  Even κ ∧ 128 ≤ κ ∧ 5 ≤ (κ : ℝ) * voteSlack η
    ∧ misfireTail η κ ≤ cut / 2 ∧ crossTail η κ ≤ crossLimit ∧ halfTail η κ ^ 4 ≤ crossLimit

open scoped Classical in
/-- The family's size: the least that `FamilyFits`, searched no further than `famHoeff`, which
Hoeffding's tails show fits whenever the claim's premises hold. -/
noncomputable def famCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ crossLimit : ℝ) : ℕ :=
  Nat.find (⟨famHoeff η populations indecisionLimit εcov δ crossLimit, Or.inr rfl⟩ :
    ∃ κ, FamilyFits η (cutBudget η indecisionLimit εcov) crossLimit κ
      ∨ κ = famHoeff η populations indecisionLimit εcov δ crossLimit)

/-- How many counts the band reaches either side of the centre count. -/
noncomputable def bandHalf (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ crossLimit : ℝ) : ℕ :=
  bandAt η (famCount η populations indecisionLimit εcov δ crossLimit)

/-- What one family member may flip.  The vote absorbs a `flipFrac` fraction of the family
flipping, so Markov charges the cut budget at that fraction and not at the family's size.

The screen's tail binds and is read at this, so it takes fifty-five sixty-fourths of the cut
budget; the heavy-prefix count's slack and threshold, whose tails are cheap, and the prefixes the
samplers redraw share the rest. -/
noncomputable def flipBudget (η : ℝ) (populations : Finset J) (indecisionLimit εcov _δ : ℝ) : ℝ :=
  55 * cutBudget η indecisionLimit εcov * flipFrac η / (64 * (populations.card : ℝ))

/-- The exponent rate of the drawn prefixes understating a member's flip mass by
`g = cutBudget·flipFrac/32`, relative to a mass of at most half the cut budget:

    g² / cutBudget -/
noncomputable def dirtyRate (η indecisionLimit εcov : ℝ) : ℝ :=
  (cutBudget η indecisionLimit εcov * flipFrac η / 32) ^ 2 / cutBudget η indecisionLimit εcov

/-- The screen's margin, at the flip budget.

A flip costs both the vote and the screen the gap `g` between the classes, the screen at
`g·(1 − 2η)` per flip.  The vote absorbs a fraction `f` flipping with `g·f ≥ (1 − η)·flipFrac`,
so a member flipping `flipBudget·f/flipFrac`, what Markov charges at that fraction, shows the
screen at least

    flipBudget · (1 − η) · (1 − 2η)

whatever `g`.  The test spends four one-sided deviations on it — the seed's and the
candidate's, each against the measured floor — so the factor here is capped below `2`.
`sc/scd` then has to resolve finer than `a/(4 − 2a)`, which is the `63/2` in `solvedStateAt`;
the two constants move together. -/
noncomputable def screenMargin (η : ℝ) (populations : Finset J) (indecisionLimit εcov δ : ℝ) : ℝ :=
  flipBudget η populations indecisionLimit εcov δ * ((1 - η) * sig η) * (63 / 64)

/-- Enough suffixes that a family of `k` fits inside the findable fraction, read at `9pAP/20` so
that a twentieth of the pool is left for the draws the sampler redraws. -/
noncomputable def poolCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℕ :=
  ⌈2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊

/-- The fraction of a state's prefix draws the samplers may redraw.  Every margin it is charged
against is a fraction of the cut budget, the flip margins at the flip fraction, so this is both
at once with room to spare. -/
noncomputable def rejFrac (η indecisionLimit εcov : ℝ) : ℝ :=
  cutBudget η indecisionLimit εcov * flipFrac η / 128

/-- The draws the samplers may redraw at a prefix count. -/
noncomputable def rejCount (η indecisionLimit εcov : ℝ) (m : ℕ) : ℕ :=
  ⌊rejFrac η indecisionLimit εcov * m⌋₊

/-- The suffix draws the pool's sampler may redraw: a twentieth of the findable share. -/
noncomputable def poolRej (pAP : ℝ) (M : ℕ) : ℕ :=
  ⌊pAP * M / 20⌋₊

/-- The log factor the counts share.  Every tail is a deviation bound at one of the failure
probabilities or rates in play, union-bounded over at most the pool's pairs of suffixes, so a
log of their reciprocals and of the pool's size covers all of it up to a constant. -/
noncomputable def budgetLog (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ) : ℝ :=
  Real.log (((populations.card : ℝ) + 2)
    * (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
    / (δ * α * pAP * budgetScale η indecisionLimit εcov))

/-- What a budget of `k` buys.

`sig` enters at the fourth power because the screen's tail and the gate's bind: their margins
are a rate times `sig²`, and a deviation bound squares the margin it is given.  The population
count enters squared because the screen's margin is divided by it. -/
noncomputable def budgetCap (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit k : ℝ) : ℝ :=
  k * (populations.card : ℝ) ^ 2
    * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
    / (sig η ^ 4 * budgetScale η indecisionLimit εcov ^ 2)

/-- What the harvested strings add to the budget: the table holds them in its count, so the
count must be large against them. -/
noncomputable def harvestCap (η indecisionLimit εcov : ℝ) (hsz : ℕ) : ℝ :=
  128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov + 1

/-- `4K (log(4Kk) + 2 log log(4Kk))`, past which `n ≥ 4K log(kn)`. -/
noncomputable def gateLogCount (K k : ℝ) : ℝ :=
  4 * K * (Real.log (4 * K * k) + 2 * Real.log (Real.log (4 * K * k)))

/-- The counts the round's tails ask for, summed so each is met.  The gate's two terms pay for
its deviations in `gateRoom`: the seed's and the references', at `log(32|J|(M + 3)/δ)`, take
two fifths of it, and the levels', which shrink with the prefix count `n`, the rest.  With
`K = (50/9)(gateDevD + 1)²/gateRoom²` the latter need

    n ≥ K (log(7|J|/α) + 4 log(2n + 2)),

which `n ≥ 4K (log(4Kk) + 2 log log(4Kk))` clears, `k = 4 (7|J|/α)^(1/4)`.

The heavy hits' term is a relative tail: a hit rate of at most `cutBudget` reaches
`(1 + 1/32)·cutBudget` at the rate `relRate`, `cutBudget/2112`; with it the undecided share's
tail at `indecisionLimit/2`.

The `4096/9` term is the hash sides' imbalance on a named population, `G = ⌈√(m·log(128|J|/δ))⌉`,
which is charged against the cut budget and has to fit in `3·cutBudget·m/64`. -/
noncomputable def prefCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℕ :=
  ⌈Real.log (256 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2)⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ)
        * (2112 / cutBudget η indecisionLimit εcov + 2 / indecisionLimit ^ 2)⌉₊
    + ⌈25 / 2 * (gateDevT η + gateDevU η) ^ 2
        * Real.log (32 * (populations.card : ℝ)
          * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3) / δ)
        / gateRoom η indecisionLimit εcov ^ 2⌉₊
    + ⌈gateLogCount (50 / 9 * (gateDevD η + 1) ^ 2 / gateRoom η indecisionLimit εcov ^ 2)
        (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α)))⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / dirtyRate η indecisionLimit εcov⌉₊
    + ⌈2900 * Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / (cutBudget η indecisionLimit εcov * flipFrac η)⌉₊
    + ⌈4096 * Real.log (128 * (populations.card : ℝ) / δ)
        / (9 * cutBudget η indecisionLimit εcov ^ 2)⌉₊
    + ⌈128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov⌉₊
    + 1

/-- The fewest prefixes at which a stop is covered: the FNR test's tail at
`δ/(128·|populations|)`.  The gate pays for its own soundness at every count. -/
noncomputable def validCount (_η : ℝ) (populations : Finset J)
    (indecisionLimit _εcov δ _pAP _crossLimit : ℝ) : ℕ :=
  ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (indecisionLimit / 2) ^ 2)⌉₊ + 1

/-- The state at a given prefix count: every other field read off the condition it has to
meet. -/
noncomputable def solvedStateAt (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) (mi : ℕ) : State where
  nsuff := poolCount η populations indecisionLimit εcov δ pAP crossLimit
  npref := mi
  k := famCount η populations indecisionLimit εcov δ crossLimit + 1
  lo := ⌈(famCount η populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊
    - bandHalf η populations indecisionLimit εcov δ crossLimit - 1
  hi := ⌈(famCount η populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊
    + bandHalf η populations indecisionLimit εcov δ crossLimit
  sc := ⌈((⌈63 / (2 * screenMargin η populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ)
    * screenMargin η populations indecisionLimit εcov δ⌉₊
  scd := ⌈63 / (2 * screenMargin η populations indecisionLimit εcov δ)⌉₊ + 1

/-- How many times the loop runs the gate: the prefix count halves down to one.  There is no
other state the loop can return at. -/
noncomputable def ladderLen (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℕ :=
  Nat.log 2 (prefCount η populations indecisionLimit εcov δ α pAP crossLimit hsz) + 1

/-- The states the loop runs the gate at: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) (hsz : ℕ) : Finset State :=
  (Finset.range (ladderLen η populations indecisionLimit εcov δ α pAP crossLimit hsz)).image
    (fun i => solvedStateAt η populations indecisionLimit εcov δ pAP crossLimit
      (prefCount η populations indecisionLimit εcov δ α pAP crossLimit hsz / 2 ^ i))

/-- What one tested state may cost, summed over the populations: the samplers redrawing more
than `rejCount` of the state's draws, priced by Markov's inequality at `ρ` a chance so linearly in
the draws, and the draws missing the undecided set.  The gate's own tails are not here: they
are spent at `gateLevel`, which sums to `α` over the states. -/
noncomputable def stateFail (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov ρ : ℝ)
    (B : State) : ℝ :=
  (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 5) * (B.npref : ℝ) * ρ
      / rejFrac η₀ indecisionLimit εcov
    + Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2))

/-- A state can be stopped at when its thresholds are in order and it has drawn enough
prefixes to carry its share of the error budget.

At a handful of prefixes neither the screen nor the certification sample can see a dirty
family, so the guarantee cannot cover a stop there. -/
structure Capped (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov δ ρ pAP : ℝ)
    (N V : ℕ) (B : State) : Prop where
  /-- Reject at or below accept, so the two gate sides are disjoint. -/
  lohi : B.lo ≤ B.hi
  /-- There are prefixes to read the rate off. -/
  mpos : 0 < B.npref
  /-- Two accept-preserving draws more than the pool's sampler redraws are expected in the
  pool, so one of them is not the seed and the floor has a clean candidate to sit at. -/
  found : (2 : ℝ) + poolRej pAP B.nsuff ≤ (B.nsuff : ℝ) * (pAP / 2)
  /-- Two shares, each summing to at most `δ/8` over the ladder.  One is proportional to the
  prefix count, whose sum halves down from the top count `N`; it is what the top rung and the
  collision terms, which grow with the count, are paid from.  The other goes to the rungs at or
  above `V` in inverse proportion to their counts, whose sum halves up from `V`; it is what a low
  rung's tails are paid from, and it asks nothing of `N`.

  A share uniform over the ladder would make every count clear `log L`, and `L` is read off the
  top count. -/
  share : stateFail η₀ populations indecisionLimit εcov ρ B
    ≤ δ * (B.npref : ℝ) / (8 * N)
      + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0

open scoped Classical in

/-- The rungs of the ladder that carry their share.  A `Finset`, so the union bound over it is
a finite sum and no summable weight over all budgets is needed.  A side of the hash holds at
least a third of a tame population, so its draws' atoms are at most `3ρ`. -/
noncomputable def stoppable (η₀ : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) (hsz : ℕ) (ρ : ℝ) : Finset State :=
  (schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).filter
    (Capped η₀ populations indecisionLimit εcov δ (3 * ρ) pAP
      (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
      (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit))

/-- How much collision mass the populations may carry.  A state redraws `m·ρ/rejFrac` of its
`m` draws in expectation, which its share `δ·m/(8N)` pays for, so the cap is linear in the top
count `N`. -/
noncomputable def collisionCap (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℝ :=
  δ * budgetScale η indecisionLimit εcov ^ 2
    / (2 ^ 20 * ((populations.card : ℝ) + 1) ^ 2
      * (prefCount η populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ))

/-- How much collision mass the suffix measure may carry: the pool redraws `M·ρ/pAP` of its `M`
draws, a twentieth of the findable share. -/
noncomputable def poolCollisionCap (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℝ :=
  δ * pAP / (2 ^ 12 * ((populations.card : ℝ) + 1)
    * (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ))

open scoped Classical in
/-- The E-L\* clustering algorithm is PAC-correct: with probability `≥ 1 − δ − α` the loop
terminates, and the family it returns — at whatever state it stops — cuts each tame population
the way the noiseless oracle does on all but its gate's limit, `εcov` on the uniform pool and a
half elsewhere, twice the redraw fraction, and on a named population `εcov/1024` for reading it
through one side of the hash, and leaves at most `2·indecisionLimit` of each undecided.

The algorithm is not told the noise rate, only an upper bound `η₀` on it — `η₀` is what
`min_signal_strength` gives `build_pst`, and every computed field of `State` is solved from
it, never from `O.η`.  The gate reads each class's rate at `η₀`, so it needs none of them.

The hypotheses, in the order they appear: both of the oracle's noise rates are at most `η₀`,
which has signal; the uniform pool is a population; the populations, the harvested strings and
the suffix measure are supported on a `Flat` pair of sets; no string has more than `ρ` of a tame
population and `pAP` of the suffix measure is accept-preserving; `indecisionLimit`, `α`, `εcov`
and `δ` are in range; and `ρ` fits `collisionCap` and `Dsf`'s collision mass fits
`poolCollisionCap`. -/
def ClusteringCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (rule : Clusterer S) (populations tame : Finset J) (uni : J)
    (H : J → Finset S) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre Suf : Set S) (η₀ indecisionLimit εcov α δ ρ pAP crossLimit k : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  uni ∈ tame →
  tame ⊆ populations →
  Flat Pre Suf →
  (∀ j, ∀ p ∈ H j, p ∈ Pre) →
  (∀ j ∈ populations, D j Preᶜ = 0) →
  Dsf Sufᶜ = 0 →
  (∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ) →
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
  (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card : ℝ)
    ≤ budgetCap populations η₀ indecisionLimit εcov δ α pAP crossLimit k
      + harvestCap η₀ indecisionLimit εcov (allHarvest H).card →
  ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card →
  collisionMass Dsf
    ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit →
  1 - δ - α ≤ (runMeasure μ D Dsf).real
    {x | (∃ B : {B : State // B ∈ stoppable η₀ populations
        indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card ρ},
        streams H uni x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val) ∧
      ∀ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ
        α pAP crossLimit (allHarvest H).card ρ},
        streams H uni x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val →
        ∀ j ∈ tame, 1 - gateLimit uni εcov j - 2 * rejFrac η₀ indecisionLimit εcov
            - (if j = uni then 0 else εcov / 1024)
          ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
              (familyBy rule O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
          ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
              (familyBy rule O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
            ≤ 2 * indecisionLimit}

end OrthoDFA
