import OrthoDFA.Proofs.Clusterer
import OrthoDFA.Proofs.GateCharge
import OrthoDFA.Proofs.Coins
import OrthoDFA.Proofs.RedrawTail
import OrthoDFA.LowLevelClaims.BandKappa

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

/-- The smallest rate any of the round's tails resolves, the gate's charge on a misfire
included.  Only logs read it: the counts' polynomial part is in `shiftScale`.

Stated in the rates themselves rather than in `cutBudget`, which is one particular way of
taking their minimum. -/
noncomputable def budgetScale (η indecisionLimit εcov : ℝ) : ℝ :=
  min (min εcov (1 / 2) * gateA η ^ 2) indecisionLimit

/-- How far from the centre the vote's count is read.  A class at rate `r` leaves its prefixes'
vote `1/2 − r ≥ sig` from the centre, so a family a fraction `(1/2 − r − voteSlack)/gap` of which
flips still clears it (`clsFrac`). -/
noncomputable def voteSlack (η : ℝ) : ℝ := 4 * sig η / 5

/-- The share of a population the vote may misfire on, among the prefixes the family is usable
at.  `indecisionLimit/2` is what the FNR test tolerates, and the last term is what the gate's
charge leaves room for: a misfire is charged at most `1/a²` (`sideCharge_le_gateA`), on either
side, so `5/8` of this takes at most `5/256` of the gate's limit per side. -/
noncomputable def cutBudget (η indecisionLimit εcov : ℝ) : ℝ :=
  min (εcov / 8) (min (sig η / 10) (min (indecisionLimit / 2)
    (gateA η ^ 2 * min εcov (1 / 2) / 32)))

/-- What the gate's limit leaves its deviations, per prefix, once the charge of a clean family and
its sampler's redraws are paid. -/
noncomputable def gateRoom (_η _indecisionLimit εcov : ℝ) : ℝ := min εcov (1 / 2) / 2

/-- The rate the count rule resolves, which no factor of the gate's charge enters. -/
noncomputable def shiftScale (indecisionLimit εcov : ℝ) : ℝ :=
  min (min εcov (1 / 2)) indecisionLimit

/-- What the family may shift a population, on average over its members, in the count rule's own
units: Markov per class charges a class's heavy mass at `σ_b ≥ (2/5)·sig²`, and the gate charges
it at most `5/(2 sig²)·σ_b` a prefix, so `sig²·shiftScale` resolves both the FNR test and the
gate. -/
noncomputable def shiftBudget (η indecisionLimit εcov : ℝ) : ℝ :=
  sig η ^ 2 * shiftScale indecisionLimit εcov / 256

/-- Hoeffding's size for the family, which `famCount` never exceeds.  Its vote fails at
`exp (-κ·voteSlack²/2)`: a clean vote sits `s` from the centre, a flipping `clsFrac` of the
family spends all but `voteSlack` of that, and the band's far edge takes half of the rest.  That
rate only has to sit under half the cut budget: the misfires on the certification sample are
independent given the family, so their count concentrates below the budget.

The band is about `κ·voteSlack` wide and a crossing vote travels all of it, so its tail is
`exp (-2κ·voteSlack²)`, which the second term puts under `crossLimit`, with `e⁴` to spare for the
loop's band check.  Each tail asks for its own size, and the family takes the larger.  The third
pays for the seed: the family returned is one member larger than the band is centred for, which
its vote pays for out of the slack, at `decSlack`.

The size is even, and the `+ 1` is inside the doubling, because `clsFrac` spends its side of
the margin exactly: the band is centred at `κ/2`, and an odd `κ` would round that up past what
`voteSlack` has left to pay with. -/
noncomputable def famHoeff (η : ℝ) (_populations : Finset J)
    (indecisionLimit εcov _δ crossLimit : ℝ) : ℕ :=
  2 * (max ⌈Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2⌉₊
      ⌈(Real.log (1 / crossLimit) + 4) / (4 * voteSlack η ^ 2)⌉₊
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
of it, and the family is at least 32 strong, which keeps the pool at 64 for the budget's
bounds.

The last is for the loop's band check, at `ζ = crossLimit^(1/4)`: whatever the read's two rates,
it is on its rarer decided side at most `ζ` times as often as undecided (`BandKappaBound`). -/
def FamilyFits (η cut crossLimit : ℝ) (κ : ℕ) : Prop :=
  Even κ ∧ 32 ≤ κ ∧ 5 ≤ (κ : ℝ) * voteSlack η
    ∧ misfireTail η κ ≤ cut / 2 ∧ crossTail η κ ≤ crossLimit
    ∧ bandKappa κ (loAt η κ) (κ / 2) (hiAt η κ + 1) ^ 4 ≤ crossLimit

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

/-- Enough suffixes that a family of `k` fits inside the findable fraction, read at `9pAP/20` so
that a twentieth of the pool is left for the draws the sampler redraws. -/
noncomputable def poolCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℕ :=
  ⌈2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP⌉₊
    + ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊

/-- The accept-preserving rows a pool of `M` guarantees besides the seed: half its findable
share, the rest spent on the count's deviation and the sampler's redraws. -/
noncomputable def apFloor (pAP : ℝ) (M : ℕ) : ℝ := M * pAP / 2 - 1

/-- How far above the seed's prediction the clean rows are read, in units of `√(|J|·m)`:

    u = √(log(a/((a − κ + 1)ε))/2),   a = apFloor,   ε = δ/16,

so that more than `a − κ` of `a` clean rows clear it with chance at most `ε`, by Markov. -/
noncomputable def cleanLevel (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℝ :=
  Real.sqrt (Real.log (apFloor pAP (poolCount η populations indecisionLimit εcov δ pAP crossLimit)
    / ((apFloor pAP (poolCount η populations indecisionLimit εcov δ pAP crossLimit)
      - famCount η populations indecisionLimit εcov δ crossLimit + 1) * (δ / 16))) / 2)

/-- How far below its mean a heavy row has to read to be picked, in units of `√(|J|·m)`:

    x = max(2, √((5/2) log(15(M + 1)/(κ ε)))),   ε = δ/16,

the `15` being `θ/(1 − θ)` at `θ = 15/16`, the share of the table budget the heavy rows are
charged at. -/
noncomputable def heavyLevel (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℝ :=
  max 2 (Real.sqrt (5 / 2 * Real.log (15
    * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1)
    / ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) * (δ / 16)))))

/-- The fraction of a state's prefix draws the samplers may redraw.  A redraw hides at most a
flip's shift on a class, which Markov charges at `w_b/σ_b ≤ 5/sig` against the FNR test and the
gate at `5/(2 sig²)` against its limit: the first two terms.  Twice it is what the redraws can bias
the verdict by, which the third holds to `εs/2`. -/
noncomputable def rejFrac (η indecisionLimit εcov εs : ℝ) : ℝ :=
  min (min (sig η ^ 2 * min εcov (1 / 2) / 1024) (sig η * indecisionLimit / 128)) (εs / 4)

/-- The exponent rate of the drawn prefixes understating a member's shift mass on a class by
`g = shiftBudget`, relative to a mass of at most `shiftBudget + rejFrac + g`:

    g²/(4 (shiftBudget + rejFrac)) -/
noncomputable def dirtyRate (η indecisionLimit εcov εs : ℝ) : ℝ :=
  shiftBudget η indecisionLimit εcov ^ 2
    / (4 * (shiftBudget η indecisionLimit εcov + rejFrac η indecisionLimit εcov εs))

/-- The draws the samplers may redraw at a prefix count. -/
noncomputable def rejCount (η indecisionLimit εcov εs : ℝ) (m : ℕ) : ℕ :=
  ⌊rejFrac η indecisionLimit εcov εs * m⌋₊

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

`sig` enters at the fourth power because the count rule's tail and the gate's bind: their
margins are a rate times `sig²`, and a deviation bound squares the margin it is given. -/
noncomputable def budgetCap (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit k : ℝ) : ℝ :=
  k * (populations.card : ℝ)
    * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
    / (sig η ^ 4 * shiftScale indecisionLimit εcov ^ 2)

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
`K = (200/9)(gateDevD + 1)²/gateRoom²` the latter need

    n ≥ K (log(7|J|/α) + 4 log(2n + 2)),

which `n ≥ 4K (log(4Kk) + 2 log log(4Kk))` clears, `k = 4 (7|J|/α)^(1/4)`.

The first term is the count rule's: a heavy row's shift, `(15/16)·shiftBudget` a prefix, has to
clear `(u + x)√(|J| m)` on a table of at most `|J| m` prefixes.

The class tails are relative: a class's heavy hit rate `q ≥ 2·shiftBudget` reaches
`(1 + 1/32)·q` at the rate `relRate`, at least `shiftBudget/1056`; with them the undecided share's
tail at `indecisionLimit/2`.  The `dirtyRate` term is a member's shift on a class read off its
first draws, and the `64/shiftBudget` term a named population's heavy prefixes, Markov on the
window and the hash's split of it.

The `4096/9` term is the hash sides' imbalance on a named population, `G = ⌈√(m·log(128|J|/δ))⌉`,
which is charged against the cut budget and has to fit in `3·cutBudget·m/64`.

The `εs` term is the redraws' tail when `rejFrac` is `εs/4`: `rejFrac·m·log 2` must clear
`log(256|J|/δ)`. -/
noncomputable def prefCount (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℕ :=
  ⌈(populations.card : ℝ) * (cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η populations indecisionLimit εcov δ pAP crossLimit) ^ 2
      / (15 / 16 * shiftBudget η indecisionLimit εcov) ^ 2⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ)
        * (1056 / shiftBudget η indecisionLimit εcov + 2 / indecisionLimit ^ 2)⌉₊
    + ⌈50 * (gateDevT η + gateDevU η) ^ 2
        * Real.log (32 * (populations.card : ℝ)
          * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3) / δ)
        / gateRoom η indecisionLimit εcov ^ 2⌉₊
    + ⌈gateLogCount (200 / 9 * (gateDevD η + 1) ^ 2 / gateRoom η indecisionLimit εcov ^ 2)
        (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α)))⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / dirtyRate η indecisionLimit εcov εs⌉₊
    + ⌈64 * Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / shiftBudget η indecisionLimit εcov⌉₊
    + ⌈4096 * Real.log (128 * (populations.card : ℝ) / δ)
        / (9 * cutBudget η indecisionLimit εcov ^ 2)⌉₊
    + ⌈6 * Real.log (256 * (populations.card : ℝ) / δ) / εs⌉₊
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
  sc := 1
  scd := 1

/-- How many times the loop runs the gate: the prefix count halves down to one.  There is no
other state the loop can return at. -/
noncomputable def ladderLen (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℕ :=
  Nat.log 2 (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz) + 1

/-- The states the loop runs the gate at: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) (hsz : ℕ) : Finset State :=
  (Finset.range (ladderLen η populations indecisionLimit εcov εs δ α pAP crossLimit hsz)).image
    (fun i => solvedStateAt η populations indecisionLimit εcov δ pAP crossLimit
      (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz / 2 ^ i))

/-- What one tested state may cost, summed over the populations: each table and certification
sampler rejecting more than `rejCount` of its draws, a binomial tail at what a draw can land on,
and the draws missing the undecided set.  The gate's own tails are not here: they are spent at
`gateLevel`, which sums to `α` over the states. -/
noncomputable def stateFail (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov εs ρ : ℝ)
    (B : State) : ℝ :=
  (populations.card : ℝ) * (rejTail (B.npref + rejCount η₀ indecisionLimit εcov εs B.npref)
        (((B.npref + rejCount η₀ indecisionLimit εcov εs B.npref : ℕ) : ℝ) * ρ)
        (rejCount η₀ indecisionLimit εcov εs B.npref)
      + rejTail (2 * B.npref + rejCount η₀ indecisionLimit εcov εs B.npref)
        (((populations.card * B.npref + (2 * B.npref + rejCount η₀ indecisionLimit εcov εs B.npref)
          : ℕ) : ℝ) * ρ)
        (rejCount η₀ indecisionLimit εcov εs B.npref)
    + Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2))

/-- A state can be stopped at when its thresholds are in order and it has drawn enough
prefixes to carry its share of the error budget.

At a handful of prefixes neither the count rule nor the certification sample can see a dirty
family, so the guarantee cannot cover a stop there. -/
structure Capped (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov εs δ ρ pAP : ℝ)
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
  share : stateFail η₀ populations indecisionLimit εcov εs ρ B
    ≤ δ * (B.npref : ℝ) / (8 * N)
      + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0

open scoped Classical in

/-- The rungs of the ladder that carry their share.  A `Finset`, so the union bound over it is
a finite sum and no summable weight over all budgets is needed.  A side of the hash holds at
least a third of a tame population, so its draws' atoms are at most `3ρ`. -/
noncomputable def stoppable (η₀ : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) (hsz : ℕ) (ρ : ℝ) : Finset State :=
  (schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).filter
    (Capped η₀ populations indecisionLimit εcov εs δ (3 * ρ) pAP
      (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
      (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit))

/-- How much any one string may hold of a tame population.  A rung's draws land on what earlier
draws or the tables hold at most `(|J| + 3)·m·ρ` of the time, which against `rejFrac` makes its
rejections a binomial tail of order `(m/N)·2^-(r+1)`; the uniform pool's draws land in a window
of `4|J|N` strings at most `e/4` of the time, `e = min(εs, indecisionLimit)/8`; and the hash
splits every population evenly but for its sub-Gaussian tail at `ρ/e²`. -/
noncomputable def collisionCap (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) (hsz : ℕ) : ℝ :=
  min (min (rejFrac η indecisionLimit εcov εs
        / (64 * ((populations.card : ℝ) + 3)
          * (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)))
      (min εs indecisionLimit
        / (128 * (populations.card : ℝ)
          * (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ))))
    (δ * min εs indecisionLimit ^ 2
      / (2 ^ 15 * (populations.card : ℝ)
        * (3 + 2 * (Nat.log 2 (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit
          hsz) : ℝ))))

/-- How much collision mass the suffix measure may carry: the pool redraws `M·ρ/pAP` of its `M`
draws, a twentieth of the findable share. -/
noncomputable def poolCollisionCap (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ pAP crossLimit : ℝ) : ℝ :=
  δ * pAP / (2 ^ 12 * ((populations.card : ℝ) + 1)
    * (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ))

open scoped Classical in
/-- The E-L\* clustering algorithm is PAC-correct: with probability `≥ 1 − δ − α` the loop
terminates, and the family the count rule returns — at whatever state it stops — cuts each tame
population the way the noiseless oracle does on all but its gate's limit, `εcov` on the uniform
pool and a half elsewhere, twice the redraw fraction, and on a named population `εs/2` for
reading it through one side of the hash, and leaves at most `2·indecisionLimit` of each
undecided.

The algorithm is not told the noise rate, only an upper bound `η₀` on it — `η₀` is what
`min_signal_strength` gives `build_pst`, and every computed field of `State` is solved from
it, never from `O.η`.  The gate reads each class's rate at `η₀`, so it needs none of them.

The hypotheses, in the order they appear: both of the oracle's noise rates are at most `η₀`,
which has signal; the uniform pool is a population; the populations, the harvested strings and
the suffix measure are supported on a `Flat` pair of sets; no string has more than `ρ` of a tame
population and `pAP` of the suffix measure is accept-preserving; `indecisionLimit`, `α`, `εcov`,
`εs` and `δ` are in range; and `ρ` fits `collisionCap` and `Dsf`'s collision mass fits
`poolCollisionCap`. -/
def ClusteringCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (H : J → Finset S) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre Suf : Set S) (η₀ indecisionLimit εcov εs α δ ρ pAP crossLimit k : ℝ),
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
  0 < εs →
  0 < δ →
  -- The budget the schedule is built from costs no more than `k` buys: a count that is
  -- logarithmic in the failure probability and polynomial in the rates it resolves.
  -- Derivable rather than assumed -- `prefCount_le_poly` discharges it -- and carried
  -- here so the statement says what the algorithm costs and not only that it works.
  (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card : ℝ)
    ≤ budgetCap populations η₀ indecisionLimit εcov δ α pAP crossLimit k
      + 6 * Real.log (256 * (populations.card : ℝ) / δ) / εs
      + harvestCap η₀ indecisionLimit εcov (allHarvest H).card →
  ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card →
  collisionMass Dsf
    ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit →
  1 - δ - α ≤ (runMeasure μ D Dsf).real
    {x | (∃ B : {B : State // B ∈ stoppable η₀ populations
        indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card ρ},
        streams H uni x ∈ retBy nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
          B.val) ∧
      ∀ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov εs δ
        α pAP crossLimit (allHarvest H).card ρ},
        streams H uni x ∈ retBy nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
          B.val →
        ∀ j ∈ tame, 1 - gateLimit uni εcov j - 2 * rejFrac η₀ indecisionLimit εcov εs
            - (if j = uni then 0 else εs / 2)
          ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
              (familyBy nearestClusterer O.mq populations (streams H uni x) B.val) p (oracleNoise
                x.1)}
          ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
              (familyBy nearestClusterer O.mq populations (streams H uni x) B.val) p (oracleNoise
                x.1)}
            ≤ 2 * indecisionLimit}

end OrthoDFA
