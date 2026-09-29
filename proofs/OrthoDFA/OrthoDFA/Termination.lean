import OrthoDFA.ReturnAccuracy

/-!
# The learner stops

Each round the clustering picks a family from the populations so far, and the L\* stage builds a
hypothesis with it.  A gate compares the hypothesis with the family's cut; past the gate, the
merge check reads the hypothesis's states.  A round the gate refuses keeps what it harvests, and
a round whose check fails keeps the failed state's off-label side; both become populations.

Two kinds of target state hold a round up: those the family often leaves undecided, and those
it decides the wrong way more often than the right one.  A kept harvest is heavy on the first
kind, and a kept side on the second.  Every family after it has to read that population as the
gates allow, which is incompatible with reading those same states badly again.  How often counts
as often is one of finitely many thresholds `θs`, and a population may put at most `cap t` on the
states left undecided more than `t` of the time.  So no two refused rounds find the same set of
undecided states at the same threshold, and no two failed checks the same set of misread ones,
and at most `(|θs| + 1) · 2^|Q|` rounds do not return, besides the refused rounds that harvest
nothing heavy enough.  Those are not rare: the pass stops on a run of clean probes, so a round
can stop short, be refused narrowly, and find nothing undecided.  Each is idle with probability
at most `δs` given the rounds before it, so `W` of them cost `δs ^ W` for each set of rounds
they could be.

The clustering, the stage and the check are each held to a spec, and nothing else is assumed of
them.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q : Type*} [Fintype Q]

/-- The chance, over the oracle's noise, that `F` decides `p` the right way. -/
noncomputable def rightProb (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) : ℝ :=
  μ.real {ω | decided O.mq lo hi F p ω ∧ cutCorrect O lo hi F p ω}

/-- The target states the family leaves undecided more than `θ` of the time. -/
def undecidedStates (A : DFA S Q) (O : Oracle μ S) (θ : ℝ) (B : State) (F : Finset S) : Set Q :=
  {q | ∃ p, A.state p = q ∧ θ < undecidedProb O B.lo B.hi F p}

/-- The target states the family decides the wrong way more often than the right one. -/
def misreadStates (A : DFA S Q) (O : Oracle μ S) (B : State) (F : Finset S) : Set Q :=
  {q | ∃ p, A.state p = q ∧ rightProb O B.lo B.hi F p < miscutProb O B.lo B.hi F p}

/-- Round `r`'s populations: the first ones, and every harvest and every side kept before. -/
def poolsAt {Θ J : Type*} (populations : Finset J) (D : J → Measure S)
    (harvest side : ℕ → Θ → Measure S) (kept sideKept : ℕ → Θ → Prop) (r : ℕ) (θ : Θ) :
    Set (Measure S) :=
  {D' | (∃ j ∈ populations, D' = D j)
    ∨ (∃ i < r, kept i θ ∧ D' = harvest i θ)
    ∨ ∃ i < r, sideKept i θ ∧ D' = side i θ}

/-- Round `r` is refused without keeping a harvest that puts more than `cap t` on the states
the family leaves undecided more than `t` of the time, for any `t` in `θs`. -/
def idleRefusal {Θ : Type*} (A : DFA S Q) (O : Oracle μ S) (family : ℕ → Θ → State × Finset S)
    (gate kept : ℕ → Θ → Prop) (harvest : ℕ → Θ → Measure S) (θs : Finset ℝ) (cap : ℝ → ℝ)
    (r : ℕ) : Set Θ :=
  {θ | ¬ gate r θ ∧ ¬ (kept r θ ∧ ∃ t ∈ θs, cap t < (harvest r θ).real
      {p | A.state p ∈ undecidedStates A O t (family r θ).1 (family r θ).2})}

/-- Round `r` settles on the cut `family r`; `gate r` says whether its hypothesis gets past the
gate, and `fails r` whether the merge check then fails some state.  A round returns when it gets
past the gate and nothing fails.  A refused round may keep its harvest, `harvest r`, and a round
whose check fails may keep the failed state's off-label side, `side r`.  `ℱ r` is what is known
before round `r`.  Suppose each round:

* but for probability `δc`, every population so far puts at most `cap t` on the states the family
  leaves undecided more than `t` of the time, for each `t` in `θs`, and at most `cM` on those it
  misreads.  A population the family leaves undecided at most `2·indecisionLimit + slack` of the
  time on average gives `cap t = (2·indecisionLimit + slack)/t`;
* but for probability `δs` given `ℱ r`, a refused round keeps a harvest putting more than `cap t`
  on the states the family leaves undecided more than `t` of the time, for some `t` in `θs`;
* but for probability `δa`, a failed check keeps a side putting more than `ρ` on states the
  family misreads.

Then none of the first `(|θs| + 1) · 2^|Q| + W` rounds returns with probability at most
`((|θs| + 1) · 2^|Q| + W)·(δc + δa) + C((|θs| + 1) · 2^|Q| + W, W)·δs^W`. -/
def Termination : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q : Type*} [Fintype Q] {J Θ : Type*}
    [mΘ : MeasurableSpace Θ] (P : Measure Θ) [IsProbabilityMeasure P] (ℱ : Filtration ℕ mΘ)
    (A : DFA S Q) (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (family : ℕ → Θ → State × Finset S) (gate : ℕ → Θ → Prop) (fails : ℕ → Θ → Prop)
    (harvest side : ℕ → Θ → Measure S) (kept sideKept : ℕ → Θ → Prop)
    (θs : Finset ℝ) (cap : ℝ → ℝ) (cM ρ δc δs δa : ℝ) (W : ℕ),
  cM < ρ →
  (∀ r, P.real {θ | ∃ D' ∈ poolsAt populations D harvest side kept sideKept r θ,
      (∃ t ∈ θs, cap t < D'.real
          {p | A.state p ∈ undecidedStates A O t (family r θ).1 (family r θ).2})
      ∨ cM < D'.real
          {p | A.state p ∈ misreadStates A O (family r θ).1 (family r θ).2}} ≤ δc) →
  (∀ r, MeasurableSet[ℱ (r + 1)] (idleRefusal A O family gate kept harvest θs cap r)) →
  (∀ r E, MeasurableSet[ℱ r] E →
      P.real (E ∩ idleRefusal A O family gate kept harvest θs cap r) ≤ δs * P.real E) →
  (∀ r, P.real {θ | gate r θ ∧ fails r θ ∧ ¬ (sideKept r θ ∧ ρ < (side r θ).real
      {p | A.state p ∈ misreadStates A O (family r θ).1 (family r θ).2})} ≤ δa) →
  P.real {θ | ∀ r < (θs.card + 1) * 2 ^ Fintype.card Q + W, ¬ gate r θ ∨ fails r θ}
    ≤ ((θs.card + 1) * 2 ^ Fintype.card Q + W) * (δc + δa)
      + (((θs.card + 1) * 2 ^ Fintype.card Q + W).choose W : ℝ) * δs ^ W

end OrthoDFA
