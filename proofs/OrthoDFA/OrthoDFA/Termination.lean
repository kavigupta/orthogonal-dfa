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
as often is one of finitely many thresholds `θs`, and a population may put at most `cap k t` on
the states left undecided more than `t` of the time, `k` being how often the limits have been
halved.  The caps only shrink, so no two refused rounds find the same set of undecided states at
the same threshold, and no two failed checks the same set of misread ones.

A probe is checked only if every read on its way is decided, so a family undecided as often as
the limit allows can leave most of a round's probes unchecked and its harvest light.  Such a
round halves the limits for every later family; the first `kstar` of them count as progress.  So
at most `(|θs| + 1) · 2^|Q| + kstar` rounds do not return, besides the idle ones: refused,
harvesting nothing heavy enough, and not among those halvings.  Each is idle with probability at
most `δs` given the rounds before it, so `W` of them cost `δs ^ W` for each set of rounds they
could be.

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

/-- Round `r` is refused without keeping a harvest that puts more than `cap k t` on the states
the family leaves undecided more than `t` of the time, for any `t` in `θs`, where `k` is the
number of halvings before it; and it is not one of the first `kstar` rounds to halve. -/
def idleRefusal {Θ : Type*} (A : DFA S Q) (O : Oracle μ S) (family : ℕ → Θ → State × Finset S)
    (gate kept halves : ℕ → Θ → Prop) (halvings : ℕ → Θ → ℕ) (kstar : ℕ)
    (harvest : ℕ → Θ → Measure S) (θs : Finset ℝ) (cap : ℕ → ℝ → ℝ) (r : ℕ) : Set Θ :=
  {θ | ¬ gate r θ ∧ ¬ (kept r θ ∧ ∃ t ∈ θs, cap (halvings r θ) t < (harvest r θ).real
      {p | A.state p ∈ undecidedStates A O t (family r θ).1 (family r θ).2})
    ∧ ¬ (halves r θ ∧ halvings r θ < kstar)}

/-- Round `r` settles on the cut `family r`; `gate r` says whether its hypothesis gets past the
gate, and `fails r` whether the merge check then fails some state.  A round returns when it gets
past the gate and nothing fails.  A refused round may keep its harvest, `harvest r`, and a round
whose check fails may keep the failed state's off-label side, `side r`.  A refused round
`halves r` when most of its probes since its last split went unchecked; `halvings r` counts the
rounds before `r` that did, and each halves the limits every later family is held to, so the
caps `cap k` shrink as `k` grows.  `ℱ r` is what is known before round `r`.  Suppose each round:

* but for probability `δc`, every population so far puts at most `cap k t` on the states the
  family leaves undecided more than `t` of the time, for each `t` in `θs`, and at most `cM` on
  those it misreads, where `k` is the number of halvings before the round;
* but for probability `δs` given `ℱ r`, a refused round keeps a harvest putting more than
  `cap k t` on the states the family leaves undecided more than `t` of the time, for some `t` in
  `θs`, or is one of the first `kstar` rounds to halve;
* but for probability `δa`, a failed check keeps a side putting more than `ρ` on states the
  family misreads.

Then none of the first `(|θs| + 1) · 2^|Q| + kstar + W` rounds returns with probability at most
`((|θs| + 1) · 2^|Q| + kstar + W)·(δc + δa) + C((|θs| + 1) · 2^|Q| + kstar + W, W)·δs^W`. -/
def Termination : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q : Type*} [Fintype Q] {J Θ : Type*}
    [mΘ : MeasurableSpace Θ] (P : Measure Θ) [IsProbabilityMeasure P] (ℱ : Filtration ℕ mΘ)
    (A : DFA S Q) (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (family : ℕ → Θ → State × Finset S) (gate : ℕ → Θ → Prop) (fails : ℕ → Θ → Prop)
    (harvest side : ℕ → Θ → Measure S) (kept sideKept halves : ℕ → Θ → Prop)
    (halvings : ℕ → Θ → ℕ) (kstar : ℕ)
    (θs : Finset ℝ) (cap : ℕ → ℝ → ℝ) (cM ρ δc δs δa : ℝ) (W : ℕ),
  cM < ρ →
  (∀ r θ, halvings r θ ≤ halvings (r + 1) θ) →
  (∀ r θ, halves r θ → halvings (r + 1) θ = halvings r θ + 1) →
  (∀ k t, cap (k + 1) t ≤ cap k t) →
  (∀ r, P.real {θ | ∃ D' ∈ poolsAt populations D harvest side kept sideKept r θ,
      (∃ t ∈ θs, cap (halvings r θ) t < D'.real
          {p | A.state p ∈ undecidedStates A O t (family r θ).1 (family r θ).2})
      ∨ cM < D'.real
          {p | A.state p ∈ misreadStates A O (family r θ).1 (family r θ).2}} ≤ δc) →
  (∀ r, MeasurableSet[ℱ (r + 1)]
      (idleRefusal A O family gate kept halves halvings kstar harvest θs cap r)) →
  (∀ r E, MeasurableSet[ℱ r] E →
      P.real (E ∩ idleRefusal A O family gate kept halves halvings kstar harvest θs cap r)
        ≤ δs * P.real E) →
  (∀ r, P.real {θ | gate r θ ∧ fails r θ ∧ ¬ (sideKept r θ ∧ ρ < (side r θ).real
      {p | A.state p ∈ misreadStates A O (family r θ).1 (family r θ).2})} ≤ δa) →
  P.real {θ | ∀ r < (θs.card + 1) * 2 ^ Fintype.card Q + kstar + W, ¬ gate r θ ∨ fails r θ}
    ≤ ((θs.card + 1) * 2 ^ Fintype.card Q + kstar + W) * (δc + δa)
      + (((θs.card + 1) * 2 ^ Fintype.card Q + kstar + W).choose W : ℝ) * δs ^ W

end OrthoDFA
