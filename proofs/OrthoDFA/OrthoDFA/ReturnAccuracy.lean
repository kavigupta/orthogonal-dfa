import OrthoDFA.ClusteringQuality
import Mathlib.Data.Nat.Nth

/-!
# What the learner returns is worth

The learner returns a hypothesis only once a merge check passes at each of its states, and
recomputes its accept labels from fresh draws first (`denoise_accept_labels`).  Together they
make it accurate however the L\* stage built it, so the stage here is arbitrary: it need only
read the noise at few strings.

Known modelling gap.  The learner here returns the first round whose check passes.
`counterexample_driven_synthesis` returns the round `BestRound` ranks highest, checks only
rounds that clear `acc_threshold`, and returns unchecked on a stall.

Known modelling gap.  `denoise_accept_labels` tests each state's accept rate against
`decision_boundary` at a significance level and keeps the discovery label when the test does
not decide; here the label is the majority of the reads, against `½`.  It draws distinct
strings by path counting, where these are the sampler's draws that reach the state, and may
repeat.

Known modelling gap.  The merge check is held to its power and nothing else.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]

/-- The sampler conditioned on reaching `h`; the sampler itself where nothing reaches it. -/
noncomputable def reaching (H : DFA S R) (Dsamp : Measure S) (h : R) : Measure S :=
  if Dsamp {w | H.state w = h} = 0 then Dsamp else Dsamp[|{w | H.state w = h}]

/-- The share of the strings reaching `h` whose true label is the rarer one. -/
noncomputable def minorityShare (A : DFA S Q) (H : DFA S R) (Dsamp : Measure S) (h : R) : ℝ :=
  min ((reaching H Dsamp h).real {w | A.state w ∈ A.accept})
    ((reaching H Dsamp h).real {w | A.state w ∉ A.accept})

/-- `h` is labelled accepting when more than half of the reads at its `n` draws accept. -/
def denoisedLabel (O : Oracle μ S) (n : ℕ) (draws : ℕ → S) (ω : Ω) : Prop :=
  n < 2 * ((Finset.range n).filter fun i => O.mq (draws i) ω = 1).card

/-- The chance a sampled string is labelled the way the target labels it. -/
noncomputable def accuracy (A : DFA S Q) (H : DFA S R) (label : R → Prop)
    (Dsamp : Measure S) : ℝ :=
  Dsamp.real {w | label (H.state w) ↔ A.state w ∈ A.accept}

/-- How far from `½` the reads of a state sit when its minority share is at most `w`. -/
noncomputable def denoiseMargin (η₀ w : ℝ) : ℝ := 1 / 2 - η₀ - w * (1 - η₀)

open scoped Classical in
/-- The draws of `u` that reach `h`, in order; `u` itself when only finitely many do. -/
noncomputable def hitsOf (H : DFA S R) (h : R) (u : ℕ → S) : ℕ → S :=
  if {i | H.state (u i) = h}.Infinite then fun k => u (Nat.nth (fun i => H.state (u i) = h) k)
  else u

/-- The noise at `U` is `b`. -/
def pinned (O : Oracle μ S) (U : Finset S) (b : S → ℝ) : Set Ω :=
  {ω | ∀ s ∈ U, O.noise s ω = b s}

/-- `f` reads the noise only at `Q`, choosing `Q` as it goes: whatever agrees with `ω` on
`Q ω` has the same `Q` and the same `f`. -/
def ReadsOnly {α : Type*} (O : Oracle μ S) (Q : Ω → Finset S) (f : Ω → α) : Prop :=
  ∀ ω ω', (∀ s ∈ Q ω, O.noise s ω = O.noise s ω') → Q ω' = Q ω ∧ f ω' = f ω

/-- The noise, then per round the merge check's draws and one sampler stream per state. -/
noncomputable def loopMeasure (K : ℕ) (Dsamp : Measure S) {X : Type*} [MeasurableSpace X]
    (ν : Measure X) : Measure (Ω × (Fin K → X) × (Fin K → R → ℕ → S)) :=
  μ.prod ((Measure.pi fun _ : Fin K => ν).prod
    (Measure.pi fun _ : Fin K => Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp))

/-- Round `r` of the L\* stage settles on `stage r`, a hypothesis `hyp (stage r)` and whatever
else the merge check reads, from the noise and the earlier rounds' check draws, having read the
noise at `queried r`: at most `T` strings, however it picks them.  Whatever the stage does, the
chance that some round's check `passes` at every state while its hypothesis, labels denoised
from that round's streams, is less than `1 − w − ε` accurate is at most `K·(δ + |R|·β)`.

The premises are the oracle's noise bound; a minority share `w` small enough that denoising
still reads the majority; a sampler with no atom above `κ`; enough draws `n`, few enough
repeats of each other or of what the stage read, for denoising to be right at every state
carrying at least `ε/|R|`; and a check that passes such a state with minority share at least `w`
with probability at most `β`, whatever noise the stage has read. -/
def ReturnAccuracy : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]
    {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X] {C : Type*}
    (A : DFA S Q) (O : Oracle μ S) (Dsamp : Measure S) (ν : Measure X) (K T : ℕ)
    (stage : Fin K → Ω → (Fin K → X) → C) (hyp : C → DFA S R)
    (queried : Fin K → Ω → (Fin K → X) → Finset S) (passes : C → R → Set (Ω × X))
    (η₀ w ε δ κ β : ℝ) (n : ℕ),
  IsProbabilityMeasure Dsamp → IsProbabilityMeasure ν →
  O.L = {v | A.state v ∈ A.accept} →
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  0 ≤ w →
  0 < denoiseMargin η₀ w →
  0 < ε →
  0 < δ →
  (∀ a, Dsamp.real {a} ≤ κ) →
  Real.log (2 * Fintype.card R / δ) / (2 * denoiseMargin η₀ w ^ 2) ≤ n →
  κ * (Fintype.card R : ℝ) ^ 2 * n * (n + 2 * T) ≤ δ * ε →
  0 ≤ β →
  (∀ r ω x x', (∀ i < r, x i = x' i) →
    stage r ω x = stage r ω x' ∧ queried r ω x = queried r ω x') →
  (∀ r x, ReadsOnly O (fun ω => queried r ω x) (fun ω => stage r ω x)) →
  (∀ r ω x, (queried r ω x).card ≤ T) →
  (∀ c h U b, U.card ≤ T →
    ε / Fintype.card R ≤ Dsamp.real {v | (hyp c).state v = h} →
    w ≤ minorityShare A (hyp c) Dsamp h →
    ((μ[|pinned O U b]).prod ν).real (passes c h) ≤ β) →
  (loopMeasure (μ := μ) (R := R) K Dsamp ν).real
      {z | ∃ r, (∀ h, (z.1, z.2.1 r) ∈ passes (stage r z.1 z.2.1) h)
        ∧ accuracy A (hyp (stage r z.1 z.2.1))
            (fun h => denoisedLabel O n (hitsOf (hyp (stage r z.1 z.2.1)) h (z.2.2 r h)) z.1)
            Dsamp
          < 1 - w - ε}
    ≤ K * (δ + Fintype.card R * β)

end OrthoDFA
