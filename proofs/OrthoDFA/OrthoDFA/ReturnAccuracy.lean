import OrthoDFA.ClusteringQuality

/-!
# What a returned hypothesis is worth

A hypothesis's accept labels are recomputed from fresh draws before it is returned
(`denoise_accept_labels`), and a merge check reads each of its states.  Together they make
the hypothesis accurate however it was built, so nothing here depends on the L\* stage that
built it.

Known modelling gap.  `denoise_accept_labels` tests each state's accept rate against
`decision_boundary` at a significance level and keeps the discovery label when the test does
not decide; here the label is the majority of the reads, against `½`.  It also draws distinct
strings, where these draws are independent and may repeat.

Known modelling gap.  The merge check is held to its power and nothing else: a state whose
minority share is at least `w` passes it with probability at most `β`.
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

/-- The noise, `n` draws at every state of `H`, and whatever the merge check draws. -/
noncomputable def returnMeasure (H : DFA S R) (Dsamp : Measure S) {X : Type*}
    [MeasurableSpace X] (ν : Measure X) : Measure (Ω × (R → ℕ → S) × X) :=
  μ.prod ((Measure.pi fun h => Measure.infinitePi fun _ : ℕ => reaching H Dsamp h).prod ν)

/-- Whatever hypothesis `H` the L\* stage built, it is rarely both passed by the merge check at
every state and less than `1 − w − ε` accurate once its labels are denoised.

The premises are the oracle's noise bound; a minority share `w` small enough that denoising
still reads the majority; a sampler with no atom above `κ`; and enough draws `n`, few enough
repeats, for denoising to be right at every state carrying at least `ε/|R|`. -/
def ReturnAccuracy : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]
    {X : Type*} [MeasurableSpace X]
    (A : DFA S Q) (H : DFA S R) (O : Oracle μ S) (Dsamp : Measure S) (ν : Measure X)
    (passes : R → Set (Ω × (R → ℕ → S) × X)) (η₀ w ε δ κ β : ℝ) (n : ℕ),
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
  κ * (Fintype.card R : ℝ) ^ 2 * (n : ℝ) ^ 2 ≤ δ * ε →
  0 ≤ β →
  (∀ h, w ≤ minorityShare A H Dsamp h →
    (returnMeasure (μ := μ) H Dsamp ν).real (passes h) ≤ β) →
  (returnMeasure (μ := μ) H Dsamp ν).real
      {z | (∀ h, z ∈ passes h)
        ∧ accuracy A H (fun h => denoisedLabel O n (z.2.1 h) z.1) Dsamp < 1 - w - ε}
    ≤ δ + Fintype.card R * β

end OrthoDFA
