import OrthoDFA.Certificate
import OrthoDFA.Termination

/-!
# The learner

Each round clusters a family over the populations so far and has the L\* stage build a
hypothesis with it.  A gate compares the hypothesis with the family's cut on fresh draws; past
it, the hypothesis is denoised and the certificate reads it.  The first round the certificate
passes is returned.  Every state carrying `ε/|R|` of every round's hypothesis becomes a
population, and so does what the stage harvests, when a yield test keeps it.  A round whose pass
left most of its last probes unchecked halves the indecision limit every later family is held to.

Everything but the L\* stage is the algorithm modelled in `Clustering`, `Certificate` and
`ReturnAccuracy`.

Known modelling gap.  The gate is `estimate_agreement_rate` against `acc_threshold`, which
checks that each draw's walk ends in the leaf the tree sifts it to.  Here only the root's
decision is checked: the family's vote, read at the middle of the cut's band as `oracle_decider`
reads it at `decision_boundary`, so every draw is decided.

Known modelling gap.  Each round draws its suffixes afresh, where `suffix_pool` persists
across rounds.  A population is only ever judged by suffixes it was not selected with.

Known modelling gap.  `retire_states` drops each round's `("state", leaf)` populations; here
they are kept.

Known modelling gap.  A harvest is a function of a probe: `BoundarySource` walks each probe
through the round's tree and keeps the strings it cannot place.  Here a harvest decides with its
own reads of the probe, never the string it returns.

Known modelling gap.  A run also stops once its rounds stall, and `CERTIFICATE_PATIENCE` rounds
after the gate first lets one past; here the rounds run on.

Known modelling gap.  Each round draws `M` strings per population from the sampler and keeps
those reaching the population's state, where `StateSource` and `SplitSource` sample reaching
strings directly; a population short of draws reads `ε` in their place.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]

instance : Nonempty State := ⟨⟨0, 0, 0, 0, 0, 0, 0⟩⟩

/-- The draws of `a`, then `ε`. -/
def padded {M : ℕ} (a : Fin M → S) : ℕ → S := fun i => if h : i < M then a ⟨i, h⟩ else 1

open scoped Classical in
/-- The draws of `a` reaching `h`, in order, then `ε`. -/
noncomputable def hitsIn (H : DFA S R) (h : R) {M : ℕ} (a : Fin M → S) : ℕ → S :=
  fun i => ((List.ofFn a).filter fun p => H.state p = h).getD i 1

/-- What a harvest makes of a probe, reading the oracle as it goes: a string to keep, or
nothing. -/
abbrev Harvester (S : Type*) := S → (S → ℝ) → Option S

/-- A round's outcome: its hypothesis, denoised; whether it got past the gate; whether the
certificate passed it; its harvest if the yield test kept it; and whether its pass halves the
limits. -/
abbrev Outcome (S R : Type*) [Stringlike S] :=
  DFA S R × Prop × Prop × Option (Harvester S) × Prop

/-- A population: the sampler, `none`; the strings reaching state `h` of round `i`'s hypothesis,
`some (i, some h)`; or round `i`'s harvest, `some (i, none)`. -/
abbrev Pop (K : ℕ) (R : Type*) := Option (Fin K × Option R)

/-- What `hv` keeps of the draws of `a`, in order, then `ε`. -/
noncomputable def harvestedIn (O : Oracle μ S) (ω : Ω) (hv : Harvester S) {M : ℕ}
    (a : Fin M → S) : ℕ → S :=
  fun i => ((List.ofFn a).filterMap fun s => hv s fun t => O.mq t ω).getD i 1

open scoped Classical in
/-- After the rounds `past`: the sampler; each state of a round's hypothesis carrying
`heavy/|R|` of the sampler; and each harvest kept. -/
noncomputable def populationsAfter {K : ℕ} (Dsamp : Measure S) (heavy : ℝ)
    (past : List (Outcome S R)) : Finset (Pop K R) :=
  Finset.univ.filter fun j => match j with
    | none => True
    | some (i, some h) => ∃ o ∈ past[i.val]?,
      heavy / Fintype.card R ≤ Dsamp.real {v | o.1.state v = h}
    | some (i, none) => ∃ o ∈ past[i.val]?, o.2.2.2.1.isSome

/-- Population `j`'s draws out of the sampler's draws `a`. -/
noncomputable def populationDraws (O : Oracle μ S) (ω : Ω) {K : ℕ} (past : List (Outcome S R))
    (j : Pop K R) {M : ℕ} (a : Fin M → S) : ℕ → S :=
  match j with
  | none => padded a
  | some (i, some h) => match past[i.val]? with
    | some o => hitsIn o.1 h a
    | none => padded a
  | some (i, none) => match past[i.val]? with
    | some (_, _, _, some hv, _) => harvestedIn O ω hv a
    | _ => padded a

open scoped Classical in
/-- How many of the rounds `past` halved the limits. -/
noncomputable def halvingsIn (past : List (Outcome S R)) : ℕ :=
  (past.filter fun o => decide o.2.2.2.2).length

/-- What the clustering draws: suffixes, then prefixes and certification prefixes per
population. -/
abbrev ClusterDraws (S : Type*) (K : ℕ) (R : Type*) :=
  ((ℕ → S) × (Pop K R → ℕ → S)) × (Pop K R → ℕ → S)

/-- One round's draws: `M` suffixes from `Dsf`, `M` sampler draws per population for its
prefixes and as many for its certification prefixes, the stage's own draws, the gate's `G`
sampler draws, the yield test's `Y`, one sampler stream per state for denoising, and the
certificate's sampler stream. -/
abbrev RoundDraws (S : Type*) (K : ℕ) (R Xs : Type*) (M G Y : ℕ) :=
  (Fin M → S) × ((Pop K R → Fin M → S) × (Pop K R → Fin M → S))
    × Xs × (Fin G → S) × (Fin Y → S) × ((R → ℕ → S) × (ℕ → S))

/-- The family's vote on `p`, read at the middle of the cut's band, is the way `H` labels it. -/
def cutAgrees (O : Oracle μ S) (B : State) (F : Finset S) (H : DFA S R) (p : S) (ω : Ω) :
    Prop :=
  (B.lo + B.hi < 2 * voteCount O.mq F p ω ↔ H.state p ∈ H.accept)

/-- The share of the sampler where `H` and the family's vote do not agree. -/
noncomputable def cutDisagreement (O : Oracle μ S) (B : State) (F : Finset S) (H : DFA S R)
    (Dsamp : Measure S) (ω : Ω) : ℝ :=
  Dsamp.real {p | ¬ cutAgrees O B F H p ω}

/-- The chance `hv` keeps a sampler draw, read with fresh noise. -/
noncomputable def harvestYield (O : Oracle μ S) (hv : Harvester S) (Dsamp : Measure S) : ℝ :=
  (Dsamp.prod μ).real {q | (hv q.1 fun t => O.mq t q.2).isSome}

/-- What `hv` keeps of a sampler draw read with fresh noise, given that it keeps it. -/
noncomputable def harvestLaw (O : Oracle μ S) (hv : Harvester S) (Dsamp : Measure S) :
    Measure S :=
  Measure.sum fun a : S => ENNReal.ofReal
      ((Dsamp.prod μ).real {q | hv q.1 (fun t => O.mq t q.2) = some a}
        / harvestYield O hv Dsamp)
    • Measure.dirac a

/-- `H` with each state labelled by the majority of `nd` reads of the strings of `u h` reaching
it: `denoise_accept_labels`. -/
noncomputable def denoised (O : Oracle μ S) (H : DFA S R) (nd : ℕ) (u : R → ℕ → S) (ω : Ω) :
    DFA S R :=
  { H with accept := {h | denoisedLabel O nd (hitsOf H h (u h)) ω} }

/-- The learner's knobs outside the clustering, and its L\* stage: `stage F B c ω s` is the
hypothesis built from the cut of `F` at `B`, the round's clustering draws `c`, the noise, and the
stage's own draws `s`; `harvest F B c ω s` is what it harvests, and `halves F B c ω s` whether most
of its pass's last probes went unchecked. -/
structure Learner (Ω S R Xs : Type*) [Stringlike S] (K : ℕ) where
  stage : Finset S → State → ClusterDraws S K R → Ω → Xs → DFA S R
  harvest : Finset S → State → ClusterDraws S K R → Ω → Xs → Harvester S
  halves : Finset S → State → ClusterDraws S K R → Ω → Xs → Prop
  /-- Only states carrying `heavy/|R|` of the sampler become populations. -/
  heavy : ℝ
  M : ℕ
  /-- The gate lets a round past when at most `gth` of its `G` draws disagree. -/
  G : ℕ
  gth : ℕ
  /-- The harvest is kept when it keeps at least `yth` of `Y` draws. -/
  Y : ℕ
  yth : ℕ
  /-- Draws per state when denoising. -/
  nd : ℕ
  /-- The certificate's signal, error, and level spread over the rounds. -/
  signal : ℝ
  certError : ℝ
  certLevel : ℝ

/-- One round after the rounds `past`, at the limit halved once for each of them that halved
it.  The family is cut at some state of the schedule the clustering's gate returns at. -/
noncomputable def round (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : ℕ → Finset (Pop K R) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (Outcome S R)) (ω : Ω)
    (d : RoundDraws S K R Xs L.M L.G L.Y) : Outcome S R :=
  let pops := populationsAfter Dsamp L.heavy past
  let k := halvingsIn past
  let c : ClusterDraws S K R :=
    ((padded d.1, fun j => populationDraws O ω past j (d.2.1.1 j)),
      fun j => populationDraws O ω past j (d.2.1.2 j))
  let B := Classical.epsilon fun B =>
    B ∈ schedule k pops ∧ ((ω, c) : Run Ω S (Pop K R)) ∈ ret O.mq pops none (indecisionLimit / 2 ^ k) α B
  let F := familyAt O.mq pops (ω, c) B
  let H := L.stage F B c ω d.2.2.1
  let hv := L.harvest F B c ω d.2.2.1
  let Hd := denoised O H L.nd d.2.2.2.2.2.1 ω
  open scoped Classical in
  (Hd, ((Finset.univ.filter fun i => ¬ cutAgrees O B F H (d.2.2.2.1 i) ω).card ≤ L.gth),
    certifies O Hd L.signal (lookLevel L.certLevel past.length) L.certError d.2.2.2.2.2.2 ω,
    (if L.yth ≤ (Finset.univ.filter fun i =>
        (hv (d.2.2.2.2.1 i) fun t => O.mq t ω).isSome).card then some hv else none),
    L.halves F B c ω d.2.2.1)

/-- The first `r` rounds. -/
noncomputable def history (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : ℕ → Finset (Pop K R) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.Y) :
    ℕ → List (Outcome S R)
  | 0 => []
  | r + 1 =>
    if h : r < K then
      history O Dsamp schedule indecisionLimit α L ω d r
        ++ [round O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L ω d r) ω (d ⟨r, h⟩)]
    else history O Dsamp schedule indecisionLimit α L ω d r

/-- The noise and each round's draws. -/
noncomputable def learnerMeasure (Dsamp Dsf : Measure S) {K : ℕ} {Xs : Type*}
    [MeasurableSpace Xs] (νs : Measure Xs) (L : Learner Ω S R Xs K) :
    Measure (Ω × (Fin K → RoundDraws S K R Xs L.M L.G L.Y)) :=
  μ.prod (Measure.pi fun _ : Fin K =>
      (Measure.pi fun _ : Fin L.M => Dsf).prod
        (((Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp).prod
            (Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp)).prod
          (νs.prod ((Measure.pi fun _ : Fin L.G => Dsamp).prod
            ((Measure.pi fun _ : Fin L.Y => Dsamp).prod
              ((Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp).prod
                (Measure.infinitePi fun _ : ℕ => Dsamp)))))))

end OrthoDFA
