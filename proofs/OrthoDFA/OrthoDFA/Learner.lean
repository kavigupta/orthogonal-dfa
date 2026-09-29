import OrthoDFA.Check
import OrthoDFA.Termination

/-!
# The learner

Each round clusters a family over the populations so far and has the L\* stage build a
hypothesis with it.  A gate compares the hypothesis with the family's cut on fresh draws; past
it, the merge check runs at every state carrying `ε/|R|` of the sampler.  The first round past
the gate in which no state fails is returned, its labels denoised.  Every state carrying `ε/|R|`
of every round's hypothesis becomes a population, and so does what the stage harvests, when a
yield test keeps it.

Everything but the L\* stage is the algorithm modelled in `Clustering`, `Check` and
`ReturnAccuracy`.  The stage is arbitrary, held to reading the noise at few strings, and,
whenever the family cuts nearly all of the sampler well, to agreeing with the family's cut or
harvesting enough of a state the family leaves undecided.

Known modelling gap.  The gate is `estimate_agreement_rate` against `acc_threshold`, which
checks that each draw's walk ends in the leaf the tree sifts it to.  Here only the root's
decision is checked, which is the family's cut, and it is read with the cut's band rather than
`oracle_decider` at `decision_boundary`.  A draw the tree cannot place is left out there and
counts as a disagreement here.

Known modelling gap.  Each round draws its suffixes afresh, where `suffix_pool` persists
across rounds.  A population is only ever judged by suffixes it was not selected with.

Known modelling gap.  `retire_states` drops each round's `("state", leaf)` populations; here
they are kept.

Known modelling gap.  A harvest is a function of a probe: `BoundarySource` walks each probe
through the round's tree and keeps the strings it cannot place.  Here a harvest decides with its
own reads of the probe, never the string it returns.

Known modelling gap.  Each round draws `M` strings per population from the sampler and keeps
those reaching the population's state, where `StateSource` and `SplitSource` sample reaching
strings directly; a population short of draws reads `ε` in their place.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]

instance : Nonempty State := ⟨⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩⟩

/-- The draws of `a`, then `ε`. -/
def padded {M : ℕ} (a : Fin M → S) : ℕ → S := fun i => if h : i < M then a ⟨i, h⟩ else 1

open scoped Classical in
/-- The draws of `a` reaching `h`, in order, then `ε`. -/
noncomputable def hitsIn (H : DFA S R) (h : R) {M : ℕ} (a : Fin M → S) : ℕ → S :=
  fun i => ((List.ofFn a).filter fun p => H.state p = h).getD i 1

/-- What a harvest makes of a probe, reading the oracle as it goes: a string to keep, or
nothing. -/
abbrev Harvester (S : Type*) := S → (S → ℝ) → Option S

/-- A round's outcome: its hypothesis, whether it got past the gate, the states the check
failed, and its harvest if the yield test kept it. -/
abbrev Outcome (S R : Type*) [Stringlike S] := DFA S R × Prop × Finset R × Option (Harvester S)

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
    | some (i, none) => ∃ o ∈ past[i.val]?, o.2.2.2.isSome

/-- Population `j`'s draws out of the sampler's draws `a`. -/
noncomputable def populationDraws (O : Oracle μ S) (ω : Ω) {K : ℕ} (past : List (Outcome S R))
    (j : Pop K R) {M : ℕ} (a : Fin M → S) : ℕ → S :=
  match j with
  | none => padded a
  | some (i, some h) => match past[i.val]? with
    | some o => hitsIn o.1 h a
    | none => padded a
  | some (i, none) => match past[i.val]? with
    | some (_, _, _, some hv) => harvestedIn O ω hv a
    | _ => padded a

/-- What the clustering draws: suffixes, then prefixes and certification prefixes per
population. -/
abbrev ClusterDraws (S : Type*) (K : ℕ) (R : Type*) :=
  ((ℕ → S) × (Pop K R → ℕ → S)) × (Pop K R → ℕ → S)

/-- One round's draws: `M` suffixes from `Dsf`, `M` sampler draws per population for its
prefixes and as many for its certification prefixes, the stage's own draws, the gate's `G`
sampler draws, the yield test's `Y`, and the check's `N` sampler draws and `m` suffixes. -/
abbrev RoundDraws (S : Type*) (K : ℕ) (R Xs : Type*) (M G Y N m : ℕ) :=
  (Fin M → S) × ((Pop K R → Fin M → S) × (Pop K R → Fin M → S))
    × Xs × (Fin G → S) × (Fin Y → S) × ((Fin N → S) × (Fin m → S))

/-- The family's cut at `B` decides `p`, and the way `H` labels it. -/
def cutAgrees (O : Oracle μ S) (B : State) (F : Finset S) (H : DFA S R) (p : S) (ω : Ω) :
    Prop :=
  (B.hi < voteCount O.mq F p ω ∧ H.state p ∈ H.accept)
    ∨ (voteCount O.mq F p ω ≤ B.lo ∧ H.state p ∉ H.accept)

/-- The share of the sampler where `H` and the family's cut do not agree. -/
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

/-- The learner's knobs outside the clustering, and its L\* stage: `stage F B c ω s` is the
hypothesis built from the cut of `F` at `B`, the round's clustering draws `c`, the noise, and the
stage's own draws `s`, and `harvest F B c ω s` what it harvests. -/
structure Learner (Ω S R Xs : Type*) [Stringlike S] (K : ℕ) where
  stage : Finset S → State → ClusterDraws S K R → Ω → Xs → DFA S R
  harvest : Finset S → State → ClusterDraws S K R → Ω → Xs → Harvester S
  /-- Only states carrying `heavy/|R|` of the sampler are checked. -/
  heavy : ℝ
  M : ℕ
  /-- The gate lets a round past when at most `gth` of its `G` draws disagree. -/
  G : ℕ
  gth : ℕ
  /-- The harvest is kept when it keeps at least `yth` of `Y` draws. -/
  Y : ℕ
  yth : ℕ
  N : ℕ
  m : ℕ
  n : ℕ
  t : ℝ
  /-- Draws per state when denoising. -/
  nd : ℕ

/-- One round after the rounds `past`.  The family is cut at some state of the schedule the
clustering's gate returns at. -/
noncomputable def round (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : Finset (Pop K R) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (Outcome S R)) (ω : Ω)
    (d : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) : Outcome S R :=
  let pops := populationsAfter Dsamp L.heavy past
  let c : ClusterDraws S K R :=
    ((padded d.1, fun j => populationDraws O ω past j (d.2.1.1 j)),
      fun j => populationDraws O ω past j (d.2.1.2 j))
  let B := Classical.epsilon fun B =>
    B ∈ schedule pops ∧ ((ω, c) : Run Ω S (Pop K R)) ∈ ret O.mq pops indecisionLimit α B
  let F := clusterAt O.mq pops (ω, c) B
  let H := L.stage F B c ω d.2.2.1
  let hv := L.harvest F B c ω d.2.2.1
  open scoped Classical in
  (H, ((Finset.univ.filter fun i => ¬ cutAgrees O B F H (d.2.2.2.1 i) ω).card ≤ L.gth),
    (Finset.univ.filter fun h =>
      L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
        ∧ checkFails O H L.n L.t h d.2.2.2.2.2 ω),
    if L.yth ≤ (Finset.univ.filter fun i =>
        (hv (d.2.2.2.2.1 i) fun t => O.mq t ω).isSome).card then some hv else none)

/-- The first `r` rounds. -/
noncomputable def history (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : Finset (Pop K R) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) :
    ℕ → List (Outcome S R)
  | 0 => []
  | r + 1 =>
    if h : r < K then
      history O Dsamp schedule indecisionLimit α L ω d r
        ++ [round O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L ω d r) ω (d ⟨r, h⟩)]
    else history O Dsamp schedule indecisionLimit α L ω d r

/-- The noise, each round's draws, and one sampler stream per state for denoising. -/
noncomputable def learnerMeasure (Dsamp Dsf : Measure S) {K : ℕ} {Xs : Type*}
    [MeasurableSpace Xs] (νs : Measure Xs) (L : Learner Ω S R Xs K) :
    Measure (Ω × (Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) × (Fin K → R → ℕ → S)) :=
  μ.prod ((Measure.pi fun _ : Fin K =>
      (Measure.pi fun _ : Fin L.M => Dsf).prod
        (((Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp).prod
            (Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp)).prod
          (νs.prod ((Measure.pi fun _ : Fin L.G => Dsamp).prod
            ((Measure.pi fun _ : Fin L.Y => Dsamp).prod
              ((Measure.pi fun _ : Fin L.N => Dsamp).prod
                (Measure.pi fun _ : Fin L.m => Dsf))))))).prod
    (Measure.pi fun _ : Fin K => Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp))

/-- The E-L\* learner is correct, given an L\* stage that, whenever the family cuts nearly all
of the sampler well, agrees with the family's cut or harvests enough of a state the family
leaves undecided.  With probability at most the bound below, either none of the
`K ≥ 2·|Q| + 1` rounds returns, or one returns a hypothesis less than `1 − w − ε` accurate once
denoised.

The clustering's parameters are `ClusteringQualityGuarantee`'s, and a schedule exists for them.
The stage reads the noise at at most `Ts` strings, and its harvest at at most `Th` per probe, so
every round reads it at at most `T`.  A harvest's reads, and what it keeps, are spread over the
sampler (`κh`, `κa`); what it keeps is a prefix, never a string it read.  Given the reads so far,
if the family cuts less than `x₀` of the sampler badly, the stage both disagrees with its cut on
more than `ζ₀` of the sampler and fails to keep, at yield `π₁`, more than `θh` of some state the
family leaves undecided, with probability at most `δs`.  Per round, the bound charges:
* the clustering: its `δc`, a population short of draws, a harvest kept at yield below `π₀`,
  and a read landing where another read did;
* the stage `δs`, the gate refusing it all the same, and the yield test dropping a harvest of
  yield `π₁`;
* the gate letting through a hypothesis that disagrees with the cut on more than `ζ₂`, and the
  cut itself misreading more than `ζ₁` of what it cuts well;
* the check failing a state whose minority share is below `wₛ` (`CheckGuarantee`);
* the return: `ReturnAccuracy`'s `δ + |R|·β`, with `CheckGuarantee`'s `β`. -/
def LearnerCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]
    (A : DFA S Q) (O : Oracle μ S) (Pre Suf : Set S) (K : ℕ)
    (η₀ indecisionLimit εcov α δc pAP tolerance : ℝ),
  O.L = {w | A.state w ∈ A.accept} →
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  Flat Pre Suf →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δc →
  δc ≤ 1 →
  0 < tolerance →
  ∃ cap : ℝ,
    0 < cap ∧
    ∀ (Dsamp Dsf : Measure S) (ε κ : ℝ),
      IsProbabilityMeasure Dsamp → IsProbabilityMeasure Dsf →
      Dsamp Preᶜ = 0 →
      Dsf Sufᶜ = 0 →
      (∀ a, Dsamp.real {a} ≤ κ) →
      pAP ≤ Dsf.real {v | v ≠ 1 ∧ ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      0 < ε →
      ε ≤ 1 →
      κ * Fintype.card R / ε ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ schedule : Finset (Pop K R) → Finset State,
        ∀ {Xs : Type*} [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
          (νs : Measure Xs) (L : Learner Ω S R Xs K)
          (stageReads : Finset S → State → ClusterDraws S K R → Ω → Xs → Finset S)
          (harvestReads : Finset S → State → ClusterDraws S K R → Ω → Xs → S → (S → ℝ) →
            Finset S)
          (P Ts Th : ℕ) (ζ₀ ζ₁ ζ₂ x₀ wₛ w δs δ κf κh κa π₀ π₁ θh : ℝ),
        IsProbabilityMeasure νs →
        L.heavy = ε →
        (∀ a, Dsf.real {a} ≤ κf) →
        (∀ pops, ∀ B ∈ schedule pops, B.npref ≤ P ∧ B.nsuff ≤ L.M) →
        (P : ℝ) ≤ L.M * ε / Fintype.card R →
        (P : ℝ) ≤ L.M * π₀ →
        (∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
          (fun ω => (L.stage F B c ω s, L.harvest F B c ω s))) →
        (∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts) →
        (∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
          harvestReads F B c ω s p f' = harvestReads F B c ω s p f
            ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f) →
        (∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th) →
        (∀ F B c ω s f t, Dsamp.real {p | t ∈ harvestReads F B c ω s p f} ≤ κh) →
        (∀ F B c ω s f a, Dsamp.real {p | L.harvest F B c ω s p f = some a} ≤ κa) →
        (∀ F B c ω s p f a, L.harvest F B c ω s p f = some a →
          a ∈ Pre ∧ a ∉ harvestReads F B c ω s p f) →
        0 < π₀ →
        π₀ * L.Y ≤ L.yth →
        (L.yth : ℝ) ≤ π₁ * L.Y →
        κa ≤ π₀ * (κ * Fintype.card R / ε) →
        2 * εcov / tolerance ≤ θh →
        4 * indecisionLimit / tolerance ≤ θh →
        let J := Fintype.card (Pop K R)
        let T := K * (2 * J * L.M * (L.M + 1) + Ts + 2 * J * L.M * Th + L.G * (L.M + 1)
          + L.Y * Th + L.N * (L.m + 1))
        (∀ F B c (U : Finset S) b, U.card ≤ T →
          badlyCut O tolerance B F Dsamp < x₀ →
          ((μ[|pinned O U b]).prod νs).real
              {q | ζ₀ < cutDisagreement O B F (L.stage F B c q.1 q.2) Dsamp q.1
                ∧ ¬ (π₁ ≤ harvestYield O (L.harvest F B c q.1 q.2) Dsamp
                  ∧ ∃ qq, (∃ p, A.state p = qq
                      ∧ tolerance < undecidedProb O B.lo B.hi F p)
                    ∧ θh < stateMass A (harvestLaw O (L.harvest F B c q.1 q.2) Dsamp) qq)}
            ≤ δs) →
        0 ≤ δs →
        ζ₀ * L.G ≤ L.gth →
        (L.gth : ℝ) ≤ ζ₂ * L.G →
        0 ≤ ζ₂ →
        0 < ζ₁ →
        2 * εcov / tolerance < (wₛ - (ζ₂ + ζ₁) * Fintype.card R / ε) / Fintype.card Q →
        4 * indecisionLimit / tolerance
          < (wₛ - (ζ₂ + ζ₁) * Fintype.card R / ε) / Fintype.card Q →
        2 * εcov / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q) →
        4 * indecisionLimit / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q) →
        2 * Fintype.card Q + 1 ≤ K →
        0 ≤ L.t →
        (2 * L.n : ℝ) ≤ L.N * ε / Fintype.card R →
        2 * L.t ≤ L.n * w * (1 - 2 * η₀) ^ 2 →
        0 ≤ w →
        0 < denoiseMargin η₀ w →
        0 < δ →
        Real.log (2 * Fintype.card R / δ) / (2 * denoiseMargin η₀ w ^ 2) ≤ L.nd →
        κ * (Fintype.card R : ℝ) ^ 2 * L.nd * (L.nd + 2 * T) ≤ δ * ε →
        let repeats := κ * Fintype.card R / ε * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T)
        let β := (1 - pAP) ^ L.m
          + Real.exp (-2 * (L.N * ε / Fintype.card R - 2 * L.n) ^ 2 / L.N)
          + Real.exp (-(L.n * w * (1 - 2 * η₀) ^ 2 - 2 * L.t) ^ 2 / (2 * L.n)) + repeats
        (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
            {z | let rounds := history O Dsamp schedule indecisionLimit α L z.1 z.2.1 K
              (∀ o ∈ rounds, ¬ (o.2.1 ∧ o.2.2.1 = ∅))
              ∨ ∃ r : Fin K, ∃ o ∈ rounds[r.val]?, o.2.1 ∧ o.2.2.1 = ∅
                ∧ accuracy A o.1
                    (fun h => denoisedLabel O L.nd (hitsOf o.1 h (z.2.2 r h)) z.1) Dsamp
                  < 1 - w - ε}
          ≤ K * (δc
            + 2 * J * Real.exp (-2 * (L.M * ε / Fintype.card R - P) ^ 2 / L.M)
            + 2 * J * Real.exp (-2 * (L.M * π₀ - P) ^ 2 / L.M)
            + Real.exp (-2 * L.Y * (L.yth / L.Y - π₀) ^ 2)
            + 2 * J * L.M * (L.M + 1) * T * (κ * Fintype.card R / ε)
            + 2 * J * L.M * (T + 2 * J * L.M * (L.M + 1) + 2 * J * L.M * Th) * κh
            + 2 * J * L.M * (L.M + 1) * Th * κf
            + δs + Real.exp (-2 * L.G * (L.gth / L.G - ζ₀) ^ 2)
            + Real.exp (-2 * L.Y * (π₁ - L.yth / L.Y) ^ 2) + L.Y * (T + L.Y * Th) * κh
            + Real.exp (-2 * L.G * (ζ₂ - L.gth / L.G) ^ 2)
            + (tolerance + T * (L.M + 1) * κ) / ζ₁
            + Fintype.card R * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ + repeats)
            + δ + Fintype.card R * β)

end OrthoDFA
