import OrthoDFA.Check
import OrthoDFA.Termination

/-!
# The learner

Each round clusters a family over the populations so far, has the L\* stage build a hypothesis
with it, and runs the merge check at every state carrying `ε/|R|` of the sampler.  The first
round in which no state fails is returned, its labels denoised; each state that fails becomes a
population.

Everything but the L\* stage is the algorithm modelled in `Clustering`, `Check` and
`ReturnAccuracy`.  The stage is arbitrary, held to reading the noise at few strings and to
labelling what the family cuts well the way the target does (`mislabelledWellCut`).

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

open scoped Classical in
/-- After the rounds `past`, each a hypothesis and the states the check failed: the sampler,
`none`, and `some (i, h)` for each state `h` round `i` failed. -/
noncomputable def populationsAfter {K : ℕ} (past : List (DFA S R × Finset R)) :
    Finset (Option (Fin K × R)) :=
  Finset.univ.filter fun j => match j with
    | none => True
    | some (i, h) => ∃ o ∈ past[i.val]?, h ∈ o.2

/-- Population `j`'s draws out of the sampler's draws `a`. -/
noncomputable def populationDraws {K : ℕ} (past : List (DFA S R × Finset R))
    (j : Option (Fin K × R)) {M : ℕ} (a : Fin M → S) : ℕ → S :=
  match j with
  | none => padded a
  | some (i, h) => match past[i.val]? with
    | some o => hitsIn o.1 h a
    | none => padded a

/-- What the clustering draws: suffixes, then prefixes and certification prefixes per
population. -/
abbrev ClusterDraws (S : Type*) (K : ℕ) (R : Type*) :=
  ((ℕ → S) × (Option (Fin K × R) → ℕ → S)) × (Option (Fin K × R) → ℕ → S)

/-- One round's draws: `M` suffixes from `Dsf`, `M` sampler draws per population for its
prefixes and as many for its certification prefixes, the stage's own draws, and the check's
`N` sampler draws and `m` suffixes. -/
abbrev RoundDraws (S : Type*) (K : ℕ) (R Xs : Type*) (M N m : ℕ) :=
  (Fin M → S) × ((Option (Fin K × R) → Fin M → S) × (Option (Fin K × R) → Fin M → S))
    × Xs × ((Fin N → S) × (Fin m → S))

/-- The learner's knobs outside the clustering, and its L\* stage: `stage F B c ω s` is the
hypothesis built from the cut of `F` at `B`, the round's clustering draws `c`, the noise, and the
stage's own draws `s`. -/
structure Learner (Ω S R Xs : Type*) [Stringlike S] (K : ℕ) where
  stage : Finset S → State → ClusterDraws S K R → Ω → Xs → DFA S R
  /-- Only states carrying `heavy/|R|` of the sampler are checked. -/
  heavy : ℝ
  M : ℕ
  N : ℕ
  m : ℕ
  n : ℕ
  t : ℝ
  /-- Draws per state when denoising. -/
  nd : ℕ

/-- One round after the rounds `past`: the hypothesis, and the states the check fails.  The
family is cut at some state of the schedule the gate returns at. -/
noncomputable def round (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (DFA S R × Finset R)) (ω : Ω)
    (d : RoundDraws S K R Xs L.M L.N L.m) : DFA S R × Finset R :=
  let pops := populationsAfter past
  let c : ClusterDraws S K R :=
    ((padded d.1, fun j => populationDraws past j (d.2.1.1 j)),
      fun j => populationDraws past j (d.2.1.2 j))
  let B := Classical.epsilon fun B =>
    B ∈ schedule pops ∧ ((ω, c) : Run Ω S (Option (Fin K × R))) ∈ ret O.mq pops indecisionLimit α B
  let H := L.stage (clusterAt O.mq pops (ω, c) B) B c ω d.2.2.1
  open scoped Classical in
  (H, Finset.univ.filter fun h =>
    L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
      ∧ checkFails O H L.n L.t h d.2.2.2 ω)

/-- The first `r` rounds. -/
noncomputable def history (O : Oracle μ S) (Dsamp : Measure S) {K : ℕ} {Xs : Type*}
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.N L.m) :
    ℕ → List (DFA S R × Finset R)
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
    Measure (Ω × (Fin K → RoundDraws S K R Xs L.M L.N L.m) × (Fin K → R → ℕ → S)) :=
  μ.prod ((Measure.pi fun _ : Fin K =>
      (Measure.pi fun _ : Fin L.M => Dsf).prod
        (((Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin L.M => Dsamp).prod
            (Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin L.M => Dsamp)).prod
          (νs.prod ((Measure.pi fun _ : Fin L.N => Dsamp).prod
            (Measure.pi fun _ : Fin L.m => Dsf))))).prod
    (Measure.pi fun _ : Fin K => Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp))

/-- The E-L\* learner is correct, given an L\* stage that labels what the family cuts well.
With probability at most the bound below, either all `K ≥ 2·|Q| + 1` rounds fail the check, or
some round passes it with a hypothesis less than `1 − w − ε` accurate once denoised.

The clustering's parameters are `ClusteringQualityGuarantee`'s, and a schedule exists for them.
The stage reads the noise at at most `Ts` strings, so every round reads it at at most `T`.
Given the reads so far, it labels more than `ζ` of the sampler the family cuts well unlike the
target with probability at most `δs`.  Per round, the bound charges:
* the clustering: its `δc`, a population short of draws, and a read landing where a previous
  round read;
* the stage: `δs`;
* the check failing a state whose minority share is below `wₛ` (`CheckGuarantee`);
* the return: `ReturnAccuracy`'s `δ + |R|·β`, with `CheckGuarantee`'s `β`. -/
def LearnerCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]
    (A : DFA S Q) (O : Oracle μ S) (Pre : Set S) (K : ℕ)
    (η₀ indecisionLimit εcov α δc pAP tolerance : ℝ),
  O.L = {w | A.state w ∈ A.accept} →
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  Flat Pre →
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
      (∀ a, Dsamp.real {a} ≤ κ) →
      pAP ≤ Dsf.real {v | v ≠ 1 ∧ ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      0 < ε →
      ε ≤ 1 →
      κ * Fintype.card R / ε ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ schedule : Finset (Option (Fin K × R)) → Finset State,
        ∀ {Xs : Type*} [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
          (νs : Measure Xs) (L : Learner Ω S R Xs K)
          (stageReads : Finset S → State → ClusterDraws S K R → Ω → Xs → Finset S)
          (P Ts : ℕ) (ζ wₛ w δs δ : ℝ),
        IsProbabilityMeasure νs →
        L.heavy = ε →
        (∀ pops, ∀ B ∈ schedule pops, B.npref ≤ P ∧ B.nsuff ≤ L.M) →
        (P : ℝ) ≤ L.M * ε / Fintype.card R →
        (∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s)) →
        (∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts) →
        let T := K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts
          + L.N * (L.m + 1))
        (∀ F B c (U : Finset S) b, U.card ≤ T →
          ((μ[|pinned O U b]).prod νs).real
              {q | ζ < mislabelledWellCut A O tolerance B F (L.stage F B c q.1 q.2) Dsamp}
            ≤ δs) →
        0 ≤ ζ →
        2 * εcov / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
        4 * indecisionLimit / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
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
              (∀ o ∈ rounds, o.2.Nonempty)
              ∨ ∃ r : Fin K, ∃ o ∈ rounds[r.val]?, o.2 = ∅
                ∧ accuracy A o.1
                    (fun h => denoisedLabel O L.nd (hitsOf o.1 h (z.2.2 r h)) z.1) Dsamp
                  < 1 - w - ε}
          ≤ K * (δc
            + 2 * Fintype.card (Option (Fin K × R))
              * Real.exp (-2 * (L.M * ε / Fintype.card R - P) ^ 2 / L.M)
            + 2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) * T
              * (κ * Fintype.card R / ε)
            + δs
            + Fintype.card R * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ + repeats)
            + δ + Fintype.card R * β)

end OrthoDFA
