import OrthoDFA.TallyLoop
import OrthoDFA.RoundEnd

/-!
# One round of the tally loop

The oracle (`TriOracle`): every string's read is accept, reject or undecided, independently across
strings, with a law that depends only on the string's state in the target DFA. A read-state is
good where it reads undecided at most `1.5θ` of the time.

`TallyRoundCorrect`: over `C.budget` probes drawn from a `PrefixDistribution`, the round ends well
(`TallyEndsWell`: consistent with the hypothesis disagreeing on at most `εd` of the probes, or a
harvest more than half of whose strings are at read-states that are not good, with a replay that
is spread and yields) but for chance at most `2δ`, at settings the proof chooses, once the laws'
`κ` and `ε`, the goodness level and the probes' prefix mass are small enough. The replay's yield
is set by how often the draws' prefixes read undecided. It is derived from a detailed bound
(`Proofs/TallyRiskDefs.lean`) that need not be read to trust it.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The oracle: a read of every string, accept, reject or undecided, independent across strings,
each with its state's law in the target `M`. Every law is nonnegative and reads its rarer decided
side at most `κ` times as often as it is undecided, or at most `ε`. -/
structure TriOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : _root_.DFA α σ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : iIndepFun read μ
  dist : σ → ARU → ℝ
  law : ∀ z r, μ.real {ω | read z ω = r} = dist (M.eval z.toList) r
  κ : ℝ
  ε : ℝ
  nonneg : ∀ q v, 0 ≤ dist q v
  rare : ∀ q, min (dist q .accept) (dist q .reject) ≤ max ε (κ * dist q .undecided)

/-- A read-state is good where it reads undecided at most `1.5θ` of the time. -/
def TriOracle.Good {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {σ : Type*}
    {M : _root_.DFA α σ} (O : TriOracle μ M) (θ : ℝ) (q : σ) : Prop :=
  O.dist q .undecided ≤ 3 / 2 * θ

/-- The probes that search: their walk and the cut disagree at a decided read. -/
def searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (rd z).cut) T edges k x with
    | .pair _ | .edge _ _ | .triple _ => True
    | _ => False}

/-- The run from `s` over the draws ends with an ending satisfying `Q`. -/
def RunEnds {S X E : Type*} (step : S → X → S ⊕ (E × S)) (Q : E → S → Prop) :
    S → List X → Prop
  | _, [] => False
  | s, x :: xs => match step s x with
    | .inl s' => RunEnds step Q s' xs
    | .inr (e, s') => Q e s'

/-- The round's ending, as every level of the loop states it: success is consistent, a harvest
is its strings. -/
def tallyEnd : TEnd α → TState α → RoundEnd α
  | .success, _ => .consistent
  | .harvestStart, s => .harvest s.startH
  | .harvest e, s => .harvest (s.harv e.1 e.2)
  | .harvestPT, s => .harvest s.pt
  | .tooBig, _ => .failed

/-- The hypothesis `(T, edges)` replayed on a draw `w` from the anchor `e`: the strings a probe of
`w` walked from `e` first reads undecided. -/
def tallyReplay (rd : FreeMonoid α → ARU) (T : DTree α) (edges : EdgeTable α) (w : FreeMonoid α)
    (e : ℕ) : List (FreeMonoid α) :=
  ((firstsBy (fun z => (rd z).cut) T edges e w).filter fun q => (rd q.2).cut = none).map Prod.snd

/-- The round ends well: consistent with the hypothesis disagreeing on at most `εd` of the probes,
or a harvest more than half of whose strings are at `bad` read-states, its replay from an anchor
drawn uniformly from `a, …, L` spread and yielding `y`. -/
def TallyEndsWell {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop) (D : Measure (FreeMonoid α))
    (C : TallyCfg) (a L : ℕ) (y : ℝ) (rd : FreeMonoid α → ARU) (e : TEnd α) (s : TState α) :
    Prop :=
  EndsWell M bad D C.εd (searchAt rd C.k s.tree s.edges) a L (tallyReplay rd s.tree s.edges) y
    (tallyEnd e s)

/-- The largest mass the probes' law puts on one length-`k` prefix. -/
def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ := ⨆ u, D.real {x | prefixOf x k = u}

/-- A crude bound on `log (8 K / δ)`, `K` the most trees, edge maps and edges a tail is united
over. -/
def tallyLog (σ : Type*) [Fintype σ] (S : ℕ) (δ : ℝ) : ℝ :=
  4 * (Fintype.card σ + S + 2) * Fintype.card α
    * Real.log ((Fintype.card σ + S + 3) * Fintype.card α / δ)

/-- The mean count, over a draw from `P.D`, of the positions from `start` to `L` at which its
prefix reads undecided, a string at the state `q` of `M` reading undecided `dist q .undecided` of
the time. -/
def undecidedCount (P : PrefixDistribution α) {σ : Type*} (M : _root_.DFA α σ)
    (dist : σ → ARU → ℝ) : ℝ :=
  ∑' u : FreeMonoid α, if P.firstAnchor ≤ u.toList.length ∧ u.toList.length ≤ P.L then
    P.D.real {x | u.toList <+: x.toList} * dist (M.eval u.toList) .undecided else 0

/-- `TallyRoundGuarantee M O θ εd P δ S k`: some settings `C`, with `C.k = k`, `C.εd = εd` and a
budget of at most `K · G` probes, make the round fail to end well with chance at most `2δ`, once
the goodness level `θ`, the laws' `κ` and `ε` and the probes' prefix mass at `k` are small enough
for `n = |Q| + S` (at most `S` splits that are not genuine), `A = |Σ|` and the draws' length `L`.
`16ρ` is about the share of the probes at which the round affords a record that is not true. A
harvest's replay then yields `(V − 2√(q V log(1/δ))) / (L + 1 − start)`, `V` being the mean count
of positions from `start` to `L` at which a draw's prefix reads undecided. -/
def TallyRoundGuarantee [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ) : Prop :=
  let n : ℝ := Fintype.card σ + S
  let A : ℝ := Fintype.card α
  let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  let K : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
    + 8 * (n + 3) * A * (P.L + 1)
  let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 4) * A + 7 * K ^ 2) / δ
  let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
  let V := undecidedCount P M O.dist
  let bad := fun q => ¬ O.Good θ q
  0 < δ → δ ≤ 1 → 0 < εd → εd ≤ 1 → 1 ≤ P.L → 0 ≤ θ →
  384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
  0 < O.κ → 4 * O.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
  0 ≤ O.ε → (P.L + 1) * (n + 1) * O.ε + 3 * tallyLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
  ∃ C : TallyCfg, C.k = k ∧ C.εd = εd ∧ (C.budget : ℝ) ≤ K * G ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => P.D)
        {xs | ¬ RunEnds (tallyStep C fun z => (O.read z ω).cut)
          (TallyEndsWell M bad P.D C P.firstAnchor P.L
            ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ))) / (P.L + 1 - P.firstAnchor : ℕ))
            (O.read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ ENNReal.ofReal (2 * δ)

/-- `TallyRoundCorrect`: the round guarantee at every target, oracle, goodness level, accuracy,
probe law, confidence, count of splits that are not genuine, and start `k`. -/
def TallyRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ),
    TallyRoundGuarantee M O θ εd P δ S k

end OrthoDFA

end
