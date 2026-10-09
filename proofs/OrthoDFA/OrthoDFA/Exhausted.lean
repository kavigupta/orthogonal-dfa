import OrthoDFA.Spurious

/-!
# How often the round's split tests miss, and how often the round ends exhausted

A split test is in the power case at a key `κ` when it counts only strings nothing but `κ`'s own
tests has read, and its two sides' mean answers part by a margin `τ` beyond its threshold.

`RoundStrongPower`: in the round, which takes no key as not splitting, the chance that at some key
the first test in the power case or splitting is in the power case and does not split is at most
`e^{−τ²}` times the chances, summed over the keys, that there is such a test.

`RoundStrongExhausted`: where the budget covers `|Q| + N₁ + N₂` readings of `nr + np` probes, the
round ends exhausted with chance at most that power term, the chance that `N₁` of its readings
rerun first a draw with a read off its route, the chance that `N₂` rerun first a draw with no
such read whose test is not in the power case and does not split, and the chance of a noisy
split, which is not bounded.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

open scoped Classical in
/-- The counted strings on the side `b`. -/
noncomputable def sideOf (ts : List (FreeMonoid α × Bool)) (b : Bool) : Finset (FreeMonoid α) :=
  ((ts.filter fun p => p.2 = b).map Prod.fst).toFinset

/-- The mean of the oracle's answers over `S`. -/
noncomputable def meanBit (O : Oracle μ (FreeMonoid α)) (S : Finset (FreeMonoid α)) : ℝ :=
  (∑ w ∈ S, μ[O.mq w]) / S.card

/-- `2ab/(a + b)`. -/
noncomputable def harm (a b : ℕ) : ℝ := 2 * a * b / (a + b)

/-- The test of the sides of `ts` at threshold `θ`, with both sides read, misses by a margin `τ`
of signal: `√H` times the gap of the sides' mean answers is at least `√θ + τ`. -/
def PowerCase (O : Oracle μ (FreeMonoid α)) (θ τ : ℝ) (ts : List (FreeMonoid α × Bool)) :
    Prop :=
  (sideOf ts true).Nonempty ∧ (sideOf ts false).Nonempty
    ∧ √θ + τ ≤ √(harm (sideOf ts true).card (sideOf ts false).card)
      * (meanBit O (sideOf ts true) - meanBit O (sideOf ts false))

section Find

variable (C : StrongCfg α) (R : CutReads α)

/-- The accumulator a pass starts from. -/
def passStart (A : RoundAcc α) : RoundAcc α :=
  { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } }

open scoped Classical in
/-- The first step of the pass at which `P` holds, and the accumulator before it. -/
noncomputable def passFind (P : RoundAcc α → FreeMonoid α → Prop) :
    RoundAcc α → List (FreeMonoid α) → Option (RoundAcc α × FreeMonoid α)
  | _, [] => none
  | A, x :: xs =>
    if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then none
    else if P A x then some (A, x) else passFind P (strongStep C R A x) xs

/-- The first step of the round at which `P` holds, and the accumulator before it. -/
noncomputable def roundFind (P : RoundAcc α → FreeMonoid α → Prop) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → Option (RoundAcc α × FreeMonoid α)
  | 0, _, _, _, _ => none
  | n + 1, j, A, first, d =>
    match passFind C R P (passStart A) (first ++ List.ofFn (d 0).1) with
    | some r => some r
    | none =>
      match strongReading C R j A first (d 0) with
      | (_, .inl _) => none
      | (A', .inr lv) => roundFind P n (j + 1) A' lv (Fin.tail d)

end Find

section Test

variable (C : StrongCfg α)

/-- What a step reads before its test. -/
noncomputable def stepPre (F : Finset (FreeMonoid α)) (A : RoundAcc α) (x : FreeMonoid α) :
    Finset (FreeMonoid α) :=
  stepReads C.K F A.s.tree A.s.tree A.s.pool C.k x

/-- The split test the step of `x` from `A` reaches: its key and the strings it counts. -/
noncomputable def stepTest (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    Option (TestKey α × List (FreeMonoid α × Bool)) :=
  match probeOutcome R A.s.tree A.s.edges C.k x with
  | .edge ps fd =>
    (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) C.K.forced C.k x ps
      fd).key.map fun κ => (κ, testStrings C.K R A.s.tree A.s.pool κ.1 κ.2
        (stepSkip C.K R C.k A.s x κ))
  | _ => none

/-- The split test's threshold against `A`'s tree. -/
noncomputable def testThreshold (A : RoundAcc α) : ℝ :=
  Real.log (2 * ((A.s.tree.paths.length * Fintype.card α : ℕ) : ℝ) / C.K.splitFpr)

/-- A test at `κ` from `A` on `x`, counting `ts`: none of its strings read before but by `κ`'s
tests, and its sides' mean answers apart by `τ` beyond its threshold. -/
def FreshCase (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ)
    (κ : TestKey α) (A : RoundAcc α) (x : FreeMonoid α) (ts : List (FreeMonoid α × Bool)) :
    Prop :=
  (∀ p ∈ ts, p.1 ∉ A.s.log ∧ p.1 ∉ A.reads ∧ p.1 ∉ stepPre C F A x
      ∧ ∀ κ', (p.1, κ') ∈ A.s.tested → κ' = κ)
    ∧ PowerCase O (testThreshold C A) τ ts

/-- The step reaches a test at `κ` in the power case. -/
def CaseAt (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ) (κ : TestKey α)
    (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) : Prop :=
  ∃ ts, stepTest C R A x = some (κ, ts) ∧ FreshCase C O F τ κ A x ts

/-- The step reaches a test at `κ` in the power case or one that splits. -/
def Decisive (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ) (κ : TestKey α)
    (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) : Prop :=
  ∃ ts, stepTest C R A x = some (κ, ts) ∧ (FreshCase C O F τ κ A x ts
    ∨ verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k A.s x κ) = .split)

end Test

/-- The readings that rerun live draws: what each starts from and the draw it reruns first. -/
def rerunFirsts (es : List (RoundAcc α × List (FreeMonoid α))) :
    List (RoundAcc α × FreeMonoid α) :=
  es.filterMap fun e => e.2.head?.map (e.1, ·)

/-- `RoundStrongPower`: in the round, which takes no key as not splitting, at some key the first
test in the power case or splitting is in the power case and does not split with chance at most
`e^{−τ²}` times the chances, over the keys, that there is such a test. -/
def RoundStrongPower : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) (τ : ℝ) (Rmax : ℕ)
    (d : Fin Rmax → C.Draws),
    C.K.forced = ∅ → 0 ≤ τ →
    μ {ω | ∃ κ A x, roundFind C (readsAt O B F ω) (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0
          (startAcc C (readsAt O B F ω) seed) [] d = some (A, x)
        ∧ CaseAt C O F τ κ (readsAt O B F ω) A x
        ∧ verdict C.K (readsAt O B F ω) A.s.tree A.s.pool κ.1 κ.2
            (A.s.tree.paths.length * Fintype.card α)
            (stepSkip C.K (readsAt O B F ω) C.k A.s x κ) ≠ .split}
      ≤ ENNReal.ofReal (Real.exp (-τ ^ 2)) * ∑' κ, μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
        ≠ none}

open scoped Classical in
/-- `RoundStrongExhausted`: with the noise and the draws drawn together, and the budget at two
leaves covering `|Q| + N₁ + N₂` readings of `nr + np` probes, the round ends exhausted with chance
at most `RoundStrongPower`'s bound averaged over the draws, the chance that at least `N₁` readings
rerun first a draw with a read off its route in the tree it was drawn against, the chance that at
least `N₂` rerun first a draw with no such read whose step reaches no test in the power case or
splitting, and the chance of a noisy split. The last is not bounded. -/
def RoundStrongExhausted : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (C : StrongCfg α)
    (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (seed : List (FreeMonoid α)) (τ : ℝ)
    (N₁ N₂ Rmax : ℕ),
    C.K.forced = ∅ → 0 ≤ τ → 0 < C.K.patience → Monotone C.budget →
    (Fintype.card Q + N₁ + N₂) * (C.nr + C.np) ≤ C.budget 2 →
    let ν := Measure.pi fun _ : Fin Rmax => C.drawMeasure D
    let R := fun p : Ω × (Fin Rmax → C.Draws) => readsAt O B F p.1
    let reruns := fun p : Ω × (Fin Rmax → C.Draws) =>
      rerunFirsts (strongEntries C (R p) Rmax 0 (startAcc C (R p) seed) [] p.2)
    μ.prod ν {p | (strongRun C (R p) seed Rmax p.2).1 = .exhausted}
      ≤ ENNReal.ofReal (Real.exp (-τ ^ 2)) * ∑' κ, ∫⁻ d, μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
            ≠ none} ∂ν
        + μ.prod ν {p | N₁ ≤ ((reruns p).filter fun e =>
            SpuriousAt (R p) A side rep e.1.s.tree C.k e.2).length}
        + μ.prod ν {p | N₂ ≤ ((reruns p).filter fun e =>
            ¬ SpuriousAt (R p) A side rep e.1.s.tree C.k e.2
              ∧ ¬ ∃ κ, Decisive C O F τ κ (R p) (passStart e.1) e.2).length}
        + μ.prod ν {p | noisySplits (R p) A side rep (strongRun C (R p) seed Rmax p.2).2.1.splits
            ≠ 0}

end OrthoDFA
