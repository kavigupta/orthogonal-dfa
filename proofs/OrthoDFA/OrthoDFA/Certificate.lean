import OrthoDFA.ReturnAccuracy

/-!
# The certificate

`certifies`: look `k` reads the first `|R| · 2^k` strings of a sampler stream through the
hypothesis, and certifies when the error bound over Clopper–Pearson intervals on each state's
share of the draws and on its rate of reading `1` is at most the error asked for.  It refuses once
the bound at the rates read is further above that error than the intervals' reach, or once the
intervals are too narrow for the bound to come down.

Known modelling gap.  `worst_contributions` hands the mass free of the lower bounds to the worst
states in turn, whatever the bounds sum to.  Here a bound whose mass intervals do not straddle `1`
is `1`.
-/

namespace OrthoDFA

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {S : Type*} [Stringlike S] {R : Type*} [Fintype R]

/-- `look_level`: spent over every look, it sums to `α`. -/
noncomputable def lookLevel (α : ℝ) (look : ℕ) : ℝ := α * 6 / (Real.pi * (look + 1)) ^ 2

/-- The Clopper–Pearson lower bound on the rate behind `hits` of `n`. -/
noncomputable def cpLow (hits n : ℕ) (level : ℝ) : ℝ :=
  if hits = 0 then 0 else sInf {p | p ∈ Set.Icc (0 : ℝ) 1 ∧ level / 2 < binomSfGe n p hits}

/-- The Clopper–Pearson upper bound on the rate behind `hits` of `n`. -/
noncomputable def cpHigh (hits n : ℕ) (level : ℝ) : ℝ :=
  if n ≤ hits then 1
  else sSup {p | p ∈ Set.Icc (0 : ℝ) 1 ∧ level / 2 < 1 - binomSfGe n p (hits + 1)}

/-- `error_bound`: the largest `∑ m_h e_h` over masses in `[ml, mh]` summing to `1`, rates in
`[rl, rh]`, and offsets `p` with every rate in `[p, p + band]`, where

    e_h = 1 - (r_h - p) / band  where h accepts,   (r_h - p) / band  where it rejects,
    band = max(gap, max_h rl_h - min_h rh_h). -/
noncomputable def errorBound (ml mh rl rh : R → ℝ) (acc : R → Prop) (gap : ℝ) : ℝ :=
  let band := max gap ((⨆ h, rl h) - ⨅ h, rh h)
  let least := max 0 ((⨆ h, rl h) - band)
  let most := max least (min (1 - band) (⨅ h, rh h))
  open scoped Classical in
  if (∑ h, ml h) ≤ 1 ∧ 1 ≤ ∑ h, mh h then
    sSup {x | ∃ (m r : R → ℝ) (p : ℝ), p ∈ Set.Icc least most
      ∧ (∀ h, m h ∈ Set.Icc (ml h) (mh h)) ∧ ∑ h, m h = 1
      ∧ (∀ h, r h ∈ Set.Icc (rl h) (rh h) ∧ r h ∈ Set.Icc p (p + band))
      ∧ x = ∑ h, m h * (if acc h then 1 - (r h - p) / band else (r h - p) / band)}
  else 1

open scoped Classical in
/-- Of the first `n` strings of `u`, how many reach `h`. -/
noncomputable def drawnAt (H : DFA S R) (u : ℕ → S) (n : ℕ) (h : R) : ℕ :=
  ((Finset.range n).filter fun i => H.state (u i) = h).card

open scoped Classical in
/-- Of the first `n` strings of `u`, how many reach `h` and read `1`. -/
noncomputable def onesAt (O : Oracle μ S) (H : DFA S R) (u : ℕ → S) (ω : Ω) (n : ℕ) (h : R) :
    ℕ :=
  ((Finset.range n).filter fun i => H.state (u i) = h ∧ O.mq (u i) ω = 1).card

/-- Look `k`'s strings. -/
def lookSize (R : Type*) [Fintype R] (k : ℕ) : ℕ := Fintype.card R * 2 ^ k

/-- Look `k`'s bound over the intervals, at level `lookLevel α k / 2|R|` each. -/
noncomputable def lookBound (O : Oracle μ S) (H : DFA S R) (s α : ℝ) (u : ℕ → S) (ω : Ω)
    (k : ℕ) : ℝ :=
  let n := lookSize R k
  let level := lookLevel α k / (2 * Fintype.card R)
  errorBound (fun h => cpLow (drawnAt H u n h) n level) (fun h => cpHigh (drawnAt H u n h) n level)
    (fun h => cpLow (onesAt O H u ω n h) (drawnAt H u n h) level)
    (fun h => cpHigh (onesAt O H u ω n h) (drawnAt H u n h) level)
    (fun h => h ∈ H.accept) (2 * s)

/-- Look `k` refuses: the bound at the rates read is further above `e` than the intervals reach,
or they reach no further than `e / 2`. -/
noncomputable def lookRefuses (O : Oracle μ S) (H : DFA S R) (s α e : ℝ) (u : ℕ → S) (ω : Ω)
    (k : ℕ) : Prop :=
  let n := lookSize R k
  let level := lookLevel α k / (2 * Fintype.card R)
  let share := fun h => (drawnAt H u n h : ℝ) / n
  let read := fun h => (onesAt O H u ω n h : ℝ) / max (drawnAt H u n h) 1
  let ml := fun h => cpLow (drawnAt H u n h) n level
  let mh := fun h => cpHigh (drawnAt H u n h) n level
  let rl := fun h => cpLow (onesAt O H u ω n h) (drawnAt H u n h) level
  let rh := fun h => cpHigh (onesAt O H u ω n h) (drawnAt H u n h) level
  let slack := (∑ h, mh h * (rh h - rl h)) / (2 * s) + ∑ h, (mh h - ml h)
  slack < errorBound share share read (fun h => if drawnAt H u n h = 0 then 1 else read h)
      (fun h => h ∈ H.accept) (2 * s) - e
    ∨ slack ≤ e / 2

/-- `certifies`: some look's bound is at most `e`, and no look before it refuses. -/
def certifies (O : Oracle μ S) (H : DFA S R) (s α e : ℝ) (u : ℕ → S) (ω : Ω) : Prop :=
  ∃ k, lookBound O H s α u ω k ≤ e ∧ ∀ j < k, ¬ lookRefuses O H s α e u ω j

end OrthoDFA
