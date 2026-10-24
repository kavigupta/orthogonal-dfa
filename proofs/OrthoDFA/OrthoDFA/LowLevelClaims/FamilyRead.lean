import OrthoDFA.Code.FamilyRead

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {S : Type*} [Stringlike S]

noncomputable def readProb (O : Oracle μ S) (F : Finset S) (kl kh : ℕ) (w : S) (r : ARU) : ℝ :=
  μ.real {ω | familyRead O.mq F kl kh w ω = r}

def SuffixFree {α : Type*} (F : Finset (FreeMonoid α)) : Prop :=
  ∀ v ∈ F, ∀ v' ∈ F, v.toList <:+ v'.toList → v = v'

/-! ## The band check -/

/-- The check `reads_minority_bounded` makes of a band.  Take `N` queries, `a` of them answering
1 with chance `p` and the rest with chance `r`; for every `a ≤ N`, the read of their count is
accept at most `ε` of the time, or reject at most `ε`, or undecided at least a third with accept
or reject at most `ε₂`; and it is on its rarer decided side at most `ε`, or at most `κ` times as
often as it is undecided. -/
def BandPasses (N kl kh : ℕ) (p r ε ε₂ κ : ℝ) : Prop :=
  ∀ a ≤ N,
    let dist : ARU → ℝ := fun rd => ∑ j ∈ Finset.range (N + 1),
      if readOf kl kh j = rd then
        ∑ x ∈ Finset.antidiagonal j,
          (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
            * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))
      else 0
    (dist .accept ≤ ε ∨ dist .reject ≤ ε
      ∨ ((dist .accept ≤ ε₂ ∨ dist .reject ≤ ε₂) ∧ 1 / 3 ≤ dist .undecided))
    ∧ min (dist .accept) (dist .reject) ≤ max ε (κ * dist .undecided)

/-- For every `a ≤ N`, the read of the count of `N` queries, `a` of them answering 1 with chance
`p` and the rest with chance `r`, is on its rarer decided side at most `κ` times as often as it is
undecided. -/
def RarerBounded (N kl kh : ℕ) (p r κ : ℝ) : Prop :=
  ∀ a ≤ N,
    let dist : ARU → ℝ := fun rd => ∑ j ∈ Finset.range (N + 1),
      if readOf kl kh j = rd then
        ∑ x ∈ Finset.antidiagonal j,
          (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
            * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))
      else 0
    min (dist .accept) (dist .reject) ≤ κ * dist .undecided

/-! ## The claim -/

/-- With the language a DFA's, the family suffix-free and the band passing its check: the reads
are independent across strings, and every DFA state has one distribution that all its strings
read with, which is accept at most `ε` of the time, or reject at most `ε`, or undecided at least
a third with accept or reject at most `ε₂`, and is on its rarer decided side at most `ε`, or at
most `κ` times as often as it is undecided. -/
def FamilyReadBound : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {α σ : Type*} [Countable α] (O : Oracle μ (FreeMonoid α)) (M : DFA α σ)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (ε ε₂ κ : ℝ),
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  SuffixFree F →
  BandPasses F.card kl kh (1 - O.ηIn) O.ηOut ε ε₂ κ →
  iIndepFun (fun w => familyRead O.mq F kl kh w) μ
  ∧ ∀ q : σ, ∃ dist : ARU → ℝ,
    (∀ w, M.eval w.toList = q → readProb O F kl kh w = dist)
    ∧ (dist .accept ≤ ε ∨ dist .reject ≤ ε
      ∨ ((dist .accept ≤ ε₂ ∨ dist .reject ≤ ε₂) ∧ 1 / 3 ≤ dist .undecided))
    ∧ min (dist .accept) (dist .reject) ≤ max ε (κ * dist .undecided)

end OrthoDFA
