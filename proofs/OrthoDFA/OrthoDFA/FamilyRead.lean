import OrthoDFA.Clustering
import Mathlib.Computability.DFA

/-!
# The family's read of a string

`X(w)` counts the members `v` of the suffix family `F` whose query `w·v` answers 1.  The read
accepts at `X(w) ≥ kh`, rejects at `X(w) ≤ kl`, and is undecided in between.

Across strings the reads are independent draws when no member of `F` is a proper suffix of
another, and a string's read law is fixed by how many members its DFA state sends into the
language.  So whether every state's read is nearly never accept, nearly never reject, or often
undecided comes down to a check on the parameters alone, `TrichotomyAt`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

inductive Read
  | accept
  | reject
  | undecided
  deriving DecidableEq

instance : MeasurableSpace Read := ⊤

def readOf (kl kh x : ℕ) : Read :=
  if kh ≤ x then .accept else if x ≤ kl then .reject else .undecided

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {S : Type*} [Stringlike S]

noncomputable def familyRead (mq : S → Ω → ℝ) (F : Finset S) (kl kh : ℕ) (w : S) (ω : Ω) :
    Read :=
  readOf kl kh (voteCount mq F w ω)

noncomputable def readProb (O : Oracle μ S) (F : Finset S) (kl kh : ℕ) (w : S) (r : Read) : ℝ :=
  μ.real {ω | familyRead O.mq F kl kh w ω = r}

/-! ## Strings -/

/-- Scoped, so that it never meets another `MeasurableSpace (FreeMonoid α)`.  On a countable
type the discrete σ-algebra is the only one with measurable singletons. -/
noncomputable scoped instance {α : Type*} [Countable α] : Stringlike (FreeMonoid α) where
  toMeasurableSpace := ⊤
  measurable_const_mul _ := fun _ _ => trivial
  measurable_mul_const _ := fun _ _ => trivial
  exists_injective_nat' := (inferInstance : Countable (List α)).exists_injective_nat'
  measurableSet_singleton _ := trivial
  decEq := Classical.decEq _

def SuffixFree {α : Type*} (F : Finset (FreeMonoid α)) : Prop :=
  ∀ v ∈ F, ∀ v' ∈ F, v.toList <:+ v'.toList → v = v'

/-! ## The read's law -/

/-- `P[X = k]` for `X` the sum of `Bin(a, p)` and an independent `Bin(N − a, r)`. -/
noncomputable def voteLaw (N a : ℕ) (p r : ℝ) (k : ℕ) : ℝ :=
  ∑ x ∈ Finset.antidiagonal k,
    (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
      * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))

noncomputable def readLaw (N a : ℕ) (p r : ℝ) (kl kh : ℕ) (rd : Read) : ℝ :=
  ∑ j ∈ Finset.range (N + 1), if readOf kl kh j = rd then voteLaw N a p r j else 0

/-- Every read law over `N` members, `a` of them accepting, is accept at most `ε` of the time,
or reject at most `ε`, or undecided at least a third. -/
def TrichotomyAt (N kl kh : ℕ) (ηIn ηOut ε : ℝ) : Prop :=
  ∀ a ≤ N,
    readLaw N a (1 - ηIn) ηOut kl kh .accept ≤ ε
    ∨ readLaw N a (1 - ηIn) ηOut kl kh .reject ≤ ε
    ∨ 1 / 3 ≤ readLaw N a (1 - ηIn) ηOut kl kh .undecided

/-! ## The claims -/

/-- With the language a DFA's and the family suffix-free, the reads are independent across
strings, every string in a state reads with that state's law, and wherever `TrichotomyAt` holds
every state's law is accept at most `ε` of the time, or reject at most `ε`, or undecided at least
a third. -/
def FamilyReadTrichotomy : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {α σ : Type*} [Countable α] (O : Oracle μ (FreeMonoid α)) (M : DFA α σ)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (ε : ℝ),
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  SuffixFree F →
  TrichotomyAt F.card kl kh O.ηIn O.ηOut ε →
  iIndepFun (fun w => familyRead O.mq F kl kh w) μ
  ∧ ∀ q : σ, ∃ law : Read → ℝ,
    (∀ w, M.eval w.toList = q → ∀ rd, readProb O F kl kh w rd = law rd)
    ∧ (law .accept ≤ ε ∨ law .reject ≤ ε ∨ 1 / 3 ≤ law .undecided)

/-- `FamilyReadTrichotomy` at the shipped parameters: `N = 62`, `kl = 20`, `kh = 42`, both
noise rates `1/5`, and `ε = 10⁻¹⁰`. -/
def FamilyReadShipped : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {α σ : Type*} [Countable α] (O : Oracle μ (FreeMonoid α)) (M : DFA α σ)
    (F : Finset (FreeMonoid α)),
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  SuffixFree F →
  F.card = 62 →
  O.ηIn = 1 / 5 →
  O.ηOut = 1 / 5 →
  iIndepFun (fun w => familyRead O.mq F 20 42 w) μ
  ∧ ∀ q : σ, ∃ law : Read → ℝ,
    (∀ w, M.eval w.toList = q → ∀ rd, readProb O F 20 42 w rd = law rd)
    ∧ (law .accept ≤ 1 / 10 ^ 10 ∨ law .reject ≤ 1 / 10 ^ 10 ∨ 1 / 3 ≤ law .undecided)

end OrthoDFA
