import OrthoDFA.Clustering
import Mathlib.Computability.DFA

/-!
# The family's read of a string

`X(w)` counts the members `v` of the suffix family `F` whose query `w·v` answers 1.  The read
accepts at `X(w) ≥ kh`, rejects at `X(w) ≤ kl`, and is undecided in between.

Across strings the reads are independent draws when no member of `F` is a proper suffix of
another, and a string's read law is fixed by how many members its DFA state sends into the
language.  At the shipped parameters every state's read is either nearly never accept, nearly
never reject, or undecided at least a third of the time.
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

instance {α : Type*} : MeasurableSpace (FreeMonoid α) := ⊤

noncomputable instance {α : Type*} [Countable α] : Stringlike (FreeMonoid α) where
  measurable_const_mul _ := fun _ _ => trivial
  measurable_mul_const _ := fun _ _ => trivial
  exists_injective_nat' := (inferInstance : Countable (List α)).exists_injective_nat'
  measurableSet_singleton _ := trivial
  decEq := Classical.decEq _

def SuffixFree {α : Type*} (F : Finset (FreeMonoid α)) : Prop :=
  ∀ v ∈ F, ∀ v' ∈ F, v.toList <:+ v'.toList → v = v'

/-! ## The read's law

A sum of independent bits at rates `p₁, …, p_N` has the Poisson-binomial law, built one bit at a
time by `pbStep`. -/

def pbStep (p : ℝ) (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => (1 - p) * f 0
  | k + 1 => (1 - p) * f (k + 1) + p * f k

def pbBase (k : ℕ) : ℝ := if k = 0 then 1 else 0

instance : LeftCommutative pbStep where
  left_comm p q f := by
    funext k
    rcases k with _ | _ | k <;> simp only [pbStep] <;> ring

/-- `P[X = k]` for `a` bits at rate `p` and `N − a` at rate `r`. -/
noncomputable def voteLaw (N a : ℕ) (p r : ℝ) (k : ℕ) : ℝ :=
  (Multiset.replicate a p + Multiset.replicate (N - a) r).foldr pbStep pbBase k

noncomputable def readLaw (N a : ℕ) (p r : ℝ) (kl kh : ℕ) (rd : Read) : ℝ :=
  ∑ j ∈ Finset.range (N + 1), if readOf kl kh j = rd then voteLaw N a p r j else 0

/-- Every vote law whose mean `a(1 − ηIn) + (N − a)ηOut` lies strictly inside the band is
undecided at least a third of the time. -/
def BandHolds (N kl kh : ℕ) (ηIn ηOut : ℝ) : Prop :=
  ∀ a ≤ N, (kl : ℝ) < a * (1 - ηIn) + ((N - a : ℕ) : ℝ) * ηOut →
    (a : ℝ) * (1 - ηIn) + ((N - a : ℕ) : ℝ) * ηOut < kh →
    1 / 3 ≤ readLaw N a (1 - ηIn) ηOut kl kh .undecided

/-! ## The claims -/

/-- Distinct strings query disjoint sets of strings, so their reads are independent. -/
def FamilyReadIndependent : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {α : Type*} [Countable α] (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ),
  SuffixFree F →
  iIndepFun (fun w => familyRead O.mq F kl kh w) μ

/-- Two strings in the same DFA state read with the same law. -/
def FamilyReadByState : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {α σ : Type*} [Countable α] (O : Oracle μ (FreeMonoid α)) (M : DFA α σ)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ),
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  ∀ w w', M.eval w.toList = M.eval w'.toList →
  ∀ rd, readProb O F kl kh w rd = readProb O F kl kh w' rd

/-- For any parameters whose band holds, every string's read is accept at most
`exp(−2(kh − kl)²/N)` of the time, or reject at most that, or undecided at least a third. -/
def FamilyReadTrichotomy : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] (O : Oracle μ S) (F : Finset S) (kl kh : ℕ),
  kl < kh →
  BandHolds F.card kl kh O.ηIn O.ηOut →
  ∀ w : S,
    readProb O F kl kh w .accept ≤ Real.exp (-2 * ((kh : ℝ) - kl) ^ 2 / F.card)
    ∨ readProb O F kl kh w .reject ≤ Real.exp (-2 * ((kh : ℝ) - kl) ^ 2 / F.card)
    ∨ 1 / 3 ≤ readProb O F kl kh w .undecided

/-- At the shipped parameters, `N = 62`, `kl = 20`, `kh = 42` and both noise rates `1/5`, with
the language a DFA's and the family suffix-free: the reads are independent across strings, each
string's read law depends only on its state, and every state's law is accept at most `10⁻¹⁰`
of the time, or reject at most that, or undecided at least a third. -/
def FamilyReadGuarantee : Prop :=
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
