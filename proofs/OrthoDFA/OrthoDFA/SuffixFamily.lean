import OrthoDFA.Level5Claim

/-!
# A suffix family read against a DFA target

The oracle's language is the target DFA's, the family is suffix-free, and its band passes the
band selection rule's check (`BandPasses`). A draw of the oracle's noise reads every string once
(`SuffixFamily.read`), and each target state has one distribution its strings read with
(`SuffixFamily.readDist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The oracle `O`, whose language is `M`'s; the family `F`, suffix-free, read with the band
`kl, kh`; and the band check's constants `ε, ε₂, κ`, which the band passes. -/
structure SuffixFamily {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (α σ : Type*)
    [Fintype α] where
  O : Oracle μ (FreeMonoid α)
  M : _root_.DFA α σ
  F : Finset (FreeMonoid α)
  kl : ℕ
  kh : ℕ
  ε : ℝ
  ε₂ : ℝ
  κ : ℝ
  lang : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept
  suffixFree : SuffixFree F
  band : BandPasses F.card kl kh (1 - O.ηIn) O.ηOut ε ε₂ κ

namespace SuffixFamily

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {σ : Type*} (fam : SuffixFamily μ α σ)

/-- The family's read of every string at the oracle's noise `ω`. -/
def read (ω : Ω) (z : FreeMonoid α) : ARU := familyRead fam.O.mq fam.F fam.kl fam.kh z ω

/-- The read's side, or `none` where undecided. -/
def cut (ω : Ω) (z : FreeMonoid α) : Option Bool := (fam.read ω z).cut

open scoped Classical in
/-- The distribution a state's strings read with: a string's at a state some string reaches,
else none. -/
def readDist (q : σ) : ARU → ℝ :=
  if h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q then
    readProb fam.O fam.F fam.kl fam.kh h.choose
  else fun _ => 0

end SuffixFamily

end OrthoDFA

end
