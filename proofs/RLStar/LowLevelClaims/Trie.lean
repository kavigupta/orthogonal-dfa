import RLStar.Code.TrieDFA
import RLStar.LowLevelClaims.Level3Claim
import RLStar.LowLevelClaims.CoveredStart

/-!
# Local walk starts: what a round certifies of its DFA

A probe walked past the expanded prefixes runs the round's DFA on the probe itself. So a probe
that neither searches nor stops undecided at its start or end sift ends where its own sift does,
and the DFA classifies it as the root reads it (`TrieProbeAgreeCorrect`). Over any law of strings
the DFA then misjudges at most the probes that disagree plus the strings the root misreads
(`TrieSinglePieceCorrect`), with no start, zone or edge argument.

`Level3TrieRootOne`: every tree a round grows reads `ε` at its root, as level 3's do.
`Level3TrieExpandedCard`: a round expands at most `cap` prefixes of each probe.

`Level3TrieConsistentSound` is the open half: a consistent ending certifies its rate, for any
fixed reads.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type*}

/-- The probes that, walked past `X` under `H`, search, change the hypothesis or stop undecided
at their start or end sift. -/
def trieDisagreeSet (read : FreeMonoid α → ARU) (X : Finset (FreeMonoid α)) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | x ∈ searchSet read (walkStartIn X x) H ∪ endsSet read (walkStartIn X x) H}

open scoped Classical in
/-- The strings the read leaves undecided or puts on the wrong side of `L`. -/
def misreadSet (read : FreeMonoid α → ARU) (L : Set (FreeMonoid α)) : Set (FreeMonoid α) :=
  {x | read x ≠ if x ∈ L then ARU.accept else ARU.reject}

end Random

open Random

def TrieProbeAgreeCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (H : HypothesisDFA α)
    (X : Finset (FreeMonoid α)), (∃ r a, H.tree = .node 1 r a) →
    ∀ x ∉ trieDisagreeSet read X H, H.TrieClassifies read X x (read x = .accept)

/-- `P(DFA(x) ≠ M(x)) ≤ P(probe of x disagrees) + P(root misreads x)`, for any law. -/
def TrieSinglePieceCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (H : HypothesisDFA α)
    (X : Finset (FreeMonoid α)), (∃ r a, H.tree = .node 1 r a) →
    ∀ (L : Set (FreeMonoid α)) (D : Measure (FreeMonoid α)) [IsFiniteMeasure D],
    D.real {x | ¬ H.TrieClassifies read X x (x ∈ L)}
      ≤ D.real (trieDisagreeSet read X H) + D.real (misreadSet read L)

def Level3TrieRootOne : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (cap : ℕ)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α),
    ∃ r a, (indecisiveLStarLevel3Trie C cap read draws).hypothesis.tree = .node 1 r a

def Level3TrieExpandedCard : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (cap : ℕ)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α),
    (indecisiveLStarLevel3Trie C cap read draws).expanded.card ≤ P * cap

/-- For any fixed reads, `P(consistent ∧ P_D(probe disagrees) > ε) ≤ δ` over the probes. -/
def Level3TrieConsistentSound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (cap : ℕ)
    (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
    0 < C.δ → 0 < C.disagreementRate →
    (Measure.pi fun _ : Fin C.probeBudget => D).real
      {draws | let r := indecisiveLStarLevel3Trie C cap read draws
        r.ending = .consistent
          ∧ C.disagreementRate < D.real (trieDisagreeSet read r.expanded r.hypothesis)}
      ≤ C.δ

end OrthoDFA

end
