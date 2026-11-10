import RLStar.Code.TrieDFA
import RLStar.Code.LearnerLoopTrie
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

`Level3TrieAgrees`: probes at least `walkStart` long with distinct first `walkStart` letters run
as level 3's, nothing expanded past the complete trie. `TrieLoopAgrees`: where, besides, every
harvested population's probes have first `walkStart` letters of no string the clusterings read and
of no probe of an earlier batch, the loop with local walk starts returns the loop's rounds and reads
the same strings.
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

def Level3TrieAgrees : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (cap : ℕ)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α),
    (∀ i, C.walkStart ≤ (draws i).toList.length) →
    Function.Injective (fun i => prefixOf (draws i) C.walkStart) →
    indecisiveLStarLevel3Trie C cap read draws
      = ⟨(indecisiveLStarLevel3 C read draws).hypothesis, completeTrie C.walkStart,
        (indecisiveLStarLevel3 C read draws).ending⟩

section Lift

variable {α : Type*} [Fintype α] [DecidableEq α] (k : ℕ)

def liftSampler (s : HarvestSampler α) : TrieHarvestSampler α :=
  ⟨s.family, s.kl, s.kh, s.hyp, completeTrie k, s.site⟩

def liftRound (o : LoopRound α) : TrieLoopRound α :=
  ⟨o.state, o.family, ⟨o.result.hypothesis, completeTrie k, o.result.ending⟩, o.prefs, o.suffs⟩

def liftHist (h : LoopHist α) : TrieLoopHist α :=
  ⟨h.pops.map (liftSampler k), h.prefs, h.suffs⟩

/-- The first `M` batches of `N` probes drawn by `u` from `D`: each at least `k` long, with first
`k` letters those of no string of `T` and of no probe of an earlier batch. -/
def BatchesGood (D : Measure (FreeMonoid α)) (N M : ℕ) (T : Finset (FreeMonoid α))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : Prop :=
  ∀ n < M, ∀ t, k ≤ (drawFrom D ((u n).2 t)).toList.length
    ∧ (∀ z ∈ T, prefixOf z k ≠ prefixOf (drawFrom D ((u n).2 t)) k)
    ∧ ∀ m < n, ∀ t', prefixOf (drawFrom D ((u m).2 t')) k ≠ prefixOf (drawFrom D ((u n).2 t)) k

/-- The level-3 probes `u` draws from `D` off the keys `K₀`: each at least `k` long, with distinct
first `k` letters. -/
def L3Good (D : Measure (FreeMonoid α)) (K₀ : Finset (FreeMonoid α)) {P : ℕ}
    (u : Fin P → unitInterval) : Prop :=
  (∀ i, k ≤ (drawFrom (avoidLaw D k K₀) (u i)).toList.length)
    ∧ Function.Injective fun i => prefixOf (drawFrom (avoidLaw D k K₀) (u i)) k

end Lift

/-- Round `d` after the history `h`, at the answers `rd`: every harvested population's batches are
good, and where the loop's round returns, its level-3 probes are. -/
def TrieRoundGood {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ} (settings : LoopSettings K)
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (rd : FreeMonoid α → ℝ) : Prop :=
  (∀ i : Fin K, BatchesGood settings.level3.walkStart D settings.level3.harvestReadCap
      settings.scanCap h.prefs (d.1.2.1 (some i)))
    ∧ ∀ o, (roundQ settings D Dsf h d).eval rd = some o →
      L3Good settings.level3.walkStart D
        (o.prefs.image fun p => prefixOf p settings.level3.walkStart) d.2

def TrieLoopAgrees : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ} (rd : FreeMonoid α → ℝ)
    (settings : LoopSettings K) (cap : ℕ) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget),
    (∀ r (hr : r < K) rs, (loopRoundsQ settings D Dsf draws r).eval rd = some rs →
      (∀ o ∈ rs, o.result.ending.harvested ≠ none) →
      TrieRoundGood settings D Dsf (histOf rs) (draws ⟨r, hr⟩) rd) →
    (trieLoopRoundsQ settings cap D Dsf draws K).eval rd
        = ((loopRoundsQ settings D Dsf draws K).eval rd).map
          (List.map (liftRound settings.level3.walkStart))
      ∧ (trieLoopRoundsQ settings cap D Dsf draws K).reads rd
        = (loopRoundsQ settings D Dsf draws K).reads rd

end OrthoDFA

end
