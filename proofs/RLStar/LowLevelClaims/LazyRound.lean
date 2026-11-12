import RLStar.Code.LazyRound
import RLStar.LowLevelClaims.Level3Claim
import RLStar.LowLevelClaims.Level3NoWrong

/-!
# A round with a lazy trie: what it certifies

An epoch is level 3 over the padded probes, so with no wrong read it ends well but for its level
(`EpochCorrect`). Below, `harvestMass` is the chance a probe walked from `X` harvests at the site,
and `lazyDisagree` the probes whose walk from `X` searches or stops undecided at its start or end.

A top-up retracts only where the harvest mass has fallen below half the yield, but for a chance
geometric in the block size (`TopUpCorrect`). The trie grows by at most `K` walk starts for each
listed string (`LazyRoundShape`). Each retraction expands half a yield of probe mass, and the
expanded prefixes hold at most `K` of it, so the epochs suffice (`LazyRoundCorrect`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

def padRead (read : FreeMonoid α → ARU) (z : FreeMonoid (Option α)) : ARU := read (unpad z)

/-- The target with each pad a loop. -/
def _root_.DFA.padded {σ : Type*} (M : DFA α σ) : DFA (Option α) σ :=
  ⟨fun q c => c.elim q (M.step q), M.start, M.accept⟩

/-- Every string of `X` is shorter than `K`, and every prefix of one is in `X`. -/
def TrieOK (K : ℕ) (X : Finset (FreeMonoid α)) : Prop :=
  (∀ u ∈ X, u.toList.length < K) ∧ ∀ u ∈ X, ∀ i, prefixOf u i ∈ X

/-- The chance a probe walked from `X` under `H` harvests at the site. -/
def harvestMass (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Level3Site) : ℝ :=
  D.real {x | siteHarvest (padRead read) K H s (padIn K X x) ≠ []}

/-- The probes whose walk from `X` under `H` searches or stops undecided at its start or end. -/
def lazyDisagree (read : FreeMonoid α → ARU) (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) : Set (FreeMonoid α) :=
  {x | padIn K X x ∈ searchSet (padRead read) K H ∪ endsSet (padRead read) K H}

end Random

open Random

/-- With no wrong read, an epoch fails to end well, or ends past `|Q| + 2` leaves, with chance at
most `δ`: a consistent ending has at most `ε` of the probes disagreeing, and a harvest's site at
least `y` of them harvesting. -/
def EpochCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Level3Settings)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (K : ℕ) (X : Finset (FreeMonoid α)),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
    ReadsNoWrong M side read → 0 < C.δ → C.δ < 1 → 0 < ε → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    (Measure.pi fun _ : Fin C.probeBudget => D).real
        {draws | let r := (epochQ C K X draws).eval read
          nQ + 2 < r.hypothesis.tree.leaves.length ∨
          match r.ending with
          | .consistent => ε < D.real (lazyDisagree read K X r.hypothesis)
          | .harvest s => harvestMass read D K X r.hypothesis s < y
          | .failed => True}
      ≤ C.δ

/-- Where no `K`-letter prefix holds more than `p₁` of the probes and `8 nh p₁ ≤ y`, a top-up
retracts to a trie whose harvest mass is at least `y/2`, or runs out of blocks, with chance at most

    J (1 − y/4)^b. -/
def TopUpCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (H : HypothesisDFA (Option α)) (s : Level3Site) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (K nh J b : ℕ) (y p₁ : ℝ) (X : Finset (FreeMonoid α))
    (ℓ : List (FreeMonoid α)),
    TrieOK K X → ℓ.Nodup → ℓ.length < nh → (∀ᵐ x ∂D, K ≤ x.toList.length) →
    (∀ u, D.real {x | prefixOf x K = u} ≤ p₁) → 8 * nh * p₁ ≤ y → y ≤ 1 →
    nh * (K + 1) + 1 ≤ J →
    (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D).real
        {B | match (topUpQ K nh H s (List.ofFn B) X ℓ).eval read with
          | none => True
          | some (.full _, _) => False
          | some (.retract _ X', _) => y / 2 ≤ harvestMass read D K X' H s}
      ≤ J * (1 - y / 4) ^ b

/-- A round grows its trie by at most `K` walk starts per listed string, keeps its lists free of
repeats, and ends consistent only with every list short of `nh`; each listed string is a string
the round read undecided, its pads dropped. -/
def LazyRoundShape : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
    (K nh E J b : ℕ) (X₀ : Finset (FreeMonoid α))
    (draws : Fin E → EpochDraws α C.probeBudget J b),
    TrieOK K X₀ →
    let p := lazyRoundQ C K nh X₀ draws
    let r := p.eval read
    TrieOK K r.expanded ∧ X₀ ⊆ r.expanded
      ∧ r.expanded.card ≤ X₀.card + K * ((r.lists .ends).length + (r.lists .middles).length)
      ∧ (∀ s, (r.lists s).Nodup)
      ∧ (r.ending = .consistent → ∀ s, (r.lists s).length < nh)
      ∧ (∀ s ℓ, r.ending = .harvest s ℓ → r.lists s = ℓ ∧ nh ≤ ℓ.length)
      ∧ ∀ s, ∀ w ∈ r.lists s, w ∈ p.reads read ∧ read w = .undecided

/-- With no wrong read, a round fails, ends consistent with more than `ε` of the probes
disagreeing, or ends past `|Q| + 2` leaves, with chance at most

    δ + E J (1 − y/4)^b,

where every `K`-letter prefix holds at most `p₁` of the probes, `8 nh p₁ ≤ y`, the epochs number
at least `2K/y + 1` and each epoch's blocks at least `nh (K + 1) + 1`. -/
def LazyRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Level3Settings)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (K nh E J b : ℕ) (p₁ : ℝ)
    (X₀ : Finset (FreeMonoid α)),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
    ReadsNoWrong M side read → TrieOK K X₀ → (∀ᵐ x ∂D, K ≤ x.toList.length) →
    (∀ u, D.real {x | prefixOf x K = u} ≤ p₁) → 8 * nh * p₁ ≤ y →
    0 < C.δ → C.δ < 1 → 0 < ε → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) + S ≤ C.probeBudget →
    2 * K / y + 1 ≤ E → nh * (K + 1) + 1 ≤ J →
    (Measure.pi fun _ : Fin E => (Measure.pi fun _ : Fin C.probeBudget => D).prod
        (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D)).real
        {d | let r := (lazyRoundQ C K nh X₀ d).eval read
          nQ + 2 < r.hypothesis.tree.leaves.length ∨
          match r.ending with
          | .consistent => ε < D.real (lazyDisagree read K r.expanded r.hypothesis)
          | .harvest _ _ => False
          | .failed => True}
      ≤ C.δ + E * J * (1 - y / 4) ^ b

end OrthoDFA

end
