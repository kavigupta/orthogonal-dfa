import OrthoDFA.Clustering

/-!
# Clustering rules

The proof holds for any rule that picks the family the way `Clusterer` says.  `Identify` shows
`identifyCluster` is one.
-/

namespace OrthoDFA

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

/-- A way of picking the family out of the screened candidates from their reads: `reads w` says
the oracle answered accept at `w`, `W` weighs the prefixes once per population and `ord` orders
the candidates, and
`P`, `cands` and `k` are the prefixes, the candidates and the size asked for.

The proof only uses that the pick holds the seed, lies in the candidates, has `k` members when
the candidates are that many and at most `k` otherwise, and reads only the candidates' columns on
the prefixes.  `identifyClusterer` is `identify_cluster_around`. -/
structure Clusterer (S : Type*) [Stringlike S] where
  pick : (S → Prop) → List (S → ℝ) → (S → ℕ) → Finset S → Finset S → ℕ → Finset S
  seed_mem : ∀ reads w ord P cands k, (1 : S) ∈ cands → (1 : S) ∈ pick reads w ord P cands k
  subset : ∀ reads w ord P cands k, (1 : S) ∈ cands → pick reads w ord P cands k ⊆ cands
  card_le : ∀ reads w ord P (cands : Finset S) k, Set.InjOn ord ↑cands → 0 < k →
    (pick reads w ord P cands k).card ≤ k
  card_eq : ∀ reads w ord P (cands : Finset S) k, Set.InjOn ord ↑cands → (1 : S) ∈ cands →
    k ≤ cands.card → 0 < k → (pick reads w ord P cands k).card = k
  congr : ∀ reads reads' w ord P cands k, (1 : S) ∈ cands →
    (∀ p ∈ P, ∀ v ∈ cands, (reads (p * v) ↔ reads' (p * v))) →
    pick reads w ord P cands k = pick reads' w ord P cands k

/-- The cluster without its seed. -/
noncomputable def clusterBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  (rule.pick (fun w => mq w (oracleNoise x) = 1) (prefixWeights populations B.npref x)
    (poolOrder B.nsuff x) (prefixesAt populations B.npref x) (screenedAt mq populations B x)
    B.k).erase 1

/-- `familyAt` with the family picked by `rule`. -/
noncomputable def familyBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  insert 1 (clusterBy rule mq populations x B)

open scoped Classical in
/-- `ret` with the family picked by `rule`. -/
noncomputable def retBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (indecisionLimit α : ℝ) (v n : ℕ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf j n x).filter (fun p => ¬ decided mq B.lo (B.hi + 1)
          (familyBy rule mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j n x).card)
    ∧ (∃ j ∈ populations, ∃ p ∈ certOf j n x,
        B.hi + 1 < voteCount mq (familyBy rule mq populations x B) p (oracleNoise x))
    ∧ (∃ j ∈ populations, ∃ p ∈ certOf j n x,
        voteCount mq (familyBy rule mq populations x B) p (oracleNoise x) ≤ B.lo)
    ∧ noDrift mq populations uni B.lo B.hi α (clusterBy rule mq populations x B) n v 0 x
    ∧ ∃ e : ℕ, certified mq B.lo B.hi α (clusterBy rule mq populations x B)
        (gateOf uni n e x) (oracleNoise x)
      ∧ noDrift mq populations uni B.lo B.hi α (clusterBy rule mq populations x B) n v e x}

end OrthoDFA
