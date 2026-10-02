import OrthoDFA.Clustering

/-!
# Clustering rules

The proof holds for any rule that picks the family the way `Clusterer` says.  `Lloyd` shows
`clusterAround` is one.
-/

namespace OrthoDFA

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

/-- A way of picking the family out of the screened candidates from their reads: `reads w` says
the oracle answered accept at `w`, and `P`, `cands` and `k` are the prefixes, the candidates and
the size asked for.

`identify_cluster_around` is one for each boundary it is handed.  It reads only the candidates'
columns on the representative prefixes, starts from `ε` and stops before `ε` would leave, and
returns `count` rows once the pool holds that many.  How it weighs populations, when it stops
recentring, and how it breaks ties are its own business, which is why the proof is carried out
for every such rule.  A boundary handed in from reads outside the round is independent of the
round's bits, so the claim holds with it fixed first. -/
structure Clusterer (S : Type*) [Stringlike S] where
  pick : (S → Prop) → Finset S → Finset S → ℕ → Finset S
  seed_mem : ∀ reads P cands k, (1 : S) ∈ cands → (1 : S) ∈ pick reads P cands k
  subset : ∀ reads P cands k, (1 : S) ∈ cands → pick reads P cands k ⊆ cands
  card_le : ∀ reads P cands k, 0 < k → (pick reads P cands k).card ≤ k
  card_eq : ∀ reads P cands k, (1 : S) ∈ cands → k ≤ cands.card → 0 < k →
    (pick reads P cands k).card = k
  congr : ∀ reads reads' P cands k, (1 : S) ∈ cands →
    (∀ p ∈ P, ∀ v ∈ cands, (reads (p * v) ↔ reads' (p * v))) →
    pick reads P cands k = pick reads' P cands k

/-- The cluster without its seed. -/
noncomputable def clusterBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  (rule.pick (fun w => mq w (oracleNoise x) = 1) (prefixesAt populations B.npref x)
    (screenedAt mq populations B x) B.k).erase 1

/-- `familyAt` with the family picked by `rule`. -/
noncomputable def familyBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  insert 1 (clusterBy rule mq populations x B)

open scoped Classical in
/-- `ret` with the family picked by `rule`. -/
noncomputable def retBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (v : ℕ) (indecisionLimit α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf j B.npref x).filter (fun p => ¬ decided mq B.lo (B.hi + 1)
          (familyBy rule mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.npref x).card)
    ∧ (∃ r : ℕ, noDrift mq populations uni B.lo B.hi α (clusterBy rule mq populations x B)
        B.npref v r x)
    ∧ ∃ e : ℕ, admitted mq B.lo B.hi α
        (clusterBy rule mq populations x B) (certOf uni (B.npref + e) x) (oracleNoise x)}

open scoped Classical in
/-- `retBy` with the gate read at every population, on its first `npref` draws: what the loop is
shown to reach. -/
noncomputable def retAll (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (v : ℕ) (indecisionLimit α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf j B.npref x).filter (fun p => ¬ decided mq B.lo (B.hi + 1)
          (familyBy rule mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.npref x).card)
    ∧ (∃ r : ℕ, noDrift mq populations uni B.lo B.hi α (clusterBy rule mq populations x B)
        B.npref v r x)
    ∧ ∀ j ∈ populations, admitted mq B.lo B.hi α
        (clusterBy rule mq populations x B) (certOf j B.npref x) (oracleNoise x)}

lemma retAll_subset_retBy (rule : Clusterer S) (mq : S → Ω → ℝ) {populations : Finset J}
    {uni : J} (huni : uni ∈ populations) (v : ℕ) (indecisionLimit α : ℝ) (B : State) :
    retAll rule mq populations uni v indecisionLimit α B
      ⊆ retBy rule mq populations uni v indecisionLimit α B :=
  fun _ ⟨h1, h2, hd, h3⟩ => ⟨h1, h2, hd, 0, by simpa using h3 uni huni⟩

end OrthoDFA
