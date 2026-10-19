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

`identify_cluster_around` is one for each boundary it is handed and each state of `pst.rng`.
It reads only the candidates' columns on the representative prefixes, puts `ε` first, and
returns `count` rows once the pool holds that many.  How it groups the pool, which group it
anchors on, and how it fills the family are its own business, which is why the proof is carried
out for every such rule.  A boundary or generator state handed in from outside the round is
independent of the round's bits, so the claim holds with it fixed first. -/
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

/-- `refsAt` with the family picked by `rule`. -/
noncomputable def refsBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  poolAt B.nsuff x \ familyBy rule mq populations x B

/-- The gate at population `j`, at a look of `npref + e` draws, with the family picked by
`rule`. -/
def gateAt (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ εcov α : ℝ) (B : State) (j : J) (e : ℕ) (x : Run Ω S J) : Prop :=
  certifiesAt mq η₀ (gateLimit uni εcov j) (gateLevel α populations B) B.lo B.hi
    (clusterBy rule mq populations x B) (refsBy rule mq populations x B)
    (certOf (prefixesAt populations B.npref x) j (B.npref + e) x)
    (drawsTaken (prefixesAt populations B.npref x) (fun i => certPrefix j i x) (B.npref + e))
    (oracleNoise x)

open scoped Classical in
/-- `ret` with the family picked by `rule`. -/
noncomputable def retBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided mq B.lo (B.hi + 1) (familyBy rule mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card)
    ∧ ∀ j ∈ populations, ∃ e ≤ B.npref, gateAt rule mq populations uni η₀ εcov α B j e x}

open scoped Classical in
/-- `retBy` with every population's gate read at its first `npref` draws: what the loop is shown
to reach. -/
noncomputable def retAll (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (uni : J) (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided mq B.lo (B.hi + 1) (familyBy rule mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card)
    ∧ ∀ j ∈ populations, gateAt rule mq populations uni η₀ εcov α B j 0 x}

lemma retAll_subset_retBy (rule : Clusterer S) (mq : S → Ω → ℝ) {populations : Finset J}
    {uni : J} (η₀ indecisionLimit εcov α : ℝ) (B : State) :
    retAll rule mq populations uni η₀ indecisionLimit εcov α B
      ⊆ retBy rule mq populations uni η₀ indecisionLimit εcov α B :=
  fun _ ⟨h1, h2, h3⟩ => ⟨h1, h2, fun j hj => ⟨0, Nat.zero_le _, h3 j hj⟩⟩

end OrthoDFA
