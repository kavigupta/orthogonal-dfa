import OrthoDFA.Proofs.Level3Congr
import OrthoDFA.Proofs.LoopStop
import OrthoDFA.Proofs.KappaCount
import OrthoDFA.Code.Reads

/-!
# A level-3 run's oracle queries

`familyRead` at `w` queries `O.mq` only at `{w · v : v ∈ F}`, so one read is `|F|` calls. Two
counts of a level-3 run over `xs`:

* `level3Queries` — the distinct strings it asks about (`runReadsFrom` crossed with `F`). The run's
  result is fixed by the oracle's answers there (`level3_congr_level3Queries`).
* `level3Count` — the oracle calls it makes, with multiplicity (`runReadCount · |F|`).

Both are at most `|xs| · (probeSifts L · (n + 2) + (n + 2)|Σ|(n + 2)) · |F|` on `¬GrowsRun`
(`level3Queries_card_le`, `level3Count_le`), with `n + 2 = |Q| + f + 2` on the loop's success event.
Clustering adds `schedReads_card_le`; prefix generation sifts a draw through the prior hypothesis
for another `round_harvest_card_le` calls.
-/

namespace OrthoDFA

open Random MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A family's reads of `Z` ask the oracle at most `|Z|·|F|` strings. -/
theorem readStrings_card_le (Z F : Finset (FreeMonoid α)) :
    (readStrings Z F).card ≤ Z.card * F.card :=
  Finset.card_image_le.trans (le_of_eq (Finset.card_product Z F))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Every string the level-3 run over `xs` asks `O.mq` at. -/
noncomputable def level3Queries (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) {P : ℕ} (xs : Fin P → FreeMonoid α) (ω : Ω) :
    Finset (FreeMonoid α) :=
  readStrings
    (runReadsFrom (fun z => familyRead O.mq F kl kh z ω) C initialState (List.ofFn xs)) F

/-- Level 3's result is fixed by the oracle's answers at the strings it queries. -/
theorem level3_congr_level3Queries (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) {P : ℕ} (xs : Fin P → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ p ∈ level3Queries O F kl kh C xs ω, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω') xs :=
  indecisiveLStarLevel3_congr_runReads C xs fun _z hz => familyRead_congr O h hz

/-- On `¬GrowsRun`, the level-3 run queries at most
`|xs| · (probeSifts L · (n + 2) + (n + 2)|Σ|(n + 2)) · |F|` strings. -/
theorem level3Queries_card_le (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) {P : ℕ} (xs : Fin P → FreeMonoid α) (ω : Ω) {L n : ℕ}
    (hL : ∀ i, (xs i).toList.length ≤ L)
    (hg : ¬ GrowsRun (fun z => familyRead O.mq F kl kh z ω) C n initialState (List.ofFn xs)) :
    (level3Queries O F kl kh C xs ω).card
      ≤ P * (Kappa.probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2)) * F.card := by
  refine (readStrings_card_le _ _).trans ?_
  have hstart : (initialState : Level3State α).tree.leaves.length ≤ n + 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  have hcard := runReadsFrom_card_le (r := fun z => familyRead O.mq F kl kh z ω) C
    (List.ofFn xs) initialState (fun x hx => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hL i) hstart hg
  rw [List.length_ofFn] at hcard
  exact Nat.mul_le_mul_right _ hcard

/-- A level-3 run whose hypothesis tree lands in `classSet nQ` queries at most the level-3
polynomial, with `nQ + 2` the tree's leaf bound. -/
theorem round_level3_card_le (O : Oracle μ (FreeMonoid α)) (C : Level3Settings)
    {P : ℕ} (F : Finset (FreeMonoid α)) (kl kh : ℕ) (xs : Fin P → FreeMonoid α) (ω : Ω) {L nQ : ℕ}
    (hL : ∀ i, (xs i).toList.length ≤ L)
    (htree : (indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs).hypothesis.tree
      ∈ classSet nQ) :
    (level3Queries O F kl kh C xs ω).card
      ≤ P * (Kappa.probeSifts L * (nQ + 2) + (nQ + 2) * Fintype.card α * (nQ + 2)) * F.card := by
  refine level3Queries_card_le O F kl kh C xs ω hL
    (not_growsRun_of_runProbes_le (List.ofFn xs) initialState ?_)
  exact leaves_length_le_of_mem_classSet nQ _ htree

/-- On `¬GrowsRun`, the level-3 run makes at most
`|xs| · (probeSifts L · (n + 2) + (n + 2)|Σ|(n + 2)) · |F|` oracle calls. -/
theorem level3Count_le (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) {P : ℕ} (xs : Fin P → FreeMonoid α) (ω : Ω) {L n : ℕ}
    (hL : ∀ i, (xs i).toList.length ≤ L)
    (hg : ¬ GrowsRun (fun z => familyRead O.mq F kl kh z ω) C n initialState (List.ofFn xs)) :
    level3Count O F kl kh C xs ω
      ≤ P * (Kappa.probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2)) * F.card := by
  unfold level3Count
  have hstart : (initialState : Level3State α).tree.leaves.length ≤ n + 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  have hcard := runReadCount_le (r := fun z => familyRead O.mq F kl kh z ω) C
    (List.ofFn xs) initialState (fun x hx => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hL i) hstart hg
  rw [List.length_ofFn] at hcard
  exact Nat.mul_le_mul_right _ hcard

/-- A level-3 run whose hypothesis tree lands in `classSet nQ` makes at most the level-3
polynomial of oracle calls, with `nQ + 2` the tree's leaf bound. -/
theorem round_level3_count_le (O : Oracle μ (FreeMonoid α)) (C : Level3Settings)
    {P : ℕ} (F : Finset (FreeMonoid α)) (kl kh : ℕ) (xs : Fin P → FreeMonoid α) (ω : Ω) {L nQ : ℕ}
    (hL : ∀ i, (xs i).toList.length ≤ L)
    (htree : (indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs).hypothesis.tree
      ∈ classSet nQ) :
    level3Count O F kl kh C xs ω
      ≤ P * (Kappa.probeSifts L * (nQ + 2) + (nQ + 2) * Fintype.card α * (nQ + 2)) * F.card := by
  refine level3Count_le O F kl kh C xs ω hL
    (not_growsRun_of_runProbes_le (List.ofFn xs) initialState ?_)
  exact leaves_length_le_of_mem_classSet nQ _ htree

/-- One harvest draw sifts `x` through the hypothesis tree `T` via `familyRead`, asking the oracle
at most `probeSifts L · |mids T| · |F|` strings. -/
theorem harvest_reads_card_le (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (T : DTree α) (E : Edges α) (ω : Ω) {x : FreeMonoid α} {L : ℕ}
    (hx : x.toList.length ≤ L) :
    (readStrings (Kappa.probeReads (fun z => familyRead O.mq F kl kh z ω) T E k x) F).card
      ≤ Kappa.probeSifts L * (mids T).card * F.card :=
  (readStrings_card_le _ F).trans (Nat.mul_le_mul_right _ (Spur.probeReads_card_mids _ hx))

/-- The same bound in the harvest hypothesis tree's leaf count, since a tree has fewer midfix
nodes than leaves. -/
theorem harvest_reads_card_le_leaves (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (T : DTree α) (E : Edges α) (ω : Ω) {x : FreeMonoid α} {L : ℕ}
    (hx : x.toList.length ≤ L) :
    (readStrings (Kappa.probeReads (fun z => familyRead O.mq F kl kh z ω) T E k x) F).card
      ≤ Kappa.probeSifts L * T.leaves.length * F.card :=
  (harvest_reads_card_le O F kl kh k T E ω hx).trans
    (Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (by have := mids_card T; omega)))

/-- Generating one prefix from a harvest population whose hypothesis tree lands in `classSet nQ`
sifts a draw through that tree for at most `probeSifts L · (nQ + 2) · |F|` oracle calls. -/
theorem round_harvest_card_le (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (T : DTree α) (E : Edges α) (ω : Ω) {x : FreeMonoid α} {L nQ : ℕ}
    (hx : x.toList.length ≤ L) (htree : T ∈ classSet nQ) :
    (readStrings (Kappa.probeReads (fun z => familyRead O.mq F kl kh z ω) T E k x) F).card
      ≤ Kappa.probeSifts L * (nQ + 2) * F.card :=
  (harvest_reads_card_le_leaves O F kl kh k T E ω hx).trans
    (Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (leaves_length_le_of_mem_classSet nQ T htree)))

/-- A round's clustering queries at most `|prefixes| · (|suffixes| + 1)` strings. -/
theorem schedReads_card_le {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    (schedReads sched pops x).card
      ≤ (roundPrefs sched pops x).card * ((roundSuffs sched x).card + 1) := by
  classical
  unfold schedReads readSet
  refine Finset.card_image_le.trans ?_
  rw [Finset.card_product]
  gcongr
  exact Finset.card_insert_le _ _

end OrthoDFA
