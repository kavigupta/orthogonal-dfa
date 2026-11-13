import RLStar.Proofs.LoopMeasurable
import RLStar.Proofs.QueryLevel3
import RLStar.LowLevelClaims.LazyRound

/-!
# Where a lazy round reads

A lazy round started from the trie of the clustering's prefixes `CP`, over probes whose first `k`
letters are no prefix's of `CP`, walks each probe from a prefix of it no prefix of `CP` shares
(`LazyKeyOff`), and reads only such strings, at most `probes + 1 + T + 2` long
(`reads_lazyRoundQ_keyOff`). With every epoch's tree at most `n + 2` leaves it asks at most
`lazyReadCap` strings (`card_reads_lazyRoundQ_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Some prefix of `z`, at most `k` letters, is no prefix of a string of `CP`. -/
def LazyKeyOff (k : ℕ) (CP : Finset (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  ∃ i ≤ k, i ≤ z.toList.length ∧ ∀ p ∈ CP, i ≤ p.toList.length → prefixOf p i ≠ prefixOf z i

open scoped Classical in
/-- The strings at most `N'` long, `LazyKeyOff` but for the family's suffix. -/
def lazyZone (N' k : ℕ) (CP F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter fun w =>
    ∃ z, LazyKeyOff k CP z ∧ z.toList.length ≤ N' ∧ ∃ v ∈ F, w = z * v

/-- Every probe and block draw of `d` satisfies `Φ`. -/
def LazyDrawsAll {E P J b : ℕ} (Φ : FreeMonoid α → Prop) (d : Fin E → Random.EpochDraws α P J b) :
    Prop :=
  ∀ e, (∀ i, Φ ((d e).1 i)) ∧ ∀ j t, Φ ((d e).2 j t)

/-- What a lazy round asks with every epoch's tree at most `nQ + 2` leaves, its probes and block
draws at most `Lp` long: per epoch, level 3 and the top-up's sifts, and level 3's refinements. -/
def lazyReadCap (C : Random.Level3Settings) (E J b Lp nQ nα : ℕ) : ℕ :=
  E * ((C.probeBudget + J * b) * (Random.Kappa.probeSifts (Lp + C.walkStart) * (nQ + 3))
    + (nQ + 2) * ((nQ + 2) * (nα + 1) * (nQ + 2)))

theorem trieOK_prefixTrie (k : ℕ) (CP : Finset (FreeMonoid α)) :
    Random.TrieOK k (Random.prefixTrie k CP) := by
  sorry -- closes by: prefixOf_prefixOf_of_le, length_prefixOf

/-- Over probes at least `k` long whose first `k` letters avoid `CP`'s, the round from `CP`'s trie
reads only `LazyKeyOff` strings at most `probes + 1 + T + 2` long. -/
theorem reads_lazyRoundQ_keyOff (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (CP : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hk : C.walkStart = k)
    (hd : LazyDrawsAll (fun x => k ≤ x.toList.length
      ∧ Avoids k ↑(CP.image fun p => prefixOf p k) x) d) :
    ∀ z ∈ (Random.lazyRoundQ C k nh (Random.prefixTrie k CP) d).reads read,
      LazyKeyOff k CP z ∧ z.toList.length ≤ lazyProbesLen d + 1 + C.probeBudget + 2 := by
  sorry -- closes by: reads_probeOrAgreeQ_sub, reads_siteHarvestQ_sub, reads_subst, unpad algebra

/-- With every epoch's tree at most `nQ + 2` leaves, a round over probes at most `Lp` long asks at
most `lazyReadCap` strings. -/
theorem card_reads_lazyRoundQ_le (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hk : C.walkStart = k) (nQ Lp : ℕ)
    (hL : LazyDrawsAll (fun x => x.toList.length ≤ Lp) d)
    (hpk : ((Random.lazyRoundQ C k nh X₀ d).eval read).peak ≤ nQ + 2) :
    ((Random.lazyRoundQ C k nh X₀ d).reads read).card
      ≤ lazyReadCap C E J b Lp nQ (Fintype.card α) := by
  sorry -- closes by: card_reads_level3Q_le over Option α per epoch, reads_siteHarvestQ_sub

/-- Unless it fails, a round's listed strings are at most `probes + 1 + T + 2` long. -/
theorem length_lazy_list_le (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hnh : 0 < nh) (hX₀ : Random.TrieOK k X₀) :
    ∀ s, ∀ w ∈ ((Random.lazyRoundQ C k nh X₀ d).eval read).lists s,
      w.toList.length ≤ lazyProbesLen d + 1 + C.probeBudget + 2 := by
  sorry -- closes by: LazyRoundShape (listed strings are asked), the length half of the read bound

end OrthoDFA

end
