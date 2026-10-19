import OrthoDFA.Proofs.HypothesisDFA

/-!
# A probe's first `k` letters

A string is `KeyOff` where its first `k` letters avoid a set `K₀`; a probe avoids `K₀` where it is
shorter than `k` or `KeyOff`. Every prefix past the first `k` letters of an avoiding probe,
followed by anything, is `KeyOff`.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `z` is at least `k` long and its first `k` letters are not in `K₀`. -/
def KeyOff (k : ℕ) (K₀ : Set (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  k ≤ z.toList.length ∧ prefixOf z k ∉ K₀

/-- A probe level 5 may draw: shorter than `k`, or its first `k` letters off `K₀`. -/
def Avoids (k : ℕ) (K₀ : Set (FreeMonoid α)) (x : FreeMonoid α) : Prop :=
  k ≤ x.toList.length → prefixOf x k ∉ K₀

lemma length_prefixOf (x : FreeMonoid α) (j : ℕ) :
    (prefixOf x j).toList.length = min j x.toList.length := by
  simp [prefixOf]

lemma prefixOf_mul_of_le {u : FreeMonoid α} {k : ℕ} (h : k ≤ u.toList.length) (m : FreeMonoid α) :
    prefixOf (u * m) k = prefixOf u k := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
  rw [List.take_append_of_le_length h]

lemma prefixOf_prefixOf_of_le (x : FreeMonoid α) {j k : ℕ} (h : k ≤ j) :
    prefixOf (prefixOf x j) k = prefixOf x k := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take, min_eq_left h]

lemma prefixOf_mul_of {x : FreeMonoid α} {j : ℕ} {c : α} (hc : x.toList[j]? = some c) :
    prefixOf x j * FreeMonoid.of c = prefixOf x (j + 1) := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul, FreeMonoid.toList_of]
  rw [List.take_succ, hc]
  rfl

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

lemma keyOff_mul {u : FreeMonoid α} (h : KeyOff k K₀ u) (m : FreeMonoid α) :
    KeyOff k K₀ (u * m) :=
  ⟨by simp only [FreeMonoid.toList_mul, List.length_append]; have := h.1; omega,
    by rw [prefixOf_mul_of_le h.1]; exact h.2⟩

lemma keyOff_prefixOf {x : FreeMonoid α} (h : KeyOff k K₀ x) {j : ℕ} (hj : k ≤ j) :
    KeyOff k K₀ (prefixOf x j) :=
  ⟨by rw [length_prefixOf]; exact le_min hj h.1, by rw [prefixOf_prefixOf_of_le x hj]; exact h.2⟩

lemma keyOff_of_avoids {x : FreeMonoid α} (h : Avoids k K₀ x) (hk : k ≤ x.toList.length) :
    KeyOff k K₀ x :=
  ⟨hk, h hk⟩

lemma keyOff_of_prefix {x : FreeMonoid α} (hx : Avoids k K₀ x) {j : ℕ} (hkj : k ≤ j)
    (hj : j ≤ x.toList.length) (m : FreeMonoid α) : KeyOff k K₀ (prefixOf x j * m) :=
  keyOff_mul (keyOff_prefixOf (keyOff_of_avoids hx (hkj.trans hj)) hkj) m

end OrthoDFA
