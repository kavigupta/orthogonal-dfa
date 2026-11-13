import RLStar.LowLevelClaims.TrieMass
import RLStar.Proofs.TrieTable

/-!
# Probe mass through a trie

Strings of one length have disjoint cylinders, so each length below `K` adds at most `1`. A walk
start moves only where the prefix it stops at for `X` was added in `Y`.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

theorem trieMassBound_holds : TrieMassBound := by
  intro α _ _ D _ X K hX
  have hj : ∀ j : ℕ, ∑ u ∈ X with u.toList.length = j,
      D.real {x | prefixOf x u.toList.length = u} ≤ 1 := fun j => by
    calc ∑ u ∈ X with u.toList.length = j, D.real {x | prefixOf x u.toList.length = u}
        = ∑ u ∈ X with u.toList.length = j, D.real ((fun x => prefixOf x j) ⁻¹' {u}) :=
          Finset.sum_congr rfl fun u hu => by
            have h := (Finset.mem_filter.1 hu).2
            subst h
            rfl
      _ = D.real ((fun x => prefixOf x j) ⁻¹' ↑(X.filter fun u => u.toList.length = j)) :=
          sum_measureReal_preimage_singleton _ fun _ _ => (Set.to_countable _).measurableSet
      _ ≤ 1 := measureReal_le_one
  calc ∑ u ∈ X, D.real {x | prefixOf x u.toList.length = u}
      = ∑ j ∈ Finset.range K, ∑ u ∈ X with u.toList.length = j,
          D.real {x | prefixOf x u.toList.length = u} :=
        (Finset.sum_fiberwise_of_maps_to (g := fun u : FreeMonoid α => u.toList.length)
          (fun u hu => Finset.mem_range.2 (hX u hu)) _).symm
    _ ≤ ∑ _j ∈ Finset.range K, (1 : ℝ) := Finset.sum_le_sum fun j _ => hj j
    _ = K := by simp

theorem walkStartMoves_holds : WalkStartMoves := by
  intro α _ _ D _ X Y hXY
  refine le_trans (measureReal_mono ?_ (measure_ne_top D _))
    (measureReal_biUnion_finset_le (Y \ X) fun u => {x | prefixOf x u.toList.length = u})
  intro x hx
  obtain ⟨hkX, hinX, houtX⟩ := Random.walkStartIn_spec X x
  obtain ⟨hkY, hinY, houtY⟩ := Random.walkStartIn_spec Y x
  rcases lt_or_gt_of_ne hx with hkm | hmk
  · have hk := lt_of_lt_of_le hkm hkY
    refine Set.mem_iUnion₂.2 ⟨prefixOf x (Random.walkStartIn X x),
      Finset.mem_sdiff.2 ⟨hinY _ hkm, houtX hk⟩, ?_⟩
    have hlen : (prefixOf x (Random.walkStartIn X x)).toList.length = Random.walkStartIn X x := by
      simp [prefixOf, hk.le]
    change prefixOf x (prefixOf x (Random.walkStartIn X x)).toList.length
      = prefixOf x (Random.walkStartIn X x)
    rw [hlen]
  · exact absurd (hXY (hinX _ hmk)) (houtY (lt_of_lt_of_le hmk hkX))

end OrthoDFA
