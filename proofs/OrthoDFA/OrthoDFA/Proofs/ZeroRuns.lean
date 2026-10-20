import OrthoDFA.Proofs.LoopRuns

/-!
# Level 3's run where its tree stays small

No step drops a leaf, so a run ending with at most `n + 2` leaves keeps its midfixes within `n`
throughout, and reads only a probe's prefix followed by at most `n + 1` letters: reads agreeing
there give the same run (`level3_congr_small`).
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings)

theorem leaves_le_step {read : FreeMonoid α → ARU} {s s' : Level3State α} {x : FreeMonoid α}
    (hs : Inv read s) (h : applyProbe read C s x = .inl s') :
    s.tree.leaves.length ≤ s'.tree.leaves.length := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, -, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, -, -, hp, -, -, rfl⟩ | ⟨rfl, -, -⟩
  · exact le_rfl
  · simp only [splitLeaf, newHypothesis]
    rw [DTree.length_leaves_splitAt _ _ _ hp]
    omega
  · exact le_rfl

theorem leaves_le_runProbes {read : FreeMonoid α → ARU} {X : List (FreeMonoid α)} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv read C X s →
      s.tree.leaves.length ≤ (runProbes read C s xs).1.tree.leaves.length
  | [], s, _, _ => le_rfl
  | x :: xs, s, hX, hs => by
    simp only [runProbes]
    rcases ha : applyProbe read C s x with s' | e
    · have hs' := (runInv_step read C (hX x (by simp)) hs ha).1
      exact (leaves_le_step C hs.inv ha).trans
        (leaves_le_runProbes xs s' (fun y hy => hX y (by simp [hy])) hs')
    · exact le_rfl

/-- Reads agreeing on every prefix of a probe past its first `walkStart` letters, followed by at
most `n + 1` letters, give the same run where it ends with at most `n + 2` leaves. -/
theorem runProbes_congr_small {r r' : FreeMonoid α → ARU} {X : List (FreeMonoid α)} {n : ℕ}
    (h : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ n + 1 → r (prefixOf x j * m) = r' (prefixOf x j * m)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv r C X s →
      (runProbes r C s xs).1.tree.leaves.length ≤ n + 2 →
        runProbes r C s xs = runProbes r' C s xs
  | [], _, _, _, _ => rfl
  | x :: xs, s, hX, hs, hB => by
    have hxX := hX x (by simp)
    have hle := leaves_le_runProbes C (x :: xs) s hX hs
    have hstep : applyProbe r C s x = applyProbe r' C s x :=
      Kappa.applyProbe_congr' C (fun e t w he => hs.keys e t w he)
        (step_reads_agree C h hxX hs (by omega))
    have hB' := hB
    simp only [runProbes] at hB' ⊢
    rw [← hstep]
    rcases ha : applyProbe r C s x with s' | e
    · rw [ha] at hB'
      obtain ⟨hs', -⟩ := runInv_step r C hxX hs ha
      exact runProbes_congr_small h xs s' (fun y hy => hX y (by simp [hy])) hs' hB'
    · rfl

/-- `runProbes_congr_small` from the start. -/
theorem level3_congr_small {r r' : FreeMonoid α → ARU} {T n : ℕ} (xs : Fin T → FreeMonoid α)
    (h : ∀ i j, C.walkStart ≤ j → j ≤ (xs i).toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ n + 1 → r (prefixOf (xs i) j * m) = r' (prefixOf (xs i) j * m))
    (hB : (indecisiveLStarLevel3 C r xs).hypothesis.tree.leaves.length ≤ n + 2) :
    indecisiveLStarLevel3 C r xs = indecisiveLStarLevel3 C r' xs := by
  have hrun := runProbes_congr_small C (X := List.ofFn xs) (n := n)
    (fun x hx j hj hjx m hm => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact h i j hj hjx m hm)
    (List.ofFn xs) initialState (fun _ h => h) (runInv_start r C _) hB
  simp only [indecisiveLStarLevel3, hrun]

end Random

end OrthoDFA
