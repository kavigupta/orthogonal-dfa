import OrthoDFA.Proofs.LoopCheckRuns
import OrthoDFA.Proofs.LoopRenoise

/-!
# The checked round read through a re-noised family

After `T` draws the checked run's tree has at most `T + 2` leaves, so its midfixes are at most `T`
long, and the run reads only strings at most a draw's length plus `T + 1` long
(`checkResult_congr_short`). So whether it fails its claim is measurable (`checkFail`), and over a
family read through any noise it does so with chance at most `Level4CheckRoundCorrect`'s bound
(`checkFail_le`): a family's reads are a `KappaReadOracle` (`FamilyKappaOracle`), at `κ = 0` level
4's oracle.
-/

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace Random

variable (read : FreeMonoid α → ARU) (C : Level3Settings) (K : CheckSettings)

/-- After the draws `ds` the run's tree has at most `|ds| + 2` leaves and keeps the invariant. -/
theorem runChecked_tree {X Y : List (FreeMonoid α)} :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α), (∀ d ∈ ds, d.1 ∈ X ∧ ∀ i, d.2 i ∈ Y) →
      CheckRunInv read C X Y s →
      CheckRunInv read C X Y (runChecked read C K s ds).1
        ∧ (runChecked read C K s ds).1.tree.leaves.length ≤ s.tree.leaves.length + ds.length
  | [], s, _, hs => ⟨hs, by simp [runChecked]⟩
  | d :: ds, s, hd, hs => by
    simp only [runChecked]
    rcases ha : applyCheck read C K s d with s' | e
    · obtain ⟨hs', hl, -⟩ := checkRunInv_step read C K (hd d (by simp)).1 (hd d (by simp)).2 hs ha
      obtain ⟨h1, h2⟩ := runChecked_tree ds s' (fun d' hd' => hd d' (by simp [hd'])) hs'
      refine ⟨h1, ?_⟩
      simp only [List.length_cons]
      omega
    · exact ⟨hs, by simp⟩

/-- After `T` draws the run's midfixes are at most `T` long. -/
theorem runChecked_mids (ds : List (CheckDraw α K)) :
    ∀ m ∈ mids (runChecked read C K initialState ds).1.tree, m.toList.length ≤ ds.length := by
  obtain ⟨h1, h2⟩ := runChecked_tree read C K (X := ds.map Prod.fst)
    (Y := ds.flatMap fun d => List.ofFn d.2) ds initialState
    (fun d hd => ⟨List.mem_map_of_mem hd, fun i => List.mem_flatMap.2 ⟨d, hd, by simp⟩⟩)
    (checkRunInv_start read C _ _)
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  rw [h0] at h2
  exact fun m hm => ((classSet_props _ _ h1.inv.cls).1 m hm).trans (by omega)

/-- The draws' longest string. -/
def drawsLen {T : ℕ} (ds : Fin T → CheckDraw α K) : ℕ :=
  Finset.univ.sup fun i => (ds i).1.toList.length ⊔ Finset.univ.sup fun j => ((ds i).2 j).toList.length

omit [Fintype α] [DecidableEq α] in
theorem le_drawsLen {T : ℕ} (ds : Fin T → CheckDraw α K) (i : Fin T) :
    (ds i).1.toList.length ≤ drawsLen K ds ∧ ∀ j, ((ds i).2 j).toList.length ≤ drawsLen K ds := by
  have h := Finset.le_sup (f := fun i => (ds i).1.toList.length ⊔
    Finset.univ.sup fun j => ((ds i).2 j).toList.length) (Finset.mem_univ i)
  refine ⟨le_trans le_sup_left h, fun j => le_trans ?_ (le_trans le_sup_right h)⟩
  exact Finset.le_sup (f := fun j => ((ds i).2 j).toList.length) (Finset.mem_univ j)

/-- Reads agreeing on every string at most `drawsLen + T + 1` long give the same round. -/
theorem checkResult_congr_short {r r' : FreeMonoid α → ARU}
    (ds : Fin C.probeBudget → CheckDraw α K)
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ drawsLen K ds + C.probeBudget + 1 → r z = r' z) :
    indecisiveLStarCheck C K r ds = indecisiveLStarCheck C K r' ds := by
  have hlen : ∀ (y : FreeMonoid α) (j : ℕ) (m : FreeMonoid α), y.toList.length ≤ drawsLen K ds →
      m.toList.length ≤ C.probeBudget + 1 → r (prefixOf y j * m) = r' (prefixOf y j * m) := by
    intro y j m hy hm
    refine h _ ?_
    have : (prefixOf y j).toList.length ≤ y.toList.length := by simp [prefixOf]
    simp only [FreeMonoid.toList_mul, List.length_append]
    omega
  have hrun := runChecked_congr_zone C K (r := r) (r' := r') (X := (List.ofFn ds).map Prod.fst)
    (Y := (List.ofFn ds).flatMap fun d => List.ofFn d.2) (n := C.probeBudget)
    (fun x hx j _ _ m hm => by
      obtain ⟨d, hd, rfl⟩ := List.mem_map.1 hx
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hd
      exact hlen _ j m (le_drawsLen K ds i).1 hm)
    (fun y hy j _ m hm => by
      obtain ⟨d, hd, hy⟩ := List.mem_flatMap.1 hy
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hd
      obtain ⟨j', rfl⟩ := List.mem_ofFn.1 hy
      exact hlen _ j m ((le_drawsLen K ds i).2 j') hm)
    (List.ofFn ds) initialState
    (fun d hd => ⟨List.mem_map_of_mem hd, fun i => List.mem_flatMap.2 ⟨d, hd, by simp⟩⟩)
    (checkRunInv_start r C _ _) (by
      have := runChecked_mids r C K (List.ofFn ds)
      rw [List.length_ofFn] at this
      exact this)
  simp only [indecisiveLStarCheck, hrun]

end Random

end OrthoDFA
