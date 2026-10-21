import OrthoDFA.AllowanceClaim
import OrthoDFA.Proofs.KappaRound

/-!
# Level 3 within a leaf allowance: the proof

The target padded with `f` states no string reaches (`DFA.pad`) has the same runs and the same
indecision, and `|Q| + f` states. The κ claim's frame (`eps_round_le_of`) takes any invariant that
keeps the tree in level 3's class; here it is that the tree has at most `|Q| + f + 2` leaves, kept
by every step that does not grow it past them (`keepsInv_grows`).
-/

namespace OrthoDFA

/-- `M` with `f` more states, each looping on every letter, none reachable from `M`'s. -/
def DFA.pad {α σ : Type*} (M : _root_.DFA α σ) (f : ℕ) : _root_.DFA α (σ ⊕ Fin f) where
  step q a := q.elim (fun q => .inl (M.step q a)) .inr
  start := .inl M.start
  accept := .inl '' M.accept

namespace Random

theorem pad_evalFrom {α σ : Type*} (M : _root_.DFA α σ) (f : ℕ) :
    ∀ (l : List α) (q : σ), (DFA.pad M f).evalFrom (.inl q) l = .inl (M.evalFrom q l)
  | [], _ => rfl
  | a :: l, q => by
    simp only [_root_.DFA.evalFrom, List.foldl_cons] at pad_evalFrom ⊢
    exact pad_evalFrom M f l (M.step q a)

theorem pad_eval {α σ : Type*} (M : _root_.DFA α σ) (f : ℕ) (l : List α) :
    (DFA.pad M f).eval l = .inl (M.eval l) :=
  pad_evalFrom M f l M.start

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

open Eps

/-- A step that changes the hypothesis without growing the tree past `|Q| + 2` leaves keeps the
invariant and lowers `Ψr`. -/
theorem keepsInv_grows {σ : Type*} [Fintype σ] :
    KeepsInv read C (σ := σ) (fun s => s.tree.leaves.length ≤ Fintype.card σ + 2)
      (Grows read C (Fintype.card σ)) := by
  intro s s' x hs hJ hW h hne
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · exact ⟨⟨edgesOK_upd read hs.edges hu huc, hs.cls, hs.two⟩, hJ, rfl, rfl,
      Level2.Ψ_lt_learn _ (s := Level2.newHypothesis s.tree s.edges 0) hp hE _ 0⟩
  · have hJ' : (splitLeaf read s p c t t₀).tree.leaves.length ≤ Fintype.card σ + 2 := by
      by_contra hc
      exact hW ⟨_, h, by omega⟩
    have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have ht := DTree.sift_inl_mem read _ _ t huc
    have ht₀ := DTree.sift_inl_mem read _ _ t₀ (hs.edges p hp c t₀ w₀ hE).2
    refine ⟨⟨edgesOK_retarget read hs.edges hp _, ?_, ?_⟩, hJ', rfl, rfl, ?_⟩
    · simp only [splitLeaf, newHypothesis]
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hs.two; omega]
      exact classSet_split hs.cls hp ht ht₀
    · simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega
    · exact Level2.Ψ_lt_split _ (s := Level2.newHypothesis s.tree s.edges 0) hp hJ' _ 0
  · exact absurd ⟨rfl, rfl⟩ hne

omit [Fintype α] in
theorem anyB_grows (n : ℕ) :
    ∀ (s : Level3State α) (xs : List (FreeMonoid α)),
      AnyB (applyProbe read C) (Grows read C n) s xs ↔ GrowsRun read C n s xs
  | _, [] => Iff.rfl
  | s, x :: xs => by
    simp only [AnyB, GrowsRun]
    exact or_congr Iff.rfl (exists_congr fun s' => and_congr Iff.rfl (anyB_grows n s' xs))

end Random

open MeasureTheory Random Eps

theorem level3AllowanceCorrect_holds : Level3AllowanceCorrect := by
  intro α _ _ σ _ M U Ω _ μ _ read hmeas hind hund P C p₀ f
  dsimp only
  intro hpre hδ hδ1 hy0 hyρ hρ1 hP
  set M' := DFA.pad M f
  set U' : σ ⊕ Fin f → ℝ := Sum.elim U fun _ => 0
  have hcard : Fintype.card (σ ⊕ Fin f) = Fintype.card σ + f := by simp
  have heval : ∀ z : FreeMonoid α, M'.eval z.toList = .inl (M.eval z.toList) := fun z =>
    pad_eval M f z.toList
  have hund' : ∀ z, μ.real {ω | read z ω = .undecided} = U' (M'.eval z.toList) := by
    intro z; rw [heval]; exact hund z
  have hp₀ : 0 < p₀ := (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  obtain ⟨kn, hA, hθg, hηm0, hη0, hAe, hAm, hround⟩ := round_knobs (σ := σ ⊕ Fin f) (α := α)
    C P.L p₀ hp₀ hδ hδ1 hy0 hyρ hρ1 (by rw [hcard]; exact_mod_cast hP)
  rw [hcard] at hη0 hAe hAm hround
  have key := eps_round_le_of C M' U'
    (A := ((Fintype.card σ + f : ℕ) + 1 : ℝ) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 P.L)))
    (sl := p₀ * Real.log (16 * ((Fintype.card σ + f : ℕ) + 1)
      * ((1 + ((Fintype.card σ + f) + 1) ^ 3 * Fintype.card α) ^ (Fintype.card σ + f)
        * ((Fintype.card σ + f) + 3) ^ (((Fintype.card σ + f) + 2) * Fintype.card α) : ℕ)
      / C.δ) / C.harvestRate)
    (ε := C.disagreementRate) μ read hmeas hind hund'
    (fun s => s.tree.leaves.length ≤ Fintype.card (σ ⊕ Fin f) + 2)
    (fun _ _ ht _ h => by beta_reduce; rw [ht]; exact h)
    (by simp [initialState, newHypothesis, DTree.leaves])
    (fun _ s hs h => classSet_mono' (by omega) hs.cls)
    (fun rd => Grows rd C (Fintype.card (σ ⊕ Fin f))) (fun rd => keepsInv_grows rd C) P _ p₀ _
    kn hA hp₀ hpre hy0 (by linarith) hη0 (by rw [hcard]; exact hAe) (by rw [hcard]; exact hAm)
    le_rfl hθg hηm0
  rw [hcard] at key
  refine le_trans (le_of_eq ?_)
    (key.trans (add_le_add (ENNReal.ofReal_le_ofReal hround) (le_of_eq ?_)))
  · congr 1; funext ω; congr 1; ext d
    simp only [Set.mem_ofPred_eq]
    unfold EndsWell
    simp only [heval, U', Sum.elim_inl]
  · congr 1; funext ω; congr 1; ext d
    exact anyB_grows (read · ω) C _ _ _

end OrthoDFA
