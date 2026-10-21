import OrthoDFA.Proofs.DFABridge
import OrthoDFA.Proofs.StateAgree

/-!
# Every string reads as a covered state's

Where the start `q*` re-roots the target on every string of each length from `a = ⌈L/2⌉` to `2L`
with its run in the covered states (`CoveredAccurate`), every state reachable from `q*` is covered,
being reachable in fewer than `|Q|` letters (`covered_of_reach`), and a string at least `a` long
reaches a state equivalent to the one `q*` reaches along it, the two agreeing on every word
shorter than `|Q|²` (`equiv_of_long`). A string `z` shorter than `a` followed by a suffix of the
family is a string of a length the start re-roots, so `z` and `p*z`, `p*` reaching `q*`, have the
family's votes alike. Either way `z` reads as a string at a covered state (`read_covered`), and
each covered state, holding at least `m₀` of a population the family leaves undecided at most `c`
of the time, reads undecided at most `c/m₀` of the time.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

/-- The start `q` re-roots the target on every string of each length from `⌈L/2⌉` to `2L`, its
run staying in the covered states. -/
def CoveredAccurate (M : _root_.DFA α σ) (L : ℕ) (c : ℝ) (q : σ) : Prop :=
  ∀ ℓ, (L + 1) / 2 ≤ ℓ → ℓ ≤ 2 * L → ∀ x : FreeMonoid α, x.toList.length = ℓ →
    (∀ i ≤ ℓ, M.evalFrom q (prefixOf x i).toList ∈ coveredStates M L c)
      ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)

variable (M : _root_.DFA α σ) {L : ℕ} {c : ℝ} {q : σ}

omit [DecidableEq α] in
/-- Every state reachable from the start is covered. -/
theorem covered_of_reach [Nonempty α] (hq : CoveredAccurate M L c q)
    (hσ : Fintype.card σ ≤ (L + 1) / 2) (hL : (L + 1) / 2 ≤ 2 * L) (w : List α) :
    M.evalFrom q w ∈ coveredStates M L c := by
  obtain ⟨w', hw', he⟩ := evalFrom_short M q w
  obtain ⟨a⟩ := (inferInstance : Nonempty α)
  set x : FreeMonoid α := FreeMonoid.ofList (w' ++ List.replicate ((L + 1) / 2 - w'.length) a)
  have hx : x.toList.length = (L + 1) / 2 := by
    simp only [x, FreeMonoid.toList_ofList, List.length_append, List.length_replicate]; omega
  have := (hq _ le_rfl hL x hx).1 w'.length (by omega)
  rw [← he]
  convert this using 2
  simp [x, prefixOf, List.take_append_of_le_length le_rfl]

omit [DecidableEq α] in
/-- A string at least `⌈L/2⌉` long reaches a state equivalent to the one the start reaches along
it. -/
theorem equiv_of_long (hq : CoveredAccurate M L c q)
    (hL : (L + 1) / 2 + Fintype.card σ * Fintype.card σ ≤ 2 * L) (z : List α)
    (hz : (L + 1) / 2 ≤ z.length) (w : List α) :
    M.evalFrom (M.eval z) w ∈ M.accept ↔ M.evalFrom (M.evalFrom q z) w ∈ M.accept := by
  set a := (L + 1) / 2
  have hagree : ∀ u : List α, u.length < Fintype.card σ * Fintype.card σ →
      (M.evalFrom (M.eval (z.take a)) u ∈ M.accept
        ↔ M.evalFrom (M.evalFrom q (z.take a)) u ∈ M.accept) := by
    intro u hu
    have hlen : (FreeMonoid.ofList (z.take a ++ u)).toList.length = a + u.length := by
      simp only [FreeMonoid.toList_ofList, List.length_append, List.length_take]; omega
    have h := (hq (a + u.length) le_self_add (by omega) _ hlen).2
    simp only [FreeMonoid.toList_ofList, DFA.evalFrom_of_append, DFA.eval] at h
    exact h.symm
  have hall := evalFrom_agree_of_short M hagree
  have e1 : M.eval z = M.evalFrom (M.eval (z.take a)) (z.drop a) := by
    conv_lhs => rw [← List.take_append_drop a z]
    exact DFA.evalFrom_of_append _ _ _ _
  have e2 : M.evalFrom q z = M.evalFrom (M.evalFrom q (z.take a)) (z.drop a) := by
    conv_lhs => rw [← List.take_append_drop a z]
    exact DFA.evalFrom_of_append _ _ _ _
  rw [e1, e2, ← DFA.evalFrom_of_append, ← DFA.evalFrom_of_append]
  exact hall _

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  (O : Oracle μ (FreeMonoid α)) (hlang : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)

omit [DecidableEq α] in
/-- Strings whose family's suffixes are in the language alike read alike. -/
theorem readProb_congr_lang {F : Finset (FreeMonoid α)} {kl kh : ℕ} {z z' : FreeMonoid α}
    (h : ∀ v ∈ F, z * v ∈ O.L ↔ z' * v ∈ O.L) : readProb O F kl kh z = readProb O F kl kh z' := by
  classical
  funext rd
  have e : acceptCount O.L F z = acceptCount O.L F z' := by
    unfold acceptCount
    congr 1
    exact Finset.filter_congr fun v hv => h v hv
  rw [readProb_eq_readLaw, readProb_eq_readLaw, e]

omit [DecidableEq α] in
include hlang in
/-- Every string reads as a string at a state the start reaches. -/
theorem read_covered [Nonempty α] (hq : CoveredAccurate M L c q) {p : FreeMonoid α}
    (hp : M.eval p.toList = q) {F : Finset (FreeMonoid α)} {Ls : ℕ}
    (hF : ∀ v ∈ F, v.toList.length = Ls) (hLs : (L + 1) / 2 ≤ Ls) (hLs2 : (L + 1) / 2 + Ls ≤ 2 * L)
    (hL : (L + 1) / 2 + Fintype.card σ * Fintype.card σ ≤ 2 * L) (kl kh : ℕ) (z : FreeMonoid α) :
    ∃ z' : FreeMonoid α, M.eval z'.toList = M.evalFrom q z.toList
      ∧ readProb O F kl kh z = readProb O F kl kh z' := by
  refine ⟨p * z, by simp [DFA.eval, DFA.evalFrom_of_append, ← hp], readProb_congr_lang O fun v hv => ?_⟩
  rw [hlang, hlang]
  simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
  change M.evalFrom (M.eval z.toList) v.toList ∈ M.accept
    ↔ M.evalFrom (M.evalFrom (M.eval p.toList) z.toList) v.toList ∈ M.accept
  rw [hp]
  by_cases hz : (L + 1) / 2 ≤ z.toList.length
  · exact equiv_of_long M hq hL z.toList hz v.toList
  · have hlen : (z * v).toList.length = z.toList.length + Ls := by
      simp [hF v hv]
    have h := (hq _ (by omega) (by omega) (z * v) hlen).2
    simp only [FreeMonoid.toList_mul, DFA.evalFrom_of_append, DFA.eval] at h
    exact h.symm

include hlang in
/-- Every string reads undecided at most as often as some covered state does. -/
theorem undecided_le_of_covered [Nonempty α] (hq : CoveredAccurate M L c q) {p : FreeMonoid α}
    (hp : M.eval p.toList = q) {F : Finset (FreeMonoid α)} {Ls : ℕ}
    (hF : ∀ v ∈ F, v.toList.length = Ls) (hLs : (L + 1) / 2 ≤ Ls) (hLs2 : (L + 1) / 2 + Ls ≤ 2 * L)
    (hL : (L + 1) / 2 + Fintype.card σ * Fintype.card σ ≤ 2 * L)
    (hσ : Fintype.card σ ≤ (L + 1) / 2) (kl kh : ℕ) {u : ℝ}
    (hu : ∀ q' ∈ coveredStates M L c, familyReadDist O M F kl kh q' .undecided ≤ u)
    (z : FreeMonoid α) : familyReadDist O M F kl kh (M.eval z.toList) .undecided ≤ u := by
  obtain ⟨z', hz', he⟩ := read_covered M O hlang hq hp hF hLs hLs2 hL kl kh z
  rw [← readProb_eq_familyReadDist O M hlang, he, readProb_eq_familyReadDist O M hlang, hz']
  exact hu _ (covered_of_reach M hq hσ (by omega) z.toList)

end OrthoDFA
