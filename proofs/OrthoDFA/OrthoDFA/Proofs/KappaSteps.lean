import OrthoDFA.Proofs.KappaReads
import OrthoDFA.Proofs.EpsReads

/-!
# Level 5's steps

A step reads what its probe reads (`probeReads`) and, where it splits, what the split re-sifts
(`splitReads`). Where none of that is read wrong, the step keeps `WitInv` (`kappa_step_change`),
and reads agreeing on it give the same step (`applyProbe_congr'`). The round's reads come in rounds
(`rounds'`): round `2j` is what the `j`-th probe reads, round `2j + 1` what its split re-sifts.
While nothing read so far is wrong, each round is fixed by what it and the earlier ones read.
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open Eps

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- The step reads a string, one its probe reads or its split re-sifts, on its state's wrong
side. -/
def ReadsWrong (s : Level3State α) (x : FreeMonoid α) : Prop :=
  ∃ z ∈ probeReads read s.tree s.edges C.walkStart x ∪ splitReads read C s x,
    read z = if side (M.eval z.toList) then .reject else .accept

/-- A step that changes the hypothesis and reads nothing wrong starts a fresh stretch, keeps the
invariants and lowers `Ψr`. -/
theorem kappa_step_change {s s' : Level3State α} (hs : Inv read s) (hg : WitInv M side s)
    {x : FreeMonoid α} (hx : ¬ ReadsWrong M side read C s x)
    (h : applyProbe read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  classical
  refine eps_step_change_of M side read C hs hg (fun {p c u t} ho hu huc => ?_)
    (fun z hz hw => hx ⟨z, Finset.mem_union_right _ hz, hw⟩) h hne
  have hR := probeReads_edge s.edges C.walkStart ho
  exact ⟨(sift_true_reads M side read _ _ _ hu fun w hw hw' =>
      hx ⟨w, Finset.mem_union_left _ (hR w (Finset.mem_union_left _ hw)), hw'⟩).symm,
    (sift_true_reads M side read _ _ _ huc fun w hw hw' =>
      hx ⟨w, Finset.mem_union_left _ (hR w (Finset.mem_union_right _ hw)), hw'⟩).symm⟩

omit [Fintype σ] in
/-- Reads agreeing on what a step reads give the same step, its learned edges leaving leaves. -/
theorem applyProbe_congr' {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : ∀ z ∈ probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x, r z = r' z) :
    applyProbe r C s x = applyProbe r' C s x :=
  applyProbe_congr_of C hlv
    (probeReads_congr s.edges C.walkStart fun z hz => h z (Finset.mem_union_left _ hz)).1.symm
    fun z hz => h z (Finset.mem_union_right _ hz)

/-- Round `2j` is what the `j`-th probe reads, round `2j + 1` what its split re-sifts. -/
noncomputable def rounds' (xs : List (FreeMonoid α)) (i : ℕ) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  match stateAt C read initialState xs (i / 2), xs[i / 2]? with
  | some s, some x => if i % 2 = 0 then probeReads read s.tree s.edges C.walkStart x
      else splitReads read C s x
  | _, _ => ∅

omit [Fintype σ] in
theorem rounds'_even (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level3State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) :
    rounds' C xs (2 * j) read = probeReads read s.tree s.edges C.walkStart x := by
  have h1 : 2 * j / 2 = j := by omega
  simp only [rounds', h1, hs, hx]
  simp

omit [Fintype σ] in
theorem rounds'_odd (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level3State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) : rounds' C xs (2 * j + 1) read = splitReads read C s x := by
  have h1 : (2 * j + 1) / 2 = j := by omega
  simp only [rounds', h1, hs, hx]
  simp

/-- No string the rounds before `n` read is read wrong. -/
def Good' (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) (n : ℕ) : Prop :=
  ∀ i < n, ∀ z ∈ rounds' C xs i read,
    read z ≠ if side (M.eval z.toList) then .reject else .accept

/-- While nothing read is wrong, the steps keep the invariants. -/
theorem inv_of_good' (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) :
    ∀ (j : ℕ) (s : Level3State α), stateAt C read initialState xs j = some s →
      Good' M side C xs read (2 * j) → Inv read s ∧ WitInv M side s
  | 0, s, h, _ => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    exact ⟨inv_start read, witInv_start M side⟩
  | j + 1, s, h, hg => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C read initialState xs j with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[j]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe read C s₁ x with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨hI, hW⟩ := inv_of_good' xs read j s₁ hs₁ fun i hi => hg i (by omega)
    have hx' : ¬ ReadsWrong M side read C s₁ x := by
      rintro ⟨z, hz, hw⟩
      rcases Finset.mem_union.1 hz with hz | hz
      · exact hg (2 * j) (by omega) z (by rw [rounds'_even C xs read hs₁ hx]; exact hz) hw
      · exact hg (2 * j + 1) (by omega) z (by rw [rounds'_odd C xs read hs₁ hx]; exact hz) hw
    by_cases hsm : s'.tree = s₁.tree ∧ s'.edges = s₁.edges
    · exact ⟨(step_same read C hI ha hsm.1 hsm.2).1, witInv_congr M side hW hsm.1 hsm.2⟩
    · obtain ⟨h1, h2, -⟩ := kappa_step_change M side read C hI hW hx' ha hsm
      exact ⟨h1, h2⟩

/-- Reads agreeing with good ones on what the rounds before `2j` read reach the same state. -/
theorem stateAt_congr' (xs : List (FreeMonoid α)) {r r' : FreeMonoid α → ARU} :
    ∀ j, Good' M side C xs r (2 * j) → (∀ i < 2 * j, ∀ z ∈ rounds' C xs i r, r z = r' z) →
      stateAt C r initialState xs j = stateAt C r' initialState xs j
  | 0, _, _ => by simp only [stateAt]
  | j + 1, hg, ha => by
    rw [stateAt_succ, stateAt_succ, ← stateAt_congr' xs j (fun i hi => hg i (by omega))
      fun i hi => ha i (by omega)]
    rcases hs₁ : stateAt C r initialState xs j with _ | s₁
    · rfl
    rcases hx : xs[j]? with _ | x
    · rfl
    simp only [Option.bind_some]
    have hW := (inv_of_good' M side C xs r j s₁ hs₁ fun i hi => hg i (by omega)).2
    rw [applyProbe_congr' C (fun e t w h => edge_leaf M side hW h) fun z hz => ?_]
    rcases Finset.mem_union.1 hz with hz | hz
    · exact ha (2 * j) (by omega) z (by rw [rounds'_even C xs r hs₁ hx]; exact hz)
    · exact ha (2 * j + 1) (by omega) z (by rw [rounds'_odd C xs r hs₁ hx]; exact hz)

/-- While nothing read is wrong, each round is fixed by what it and the earlier ones read. -/
theorem rounds'_local (xs : List (FreeMonoid α)) (i : ℕ) (r r' : FreeMonoid α → ARU)
    (hg : Good' M side C xs r i) (ha : ∀ l ≤ i, ∀ z ∈ rounds' C xs l r, r z = r' z) :
    rounds' C xs i r = rounds' C xs i r' := by
  have hj : 2 * (i / 2) ≤ i := Nat.mul_div_le i 2
  have hst := stateAt_congr' M side C xs (i / 2) (fun l hl => hg l (by omega))
    fun l hl => ha l (by omega)
  unfold rounds'
  rw [← hst]
  rcases hs : stateAt C r initialState xs (i / 2) with _ | s
  · rfl
  rcases hx : xs[i / 2]? with _ | x
  · rfl
  simp only
  have hpr : ∀ z ∈ probeReads r s.tree s.edges C.walkStart x, r z = r' z := fun z hz =>
    ha (2 * (i / 2)) hj z (by rw [rounds'_even C xs r hs hx]; exact hz)
  obtain ⟨h1, h2⟩ := probeReads_congr s.edges C.walkStart hpr
  split_ifs with hi
  · exact h2.symm
  · simp only [splitReads, h1]

end Kappa

end Random

end OrthoDFA
