import OrthoDFA.Loop

/-!
# The L\* loop: the skeleton

`loop_succeeds` from three lemmas, each `sorry` for now:
* `loop_ends`: probes enough for every stretch the loop can run leave it ending;
* `loop_false_end`: a consistent ending or a harvest whose claim fails has chance at most the
  tests' levels, per stretch;
* `loop_too_big`: the tree grows past `Lmax` with chance at most `spurBound`, per stretch.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- The loop's run on the noise and draws `p`. -/
noncomputable def loopOut (C : LoopCfg) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) {P : ℕ} (p : Ω × (Fin P → FreeMonoid α)) :
    LoopState α × Option LoopEnd :=
  loopRun C (readsAt O B F p.1) loopStart (List.ofFn p.2)

variable (C : LoopCfg) (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (P : ℕ)

/-- Each stretch ends by the first look past `nmax`, and a stretch ends at a change, of which
there are fewer than `stretchMax`. -/
theorem loop_ends (h0 : 0 < C.n₀) (hP : stretchMax C (Fintype.card α) * (C.nmax + C.n₀) ≤ P)
    (p : Ω × (Fin P → FreeMonoid α)) : (loopOut C O B F p).2 ≠ none := by
  sorry

/-- A stretch's draws are fresh given what came before, so each test of it settles on the wrong
side of its threshold with chance at most `a` per look. -/
theorem loop_false_end (h0 : 0 < C.n₀) (hL : ∀ᵐ x ∂D, x.toList.length = C.L) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real
        {p | ((loopOut C O B F p).2 = some .agree
            ∨ ∃ c, (loopOut C O B F p).2 = some (.harvest c))
          ∧ ¬ LoopGenuine C (readsAt O B F p.1) D (loopOut C O B F p).1 (loopOut C O B F p).2}
      ≤ stretchMax C (Fintype.card α) * (4 * ((C.nmax + C.n₀) / C.n₀ : ℕ) * C.a) := by
  sorry

/-- Without a spurious split the leaves are at most `|Q| + 2`, and a spurious split needs `m`
records in one stretch whose fresh reads are wrong. -/
theorem loop_too_big {Q : Type*} [Fintype Q] (A : DFA (FreeMonoid α) Q)
    (hA : O.L = {w | A.state w ∈ A.accept}) (hF : SuffixFree F)
    (hL : ∀ᵐ x ∂D, x.toList.length = C.L) (hk : C.k ≤ C.L) (hm : 1 ≤ C.m)
    (hQ : Fintype.card Q + 3 ≤ C.Lmax) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real {p | (loopOut C O B F p).2 = some .tooBig}
      ≤ stretchMax C (Fintype.card α) * spurBound C O B F := by
  sorry

theorem loop_succeeds : LoopSucceeds := by
  intro α _ _ Ω _ μ _ Q _ C A O B F D _ P hA hF hL hk h0 hm hQ hP
  set M := μ.prod (Measure.pi fun _ : Fin P => D)
  set G := {p : Ω × (Fin P → FreeMonoid α) | LoopGenuine C (readsAt O B F p.1) D
    (loopOut C O B F p).1 (loopOut C O B F p).2}
  set E₁ := {p : Ω × (Fin P → FreeMonoid α) | ((loopOut C O B F p).2 = some .agree
      ∨ ∃ c, (loopOut C O B F p).2 = some (.harvest c))
    ∧ ¬ LoopGenuine C (readsAt O B F p.1) D (loopOut C O B F p).1 (loopOut C O B F p).2}
  set E₂ := {p : Ω × (Fin P → FreeMonoid α) | (loopOut C O B F p).2 = some .tooBig}
  have hsub : Gᶜ ⊆ E₁ ∪ E₂ := by
    intro p hp
    have hn := loop_ends C O B F P h0 hP p
    simp only [Set.mem_compl_iff, G, Set.mem_ofPred_eq] at hp
    rcases he : (loopOut C O B F p).2 with _ | e
    · exact absurd he hn
    rw [he] at hp
    cases e with
    | agree => exact .inl ⟨.inl he, by rw [he]; exact hp⟩
    | harvest c => exact .inl ⟨.inr ⟨c, he⟩, by rw [he]; exact hp⟩
    | halve => exact absurd trivial hp
    | tooBig => exact .inr he
  have h1 := loop_false_end C O B F D P h0 hL
  have h2 := loop_too_big C O B F D P A hA hF hL hk hm hQ
  have hcov : (1 : ℝ) ≤ M.real G + M.real Gᶜ := by
    rw [← probReal_univ (μ := M), ← Set.union_compl_self G]
    exact measureReal_union_le _ _
  have hc : M.real Gᶜ ≤ M.real E₁ + M.real E₂ :=
    (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_union_le _ _)
  change _ ≤ M.real G
  rw [mul_add]
  linarith

end OrthoDFA
