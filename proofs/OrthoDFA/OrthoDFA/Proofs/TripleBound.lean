import OrthoDFA.Proofs.PassReadsK

/-!
# The triples' claim

Given the pass's bits, draws with different first `k` letters read different strings, so their
contributions are independent, each pulled down on average by `fresh_first_le`. Chebyshev over
the pass's cells bounds the fluctuation by the most any one first `k` letters carry.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace Qry

variable {β : Type*} {Ω : Type*}

/-- A computation asking only strings whose answers are measurable returns, with its trace,
something measurable. -/
theorem measurableSet_run_trace {m : MeasurableSpace Ω} (cut : Ω → FreeMonoid α → Option Bool)
    {S : FreeMonoid α → Prop} (hS : ∀ w, S w → ∀ o, MeasurableSet[m] {ω | cut ω w = o}) :
    ∀ q : Qry α β, q.AsksIn S → ∀ A : Set (β × List (FreeMonoid α × Bool)),
      MeasurableSet[m] {ω | (q.run (cut ω), q.trace (cut ω)) ∈ A}
  | pure b, _, A => by
    by_cases h : (b, ([] : List (FreeMonoid α × Bool))) ∈ A
    · convert @MeasurableSet.univ Ω m using 1; ext ω; simp [run, trace, h]
    · convert @MeasurableSet.empty Ω m using 1; ext ω; simp [run, trace, h]
  | ask w g k, h, A => by
    have hset : {ω | ((ask w g k).run (cut ω), (ask w g k).trace (cut ω)) ∈ A}
        = ⋃ o : Option Bool, ({ω | cut ω w = o}
          ∩ {ω | ((k o).run (cut ω), (k o).trace (cut ω))
            ∈ (fun p : β × List (FreeMonoid α × Bool) => (p.1, (w, g) :: p.2)) ⁻¹' A}) := by
      ext ω
      simp only [run, trace, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff,
        Set.mem_preimage]
      exact ⟨fun hω => ⟨cut ω w, rfl, hω⟩, fun ⟨o, ho, hω⟩ => ho ▸ hω⟩
    rw [hset]
    exact MeasurableSet.iUnion fun o =>
      (hS w h.1 o).inter (measurableSet_run_trace cut hS (k o) (h.2 o) _)

end Qry

omit [Fintype α] [DecidableEq α] in
theorem visited_le (agrees : ℕ → Option Bool) : ∀ fuel lo hi, visited agrees fuel lo hi ≤ fuel
  | 0, _, _ => le_rfl
  | fuel + 1, lo, hi => by
    simp only [visited]
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh]; omega
    rw [if_pos hlh]
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r
    have h1 := visited_le agrees fuel ((lo + hi) / 2) hi
    have h2 := visited_le agrees fuel lo ((lo + hi) / 2)
    have h3 := visited_le agrees fuel ((lo + hi) / 2 + 1) hi
    have h4 := visited_le agrees fuel lo ((lo + hi) / 2 - 1)
    rcases v with _ | _ | _ <;> rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;>
      simp only [] <;> omega

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_triple_gt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .triple j → lo < j
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
      · obtain rfl := Outcome.triple.inj h; omega
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_triple_lt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .triple j → j < hi
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      · have := bracketAt_triple_lt agrees ps fuel _ _ j h; omega
      · have := bracketAt_triple_lt agrees ps fuel _ _ j h; omega
      · obtain rfl := Outcome.triple.inj h; omega
      · have := bracketAt_triple_lt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_lt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_lt agrees ps fuel _ _ j h; omega

section Cells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

variable {Q : Type*}

end Cells

section PassCells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

end PassCells

section Words

end Words

section Tail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

end Tail

section Draws

variable {Q : Type*}

theorem route_inr_form {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (y b : FreeMonoid α), (t.route cut y).2 = .inr b → ∃ m, b = y * m
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, y, b, h => by
    simp only [DTree.route] at h
    rcases hc : cut (y * m) with _ | _ | _ <;> simp only [hc] at h
    · exact ⟨m, (Sum.inr.inj h).symm⟩
    · rcases hr : (r.route cut y).2 with _ | b' <;> rw [hr] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h; exact route_inr_form r y b' hr
    · rcases ha : (a.route cut y).2 with _ | b' <;> rw [ha] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h; exact route_inr_form a y b' ha

end Draws

section Fail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

end Fail

end OrthoDFA
