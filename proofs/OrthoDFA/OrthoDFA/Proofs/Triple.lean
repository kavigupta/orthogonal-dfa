import OrthoDFA.Proofs.Query
import OrthoDFA.Proofs.StartAtK

/-!
# The triples' harvest

A triple's harvest is a read the cut leaves undecided. In a probe whose search ends at a triple,
every read outside the search's middles is decided, so the harvested string was first read, and
left undecided, at one of the middles (`triple_first_read`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Trace

variable (cut : FreeMonoid α → Option Bool)

omit [Fintype α] [DecidableEq α] in
theorem route_undecided : ∀ (t : DTree α) (w b : FreeMonoid α), (t.route cut w).2 = .inr b →
    b ∈ (t.route cut w).1 ∧ cut b = none
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, w, b, h => by
    simp only [DTree.route] at h ⊢
    rcases hc : cut (w * m) with _ | _ | _ <;> simp only [hc] at h ⊢
    · obtain rfl := Sum.inr.inj h
      exact ⟨by simp, hc⟩
    · rcases hr : (r.route cut w).2 with _ | b' <;> rw [hr] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h
        obtain ⟨h1, h2⟩ := route_undecided r w b' hr
        exact ⟨by simp [h1], h2⟩
    · rcases ha : (a.route cut w).2 with _ | b' <;> rw [ha] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h
        obtain ⟨h1, h2⟩ := route_undecided a w b' ha
        exact ⟨by simp [h1], h2⟩

end Trace

section Fresh

open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

end Fresh

end OrthoDFA
