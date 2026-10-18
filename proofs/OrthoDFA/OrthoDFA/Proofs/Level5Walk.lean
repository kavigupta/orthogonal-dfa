import OrthoDFA.Proofs.Level5Parts
import OrthoDFA.Proofs.Level5Probe
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.PassReads
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Level 5's walks

A record's edge is out of a leaf on the walk and points elsewhere than the leaf its next prefix
sifts to. The edge a position is charged to leaves a leaf.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Walk

variable (cut : FreeMonoid α → Option Bool)

theorem walk_mem_leaves {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {p : List Bool}
    (hp : p ∈ T.leaves) (cs : List α) : ∀ q ∈ walk edges p cs, q ∈ T.leaves :=
  walk_mem edges (fun q _ c t w h => he q c t w h) p cs hp

theorem recordBy_eq_some {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {p : List Bool} {c : α} {t : List Bool} {sp : FreeMonoid α} :
    recordBy cut k (T, edges) x = some ((p, c, t), sp)
      ↔ probeOrAgree (ofCut cut) T edges k x = .wrongEdge p c sp t := by
  unfold recordBy
  rcases probeOrAgree (ofCut cut) T edges k x with _ | ⟨p', c', u', t'⟩ | _ | _ | _ <;>
    simp only [reduceCtorEq, false_iff, iff_false, not_false_eq_true]
  constructor
  · rintro h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨⟨rfl, rfl, rfl⟩, rfl⟩ := h
    rfl
  · rintro h
    simp only [Outcome.wrongEdge.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    rfl

/-- What a record says about its walk: the edge it is at, out of a leaf on the walk, points
elsewhere than the leaf its next prefix sifts to. -/
theorem recordBy_spec {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {p : List Bool}
    {c : α} {t : List Bool} {sp : FreeMonoid α}
    (h : recordBy cut k (T, edges) x = some ((p, c, t), sp)) :
    ∃ p₀, T.sift (ofCut cut) (prefixOf x k) = .inl p₀ ∧ p ∈ walk edges p₀ (x.toList.drop k)
      ∧ T.sift (ofCut cut) (sp * FreeMonoid.of c) = .inl t
      ∧ ∃ t₀ w, edges (p, c) = some (t₀, w) ∧ t₀ ≠ t := by
  obtain ⟨p₀, fd, h₀, hkf, hfx, rfl, hc, hlt, rfl, -, ht, hE⟩ :=
    probe_wrongEdge _ T edges k x ((recordBy_eq_some cut).1 h)
  refine ⟨p₀, h₀, ?_, ?_, hE⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]
    exact List.getElem_mem hlt
  · rw [← prefixOf_succ hc, show fd - 1 + 1 = fd by omega]
    exact ht

theorem mem_searchAt {rd : FreeMonoid α → ARU} {k : ℕ} {T : DTree α} {edges : Edges α}
    {x : FreeMonoid α} :
    x ∈ searchAt rd k T edges ↔ (probeOrAgree rd T edges k x).IsSearch := by
  simp only [searchAt_eq, Set.mem_setOf_eq]
  rcases probeOrAgree rd T edges k x with _ | _ | _ | _ | ⟨zs, _ | _⟩ <;> simp [Outcome.IsSearch]

/-- The edge a position is charged to leaves a leaf. -/
theorem posEdgeFrom_mem_paths {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} {p₀ : List Bool} (hp : p₀ ∈ T.leaves) {i : ℕ} {d : List Bool × α}
    (h : posEdgeFrom edges k x p₀ i = some d) : d.1 ∈ T.leaves := by
  unfold posEdgeFrom at h
  simp only at h
  generalize hA : (walk edges p₀ (x.toList.drop k))[i - k]? = A at h
  generalize hB : x.toList[i]? = B at h
  generalize hC : (walk edges p₀ (x.toList.drop k))[i - 1 - k]? = C at h
  generalize hD : x.toList[i - 1]? = D at h
  rcases A with _ | a <;> rcases B with _ | b <;> rcases C with _ | c <;> rcases D with _ | e <;>
    simp only [Option.orElse_some, Option.orElse_none, Option.some.injEq, reduceCtorEq] at h
  all_goals first
    | (subst h; exact walk_mem_leaves he hp _ a (List.mem_of_getElem? hA))
    | (subst h; exact walk_mem_leaves he hp _ c (List.mem_of_getElem? hC))

end Walk

end OrthoDFA
