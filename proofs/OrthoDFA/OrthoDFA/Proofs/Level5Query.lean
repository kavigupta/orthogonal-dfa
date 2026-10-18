import OrthoDFA.Proofs.QueryTree
import OrthoDFA.Proofs.Level5Walk

/-!
# A probe's reads

A sift that lands off its string's true leaf read a string on its state's rarer side
(`minority_of_sift`), and over independent reads a query tree's first reads are fresh
(`edge_first_mean`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Sift

variable {σ : Type*} (G : ReadModel α σ)

/-- A decided sift off the true leaf read a string on its state's rarer side. -/
theorem minority_of_sift {cut : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α) (p : List Bool), T.sift (ofCut cut) u = .inl p →
      G.leafOf T (G.M.eval u.toList) ≠ p →
      ∃ w ∈ (T.siftWithPath cut u).1, cut w = some (!G.side (G.M.eval w.toList))
  | .leaf, u, p, h, hne => by
    simp only [DTree.sift, Sum.inl.injEq] at h
    subst h; exact absurd rfl hne
  | .node m r a, u, p, h, hne => by
    have hst : G.M.eval (u * m).toList = G.at' (G.M.eval u.toList) m := by
      simp [ReadModel.at', FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
    rw [sift_ofCut_node] at h
    simp only [DTree.siftWithPath]
    simp only [ReadModel.leafOf] at hne
    rcases hc : cut (u * m) with _ | _ | _ <;> simp only [hc] at h ⊢
    · cases h
    · simp only [map_cons_eq_inl] at h
      obtain ⟨p', rfl, hp'⟩ := h
      by_cases hs : G.side (G.at' (G.M.eval u.toList) m)
      · exact ⟨u * m, List.mem_cons_self, by rw [hst, hs, hc]; rfl⟩
      · simp only [hs, Bool.false_eq_true, if_false, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨w, hw, hcw⟩ := minority_of_sift r u p' hp' hne
        exact ⟨w, List.mem_cons_of_mem _ hw, hcw⟩
    · simp only [map_cons_eq_inl] at h
      obtain ⟨p', rfl, hp'⟩ := h
      by_cases hs : G.side (G.at' (G.M.eval u.toList) m)
      · simp only [hs, if_true, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨w, hw, hcw⟩ := minority_of_sift a u p' hp' hne
        exact ⟨w, List.mem_cons_of_mem _ hw, hcw⟩
      · exact ⟨u * m, List.mem_cons_self, by rw [hst, hc]; simp [hs]⟩

end Sift

section Mean

attribute [local instance] ARU.fintype

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- The read on a read-state's rarer side. -/
noncomputable def ReadModel.minVal (r : σ) : ARU := if G.side r then .reject else .accept

theorem cut_minVal (r : σ) (v : ARU) : v.cut = some (!G.side r) ↔ v = G.minVal r := by
  unfold ReadModel.minVal; cases v <;> cases G.side r <;> simp [ARU.cut]

theorem dist_minVal (r : σ) :
    G.readDist r (G.minVal r) = min (G.readDist r .accept) (G.readDist r .reject) := by
  unfold ReadModel.minVal ReadModel.side
  by_cases h : G.readDist r .reject < G.readDist r .accept
  · simp [h, min_eq_right h.le]
  · simp [h, min_eq_left (not_lt.1 h)]

theorem lintegral_ite_read (hmeas : ∀ w, Measurable (read w)) (w : FreeMonoid α) (v : ARU)
    (c : ENNReal) : ∫⁻ ω, (if read w ω = v then c else 0) ∂μ = c * μ {ω | read w ω = v} := by
  have : (fun ω => if read w ω = v then c else 0) = {ω | read w ω = v}.indicator fun _ => c := by
    ext ω; simp [Set.indicator_apply]
  have hS : MeasurableSet {ω | read w ω = v} := hmeas w (measurableSet_singleton v)
  rw [this, lintegral_indicator_const hS]

variable {read}

theorem measure_read (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (w : FreeMonoid α) (r : ARU) :
    μ {ω | read w ω = r} = ENNReal.ofReal (G.readDist (G.M.eval w.toList) r) := by
  rw [← hlaw, measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]

/-- The per-read edge field in mean, each string counted at its first read in the probe: a query
tree's undecided first reads labelled `e` at read-states where undecided reads are at most `θ'`
average at most `θ'` times its first reads labelled `e`. -/
theorem edge_first_mean {β : Type*} (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (P : σ → Prop)
    [DecidablePred P] (θ' : ℝ) (hP : ∀ r, P r → G.readDist r .undecided ≤ θ') (e : List Bool × α)
    (t : ProbeQuery α β) :
    ∫⁻ ω, (((t.firsts (fun y => read y ω) ∅).filter fun q =>
        q.1.2 = some e ∧ read q.2 ω = .undecided ∧ P (G.M.eval q.2.toList)).length : ENNReal) ∂μ
      ≤ ENNReal.ofReal θ' * ∫⁻ ω,
        (((t.firsts (fun y => read y ω) ∅).filter fun q => q.1.2 = some e).length : ENNReal)
            ∂μ := by
  have hcount : ∀ (rd : FreeMonoid α → ARU) (f : ReadLabel α → FreeMonoid α → ARU → Prop)
      [∀ ℓ w v, Decidable (f ℓ w v)] (c : ENNReal),
      QTree.fsum (fun ℓ w v => if f ℓ w v then c else 0) rd ∅ t
        = c * ((t.firsts rd ∅).filter fun q => f q.1 q.2 (rd q.2)).length := by
    intro rd f _ c
    unfold QTree.fsum
    induction t.firsts rd ∅ with
    | nil => simp
    | cons q l ih =>
      simp only [List.map_cons, List.sum_cons, ih, List.filter_cons]
      by_cases h : f q.1 q.2 (rd q.2)
      · simp [h, mul_add, add_comm]
      · simp [h]
  have h1 := QTree.firsts_lintegral_le (μ := μ)
    (fun ℓ w v => if ℓ.2 = some e ∧ v = .undecided ∧ P (G.M.eval w.toList) then 1 else 0)
    (fun ℓ _ _ => if ℓ.2 = some e then ENNReal.ofReal θ' else 0) t read ∅ hmeas hind
    (fun ℓ w _ => by
      by_cases hl : ℓ.2 = some e
      · by_cases hp : P (G.M.eval w.toList)
        · simp only [hl, hp, true_and, and_true, if_true]
          rw [lintegral_ite_read read hmeas, one_mul, measure_read G hlaw, lintegral_const,
            measure_univ, mul_one]
          exact ENNReal.ofReal_le_ofReal (hP _ hp)
        · simp [hp]
      · simp [hl])
  simp only [hcount, one_mul] at h1
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  convert h1 using 3 <;> simp

end Mean

end OrthoDFA
