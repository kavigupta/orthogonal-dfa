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

theorem qProbe_asksIn_ge (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qProbe t edges k x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := by
  set S : FreeMonoid α → Prop := fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m
  have hs : ∀ i g, k ≤ i → (qSift (prefixOf x i) g t).AsksIn S := fun i g hi =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, hi, m, hm, he⟩) _ (qSift_asksIn _ g t)
  have hx : (qSift x false t).AsksIn S := by
    refine Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨max k x.toList.length, le_max_left _ _, m, hm, ?_⟩)
      _ (qSift_asksIn _ false t)
    rw [he]
    congr 1
    apply FreeMonoid.toList.injective
    simp [prefixOf, List.take_of_length_le (le_max_right _ _)]
  have hb : ∀ walkAt ps fuel lo hi, k ≤ lo →
      (qBracket (qAgrees t x walkAt) ps fuel lo hi).AsksIn S := by
    intro walkAt ps fuel
    induction fuel with
    | zero => intro lo hi _; trivial
    | succ fuel ih =>
      intro lo hi hlo
      have hg : ∀ p g, lo ≤ p → (qGuard t x walkAt lo hi p g).AsksIn S := by
        intro p g hp; unfold qGuard; split_ifs
        · trivial
        · trivial
        · exact Qry.asksIn_map _ _ (hs p g (hlo.trans hp))
      by_cases hlh : lo + 1 < hi
      · rw [qBracket_unfold t x walkAt ps fuel lo hi hlh]
        refine Qry.asksIn_bind (fun o => ?_) _ (hg _ _ (by omega))
        rcases o with _ | _ | _
        · refine Qry.asksIn_bind (fun l => ?_) _ (hg _ _ (by omega))
          rcases l with _ | lv
          · trivial
          · refine Qry.asksIn_bind (fun r => ?_) _ (hg _ _ (by omega))
            rcases lv <;> rcases r with _ | _ | _
            all_goals first | trivial | exact ih _ _ (by omega)
        · exact ih _ _ (by omega)
        · exact ih _ _ (by omega)
      · rw [qBracket, if_neg hlh]
        trivial
  refine Qry.asksIn_bind (fun d => ?_) _ ?_
  · rcases d with o | d
    · trivial
    · exact hb _ d.1 _ _ _ le_rfl
  · unfold qWalk
    refine Qry.asksIn_bind (fun s0 => ?_) _ (hs k false le_rfl)
    rcases s0 with p | _
    · simp only []
      split
      · exact Qry.asksIn_bind (fun s => by rcases s with _ | _ <;> trivial) _ hx
      · refine Qry.asksIn_bind (fun s1 => ?_) _ (hs _ false (by omega))
        rcases s1 with _ | _
        · exact Qry.asksIn_bind (fun s2 => by rcases s2 with _ | _ <;> trivial) _
            (hs _ false (by omega))
        · trivial
    · trivial

section Cells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

theorem noiseAlg_mono (O : Oracle μ (FreeMonoid α)) {T T' : Set (FreeMonoid α)} (h : T ⊆ T') :
    noiseAlg O T ≤ noiseAlg O T' :=
  iSup₂_le fun w hw => le_iSup₂ (f := fun w (_ : w ∈ T') =>
    MeasurableSpace.comap (O.noise w) inferInstance) w (h hw)

open scoped Classical in
/-- The reads of noise `ω` with the bits of `vBits V c.1` set by the pattern `c.2`. -/
noncomputable def cellReads (O : Oracle μ (FreeMonoid α)) (B : State) (F V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (ω : Ω) : CutReads α :=
  ⟨B, F, fun w => if w ∈ vBits V c.1
    then O.label w + (1 - 2 * O.label w) * (if w ∈ c.2 then 1 else 0) else O.mq w ω⟩

theorem cellReads_eq (O : Oracle μ (FreeMonoid α)) (B : State) (F V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) {ω : Ω}
    (hp : noisePattern O (vBits V c.1) ω = c.2) (hc : ω ∈ cleanAll O) :
    cellReads O B F V c ω = readsAt O B F ω := by
  classical
  simp only [cellReads, readsAt, CutReads.mk.injEq, true_and]
  funext w
  by_cases hw : w ∈ vBits V c.1
  · rw [if_pos hw]
    have hn : (if w ∈ c.2 then (1 : ℝ) else 0) = O.noise w ω := by
      by_cases hwc : w ∈ c.2
      · rw [if_pos hwc]
        rw [← hp] at hwc
        exact (Finset.mem_filter.1 hwc).2.symm
      · rw [if_neg hwc]
        rcases hc w with h0 | h1
        · exact h0.symm
        · exact absurd (hp ▸ Finset.mem_filter.2 ⟨hw, h1⟩) hwc
    simp only [Oracle.mq, hn]
  · rw [if_neg hw]

/-- Under the cell's reads, a read of `w` is decided by the bits it asks off the cell's. -/
theorem measurableSet_cellCut [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α))
    (w : FreeMonoid α) (o : Option Bool) :
    MeasurableSet[noiseAlg O (↑(F.image (w * ·)) \ ↑(vBits V c.1))]
      {ω | (cellReads O B F V c ω).cut w = o} := by
  classical
  set F₁ := F.filter fun v => w * v ∈ vBits V c.1
  set F₂ := F.filter fun v => w * v ∉ vBits V c.1
  set n₁ := (F₁.filter fun v => O.label (w * v) + (1 - 2 * O.label (w * v))
    * (if w * v ∈ c.2 then 1 else 0) = 1).card
  have hcount : ∀ ω, acceptsOn F (cellReads O B F V c ω).f w
      = n₁ + (F₂.filter fun v => O.mq (w * v) ω = 1).card := by
    intro ω
    unfold acceptsOn
    rw [← Finset.card_filter_add_card_filter_not (fun v => w * v ∈ vBits V c.1),
      Finset.filter_filter, Finset.filter_filter]
    congr 1
    · simp only [F₁, n₁, Finset.filter_filter]
      congr 1
      apply Finset.filter_congr
      intro v hv
      simp only [cellReads]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨h2, by rwa [if_pos h2] at h1⟩
      · rintro ⟨h2, h1⟩; exact ⟨by rwa [if_pos h2], h2⟩
    · simp only [F₂, Finset.filter_filter]
      congr 1
      apply Finset.filter_congr
      intro v hv
      simp only [cellReads]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨h2, by rwa [if_neg h2] at h1⟩
      · rintro ⟨h2, h1⟩; exact ⟨by rwa [if_neg h2], h2⟩
  have hset : {ω | (cellReads O B F V c ω).cut w = o}
      = {ω | (fun U : Finset (FreeMonoid α) => (if B.hi < n₁ + U.card then some true
          else if n₁ + U.card ≤ B.lo then some false else none) = o)
          (F₂.filter fun v => O.mq (w * v) ω = 1)} := by
    ext ω
    simp only [Set.mem_ofPred_eq]
    rw [← hcount ω]
    rfl
  rw [hset]
  exact measurableSet_filter_pred_map O (A := F₂) (w * ·) (fun v hv => by
    simp only [F₂, Finset.mem_filter] at hv
    exact ⟨Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv.1), hv.2⟩)
    (fun U : Finset (FreeMonoid α) => (if B.hi < n₁ + U.card then some true
      else if n₁ + U.card ≤ B.lo then some false else none) = o)

end Cells

end OrthoDFA
