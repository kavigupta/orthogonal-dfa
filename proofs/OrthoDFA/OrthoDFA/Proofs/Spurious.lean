import OrthoDFA.Spurious
import OrthoDFA.Proofs.EdgeAttempts

/-!
# Where a draw's search ends at an edge: the proofs

`edge_cause`: both ends of the edge a search ends at are decided sifts, so either one of them
leaves its state's route at a decided read, or both follow it and the edge between them is wrong.
`spurious_draw`: a read off the route is in a set of the noise and the draw that no hypothesis
enters, `BadRoute` aside, so the draw's independence from the noise is all it takes.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Cause

variable {Q : Type*}

theorem wrong_of_unfaithful {R : CutReads α} {side : FreeMonoid α → Bool} {z : FreeMonoid α} :
    ∀ (t : DTree α) (x : FreeMonoid α) (p : List Bool), t.sift R.cut x = .inl p →
      ¬ FaithfulRoute R side z t x →
      ∃ m ∈ routeMids side z t, R.cut (x * m) = some (!side (z * m))
  | .leaf, _, _, _, hf => absurd trivial hf
  | .node m r a, x, p, hs, hf => by
    unfold DTree.sift DTree.route at hs
    rcases hc : R.cut (x * m) with _ | b
    · simp [hc] at hs
    by_cases hb : b = side (z * m)
    · subst hb
      have hrest : ¬ (if side (z * m) then FaithfulRoute R side z a x
          else FaithfulRoute R side z r x) := fun h => hf ⟨hc, h⟩
      cases hsd : side (z * m)
      · rw [hsd] at hc hrest
        simp only [hc] at hs hrest
        rcases h2 : (r.route R.cut x).2 with p' | b'
        · obtain ⟨m', hm', h'⟩ := wrong_of_unfaithful r x p' (by unfold DTree.sift; exact h2) hrest
          exact ⟨m', by simp [routeMids, hsd, hm'], h'⟩
        · rw [h2] at hs; simp at hs
      · rw [hsd] at hc hrest
        simp only [hc, if_true] at hs hrest
        rcases h2 : (a.route R.cut x).2 with p' | b'
        · obtain ⟨m', hm', h'⟩ := wrong_of_unfaithful a x p' (by unfold DTree.sift; exact h2) hrest
          exact ⟨m', by simp [routeMids, hsd, hm'], h'⟩
        · rw [h2] at hs; simp at hs
    · refine ⟨m, by simp [routeMids], ?_⟩
      rw [hc]
      cases b <;> cases h : side (z * m) <;> simp_all

theorem edge_cause : EdgeCause := by
  intro α _ _ Q R A side rep t edges k x ps fd h
  obtain ⟨ps₀, hi, hw, hb⟩ := probeOutcome_search R h trivial
  obtain ⟨p₀, hk, hf, hkh, hhn, hpn⟩ := walkCheck_inr R hw
  set walkAt : ℕ → List Bool := fun j => ps₀.getD (j - k) [] with hwalk
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  obtain ⟨rfl, hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) ps₀ (hi - k) k hi ps fd hkh le_rfl hpk hpn hb
  have hfdn : fd - 1 < x.toList.length := by omega
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : ((x.toList.drop k).take (hi - k))[fd - 1 - k]'(by simp; omega)
      = x.toList[fd - 1] := by
    simp only [List.getElem_take, List.getElem_drop]
    congr 1
    omega
  rw [hidx, show fd - 1 - k + 1 = fd - k by omega] at hy
  rcases hs1 : t.sift R.cut (prefixOf x (fd - 1)) with p1 | b1
  swap
  · simp [agreesAt, hs1] at hfd3
  have hp1 : p1 = walkAt (fd - 1) := by
    simpa [agreesAt, hs1] using hfd3
  rcases hs2 : t.sift R.cut (prefixOf x fd) with p2 | b2
  swap
  · simp [agreesAt, hs2] at hfd4
  have hp2 : p2 ≠ walkAt fd := by
    simpa [agreesAt, hs2] using hfd4
  by_cases hf1 : FaithfulRoute R side (rep (A.state (prefixOf x (fd - 1)))) t (prefixOf x (fd - 1))
  swap
  · obtain ⟨m, hm, hc⟩ := wrong_of_unfaithful t _ _ hs1 hf1
    exact .inl ⟨fd - 1, by omega, by omega, m, hm, hc⟩
  by_cases hf2 : FaithfulRoute R side (rep (A.state (prefixOf x fd))) t (prefixOf x fd)
  swap
  · obtain ⟨m, hm, hc⟩ := wrong_of_unfaithful t _ _ hs2 hf2
    exact .inl ⟨fd, by omega, by omega, m, hm, hc⟩
  right
  have he1 := faithfulRoute_sift t _ hf1
  have he2 := faithfulRoute_sift t _ hf2
  rw [hs1] at he1
  rw [hs2] at he2
  obtain rfl := Sum.inl.inj he1
  obtain rfl := Sum.inl.inj he2
  refine ⟨x.toList[fd - 1], List.getElem?_eq_getElem hfdn, ps.getD (fd - k) [], y, ?_, ?_⟩
  · rw [hp1]; exact hy
  · rw [← prefixOf_succ hfdn, show fd - 1 + 1 = fd by omega]
    exact fun he => hp2 (he ▸ rfl)

end Cause

section Draw

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

theorem mem_wordsUpTo {n : ℕ} {m : FreeMonoid α} (h : m.toList.length ≤ n) :
    m ∈ wordsUpTo (α := α) n := by
  unfold wordsUpTo
  refine Finset.mem_biUnion.2 ⟨m.toList.length, Finset.mem_range.2 (by omega),
    Finset.mem_image.2 ⟨fun i => m.toList[i], Finset.mem_univ _, ?_⟩⟩
  apply FreeMonoid.toList.injective
  simp [List.ofFn_getElem]

theorem cut_wrong_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (s m : FreeMonoid α) :
    μ.real {ω | (readsAt O B F ω).cut (s * m) = some (!side (rep (A.state s) * m))}
      ≤ wrongRate A O B F side rep (A.state s) m :=
  le_csSup ⟨1, by rintro _ ⟨s', -, rfl⟩; exact measureReal_le_one⟩ ⟨s, rfl, rfl⟩

open scoped Classical in
/-- The noise and draw pairs with a read misread at most `ψ₀` of the time, at a midfix up to
length `n`, decided off its side. -/
def badAll (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (k n : ℕ)
    (ψ₀ : ℝ) : Set (Ω × FreeMonoid α) :=
  {p | ∃ i, k ≤ i ∧ i ≤ p.2.toList.length ∧ ∃ m ∈ wordsUpTo n,
    wrongRate A O B F side rep (A.state (prefixOf p.2 i)) m ≤ ψ₀
      ∧ (readsAt O B F p.1).cut (prefixOf p.2 i * m)
        = some (!side (rep (A.state (prefixOf p.2 i)) * m))}

theorem spurious_sub (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (k n : ℕ)
    (ψ₀ : ℝ) (t : DTree α) (ω : Ω) (x : FreeMonoid α)
    (h : SpuriousAt (readsAt O B F ω) A side rep t k x) :
    (ω, x) ∈ badAll A O B F side rep k n ψ₀ ∨ x ∈ BadRoute A O B F side rep k n ψ₀ t := by
  obtain ⟨i, hki, hix, m, hm, hc⟩ := h
  by_cases hb : n < m.toList.length ∨ ψ₀ < wrongRate A O B F side rep (A.state (prefixOf x i)) m
  · exact .inr ⟨i, hki, hix, m, hm, hb⟩
  push Not at hb
  exact .inl ⟨i, hki, hix, m, mem_wordsUpTo hb.1, hb.2, hc⟩

open scoped Classical in
theorem badAll_slice_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (k L n : ℕ) (ψ₀ : ℝ)
    (x : FreeMonoid α) (hx : x.toList.length = L) :
    μ.real {ω | (ω, x) ∈ badAll (μ := μ) A O B F side rep k n ψ₀}
      ≤ ∑ i ∈ Finset.Icc k L, ∑ m ∈ wordsUpTo n,
        (if wrongRate A O B F side rep (A.state (prefixOf x i)) m ≤ ψ₀
          then wrongRate A O B F side rep (A.state (prefixOf x i)) m else 0) := by
  set S : ℕ → FreeMonoid α → Set Ω := fun i m =>
    {ω | (readsAt O B F ω).cut (prefixOf x i * m) = some (!side (rep (A.state (prefixOf x i)) * m))}
  have hsub : {ω | (ω, x) ∈ badAll (μ := μ) A O B F side rep k n ψ₀}
      ⊆ ⋃ i ∈ Finset.Icc k L, ⋃ m ∈ (wordsUpTo n).filter
        (fun m => wrongRate A O B F side rep (A.state (prefixOf x i)) m ≤ ψ₀), S i m := by
    rintro ω ⟨i, hki, hix, m, hm, hψ, hc⟩
    simp only [Set.mem_iUnion, Finset.mem_Icc, Finset.mem_filter]
    exact ⟨i, ⟨hki, hx ▸ hix⟩, m, ⟨hm, hψ⟩, hc⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans ?_)
  refine Finset.sum_le_sum fun i _ => (measureReal_biUnion_finset_le _ _).trans ?_
  rw [← Finset.sum_filter]
  exact Finset.sum_le_sum fun m _ => cut_wrong_le A O B F side rep _ m

theorem badAll_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k L n : ℕ) (ψ₀ : ℝ) (hlen : ∀ᵐ x ∂D, x.toList.length = L) :
    (μ.prod D).real (badAll A O B F side rep k n ψ₀) ≤ spurRate A O B F side rep D k L n ψ₀ := by
  classical
  set W := wordsOf (α := α) L
  set bad := badAll (μ := μ) A O B F side rep k n ψ₀
  have hsub : bad ⊆ (Set.univ ×ˢ (↑W)ᶜ)
      ∪ ⋃ x ∈ W, ({ω | (ω, x) ∈ bad} ×ˢ {x}) := by
    rintro ⟨ω, x⟩ hp
    by_cases hx : x ∈ W
    · exact .inr (Set.mem_biUnion hx ⟨hp, rfl⟩)
    · exact .inl ⟨trivial, hx⟩
  have hW : D.real (↑W)ᶜ = 0 := by
    rw [measureReal_def, measure_mono_null (fun x hx => ?_) (ae_iff.1 hlen), ENNReal.toReal_zero]
    simp only [Set.mem_compl_iff, Finset.mem_coe, W, mem_wordsOf] at hx
    exact hx
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [measureReal_prod_prod, hW, mul_zero, zero_add]
  refine (measureReal_biUnion_finset_le _ _).trans ?_
  unfold spurRate
  rw [integral_eq_sum_words D hlen]
  refine Finset.sum_le_sum fun x hx => ?_
  rw [measureReal_prod_prod, mul_comm]
  exact mul_le_mul_of_nonneg_left (badAll_slice_le A O B F side rep k L n ψ₀ x
    (mem_wordsOf.1 hx)) measureReal_nonneg

theorem spurious_draw : SpuriousDraw := by
  intro α _ _ Ω _ μ _ Q A O B F side rep D _ k L n ψ₀ Z _ P g T hlen hg hmap
  have hsub : {z | SpuriousAt (readsAt O B F (g z).1) A side rep (T z) k (g z).2}
      ⊆ g ⁻¹' badAll A O B F side rep k n ψ₀
        ∪ {z | (g z).2 ∈ BadRoute A O B F side rep k n ψ₀ (T z)} := by
    intro z hz
    rcases spurious_sub A O B F side rep k n ψ₀ (T z) (g z).1 (g z).2 hz with h | h
    · exact .inl h
    · exact .inr h
  have : IsProbabilityMeasure (P.map g) := by rw [hmap]; infer_instance
  have : IsProbabilityMeasure P := ⟨by
    rw [← Set.preimage_univ (f := g), ← Measure.map_apply hg MeasurableSet.univ, measure_univ]⟩
  have hpre : P.real (g ⁻¹' badAll A O B F side rep k n ψ₀)
      ≤ (P.map g).real (badAll A O B F side rep k n ψ₀) :=
    ENNReal.toReal_mono (measure_ne_top _ _) (Measure.le_map_apply hg.aemeasurable _)
  rw [hmap] at hpre
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  linarith [badAll_le (μ := μ) A O B F side rep D k L n ψ₀ hlen]

end Draw

section Round

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [Fintype α] [DecidableEq α] in
theorem map_draw (C : StrongCfg α) (D D' : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {Rmax : ℕ} (h : C.Draws → FreeMonoid α) (hh : Measurable h)
    (hD : (C.drawMeasure D).map h = D') (j : Fin Rmax) :
    (μ.prod (Measure.pi fun _ : Fin Rmax => C.drawMeasure D)).map
      (fun p => (p.1, h (p.2 j))) = μ.prod D' := by
  have hj : Measurable (fun d : Fin Rmax → C.Draws => h (d j)) :=
    hh.comp (measurable_pi_apply j)
  have e1 : (Measure.pi fun _ : Fin Rmax => C.drawMeasure D).map (fun d => h (d j)) = D' := by
    change Measure.map (h ∘ fun d : Fin Rmax → C.Draws => d j) _ = D'
    rw [← hD, ← Measure.map_map hh (measurable_pi_apply j), (measurePreserving_eval _ j).map_eq]
  have := Measure.map_prod_map μ (Measure.pi fun _ : Fin Rmax => C.drawMeasure D)
    measurable_id hj
  rw [Measure.map_id, e1] at this
  rw [this]
  rfl

omit [Fintype α] [DecidableEq α] in
theorem map_refusal (C : StrongCfg α) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (i : Fin C.nr) : (C.drawMeasure D).map (fun d => d.2.2.1 i) = D := by
  have : (fun d : C.Draws => d.2.2.1 i)
      = Function.eval i ∘ Prod.fst ∘ Prod.snd ∘ Prod.snd := rfl
  rw [this, ← Measure.map_map (measurable_pi_apply i)
      (measurable_fst.comp (measurable_snd.comp measurable_snd)),
    ← Measure.map_map measurable_fst (measurable_snd.comp measurable_snd),
    ← Measure.map_map measurable_snd measurable_snd]
  unfold StrongCfg.drawMeasure
  rw [measurePreserving_snd.map_eq, measurePreserving_snd.map_eq, measurePreserving_fst.map_eq,
    (measurePreserving_eval _ i).map_eq]

omit [Fintype α] [DecidableEq α] in
theorem map_probe (C : StrongCfg α) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (i : Fin C.np) : (C.drawMeasure D).map (fun d => d.1 i) = D := by
  have : (fun d : C.Draws => d.1 i) = Function.eval i ∘ Prod.fst := rfl
  rw [this, ← Measure.map_map (measurable_pi_apply i) measurable_fst]
  unfold StrongCfg.drawMeasure
  rw [measurePreserving_fst.map_eq, (measurePreserving_eval _ i).map_eq]

omit [IsProbabilityMeasure μ] in
theorem badRoute_leaf {Q : Type*} (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (k n : ℕ) (ψ₀ : ℝ) (x : FreeMonoid α) : x ∉ BadRoute A O B F side rep k n ψ₀ .leaf := by
  rintro ⟨i, -, -, m, hm, -⟩
  simp [routeMids] at hm

/-- One draw of the round against the tree `tr` picks from an entry, by way of `spurious_draw`. -/
theorem round_draw {Q : Type*} (C : StrongCfg α) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (n : ℕ) (ψ₀ : ℝ) {Rmax : ℕ} (j : Fin Rmax)
    (hlen : ∀ᵐ x ∂D, x.toList.length = C.L) (h : C.Draws → FreeMonoid α) (hh : Measurable h)
    (hD : (C.drawMeasure D).map h = D)
    (E : Ω × (Fin Rmax → C.Draws) → Option (RoundAcc α × List (FreeMonoid α)))
    (tr : Ω × (Fin Rmax → C.Draws) → RoundAcc α × List (FreeMonoid α) → DTree α) :
    let P := μ.prod (Measure.pi fun _ : Fin Rmax => C.drawMeasure D)
    P.real {p | ∃ e ∈ E p, SpuriousAt (readsAt O B F p.1) A side rep (tr p e) C.k (h (p.2 j))}
      ≤ spurRate A O B F side rep D C.k C.L n ψ₀
        + P.real {p | ∃ e ∈ E p, h (p.2 j) ∈ BadRoute A O B F side rep C.k n ψ₀ (tr p e)} := by
  intro P
  set T : Ω × (Fin Rmax → C.Draws) → DTree α := fun p => (E p).elim .leaf (tr p)
  have hg : Measurable (fun p : Ω × (Fin Rmax → C.Draws) => (p.1, h (p.2 j))) :=
    measurable_fst.prodMk (hh.comp ((measurable_pi_apply j).comp measurable_snd))
  have key := spurious_draw A O B F side rep D C.k C.L n ψ₀ P (fun p => (p.1, h (p.2 j))) T hlen
    hg (map_draw C D D h hh hD j)
  refine (measureReal_mono ?_ (measure_ne_top _ _)).trans
    (key.trans (add_le_add_right (measureReal_mono ?_ (measure_ne_top _ _)) _))
  · rintro p ⟨e, he, hs⟩
    simp only [Set.mem_ofPred_eq, T, Option.mem_def.1 he, Option.elim]
    exact hs
  · intro p hp
    simp only [Set.mem_ofPred_eq, T] at hp ⊢
    rcases hE : E p with _ | e
    · rw [hE] at hp
      exact absurd hp (badRoute_leaf A O B F side rep C.k n ψ₀ _)
    · rw [hE] at hp
      exact ⟨e, rfl, hp⟩

theorem round_strong_spurious : RoundStrongSpurious := by
  intro α _ _ Ω _ μ _ Q C A O B F side rep D _ seed n ψ₀ Rmax j hlen
  refine ⟨fun i => ?_, fun i => ?_⟩
  · exact round_draw C A O B F side rep D n ψ₀ j hlen (fun d => d.2.2.1 i)
      ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))
      (map_refusal C D i) _
      (fun p e => (strongPass C (readsAt O B F p.1) e.1 (e.2 ++ List.ofFn (p.2 j).1)).s.tree)
  · exact round_draw C A O B F side rep D n ψ₀ j hlen (fun d => d.1 i)
      ((measurable_pi_apply i).comp measurable_fst) (map_probe C D i) _
      (fun p e => (strongPass C (readsAt O B F p.1) e.1
        (e.2 ++ (List.ofFn (p.2 j).1).take i)).s.tree)

end Round

end OrthoDFA
