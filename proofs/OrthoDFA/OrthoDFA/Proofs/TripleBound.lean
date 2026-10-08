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

theorem visits_le (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    visits R t edges k x ≤ x.toList.length := by
  unfold visits
  rcases hw : walkCheck R t edges k x with o | ⟨ps, hi⟩
  · simp
  · obtain ⟨-, -, -, -, hhi, -⟩ := walkCheck_inr R hw
    exact (visited_le _ _ _ _).trans (by simp only []; omega)

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

theorem probeOutcome_triple_gt {R : CutReads α} {t : DTree α} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} {j : ℕ} (h : probeOutcome R t edges k x = .triple j) : k < j := by
  obtain ⟨ps, hi, -, hb⟩ := probeOutcome_search R h trivial
  exact bracketAt_triple_gt _ ps _ _ _ _ hb

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

variable {Q : Type*}

/-- The triple of `x` harvests a read at a state read undecided less than `uGood` of the time,
of a string off `Tp`. -/
def FreshTriple (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (uGood : ℝ) (R : CutReads α) (t : DTree α) (edges : Edges α)
    (Tp : Finset (FreeMonoid α)) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ j b, probeOutcome R t edges k x = .triple j ∧ tripleRead R t x j = some b
    ∧ stateIndecision A O B F (A.state b) < uGood ∧ b ∉ Tp

/-- How many of a probe's reads are of the search's middles. -/
noncomputable def tagCount (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ)
    (x : FreeMonoid α) : ℕ :=
  ((qProbe t edges k x).trace R.cut).countP fun e => e.2

open scoped Classical in
/-- A draw's share of the triples' fluctuation. -/
noncomputable def contrib (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (uGood : ℝ) (R : CutReads α) (t : DTree α) (edges : Edges α)
    (Tp : Finset (FreeMonoid α)) (k : ℕ) (x : FreeMonoid α) : ℝ :=
  (if FreshTriple A O B F uGood R t edges Tp k x then 1 else 0) - uGood * tagCount R t edges k x

/-- The strings beginning with `x`'s first `k` letters, off the cell's bits. -/
def drawBits (V : Finset (FreeMonoid α)) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α))
    (k : ℕ) (x : FreeMonoid α) : Set (FreeMonoid α) :=
  {z | (prefixOf x k).toList <+: z.toList} \ ↑(vBits V c.1)

theorem cellCut_drawBits [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (k : ℕ)
    (x : FreeMonoid α) {w : FreeMonoid α} (hw : ∃ i, k ≤ i ∧ ∃ m, w = prefixOf x i * m)
    (o : Option Bool) :
    MeasurableSet[noiseAlg O (drawBits V c k x)] {ω | (cellReads O B F V c ω).cut w = o} := by
  refine noiseAlg_mono O ?_ _ (measurableSet_cellCut O B F V c w o)
  rintro z ⟨hz, hzv⟩
  refine ⟨?_, hzv⟩
  obtain ⟨i, hi, m, rfl⟩ := hw
  simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hz
  obtain ⟨v, -, rfl⟩ := hz
  simp only [Set.mem_ofPred_eq, FreeMonoid.toList_mul, prefixOf, FreeMonoid.toList_ofList]
  exact (List.take_prefix_take_left hi).trans (List.prefix_append _ _ |>.trans
    (List.prefix_append _ _))

/-- Under the cell's reads, a draw's share depends only on the bits beginning with its first `k`
letters. -/
theorem measurable_contrib [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F V : Finset (FreeMonoid α)) (uGood : ℝ)
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (t : DTree α) (edges : Edges α)
    (Tp : Finset (FreeMonoid α)) (k : ℕ) (x : FreeMonoid α) :
    MeasurableSet[noiseAlg O (drawBits V c k x)]
        {ω | FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x}
      ∧ (∀ n, MeasurableSet[noiseAlg O (drawBits V c k x)]
        {ω | tagCount (cellReads O B F V c ω) t edges k x = n})
      ∧ Measurable[noiseAlg O (drawBits V c k x)]
        fun ω => contrib A O B F uGood (cellReads O B F V c ω) t edges Tp k x := by
  classical
  set cut : Ω → FreeMonoid α → Option Bool := fun ω => (cellReads O B F V c ω).cut
  have hS : ∀ w, (∃ i, k ≤ i ∧ ∃ m ∈ t.mids, w = prefixOf x i * m) →
      ∀ o, MeasurableSet[noiseAlg O (drawBits V c k x)] {ω | cut ω w = o} := fun w ⟨i, hi, m', _, he⟩ o =>
    cellCut_drawBits O B F V c k x ⟨i, hi, m', he⟩ o
  have hrt : ∀ A' : Set (Outcome α × List (FreeMonoid α × Bool)),
      MeasurableSet[noiseAlg O (drawBits V c k x)] {ω | ((qProbe t edges k x).run (cut ω),
        (qProbe t edges k x).trace (cut ω)) ∈ A'} :=
    Qry.measurableSet_run_trace cut hS _ (qProbe_asksIn_ge t edges k x)
  have hsift : ∀ j, k ≤ j → ∀ A' : Set ((List Bool ⊕ FreeMonoid α) × List (FreeMonoid α × Bool)),
      MeasurableSet[noiseAlg O (drawBits V c k x)] {ω | ((qSift (prefixOf x j) true t).run (cut ω),
        (qSift (prefixOf x j) true t).trace (cut ω)) ∈ A'} := fun j hj =>
    Qry.measurableSet_run_trace cut (S := fun w => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, w = prefixOf x i * m)
      hS _ (Qry.asksIn_mono (fun y ⟨m', hm, he⟩ => ⟨j, hj, m', hm, he⟩) _ (qSift_asksIn _ _ t))
  have hFT : MeasurableSet[noiseAlg O (drawBits V c k x)] {ω | FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x} := by
    have hset : {ω | FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x}
        = ⋃ j : ℕ, if k ≤ j then
            {ω | ((qProbe t edges k x).run (cut ω), (qProbe t edges k x).trace (cut ω))
              ∈ {p | p.1 = .triple j}}
            ∩ {ω | ((qSift (prefixOf x j) true t).run (cut ω),
              (qSift (prefixOf x j) true t).trace (cut ω))
              ∈ {p | ∃ b, p.1 = .inr b ∧ stateIndecision A O B F (A.state b) < uGood ∧ b ∉ Tp}}
          else ∅ := by
      ext ω
      simp only [FreshTriple, Set.mem_ofPred_eq, Set.mem_iUnion]
      constructor
      · rintro ⟨j, b, hj, hb, hg, hT⟩
        have hkj := probeOutcome_triple_gt hj
        refine ⟨j, ?_⟩
        rw [if_pos hkj.le]
        refine ⟨by simpa [cut, qProbe_run] using hj, b, ?_, hg, hT⟩
        simp only [cut, qSift_run]
        simp only [tripleRead] at hb
        rcases hs : DTree.sift (cellReads O B F V c ω).cut t (prefixOf x j) with _ | b' <;>
          rw [hs] at hb <;> simp_all
      · rintro ⟨j, hj⟩
        split_ifs at hj with hkj
        · obtain ⟨h1, b, h2, hg, hT⟩ := hj
          refine ⟨j, b, by simpa [cut, qProbe_run] using h1, ?_, hg, hT⟩
          simp only [cut, qSift_run] at h2
          simp [tripleRead, h2]
        · exact hj.elim
    rw [hset]
    refine MeasurableSet.iUnion fun j => ?_
    split_ifs with hkj
    · exact (hrt _).inter (hsift j hkj _)
    · exact @MeasurableSet.empty Ω (noiseAlg O (drawBits V c k x))
  have hcount : ∀ n, MeasurableSet[noiseAlg O (drawBits V c k x)]
      {ω | tagCount (cellReads O B F V c ω) t edges k x = n} := fun n =>
    hrt {p | p.2.countP (fun e => e.2) = n}
  have hpair : Measurable[noiseAlg O (drawBits V c k x)] fun ω =>
      (decide (FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x),
        tagCount (cellReads O B F V c ω) t edges k x) := by
    refine @measurable_to_countable' _ _ _ _ (noiseAlg O (drawBits V c k x)) _ fun y => ?_
    have : (fun ω => (decide (FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x),
        tagCount (cellReads O B F V c ω) t edges k x)) ⁻¹' {y}
        = {ω | decide (FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x) = y.1}
          ∩ {ω | tagCount (cellReads O B F V c ω) t edges k x = y.2} := by
      ext ω; simp [Prod.ext_iff]
    rw [this]
    refine MeasurableSet.inter ?_ (hcount y.2)
    obtain ⟨b, n⟩ := y
    cases b
    · have h' : {ω | decide (FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x)
          = false} = {ω | FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x}ᶜ := by
        ext ω; simp
      rw [h']; exact hFT.compl
    · have h' : {ω | decide (FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x)
          = true} = {ω | FreshTriple A O B F uGood (cellReads O B F V c ω) t edges Tp k x} := by
        ext ω; simp
      rw [h']; exact hFT
  have hg : Measurable fun p : Bool × ℕ => (if p.1 then (1 : ℝ) else 0) - uGood * p.2 :=
    measurable_of_countable _
  refine ⟨hFT, hcount, ?_⟩
  have := hg.comp hpair
  convert this using 1
  ext ω
  simp [contrib]

theorem indepFun_of_noiseAlg [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {T T' : Set (FreeMonoid α)} (h : Disjoint T T') {f g : Ω → ℝ}
    (hf : Measurable[noiseAlg O T] f) (hg : Measurable[noiseAlg O T'] g) : f ⟂ᵢ[μ] g := by
  rw [IndepFun_iff_Indep]
  exact indep_of_indep_of_le_right (indep_of_indep_of_le_left (indep_noiseAlg O h) hf.comap_le)
    hg.comap_le

/-- A set decided by `T₀`'s bits and functions of `S₁`'s and `S₂`'s, the three disjoint, factor. -/
theorem integral_indicator_mul_mul [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {T₀ S₁ S₂ : Set (FreeMonoid α)} (h01 : Disjoint T₀ S₁) (h02 : Disjoint T₀ S₂)
    (h12 : Disjoint S₁ S₂) {P : Set Ω} (hP : MeasurableSet[noiseAlg O T₀] P) {f g : Ω → ℝ}
    (hf : Measurable[noiseAlg O S₁] f) (hg : Measurable[noiseAlg O S₂] g)
    {Cf Cg : ℝ} (hfb : ∀ ω, |f ω| ≤ Cf) (hgb : ∀ ω, |g ω| ≤ Cg) :
    ∫ ω, P.indicator 1 ω * (f ω * g ω) ∂μ = μ.real P * ((∫ ω, f ω ∂μ) * ∫ ω, g ω ∂μ) := by
  have hle := noiseAlg_le O
  have hPm : Measurable[noiseAlg O T₀] (P.indicator (1 : Ω → ℝ)) :=
    Measurable.indicator measurable_const hP
  have hfg : Measurable[noiseAlg O (S₁ ∪ S₂)] fun ω => f ω * g ω :=
    (hf.mono (noiseAlg_mono O Set.subset_union_left) le_rfl).mul
      (hg.mono (noiseAlg_mono O Set.subset_union_right) le_rfl)
  have h1 : (P.indicator (1 : Ω → ℝ)) ⟂ᵢ[μ] fun ω => f ω * g ω :=
    indepFun_of_noiseAlg O (Set.disjoint_union_right.2 ⟨h01, h02⟩) hPm hfg
  have h2 : f ⟂ᵢ[μ] g := indepFun_of_noiseAlg O h12 hf hg
  have e1 := h1.integral_mul_eq_mul_integral ((hPm.mono (hle _) le_rfl).aestronglyMeasurable)
    ((hfg.mono (hle _) le_rfl).aestronglyMeasurable)
  have e2 := h2.integral_mul_eq_mul_integral ((hf.mono (hle _) le_rfl).aestronglyMeasurable)
    ((hg.mono (hle _) le_rfl).aestronglyMeasurable)
  simp only [Pi.mul_apply] at e1 e2
  rw [e1, e2, integral_indicator_one ((hle _) _ hP)]

end Cells

section PassCells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

/-- The pass on noise `ω`. -/
noncomputable def passK (ω : Ω) : KState α :=
  runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes

/-- The strings the pass on noise `ω` can read. -/
noncomputable def passReads (ω : Ω) : Finset (FreeMonoid α) :=
  passReadSet seed probes (passK O B F K k seed probes ω).tree

/-- A pattern of the bits `vBits V c.1`, clean. -/
def pcell (V : Finset (FreeMonoid α)) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    Set Ω :=
  {ω | noisePattern O (vBits V c.1) ω = c.2} ∩ noiseClean O (vBits V c.1)

/-- The pass's cell: its read set `c.1`, with those reads' bits patterned `c.2`. -/
def passCell (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) : Set Ω :=
  pcell O (F ∪ K.train F) c ∩ cleanAll O ∩ {ω | passReads O B F K k seed probes ω = c.1}

theorem passK_determined (ω ω' : Ω)
    (h : ∀ y ∈ vBits (F ∪ K.train F) (passReads O B F K k seed probes ω),
      O.noise y ω = O.noise y ω') :
    passK O B F K k seed probes ω' = passK O B F K k seed probes ω :=
  runPassK_determined O B F K k seed probes ω ω' h

theorem passCell_const [IsProbabilityMeasure μ] {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ ω : Ω}
    (h₀ : ω₀ ∈ passCell O B F K k seed probes c) (hω : ω ∈ pcell O (F ∪ K.train F) c)
    (hcl : ω ∈ cleanAll O) :
    passK O B F K k seed probes ω = passK O B F K k seed probes ω₀
      ∧ ω ∈ passCell O B F K k seed probes c := by
  obtain ⟨⟨⟨hp₀, hc₀⟩, -⟩, hT₀⟩ := h₀
  have hT₀' : passReads O B F K k seed probes ω₀ = c.1 := hT₀
  have hag : ∀ y ∈ vBits (F ∪ K.train F) (passReads O B F K k seed probes ω₀),
      O.noise y ω₀ = O.noise y ω := by
    rw [hT₀']
    exact noise_eq_of_pattern O hc₀ hω.2 (hp₀.trans hω.1.symm)
  have hs := passK_determined O B F K k seed probes ω₀ ω hag
  refine ⟨hs, ⟨⟨hω, hcl⟩, ?_⟩⟩
  change passReads O B F K k seed probes ω = c.1
  rw [← hT₀']
  simp only [passReads, hs]

theorem cut_none_eq (z : FreeMonoid α) :
    {ω | (readsAt O B F ω).cut z = none} = {ω | ¬ decided O.mq B.lo B.hi F z ω} := by
  ext ω
  simp only [Set.mem_ofPred_eq, CutReads.cut, readsAt, decided, voteCount]
  change (if B.hi < acceptsOn F (fun w => O.mq w ω) z then some true
    else if acceptsOn F (fun w => O.mq w ω) z ≤ B.lo then some false else none) = none ↔ _
  have : acceptsOn F (fun w => O.mq w ω) z = (F.filter fun v => O.mq (z * v) ω = 1).card := by
    unfold acceptsOn; congr
  rw [this]
  split_ifs <;> simp <;> omega

theorem good_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q) {uGood : ℝ} (z : FreeMonoid α)
    (hz : stateIndecision A O B F (A.state z) < uGood) :
    μ {ω | (readsAt O B F ω).cut z = none} ≤ ENNReal.ofReal uGood := by
  have hle : undecidedProb O B.lo B.hi F z ≤ stateIndecision A O B F (A.state z) :=
    le_csSup ⟨1, by rintro _ ⟨s, -, rfl⟩; exact measureReal_le_one⟩ ⟨z, rfl, rfl⟩
  rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def, cut_none_eq]
  exact ENNReal.ofReal_le_ofReal ((show μ.real {ω | ¬ decided O.mq B.lo B.hi F z ω}
    = undecidedProb O B.lo B.hi F z from rfl) ▸ hle.trans hz.le)

theorem freshTriple_firstBad (A : DFA (FreeMonoid α) Q) {uGood : ℝ} {R : CutReads α}
    {t : DTree α} {edges : Edges α} {Tp : Finset (FreeMonoid α)} {x : FreeMonoid α}
    (h : FreshTriple A O B F uGood R t edges Tp k x) :
    FirstBad ((qProbe t edges k x).trace R.cut) Tp R.cut
      fun z => stateIndecision A O B F (A.state z) < uGood := by
  obtain ⟨j, b, hj, hb, hg, hT⟩ := h
  obtain ⟨hcut, r, hr, hrb, hfirst⟩ := triple_first_read R hj hb
  refine ⟨r, hr, by rw [hrb], by rw [hrb]; exact hT, fun i hi => by rw [hrb]; exact hfirst i hi,
    by rw [hrb]; exact hcut, by rw [hrb]; exact hg⟩

theorem tagCount_le (R : CutReads α) (t : DTree α) (edges : Edges α) (x : FreeMonoid α) :
    tagCount R t edges k x ≤ t.depth * x.toList.length :=
  (qProbe_countP R t edges k x).trans (Nat.mul_le_mul_left _ (visits_le R t edges k x))

/-- Inside a cell of the pass, a draw's share is pulled down on average: its fresh triples are
outnumbered by `uGood` times its tagged reads. -/
theorem cell_mean_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q) {uGood : ℝ}
    (hu : 0 ≤ uGood) (hV : SuffixFree (F ∪ K.train F))
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ : Ω}
    (h₀ : ω₀ ∈ passCell O B F K k seed probes c) (x : FreeMonoid α) :
    μ.real (pcell O (F ∪ K.train F) c)
      * ∫ ω, contrib A O B F uGood (cellReads O B F (F ∪ K.train F) c ω)
          (passK O B F K k seed probes ω₀).tree (passK O B F K k seed probes ω₀).edges c.1 k x ∂μ
      ≤ 0 := by
  classical
  set V := F ∪ K.train F
  set s₀ := passK O B F K k seed probes ω₀
  set P := pcell O V c
  set C := passCell O B F K k seed probes c
  set FTc : Ω → Prop := fun ω => FreshTriple A O B F uGood (cellReads O B F V c ω) s₀.tree s₀.edges c.1 k x
  set tc : Ω → ℕ := fun ω => tagCount (cellReads O B F V c ω) s₀.tree s₀.edges k x
  set h : Ω → ℝ := fun ω => contrib A O B F uGood (cellReads O B F V c ω) s₀.tree s₀.edges c.1 k x
  obtain ⟨hFTm, htcm, hhm⟩ := measurable_contrib A O B F V uGood c s₀.tree s₀.edges c.1 k x
  have hle := noiseAlg_le O
  have hFTm' : MeasurableSet {ω | FTc ω} := hle _ _ hFTm
  have htcm' : Measurable fun ω => (tc ω : ℝ) := by
    refine Measurable.comp (g := fun n : ℕ => (n : ℝ)) measurable_from_top ?_
    exact measurable_to_countable' fun n => hle _ _ (htcm n)
  have hhm' : Measurable h := hhm.mono (hle _) le_rfl
  have hPm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] P :=
    (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
  have hPm' : MeasurableSet P := hle _ _ hPm
  have hbound : ∀ ω, (tc ω : ℝ) ≤ s₀.tree.depth * x.toList.length := fun ω => by
    exact_mod_cast tagCount_le k _ _ _ x
  have hint_tc : Integrable (fun ω => (tc ω : ℝ)) μ :=
    Integrable.of_bound htcm'.aestronglyMeasurable _ (ae_of_all _ fun ω => by
      rw [Real.norm_of_nonneg (Nat.cast_nonneg _)]; exact hbound ω)
  have hint_h : Integrable h μ :=
    Integrable.of_bound hhm'.aestronglyMeasurable (1 + uGood * (s₀.tree.depth * x.toList.length))
      (ae_of_all _ fun ω => by
        have h0 : (0 : ℝ) ≤ tc ω := Nat.cast_nonneg _
        have h1 := hbound ω
        simp only [h, contrib, Real.norm_eq_abs]
        split_ifs <;> rw [abs_le] <;> constructor <;> nlinarith)
  -- `P` is independent of the draw's share
  have hdisj : Disjoint (↑(vBits V c.1) : Set (FreeMonoid α)) (drawBits V c k x) :=
    Set.disjoint_left.2 fun z hz hz' => hz'.2 hz
  have hPi : Measurable[noiseAlg O ↑(vBits V c.1)] (P.indicator (1 : Ω → ℝ)) :=
    Measurable.indicator measurable_const hPm
  have hind : (P.indicator (1 : Ω → ℝ)) ⟂ᵢ[μ] h := indepFun_of_noiseAlg O hdisj hPi hhm
  have e1 : ∫ ω, P.indicator 1 ω * h ω ∂μ = μ.real P * ∫ ω, h ω ∂μ := by
    have := hind.integral_mul_eq_mul_integral (hPi.mono (hle _) le_rfl).aestronglyMeasurable
      hhm'.aestronglyMeasurable
    simp only [Pi.mul_apply] at this
    rw [this, integral_indicator_one hPm']
  -- `P` and the cell agree off a null set
  have hCP : C ⊆ P := fun ω hω => hω.1.1
  have hcl := measure_cleanAll_compl O (μ := μ)
  have hae : ∀ᵐ ω ∂μ, P.indicator (1 : Ω → ℝ) ω = C.indicator 1 ω := by
    have : ∀ᵐ ω ∂μ, ω ∈ cleanAll O := ae_iff.2 hcl
    filter_upwards [this] with ω hω
    by_cases hP : ω ∈ P
    · rw [Set.indicator_of_mem hP,
        Set.indicator_of_mem (passCell_const O B F K k seed probes h₀ hP hω).2]
    · rw [Set.indicator_of_notMem hP, Set.indicator_of_notMem fun h => hP (hCP h)]
  have e2 : ∫ ω, P.indicator 1 ω * h ω ∂μ = ∫ ω, C.indicator 1 ω * h ω ∂μ :=
    integral_congr_ae (hae.mono fun ω hω => by
      change P.indicator 1 ω * h ω = C.indicator 1 ω * h ω
      rw [hω])
  -- on the cell, the cell's reads are the oracle's
  have hon : ∀ ω ∈ C, cellReads O B F V c ω = readsAt O B F ω
      ∧ passK O B F K k seed probes ω = s₀ ∧ passReads O B F K k seed probes ω = c.1 := by
    intro ω hω
    exact ⟨cellReads_eq O B F V c hω.1.1.1 hω.1.2,
      (passCell_const O B F K k seed probes h₀ hω.1.1 hω.1.2).1, hω.2⟩
  -- the fresh triples, by `fresh_first_le`
  set st : Ω → DTree α × Edges α := fun ω =>
    ((passK O B F K k seed probes ω).tree, (passK O B F K k seed probes ω).edges)
  set C₀ := P ∩ {ω | passReads O B F K k seed probes ω = c.1}
  have hst : ∀ ω ω', (∀ y ∈ vBits V (passReads O B F K k seed probes ω),
      O.noise y ω = O.noise y ω') → st ω' = st ω
        ∧ passReads O B F K k seed probes ω' = passReads O B F K k seed probes ω := by
    intro ω ω' hag
    have hs := passK_determined O B F K k seed probes ω ω' hag
    exact ⟨by simp only [st, hs], by simp only [passReads, hs]⟩
  have hC₀ : ∀ ω ω', (∀ y ∈ vBits V (passReads O B F K k seed probes ω),
      O.noise y ω = O.noise y ω') → (ω' ∈ C₀ ↔ ω ∈ C₀) := by
    intro ω ω' hag
    have hT := (hst ω ω' hag).2
    by_cases hω : passReads O B F K k seed probes ω = c.1
    · rw [hω] at hag
      have hpat : noisePattern O (vBits V c.1) ω' = noisePattern O (vBits V c.1) ω :=
        Finset.filter_congr fun y hy => by rw [hag y hy]
      have hcln : ω' ∈ noiseClean O (vBits V c.1) ↔ ω ∈ noiseClean O (vBits V c.1) := by
        simp only [noiseClean, Set.mem_ofPred_eq]
        exact forall₂_congr fun y hy => by rw [hag y hy]
      simp only [C₀, P, pcell, Set.mem_inter_iff, Set.mem_ofPred_eq, hpat, hcln, hT]
    · simp only [C₀, Set.mem_inter_iff, Set.mem_ofPred_eq, hT, hω, and_false]
  have hfresh := fresh_first_le O B F V hV Finset.subset_union_left st
    (passReads O B F K k seed probes) hst C₀ hC₀ k x
    (fun z => stateIndecision A O B F (A.state z) < uGood) (u := ENNReal.ofReal uGood)
    fun z hz => good_le O B F A z hz
  have hsub : C ∩ {ω | FTc ω} ⊆ {ω | ω ∈ C₀ ∧ FirstBad (probeTrace O B F st k x ω)
      (passReads O B F K k seed probes ω) (readsAt O B F ω).cut
      fun z => stateIndecision A O B F (A.state z) < uGood} := by
    rintro ω ⟨hω, hft⟩
    obtain ⟨hr, hs, hT⟩ := hon ω hω
    refine ⟨⟨hω.1.1, hT⟩, ?_⟩
    have hft' : FreshTriple A O B F uGood (readsAt O B F ω) (st ω).1 (st ω).2
        (passReads O B F K k seed probes ω) k x := by
      simp only [st, hs, hT]; simpa [FTc, hr] using hft
    exact freshTriple_firstBad O B F k A hft'
  have hlin : ∫⁻ ω, C₀.indicator (fun ω => (((probeTrace O B F st k x ω).countP (·.2) : ℕ)
      : ℝ≥0∞)) ω ∂μ = ∫⁻ ω, ENNReal.ofReal (C.indicator (fun ω => (tc ω : ℝ)) ω) ∂μ := by
    have : ∀ᵐ ω ∂μ, ω ∈ cleanAll O := ae_iff.2 hcl
    refine lintegral_congr_ae (this.mono fun ω hcl' => ?_)
    beta_reduce
    by_cases hω : ω ∈ C₀
    · have hC : ω ∈ C := (passCell_const O B F K k seed probes h₀ hω.1 hcl').2
      obtain ⟨hr, hs, -⟩ := hon ω hC
      rw [Set.indicator_of_mem hω, Set.indicator_of_mem hC, ENNReal.ofReal_natCast]
      simp only [probeTrace, st, hs, tc, tagCount, hr]
    · have hC : ω ∉ C := fun h => hω ⟨h.1.1, h.2⟩
      rw [Set.indicator_of_notMem hω, Set.indicator_of_notMem hC, ENNReal.ofReal_zero]
  have hCm : MeasurableSet C := by
    have : C = P ∩ cleanAll O := by
      ext ω; constructor
      · intro hω; exact ⟨hω.1.1, hω.1.2⟩
      · rintro ⟨hP, hc'⟩; exact (passCell_const O B F K k seed probes h₀ hP hc').2
    rw [this]; exact hPm'.inter (measurableSet_cleanAll O)
  have hint_ctc : Integrable (C.indicator fun ω => (tc ω : ℝ)) μ := hint_tc.indicator hCm
  have hFTbound : μ.real (C ∩ {ω | FTc ω}) ≤ uGood * ∫ ω, C.indicator (fun ω => (tc ω : ℝ)) ω ∂μ := by
    have h1 := (measure_mono hsub).trans hfresh
    rw [hlin, ← ofReal_integral_eq_lintegral_ofReal hint_ctc
      (ae_of_all _ fun ω => Set.indicator_nonneg (fun ω _ => Nat.cast_nonneg _) ω),
      ← ENNReal.ofReal_mul hu] at h1
    rw [measureReal_def]
    exact ENNReal.toReal_le_of_le_ofReal (mul_nonneg hu (integral_nonneg fun ω =>
      Set.indicator_nonneg (fun _ _ => Nat.cast_nonneg _) ω)) h1
  -- assemble
  have e3 : ∫ ω, C.indicator 1 ω * h ω ∂μ
      = μ.real (C ∩ {ω | FTc ω}) - uGood * ∫ ω, C.indicator (fun ω => (tc ω : ℝ)) ω ∂μ := by
    have hpt : (fun ω => C.indicator 1 ω * h ω) = fun ω =>
        (C ∩ {ω | FTc ω}).indicator (fun _ => (1 : ℝ)) ω
          - uGood * C.indicator (fun ω => (tc ω : ℝ)) ω := by
      funext ω
      by_cases hC : ω ∈ C
      · rw [Set.indicator_of_mem hC, Set.indicator_of_mem hC, Pi.one_apply, one_mul]
        by_cases hf : FTc ω
        · rw [Set.indicator_of_mem (show ω ∈ C ∩ {ω | FTc ω} from ⟨hC, hf⟩)]
          simp only [h, contrib]
          rw [if_pos hf]
        · rw [Set.indicator_of_notMem (show ω ∉ C ∩ {ω | FTc ω} from fun h' => hf h'.2)]
          simp only [h, contrib]
          rw [if_neg hf]
      · rw [Set.indicator_of_notMem hC, Set.indicator_of_notMem hC,
          Set.indicator_of_notMem (show ω ∉ C ∩ {ω | FTc ω} from fun h' => hC h'.1)]
        simp
    rw [hpt, integral_sub ((integrable_const (1 : ℝ)).indicator (hCm.inter hFTm'))
      (hint_ctc.const_mul _), integral_indicator_const _ (hCm.inter hFTm'), integral_const_mul,
      smul_eq_mul, mul_one]
  rw [← e1, e2, e3]
  linarith

end PassCells

section Words

/-- The strings of length `L`. -/
noncomputable def wordsOf (L : ℕ) : Finset (FreeMonoid α) :=
  Finset.univ.image fun f : Fin L → α => FreeMonoid.ofList (List.ofFn f)

theorem mem_wordsOf {L : ℕ} {x : FreeMonoid α} : x ∈ wordsOf (α := α) L ↔ x.toList.length = L := by
  constructor
  · intro h
    obtain ⟨f, -, rfl⟩ := Finset.mem_image.1 h
    simp
  · intro h
    refine Finset.mem_image.2 ⟨fun i : Fin L => x.toList[i.1]'(by have := i.2; omega),
      Finset.mem_univ _, ?_⟩
    apply FreeMonoid.toList.injective
    simp only [FreeMonoid.toList_ofList]
    exact List.ext_get (by simp [h]) fun n h1 h2 => by simp

omit [Fintype α] [DecidableEq α] in
theorem length_prefixOf {x : FreeMonoid α} {k : ℕ} (h : k ≤ x.toList.length) :
    (prefixOf x k).toList.length = k := by
  simp [prefixOf, h]

/-- Draws of length `L` sharing `x`'s first `k` letters carry at most `prefixMax D k`. -/
theorem sum_same_prefix_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ}
    (hkL : k ≤ L) {x : FreeMonoid α} (hx : x.toList.length = L) :
    ∑ y ∈ wordsOf (α := α) L, (if prefixOf y k = prefixOf x k then D.real {y} else 0)
      ≤ prefixMax D k := by
  classical
  rw [← Finset.sum_filter]
  calc ∑ y ∈ (wordsOf (α := α) L).filter (fun y => prefixOf y k = prefixOf x k), D.real {y}
      = D.real (↑((wordsOf (α := α) L).filter fun y => prefixOf y k = prefixOf x k)) := by
        rw [sum_measureReal_singleton]
    _ ≤ D.real {y | (prefixOf x k).toList <+: y.toList} := by
        refine measureReal_mono (fun y hy => ?_) (measure_ne_top _ _)
        simp only [Finset.coe_filter, Set.mem_ofPred_eq, mem_wordsOf] at hy
        rw [Set.mem_ofPred_eq, ← hy.2]
        exact List.take_prefix _ _
    _ ≤ prefixMax D k := by
        refine le_trans (le_of_eq ?_) (le_ciSup (f := fun p : FreeMonoid α =>
          if p.toList.length = k then D.real {x | p.toList <+: x.toList} else 0)
          ⟨1, by rintro _ ⟨p, rfl⟩; simp only []; split_ifs <;> simp [measureReal_le_one]⟩
          (prefixOf x k))
        rw [if_pos (length_prefixOf (by omega))]

end Words

section Tail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

theorem disjoint_drawBits {V : Finset (FreeMonoid α)}
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {x y : FreeMonoid α}
    (hx : k ≤ x.toList.length) (hy : k ≤ y.toList.length) (hxy : prefixOf x k ≠ prefixOf y k) :
    Disjoint (drawBits V c k x) (drawBits V c k y) := by
  rw [Set.disjoint_left]
  rintro z ⟨hzx, -⟩ ⟨hzy, -⟩
  apply hxy
  apply FreeMonoid.toList.injective
  exact List.prefix_of_prefix_length_le hzx hzy (by rw [length_prefixOf hx, length_prefixOf hy])
    |>.eq_of_length (by rw [length_prefixOf hx, length_prefixOf hy])

/-- Chebyshev inside a cell of the pass. -/
theorem cell_tail_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ} (hkL : k ≤ L) {uGood ε : ℝ}
    (hu : 0 ≤ uGood) (hε : 0 < ε) (hV : SuffixFree (F ∪ K.train F))
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ : Ω}
    (h₀ : ω₀ ∈ passCell O B F K k seed probes c) :
    μ.real (passCell O B F K k seed probes c ∩ {ω | ε * (1 + uGood
        * (passK O B F K k seed probes ω).tree.depth * L) < ∑ x ∈ wordsOf (α := α) L, D.real {x}
          * contrib A O B F uGood (readsAt O B F ω) (passK O B F K k seed probes ω).tree
            (passK O B F K k seed probes ω).edges (passReads O B F K k seed probes ω) k x})
      ≤ μ.real (passCell O B F K k seed probes c) * prefixMax D k / ε ^ 2 := by
  classical
  set V := F ∪ K.train F
  set s₀ := passK O B F K k seed probes ω₀
  set P := pcell O V c
  set C := passCell O B F K k seed probes c
  set d₀ : ℝ := (s₀.tree.depth : ℝ)
  set R : ℝ := 1 + uGood * d₀ * L
  set X := wordsOf (α := α) L
  set h : FreeMonoid α → Ω → ℝ := fun x ω =>
    contrib A O B F uGood (cellReads O B F V c ω) s₀.tree s₀.edges c.1 k x
  have hle := noiseAlg_le O
  have hd₀ : 0 ≤ d₀ := Nat.cast_nonneg _
  have hR : 0 < R := by positivity
  have hhm : ∀ x, Measurable[noiseAlg O (drawBits V c k x)] (h x) := fun x =>
    (measurable_contrib A O B F V uGood c s₀.tree s₀.edges c.1 k x).2.2
  have hhm' : ∀ x, Measurable (h x) := fun x => (hhm x).mono (hle _) le_rfl
  have hhb : ∀ x ∈ X, ∀ ω, -(uGood * d₀ * L) ≤ h x ω ∧ h x ω ≤ 1 := by
    intro x hx ω
    have hl := mem_wordsOf.1 hx
    have ht : (tagCount (cellReads O B F V c ω) s₀.tree s₀.edges k x : ℝ) ≤ d₀ * L := by
      have := tagCount_le k (cellReads O B F V c ω) s₀.tree s₀.edges x
      rw [hl] at this; simp only [d₀]; exact_mod_cast this
    have ht0 : (0 : ℝ) ≤ tagCount (cellReads O B F V c ω) s₀.tree s₀.edges k x := Nat.cast_nonneg _
    simp only [h, contrib]
    split_ifs <;> constructor <;> nlinarith
  have hudL : 0 ≤ uGood * d₀ * L := by positivity
  have hhi : ∀ x ∈ X, Integrable (h x) μ := fun x hx =>
    Integrable.of_bound (hhm' x).aestronglyMeasurable R (ae_of_all _ fun ω => by
      obtain ⟨h1, h2⟩ := hhb x hx ω
      rw [Real.norm_eq_abs, abs_le]; constructor <;> nlinarith)
  set m : FreeMonoid α → ℝ := fun x => ∫ ω, h x ω ∂μ
  have hmb : ∀ x ∈ X, -(uGood * d₀ * L) ≤ m x ∧ m x ≤ 1 := fun x hx =>
    ⟨by have := integral_mono (integrable_const _) (hhi x hx) fun ω => (hhb x hx ω).1
        simpa using this,
     by have := integral_mono (hhi x hx) (integrable_const _) fun ω => (hhb x hx ω).2
        simpa using this⟩
  have hdev : ∀ x ∈ X, ∀ ω, |h x ω - m x| ≤ R := fun x hx ω => by
    obtain ⟨h1, h2⟩ := hhb x hx ω; obtain ⟨h3, h4⟩ := hmb x hx
    rw [abs_le]; constructor <;> nlinarith
  -- the cell agrees with the pattern off a null set
  have hcl := measure_cleanAll_compl O (μ := μ)
  have hcla : ∀ᵐ ω ∂μ, ω ∈ cleanAll O := ae_iff.2 hcl
  have hPm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] P :=
    (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
  have hPm' : MeasurableSet P := hle _ _ hPm
  have hCeq : C = P ∩ cleanAll O := by
    ext ω; constructor
    · intro (hω : ω ∈ C); exact ⟨hω.1.1, hω.1.2⟩
    · rintro ⟨hP, hc'⟩; exact (passCell_const O B F K k seed probes h₀ hP hc').2
  have hCm : MeasurableSet C := by rw [hCeq]; exact hPm'.inter (measurableSet_cleanAll O)
  have hCP : μ.real C = μ.real P := by
    rw [hCeq, measureReal_def, measure_inter_conull hcl, ← measureReal_def]
  rcases eq_or_lt_of_le (measureReal_nonneg : 0 ≤ μ.real P) with hP0 | hP0
  · have : μ.real C = 0 := by rw [hCP, ← hP0]
    refine (measureReal_mono Set.inter_subset_left (measure_ne_top _ _)).trans ?_
    rw [this]; simp
  -- the means are at most zero
  have hm0 : ∀ x, m x ≤ 0 := fun x => by
    have := cell_mean_le O B F K k seed probes A hu hV h₀ x
    exact nonpos_of_mul_nonpos_right this hP0
  set W : Ω → ℝ := fun ω => ∑ x ∈ X, D.real {x} * h x ω
  set M : ℝ := ∑ x ∈ X, D.real {x} * m x
  have hM : M ≤ 0 := Finset.sum_nonpos fun x _ => mul_nonpos_of_nonneg_of_nonpos measureReal_nonneg
    (hm0 x)
  -- on the cell, the true sum is `W`
  have hon : ∀ ω ∈ C, (∑ x ∈ X, D.real {x} * contrib A O B F uGood (readsAt O B F ω)
      (passK O B F K k seed probes ω).tree (passK O B F K k seed probes ω).edges
      (passReads O B F K k seed probes ω) k x) = W ω
      ∧ (passK O B F K k seed probes ω).tree.depth = s₀.tree.depth := by
    intro ω hω
    have hr := cellReads_eq O B F V c hω.1.1.1 hω.1.2
    have hs := (passCell_const O B F K k seed probes h₀ hω.1.1 hω.1.2).1
    have hT : passReads O B F K k seed probes ω = c.1 := hω.2
    refine ⟨Finset.sum_congr rfl fun x _ => ?_, by rw [hs]⟩
    simp only [h, hr, hs, hT, s₀]
  set Z : Ω → ℝ := fun ω => C.indicator (fun ω => (W ω - M) ^ 2) ω
  have hsub : C ∩ {ω | ε * (1 + uGood * (passK O B F K k seed probes ω).tree.depth * L)
      < ∑ x ∈ X, D.real {x} * contrib A O B F uGood (readsAt O B F ω)
        (passK O B F K k seed probes ω).tree (passK O B F K k seed probes ω).edges
        (passReads O B F K k seed probes ω) k x} ⊆ {ω | (ε * R) ^ 2 ≤ Z ω} := by
    rintro ω ⟨hω, hlt⟩
    obtain ⟨hsum, hdep⟩ := hon ω hω
    simp only [Set.mem_ofPred_eq, hsum, hdep] at hlt
    simp only [Set.mem_ofPred_eq, Z, Set.indicator_of_mem hω]
    have : ε * R < W ω - M := by simp only [R, d₀]; linarith
    have hεR : 0 < ε * R := mul_pos hε hR
    nlinarith
  -- the second moment
  have hWm : Measurable W := Finset.measurable_sum _ fun x _ => (hhm' x).const_mul _
  have hZm : Measurable Z := ((hWm.sub_const M).pow_const 2).indicator hCm
  have hZb : ∀ ω, |Z ω| ≤ R ^ 2 := by
    intro ω
    have hsumD : ∑ x ∈ X, D.real {x} ≤ 1 := by
      rw [sum_measureReal_singleton]; exact measureReal_le_one
    have hWM : |W ω - M| ≤ R := by
      have : W ω - M = ∑ x ∈ X, D.real {x} * (h x ω - m x) := by
        simp only [W, M, ← Finset.sum_sub_distrib, mul_sub]
      rw [this]
      refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
      calc ∑ x ∈ X, |D.real {x} * (h x ω - m x)| ≤ ∑ x ∈ X, D.real {x} * R :=
            Finset.sum_le_sum fun x hx => by
              rw [abs_mul, abs_of_nonneg measureReal_nonneg]
              exact mul_le_mul_of_nonneg_left (hdev x hx ω) measureReal_nonneg
        _ = (∑ x ∈ X, D.real {x}) * R := by rw [Finset.sum_mul]
        _ ≤ 1 * R := by gcongr
        _ = R := one_mul R
    simp only [Z, Set.indicator]
    split_ifs
    · rw [abs_of_nonneg (sq_nonneg _)]
      exact (sq_le_sq₀ (abs_nonneg _) hR.le).2 hWM |>.trans' (by rw [sq_abs])
    · simp; positivity
  have hZi : Integrable Z μ := Integrable.of_bound hZm.aestronglyMeasurable (R ^ 2)
    (ae_of_all _ fun ω => by rw [Real.norm_eq_abs]; exact hZb ω)
  have hmarkov := mul_meas_ge_le_integral_of_nonneg (ae_of_all _ fun ω =>
    Set.indicator_nonneg (fun ω _ => sq_nonneg (W ω - M)) ω) hZi ((ε * R) ^ 2)
  -- the integral, pair by pair
  have hpair : ∀ x ∈ X, ∀ y ∈ X, ∫ ω, P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y)) ∂μ
      ≤ if prefixOf x k = prefixOf y k then μ.real P * R ^ 2 else 0 := by
    intro x hx y hy
    split_ifs with hxy
    · have hpt : ∀ ω, P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y))
          ≤ P.indicator (fun _ => R ^ 2) ω := by
        intro ω
        by_cases hω : ω ∈ P
        · rw [Set.indicator_of_mem hω, Set.indicator_of_mem hω, Pi.one_apply, one_mul]
          calc (h x ω - m x) * (h y ω - m y) ≤ |h x ω - m x| * |h y ω - m y| := by
                rw [← abs_mul]; exact le_abs_self _
            _ ≤ R * R := mul_le_mul (hdev x hx ω) (hdev y hy ω) (abs_nonneg _) hR.le
            _ = R ^ 2 := by ring
        · rw [Set.indicator_of_notMem hω, Set.indicator_of_notMem hω, zero_mul]
      have hint : Integrable (fun ω => P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y))) μ :=
        Integrable.of_bound ((((measurable_const.indicator hPm').mul
          (((hhm' x).sub_const _).mul ((hhm' y).sub_const _)))).aestronglyMeasurable) (R ^ 2)
          (ae_of_all _ fun ω => by
            rw [Real.norm_eq_abs, abs_mul, abs_mul]
            have h1 : |P.indicator (1 : Ω → ℝ) ω| ≤ 1 := by
              by_cases hω : ω ∈ P <;> simp [Set.indicator, hω]
            calc |P.indicator (1 : Ω → ℝ) ω| * (|h x ω - m x| * |h y ω - m y|)
                ≤ 1 * (R * R) := mul_le_mul h1 (mul_le_mul (hdev x hx ω) (hdev y hy ω)
                  (abs_nonneg _) hR.le) (by positivity) zero_le_one
              _ = R ^ 2 := by ring)
      calc ∫ ω, P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y)) ∂μ
          ≤ ∫ ω, P.indicator (fun _ => R ^ 2) ω ∂μ :=
            integral_mono hint ((integrable_const _).indicator hPm') hpt
        _ = μ.real P * R ^ 2 := by rw [integral_indicator_const _ hPm', smul_eq_mul]
    · have hxl := mem_wordsOf.1 hx; have hyl := mem_wordsOf.1 hy
      have hdis := disjoint_drawBits k (V := V) (c := c) (by omega) (by omega) hxy
      have hd1 : Disjoint (↑(vBits V c.1) : Set (FreeMonoid α)) (drawBits V c k x) :=
        Set.disjoint_left.2 fun z hz hz' => hz'.2 hz
      have hd2 : Disjoint (↑(vBits V c.1) : Set (FreeMonoid α)) (drawBits V c k y) :=
        Set.disjoint_left.2 fun z hz hz' => hz'.2 hz
      have := integral_indicator_mul_mul O hd1 hd2 hdis hPm ((hhm x).sub_const _)
        ((hhm y).sub_const _) (hdev x hx) (hdev y hy)
      rw [this, integral_sub (hhi x hx) (integrable_const _), integral_const]
      simp [m]
  have hcov : ∫ ω, Z ω ∂μ ≤ μ.real P * R ^ 2 * prefixMax D k := by
    have hZP : ∫ ω, Z ω ∂μ = ∫ ω, P.indicator (fun ω => (W ω - M) ^ 2) ω ∂μ := by
      refine integral_congr_ae (hcla.mono fun ω hω => ?_)
      change C.indicator (fun ω => (W ω - M) ^ 2) ω = P.indicator (fun ω => (W ω - M) ^ 2) ω
      by_cases hP : ω ∈ P
      · rw [Set.indicator_of_mem hP, Set.indicator_of_mem (hCeq ▸ ⟨hP, hω⟩ : ω ∈ C)]
      · rw [Set.indicator_of_notMem hP, Set.indicator_of_notMem fun h' => hP h'.1.1]
    have hexp : ∀ ω, P.indicator (fun ω => (W ω - M) ^ 2) ω = ∑ x ∈ X, ∑ y ∈ X,
        D.real {x} * D.real {y} * (P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y))) := by
      intro ω
      have hWM : W ω - M = ∑ x ∈ X, D.real {x} * (h x ω - m x) := by
        simp only [W, M, ← Finset.sum_sub_distrib, mul_sub]
      by_cases hω : ω ∈ P
      · rw [Set.indicator_of_mem hω, hWM, sq, Finset.sum_mul_sum]
        refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
        rw [Set.indicator_of_mem hω, Pi.one_apply]
        ring
      · rw [Set.indicator_of_notMem hω]
        simp [Set.indicator_of_notMem hω]
    have hint2 : ∀ x ∈ X, ∀ y ∈ X, Integrable
        (fun ω => D.real {x} * D.real {y} * (P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y))))
        μ := by
      intro x hx y hy
      refine Integrable.const_mul ?_ _
      exact Integrable.of_bound ((((measurable_const.indicator hPm').mul
          (((hhm' x).sub_const _).mul ((hhm' y).sub_const _)))).aestronglyMeasurable) (R ^ 2)
          (ae_of_all _ fun ω => by
            rw [Real.norm_eq_abs, abs_mul, abs_mul]
            have h1 : |P.indicator (1 : Ω → ℝ) ω| ≤ 1 := by
              by_cases hω : ω ∈ P <;> simp [Set.indicator, hω]
            calc |P.indicator (1 : Ω → ℝ) ω| * (|h x ω - m x| * |h y ω - m y|)
                ≤ 1 * (R * R) := mul_le_mul h1 (mul_le_mul (hdev x hx ω) (hdev y hy ω)
                  (abs_nonneg _) hR.le) (by positivity) zero_le_one
              _ = R ^ 2 := by ring)
    rw [hZP, integral_congr_ae (ae_of_all _ hexp), integral_finset_sum _ fun x hx =>
      integrable_finset_sum _ fun y hy => hint2 x hx y hy]
    calc ∑ x ∈ X, ∫ ω, ∑ y ∈ X, D.real {x} * D.real {y}
          * (P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y))) ∂μ
        = ∑ x ∈ X, ∑ y ∈ X, D.real {x} * D.real {y}
          * ∫ ω, P.indicator 1 ω * ((h x ω - m x) * (h y ω - m y)) ∂μ := by
          refine Finset.sum_congr rfl fun x hx => ?_
          rw [integral_finset_sum _ fun y hy => hint2 x hx y hy]
          exact Finset.sum_congr rfl fun y _ => integral_const_mul _ _
      _ ≤ ∑ x ∈ X, ∑ y ∈ X, D.real {x} * D.real {y}
          * (if prefixOf x k = prefixOf y k then μ.real P * R ^ 2 else 0) :=
          Finset.sum_le_sum fun x hx => Finset.sum_le_sum fun y hy =>
            mul_le_mul_of_nonneg_left (hpair x hx y hy)
              (mul_nonneg measureReal_nonneg measureReal_nonneg)
      _ = μ.real P * R ^ 2 * ∑ x ∈ X, D.real {x}
          * ∑ y ∈ X, (if prefixOf y k = prefixOf x k then D.real {y} else 0) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [Finset.mul_sum, Finset.mul_sum]
          refine Finset.sum_congr rfl fun y _ => ?_
          by_cases hxy : prefixOf x k = prefixOf y k
          · rw [if_pos hxy, if_pos hxy.symm]; ring
          · rw [if_neg hxy, if_neg fun h' => hxy h'.symm]; ring
      _ ≤ μ.real P * R ^ 2 * ∑ x ∈ X, D.real {x} * prefixMax D k := by
          refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun x hx =>
            mul_le_mul_of_nonneg_left (sum_same_prefix_le D hkL (mem_wordsOf.1 hx))
              measureReal_nonneg) (by positivity)
      _ ≤ μ.real P * R ^ 2 * prefixMax D k := by
          rw [← Finset.sum_mul, sum_measureReal_singleton]
          refine mul_le_mul_of_nonneg_left (mul_le_of_le_one_left ?_ measureReal_le_one)
            (by positivity)
          exact Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
  have hεR : 0 < (ε * R) ^ 2 := by positivity
  calc μ.real (C ∩ {ω | ε * (1 + uGood * (passK O B F K k seed probes ω).tree.depth * L)
        < ∑ x ∈ X, D.real {x} * contrib A O B F uGood (readsAt O B F ω)
          (passK O B F K k seed probes ω).tree (passK O B F K k seed probes ω).edges
          (passReads O B F K k seed probes ω) k x})
      ≤ μ.real {ω | (ε * R) ^ 2 ≤ Z ω} := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real P * R ^ 2 * prefixMax D k / (ε * R) ^ 2 := by
        rw [le_div_iff₀ hεR, mul_comm]; exact hmarkov.trans hcov
    _ = μ.real C * prefixMax D k / ε ^ 2 := by
        rw [hCP]; field_simp

end Tail

section Draws

variable {Q : Type*}

open scoped Classical in
theorem real_eq_sum_words (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) (S : Set (FreeMonoid α)) :
    D.real S = ∑ x ∈ wordsOf (α := α) L, if x ∈ S then D.real {x} else 0 := by
  classical
  have hW : D (↑(wordsOf (α := α) L))ᶜ = 0 := by
    refine measure_mono_null (fun x hx => ?_) (ae_iff.1 hlen)
    simp only [Set.mem_compl_iff, Finset.mem_coe, mem_wordsOf] at hx
    exact hx
  rw [← Finset.sum_filter, sum_measureReal_singleton]
  rw [measureReal_def, measureReal_def, ← measure_inter_conull (s := S) hW]
  congr 2
  ext x
  simp [and_comm]

theorem integral_eq_sum_words (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) (f : FreeMonoid α → ℝ) :
    ∫ x, f x ∂D = ∑ x ∈ wordsOf (α := α) L, D.real {x} * f x := by
  classical
  have h0 : ∀ x ∉ wordsOf (α := α) L, D.real {x} = 0 := by
    intro x hx
    rw [measureReal_def, measure_mono_null (Set.singleton_subset_iff.2 (show x ∈
      {x : FreeMonoid α | ¬ x.toList.length = L} from fun h => hx (mem_wordsOf.2 h)))
      (ae_iff.1 hlen), ENNReal.toReal_zero]
  have hint : Integrable f D := by
    have hfin : ∀ᵐ x ∂D, x ∈ (wordsOf (α := α) L : Set (FreeMonoid α)) :=
      hlen.mono fun x hx => mem_wordsOf.2 hx
    refine Integrable.of_bound measurable_from_top.aestronglyMeasurable
      (∑ x ∈ wordsOf (α := α) L, |f x|) (hfin.mono fun x hx => ?_)
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (f := fun x => |f x|) (fun _ _ => abs_nonneg _) hx
  rw [integral_countable hint, tsum_eq_sum (s := wordsOf (α := α) L) fun x hx => by
    simp [h0 x hx]]
  simp [smul_eq_mul]

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

/-- Draws whose triple harvests a string the pass may have read share their first `k + 1`
letters with one of those strings. -/
theorem touched_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (R : CutReads α)
    (t : DTree α) (edges : Edges α) (Tp : Finset (FreeMonoid α)) (k : ℕ) :
    D.real {x | ∃ j b, probeOutcome R t edges k x = .triple j ∧ tripleRead R t x j = some b
        ∧ b ∈ Tp} ≤ Tp.card * prefixMax D (k + 1) := by
  classical
  have hsub : {x | ∃ j b, probeOutcome R t edges k x = .triple j ∧ tripleRead R t x j = some b
      ∧ b ∈ Tp} ⊆ ⋃ w ∈ Tp.filter (fun w => k + 1 ≤ w.toList.length),
        {x | (prefixOf w (k + 1)).toList <+: x.toList} := by
    rintro x ⟨j, b, hj, hb, hT⟩
    have hkj := probeOutcome_triple_gt hj
    have hjx : j ≤ x.toList.length := by
      obtain ⟨ps, hi, hw, hbr⟩ := probeOutcome_search R hj trivial
      obtain ⟨-, -, -, -, hhi, -⟩ := walkCheck_inr R hw
      have := bracketAt_triple_lt (α := α) _ ps _ _ _ _ hbr
      omega
    simp only [tripleRead] at hb
    rcases hs : t.sift R.cut (prefixOf x j) with _ | b' <;> rw [hs] at hb
    · simp at hb
    obtain rfl : b' = b := by simpa using hb
    obtain ⟨m, rfl⟩ := route_inr_form t _ b' hs
    refine Set.mem_biUnion (Finset.mem_filter.2 ⟨hT, by simp [prefixOf]; omega⟩) ?_
    simp only [Set.mem_ofPred_eq, prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
    rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left (by omega)]
    exact List.take_prefix _ _
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans ?_)
  have hpm : 0 ≤ prefixMax D (k + 1) :=
    Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
  calc ∑ w ∈ Tp.filter (fun w => k + 1 ≤ w.toList.length),
        D.real {x | (prefixOf w (k + 1)).toList <+: x.toList}
      ≤ ∑ _w ∈ Tp.filter (fun w => k + 1 ≤ w.toList.length), prefixMax D (k + 1) :=
        Finset.sum_le_sum fun w hw => by
          refine le_trans (le_of_eq ?_) (le_ciSup (f := fun p : FreeMonoid α =>
            if p.toList.length = k + 1 then D.real {x | p.toList <+: x.toList} else 0)
            ⟨1, by rintro _ ⟨p, rfl⟩; simp only []; split_ifs <;> simp [measureReal_le_one]⟩
            (prefixOf w (k + 1)))
          rw [if_pos (length_prefixOf (Finset.mem_filter.1 hw).2)]
    _ = (Tp.filter (fun w => k + 1 ≤ w.toList.length)).card * prefixMax D (k + 1) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ Tp.card * prefixMax D (k + 1) := by
        have := Finset.card_filter_le Tp (fun w => k + 1 ≤ w.toList.length)
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast this) hpm

end Draws

section Fail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

open scoped Classical in
/-- Where the triples' claim fails, the draws' shares sum past the fluctuation allowed. -/
theorem triple_fail_sum (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) {uGood ε : ℝ} (hu : 0 ≤ uGood) (R : CutReads α) (t : DTree α)
    (edges : Edges α) (Tp : Finset (FreeMonoid α)) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k L : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length = L)
    (h : ¬ D.real {x | ∃ j b, probeOutcome R t edges k x = .triple j
        ∧ tripleRead R t x j = some b ∧ stateIndecision A O B F (A.state b) < uGood}
      ≤ uGood * t.depth * ∫ x, (visits R t edges k x : ℝ) ∂D
        + Tp.card * prefixMax D (k + 1) + ε * (1 + uGood * t.depth * L)) :
    ε * (1 + uGood * t.depth * L)
      < ∑ x ∈ wordsOf (α := α) L, D.real {x} * contrib A O B F uGood R t edges Tp k x := by
  set X := wordsOf (α := α) L
  have h1 : D.real {x | ∃ j b, probeOutcome R t edges k x = .triple j
      ∧ tripleRead R t x j = some b ∧ stateIndecision A O B F (A.state b) < uGood}
      ≤ D.real {x | FreshTriple A O B F uGood R t edges Tp k x}
        + D.real {x | ∃ j b, probeOutcome R t edges k x = .triple j
          ∧ tripleRead R t x j = some b ∧ b ∈ Tp} := by
    refine (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)).trans
      (measureReal_union_le _ _)
    obtain ⟨j, b, hj, hb, hg⟩ := hx
    by_cases hT : b ∈ Tp
    · exact .inr ⟨j, b, hj, hb, hT⟩
    · exact .inl ⟨j, b, hj, hb, hg, hT⟩
  have h2 := touched_le D R t edges Tp k
  have h3 : D.real {x | FreshTriple A O B F uGood R t edges Tp k x}
      = ∑ x ∈ X, D.real {x} * (if FreshTriple A O B F uGood R t edges Tp k x then 1 else 0) := by
    rw [real_eq_sum_words D hlen]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [Set.mem_ofPred_eq]
    split_ifs <;> simp
  have h4 := integral_eq_sum_words D hlen fun x => (visits R t edges k x : ℝ)
  have h5 : ∑ x ∈ X, D.real {x} * (tagCount R t edges k x : ℝ)
      ≤ t.depth * ∑ x ∈ X, D.real {x} * (visits R t edges k x : ℝ) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun x _ => ?_
    have := qProbe_countP R t edges k x
    have h' : (tagCount R t edges k x : ℝ) ≤ t.depth * visits R t edges k x := by
      exact_mod_cast this
    nlinarith [measureReal_nonneg (μ := D) (s := {x})]
  have h6 : ∑ x ∈ X, D.real {x} * contrib A O B F uGood R t edges Tp k x
      = ∑ x ∈ X, D.real {x} * (if FreshTriple A O B F uGood R t edges Tp k x then 1 else 0)
        - uGood * ∑ x ∈ X, D.real {x} * (tagCount R t edges k x : ℝ) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [contrib]
    ring
  rw [h6, ← h3]
  rw [h4] at h
  have h7 : uGood * ∑ x ∈ X, D.real {x} * (tagCount R t edges k x : ℝ)
      ≤ uGood * t.depth * ∑ x ∈ X, D.real {x} * (visits R t edges k x : ℝ) := by
    rw [mul_assoc]; exact mul_le_mul_of_nonneg_left h5 hu
  push Not at h
  linarith

end Fail

end OrthoDFA
