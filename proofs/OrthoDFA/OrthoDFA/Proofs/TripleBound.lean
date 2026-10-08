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
    Measurable[noiseAlg O (drawBits V c k x)]
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

end OrthoDFA
