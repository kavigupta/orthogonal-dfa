import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.GateFlip

/-!
# `RoundAtK`, `WalkYield` and `SourceSpread`

A decision on a batch against a fixed hypothesis is a Hoeffding tail over the batch.  A refusal's
probes reach the split test or stop at a string the cut cannot place, because the pass keeps its
edges learned: every edge's witness sits at its leaf and its extension at its target.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Batch

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
/-- The batch's count of `P`, as the sum of its indicators. -/
theorem sum_indicator_eq {n : ℕ} (P : FreeMonoid α → Prop) (b : Fin n → FreeMonoid α) :
    ∑ i, (if P (b i) then (1 : ℝ) else 0) = ((Finset.univ.filter fun i => P (b i)).card : ℝ) := by
  rw [Finset.sum_boole]

open scoped Classical in
theorem batch_indicator_facts (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) :
    (∀ i, AEMeasurable (fun b : Fin n → FreeMonoid α => if P (b i) then (1 : ℝ) else 0)
        (Measure.pi fun _ : Fin n => D))
      ∧ iIndepFun (fun i (b : Fin n → FreeMonoid α) => if P (b i) then (1 : ℝ) else 0)
        (Measure.pi fun _ : Fin n => D)
      ∧ (∀ i, ∀ᵐ b ∂(Measure.pi fun _ : Fin n => D),
          (if P (b i) then (1 : ℝ) else 0) ∈ Set.Icc (0 : ℝ) 1)
      ∧ ∀ i, (Measure.pi fun _ : Fin n => D)[fun b => if P (b i) then (1 : ℝ) else 0]
          = D.real {x | P x} := by
  refine ⟨fun i => (measurable_of_countable _).aemeasurable, ?_, fun i => ?_, fun i => ?_⟩
  · exact iIndepFun_pi (μ := fun _ : Fin n => D)
      (X := fun _ (y : FreeMonoid α) => if P y then (1 : ℝ) else 0)
      fun _ => (measurable_of_countable _).aemeasurable
  · exact Filter.Eventually.of_forall fun b => by split_ifs <;> simp
  · have hS : MeasurableSet {x : FreeMonoid α | P x} := MeasurableSpace.measurableSet_top
    have heq : (fun b : Fin n → FreeMonoid α => if P (b i) then (1 : ℝ) else 0)
        = (Function.eval i ⁻¹' {x | P x}).indicator 1 := by
      ext b; simp [Set.indicator_apply]
    rw [heq, integral_indicator_one ((measurable_pi_apply i) hS),
      measureReal_def, measureReal_def,
      ← Measure.map_apply (measurable_pi_apply i) hS, (measurePreserving_eval _ i).map_eq]

/-- A predicate holding on at most `θ − δ` of draws holds on more than `θ` of a batch of `n`
with chance at most `exp(−2nδ²)`. -/
theorem share_gt_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin n => D).real {b | θ < share b P} ≤ Real.exp (-2 * n * δ ^ 2) := by
  classical
  obtain ⟨hm, hi, hI, hmean⟩ := batch_indicator_facts D P n
  have key := sumUpper_le _ Finset.univ (θ - δ) δ hm hi hI (by
    simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact le_of_eq rfl |>.trans (by gcongr)) hδ
  simp only [Finset.card_univ, Fintype.card_fin, sub_add_cancel] at key
  refine le_trans (measureReal_mono fun b hb => ?_) key
  simp only [Set.mem_ofPred_eq, share] at hb ⊢
  rw [sum_indicator_eq]
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp at hb ⊢
  · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    rw [lt_div_iff₀ hn'] at hb
    linarith

/-- One holding on at least `ε + δ` of draws holds on at most `ε` of a batch with chance at most
`exp(−2nδ²)`. -/
theorem share_le_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {ε δ : ℝ} (hδ : 0 ≤ δ)
    (h : ε + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin n => D).real {b | share b P ≤ ε} ≤ Real.exp (-2 * n * δ ^ 2) := by
  classical
  obtain ⟨hm, hi, hI, hmean⟩ := batch_indicator_facts D P n
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [CharP.cast_eq_zero, mul_zero, zero_mul, Real.exp_zero]
    exact measureReal_le_one
  have key := sumLower_le _ Finset.univ (ε + δ) δ hm hi hI (by
    simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    gcongr) hδ
  simp only [Finset.card_univ, Fintype.card_fin, add_sub_cancel_right] at key
  refine le_trans (measureReal_mono fun b hb => ?_) key
  simp only [Set.mem_ofPred_eq, share] at hb ⊢
  rw [sum_indicator_eq]
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rwa [div_le_iff₀ hn', mul_comm] at hb

end Batch

section Learned

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness sits at the edge's leaf, and its extension by the letter at the edge's
target. -/
def Learned (t : DTree α) (edges : Edges α) : Prop :=
  ∀ p c q y, edges p c = some (q, y) →
    t.sift R.cut y = .inl p ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q

omit [Fintype α] in
theorem route_splitAt {cut : FreeMonoid α → Option Bool} (d : FreeMonoid α) :
    ∀ (t : DTree α) (x : FreeMonoid α) (s p : List Bool), (t.route cut x).2 = .inl p → p ≠ s →
      ((t.splitAt d s).route cut x).2 = .inl p
  | .leaf, x, s, p, h, hps => by
    simp only [DTree.route, Sum.inl.injEq] at h
    subst h
    rcases s with _ | ⟨b, s⟩
    · exact absurd rfl hps
    · rfl
  | .node m r a, x, s, p, h, hps => by
    simp only [DTree.route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> rw [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with p' | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, hr]
        · have := route_splitAt d r x s p' hr (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
        · simp [DTree.splitAt, DTree.route, hc, hr]
      · simp at h
    · rcases ha : (a.route cut x).2 with p' | b <;> rw [ha] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · have := route_splitAt d a x s p' ha (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
      · simp at h

omit [Fintype α] in
theorem sift_splitAt {cut : FreeMonoid α → Option Bool} {t : DTree α} {x : FreeMonoid α}
    {d : FreeMonoid α} {s p : List Bool} (h : t.sift cut x = .inl p) (hps : p ≠ s) :
    (t.splitAt d s).sift cut x = .inl p :=
  route_splitAt d t x s p h hps

theorem decisiveTarget_learned {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {c : α} {cur : Option (List Bool)} {q : List Bool} {y : FreeMonoid α}
    (h : decisiveTarget K R t pool path c cur = some (q, y)) :
    t.sift R.cut y = .inl path ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q := by
  refine ⟨?_, ?_⟩
  · have hm := decisiveTarget_mem K R h
    have := (List.mem_filter.1 (List.mem_of_mem_take hm)).2
    simpa using this
  · unfold decisiveTarget at h
    dsimp only at h
    generalize hV : List.filterMap _ (members K R t pool path) = V at h
    rcases foldl_best_mem _ (fun o v r hr => by
        rcases o with _ | b
        · exact .inl (Option.some.inj hr).symm
        · dsimp only at hr
          split_ifs at hr
          · exact .inl (Option.some.inj hr).symm
          · exact .inr hr) V none (q, y) h with hm | hm
    · rw [← hV] at hm
      obtain ⟨a, -, hfa⟩ := List.mem_filterMap.1 hm
      split at hfa
      · rename_i p hp
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hfa)
        exact hp
      · simp at hfa
    · simp at hm

theorem closeEdges_learned {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (h : Learned R t edges) : Learned R t (closeEdges K R t pool edges) := by
  intro p c q y he
  simp only [closeEdges] at he
  rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', y'⟩ <;>
    rw [hd] at he
  · exact h p c q y (by simpa using he)
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj (by simpa using he))
    exact decisiveTarget_learned K R hd

theorem probeStepK_learned {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (h : Learned R s.tree s.edges) :
    Learned R (probeStepK K R k s x).tree (probeStepK K R k s x).edges := by
  simp only [probeStepK]
  split
  · split
    · rename_i d s1 y sprime _
      refine closeEdges_learned K R fun p c q w he => ?_
      rcases hE : s.edges p c with _ | ⟨q', w'⟩ <;> simp only [hE] at he
      · simp at he
      · split_ifs at he with hcl
        all_goals first
          | (simp at he; done)
          | (simp only [Option.some.injEq, Prod.mk.injEq] at he
             obtain ⟨rfl, rfl⟩ := he
             have hcl' : p ≠ s1 ∧ q' ≠ s1 := by tauto
             obtain ⟨h1, h2⟩ := h p c q' w' hE
             exact ⟨sift_splitAt h1 hcl'.1, sift_splitAt h2 hcl'.2⟩)
    · exact closeEdges_learned K R h
    · exact closeEdges_learned K R h
  · exact closeEdges_learned K R h

theorem runPassK_learned (k : ℕ) (seed probes : List (FreeMonoid α)) :
    Learned R (runPassK K R k (initialK K R seed) probes).tree
      (runPassK K R k (initialK K R seed) probes).edges := by
  unfold runPassK
  have h0 : Learned R (initialK K R seed).tree (initialK K R seed).edges :=
    closeEdges_learned K R fun _ _ _ _ he => by simp at he
  generalize initialK K R seed = s at h0 ⊢
  induction probes generalizing s with
  | nil => exact h0
  | cons x xs ih =>
    simp only [List.foldl_cons]
    refine ih _ ?_
    split
    · exact h0
    · exact probeStepK_learned K R h0

theorem parting_none {cut : FreeMonoid α → Option Bool} {x y pre : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = none →
      ∃ p, t.sift cut (x * pre) = .inl p ∧ t.sift cut (y * pre) = .inl p
  | .leaf, _ => ⟨[], rfl, rfl⟩
  | .node m r a, h => by
    simp only [DTree.parting] at h
    unfold DTree.sift DTree.route
    simp only [mul_assoc]
    rcases hx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hx, hy, reduceCtorEq] at h
    · obtain ⟨p, h1, h2⟩ := parting_none r h
      unfold DTree.sift at h1 h2
      exact ⟨false :: p, by simp [h1], by simp [h2]⟩
    · obtain ⟨p, h1, h2⟩ := parting_none a h
      unfold DTree.sift at h1 h2
      exact ⟨true :: p, by simp [h1], by simp [h2]⟩

omit [Fintype α] [DecidableEq α] in
theorem follow_inl {edges : Edges α} :
    ∀ (cs : List α) (p : List Bool) (ps : List (List Bool)), follow edges p cs = .inl ps →
      ps.length = cs.length + 1 ∧ ps.getD 0 [] = p
        ∧ ∀ i (hi : i < cs.length), ∃ y, edges (ps.getD i []) cs[i] = some (ps.getD (i + 1) [], y)
  | [], p, ps, h => by
    simp only [follow, Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, fun i hi => absurd hi (by simp)⟩
  | c :: cs, p, ps, h => by
    simp only [follow] at h
    rcases he : edges p c with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp at h
    · simp only [] at h
      rcases hf : follow edges q cs with ps' | r <;> rw [hf] at h
      · simp only [Sum.inl.injEq] at h
        subst h
        obtain ⟨hl, hh, hs⟩ := follow_inl cs q ps' hf
        refine ⟨by simp [hl], rfl, fun i hi => ?_⟩
        rcases i with _ | i
        · exact ⟨y, by simp only [List.getD_cons_zero, List.getD_cons_succ, hh,
            List.getElem_cons_zero]; exact he⟩
        · obtain ⟨y', hy'⟩ := hs i (by simpa using hi)
          exact ⟨y', by simpa using hy'⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem bisectAt_spec (place walk : ℕ → List Bool) :
    ∀ fuel lo hi, lo < hi → hi - lo ≤ fuel → place lo = walk lo → place hi ≠ walk hi →
      lo < bisectAt place walk fuel lo hi ∧ bisectAt place walk fuel lo hi ≤ hi
        ∧ place (bisectAt place walk fuel lo hi - 1) = walk (bisectAt place walk fuel lo hi - 1)
        ∧ place (bisectAt place walk fuel lo hi) ≠ walk (bisectAt place walk fuel lo hi)
  | 0, lo, hi, hlt, hf, _, _ => by omega
  | fuel + 1, lo, hi, hlt, hf, hlo, hhi => by
    simp only [bisectAt]
    split_ifs with h1 h2
    · have := bisectAt_spec place walk fuel ((lo + hi) / 2) hi (by omega) (by omega) h2 hhi
      exact ⟨by omega, this.2.1, this.2.2⟩
    · have := bisectAt_spec place walk fuel lo ((lo + hi) / 2) (by omega) (by omega) hlo h2
      exact ⟨this.1, by omega, this.2.2⟩
    · have : hi = lo + 1 := by omega
      subst this
      exact ⟨by omega, le_rfl, by simpa using hlo, hhi⟩

theorem kCheck_disagree {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} (h : kCheck R t edges k x = .disagree ps) :
    ∃ p₀ a, t.sift R.cut (prefixOf x k) = .inl p₀ ∧ follow edges p₀ (x.toList.drop k) = .inl ps
      ∧ t.sift R.cut x = .inl a ∧ some a ≠ ps.getLast? := by
  unfold kCheck at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hw] at h
  · simp at h
  · simp at h
  · simp only [] at h
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at h
    · simp only [] at h
      split_ifs at h with ha
      simp only [KCheck.disagree.injEq] at h
      subst h
      unfold kWalk at hw
      rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
      · simp only [] at hw
        rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hw
        · simp only [KWalk.reached.injEq] at hw
          subst hw
          exact ⟨p₀, a, rfl, hf, rfl, ha⟩
        · simp at hw
      · simp at hw
    · simp at h

theorem seedStep_ne_dropped {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)}
    (h : kCheck R t edges k x = .disagree ps) :
    seedStep K R t pool edges k x ps ≠ .dropped := by
  obtain ⟨p₀, a, hk, hf, hx, hne⟩ := kCheck_disagree R h
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  set n := x.toList.length with hn
  set walkAt : ℕ → List Bool := fun j => ps.getD (j - k) [] with hwalk
  have hlast : ps.getLast? = some (ps.getD (n - k) []) := by
    simp only [List.length_drop] at hlen
    have hidx : n - k < ps.length := by omega
    rw [List.getLast?_eq_getElem?, show ps.length - 1 = n - k by omega,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx]
    rfl
  have hkn : k < n := by
    by_contra hkn
    have hxk : prefixOf x k = x := by
      apply FreeMonoid.toList.injective
      simp [prefixOf, List.take_of_length_le (not_lt.1 hkn)]
    rw [hxk, hx] at hk
    obtain rfl := Sum.inl.inj hk
    simp only [List.length_drop] at hlen
    have : ps.getD (n - k) [] = a := by
      rw [show n - k = 0 by omega, hhead]
    exact hne (by rw [hlast, this])
  have hpk : place R t x k = walkAt k := by
    simp only [place, hk, Sum.elim_inl, id, hwalk, Nat.sub_self]
    exact hhead.symm
  have hpn : place R t x n ≠ walkAt n := by
    have : prefixOf x n = x := prefixOf_length x
    simp only [place, this, hx, Sum.elim_inl, id, hwalk]
    intro he
    exact hne (by rw [hlast, he])
  obtain ⟨hfd1, hfd2, hfd3, hfd4⟩ :=
    bisectAt_spec (place R t x) walkAt (n - k) k n hkn le_rfl hpk hpn
  unfold seedStep
  simp only []
  generalize hfd : bisectAt (place R t x) (fun j => ps.getD (j - k) []) (n - k) k n = fd at *
  have hfdn : fd - 1 < n := by omega
  rw [List.getElem?_eq_getElem hfdn]
  simp only []
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : (x.toList.drop k)[fd - 1 - k]'(by simp; omega) = x.toList[fd - 1] := by
    simp only [List.getElem_drop]
    congr 1
    omega
  rw [hidx, show fd - 1 - k + 1 = fd - k by omega] at hy
  rw [hy]
  simp only [ne_eq, not_true_eq_false, if_false]
  obtain ⟨hy1, hy2⟩ := hl _ _ _ _ hy
  rcases hsp : t.sift R.cut (prefixOf x (fd - 1)) with p | b
  · simp only []
    have hp : p = ps.getD (fd - 1 - k) [] := by
      have := hfd3
      simp only [place, hsp, Sum.elim_inl, id, hwalk] at this
      exact this
    subst hp
    simp only [ne_eq, not_true_eq_false, false_or, hy1, not_true_eq_false, if_false]
    rcases hpt : t.parting R.cut y (prefixOf x (fd - 1)) (FreeMonoid.of x.toList[fd - 1]) with
      _ | d | b
    · exfalso
      obtain ⟨q, h1, h2⟩ := parting_none t hpt
      rw [hy2] at h1
      rw [← prefixOf_succ hfdn, show fd - 1 + 1 = fd by omega] at h2
      apply hfd4
      simp only [place, ← h1, h2, Sum.elim_inl, id, hwalk]
    · simp only []
      split <;> simp
    · simp
  · simp

end Learned

theorem prod_real_rect {β γ : Type*} [MeasurableSpace β] [MeasurableSpace γ]
    (μ : Measure β) (ν : Measure γ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (A : Set β) (B : Set γ) : (μ.prod ν).real (A ×ˢ B) = μ.real A * ν.real B := by
  simp only [measureReal_def, Measure.prod_prod, ENNReal.toReal_mul]

theorem round_at_k_holds : RoundAtK := by
  intro α _ _ K R D _ k nw ng seed probes θw θc ε δ hδ
  simp only []
  set s₀ := initialK K R seed
  set s := runPassK K R k s₀ probes
  have hl : Learned R s.tree s.edges := runPassK_learned K R k seed probes
  set Pw := fun x => (kWalk R s₀.tree s₀.edges k x).isBlocked
  set Pc := fun x => (kCheck R s.tree s.edges k x).isBlocked
  set Pd := fun x => (kCheck R s.tree s.edges k x).isDisagreement
  set νw := Measure.pi fun _ : Fin nw => D
  set νg := Measure.pi fun _ : Fin ng => D
  set A := {bw : Fin nw → FreeMonoid α | θw < share bw Pw ∧ D.real {x | Pw x} < θw - δ}
  set B := {bg : Fin ng → FreeMonoid α | θc < share bg Pc ∧ D.real {x | Pc x} < θc - δ}
  set C := {bg : Fin ng → FreeMonoid α | share bg Pd ≤ ε ∧ ε + δ < D.real {x | Pd x}}
  have hsub : {b : (Fin nw → FreeMonoid α) × (Fin ng → FreeMonoid α) |
      ¬ RoundAtKHolds K R s₀.tree s₀.edges s D k θw θc ε δ b.1 b.2}
      ⊆ (A ×ˢ Set.univ ∪ Set.univ ×ˢ B) ∪ Set.univ ×ˢ C := by
    rintro ⟨bw, bg⟩ hb
    simp only [Set.mem_ofPred_eq, RoundAtKHolds] at hb
    split_ifs at hb with h1 h2 h3
    · exact .inl (.inl ⟨⟨h1, not_le.1 hb⟩, trivial⟩)
    · exact .inl (.inr ⟨trivial, h2, not_le.1 hb⟩)
    · exact .inr ⟨trivial, h3, not_le.1 hb⟩
    · simp only [not_forall, not_not] at hb
      obtain ⟨i, ps, hc, hd⟩ := hb
      exact absurd hd (seedStep_ne_dropped K R hl hc)
  have hA : νw.real A ≤ Real.exp (-2 * nw * δ ^ 2) := by
    by_cases h : D.real {x | Pw x} < θw - δ
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_gt_le D Pw nw hδ h.le)
    · rw [show A = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have hB : νg.real B ≤ Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h : D.real {x | Pc x} < θc - δ
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_gt_le D Pc ng hδ h.le)
    · rw [show B = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have hC : νg.real C ≤ Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h : ε + δ < D.real {x | Pd x}
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_le_le D Pd ng hδ h.le)
    · rw [show C = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have h1 := measureReal_union_le (μ := νw.prod νg) (A ×ˢ Set.univ ∪ Set.univ ×ˢ B)
    (Set.univ ×ˢ C)
  have h2 := measureReal_union_le (μ := νw.prod νg) (A ×ˢ Set.univ) (Set.univ ×ˢ B)
  rw [prod_real_rect, prod_real_rect] at h2
  rw [prod_real_rect] at h1
  simp only [probReal_univ, mul_one, one_mul] at h1 h2
  refine (measureReal_mono hsub).trans ?_
  linarith

section Sources

variable (R : CutReads α)

theorem walkOutput_isSome_iff (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (walkOutput R t edges k x).isSome
      ↔ (kWalk R t edges k x).isBlocked ∧ ¬ wrongEarlier R t edges k x := by
  unfold walkOutput wrongEarlier
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps
  · simp [KWalk.isBlocked]
  · have key : (∃ s' c' j' p, KWalk.edge s c j = KWalk.edge s' c' j'
        ∧ (t.sift R.cut (prefixOf x (j' + 1))).isLeft
        ∧ t.sift R.cut (prefixOf x j') = .inl p ∧ p ≠ s')
        ↔ (t.sift R.cut (prefixOf x (j + 1))).isLeft
          ∧ ∃ p, t.sift R.cut (prefixOf x j) = .inl p ∧ p ≠ s :=
      ⟨fun ⟨_, _, _, p, he, h1, h2, h3⟩ => by cases he; exact ⟨h1, p, h2, h3⟩,
        fun ⟨h1, p, h2, h3⟩ => ⟨s, c, j, p, rfl, h1, h2, h3⟩⟩
    rw [key]
    simp only [KWalk.isBlocked, true_and]
    rcases h1 : t.sift R.cut (prefixOf x (j + 1)) with p1 | b1
    · rcases h2 : t.sift R.cut (prefixOf x j) with p2 | b2
      · by_cases hp : p2 = s
        · simp [hp]
        · simp [hp]
      · simp
    · simp
  · simp [KWalk.isBlocked]

theorem walk_yield_holds : WalkYield := by
  intro α _ _ R D _ t edges k
  have hsub : {x | wrongEarlier R t edges k x} ⊆ {x | (kWalk R t edges k x).isBlocked} := by
    rintro x ⟨s, c, j, p, hw, -⟩
    simp [hw, KWalk.isBlocked]
  have heq : {x | (walkOutput R t edges k x).isSome}
      = {x | (kWalk R t edges k x).isBlocked} \ {x | wrongEarlier R t edges k x} := by
    ext x
    simp only [Set.mem_ofPred_eq, Set.mem_sdiff]
    exact walkOutput_isSome_iff R t edges k x
  rw [heq, measureReal_sdiff hsub MeasurableSpace.measurableSet_top]

theorem route_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x b : FreeMonoid α), (t.route cut x).2 = .inr b → ∃ m, b = x * m
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, x, b, h => by
    simp only [DTree.route] at h
    split at h
    · exact ⟨m, (Sum.inr.inj h).symm⟩
    · rcases ha : (a.route cut x).2 with p | b' <;> rw [ha] at h
      · simp at h
      · exact route_inr a x b' ha |>.imp fun m hm => by simp at h; rw [← h, hm]
    · rcases hr : (r.route cut x).2 with p | b' <;> rw [hr] at h
      · simp at h
      · exact route_inr r x b' hr |>.imp fun m hm => by simp at h; rw [← h, hm]

theorem take_prefixOf {x : FreeMonoid α} {i k : ℕ} (hk : k ≤ i) :
    (prefixOf x i).toList.take k = x.toList.take k := by
  simp [prefixOf, List.take_take, min_eq_left hk]

theorem walkOutput_prefix {t : DTree α} {edges : Edges α} {k : ℕ} {x u : FreeMonoid α}
    (h : walkOutput R t edges k x = some u) : u.toList.take k = x.toList.take k := by
  unfold walkOutput at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps <;> rw [hw] at h
  · simp only [Option.some.injEq] at h
    rw [← h, take_prefixOf le_rfl]
  · have hj : k ≤ j := by
      unfold kWalk at hw
      split at hw
      · simp at hw
      · split at hw
        · simp at hw
        · simp only [KWalk.edge.injEq] at hw
          omega
    simp only [] at h
    rcases h1 : t.sift R.cut (prefixOf x (j + 1)) with p1 | b1 <;> rw [h1] at h
    · simp only [] at h
      rcases h2 : t.sift R.cut (prefixOf x j) with p2 | b2 <;> rw [h2] at h
      · simp only [] at h
        split_ifs at h
        simp only [Option.some.injEq] at h
        rw [← h, take_prefixOf hj]
      · simp only [Option.some.injEq] at h
        rw [← h, take_prefixOf hj]
    · simp only [Option.some.injEq] at h
      rw [← h, take_prefixOf (by omega)]
  · simp at h

theorem source_spread_holds : SourceSpread := by
  intro α _ _ R D _ t edges k L u hkL hlen
  rw [measureReal_def, measureReal_def]
  refine ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae ?_)
  filter_upwards [hlen] with x hx hu
  change walkOutput R t edges k x = some u ∨ kCheck R t edges k x = .blocked (some u) at hu
  change u.toList.take k <+: x.toList
  have hpre : ∀ v : FreeMonoid α, walkOutput R t edges k x = some v →
      v.toList.take k <+: x.toList := fun v hv => by
    rw [walkOutput_prefix R hv]; exact List.take_prefix _ _
  rcases hu with hu | hu
  · exact hpre u hu
  · unfold kCheck at hu
    rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps <;> rw [hw] at hu
    · exact hpre u (by simpa using hu)
    · exact hpre u (by simpa using hu)
    · simp only [] at hu
      rcases hs : t.sift R.cut x with a | b <;> rw [hs] at hu
      · simp only [] at hu
        split_ifs at hu
      · simp only [KCheck.blocked.injEq, Option.some.injEq] at hu
        subst hu
        obtain ⟨m, rfl⟩ := route_inr _ _ _ hs
        simp only [FreeMonoid.toList_mul]
        rw [List.take_append_of_le_length (by omega)]
        exact List.take_prefix _ _

end Sources

end OrthoDFA
