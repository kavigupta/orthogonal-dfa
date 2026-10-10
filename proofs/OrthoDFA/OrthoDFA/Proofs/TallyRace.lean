import OrthoDFA.Proofs.TallyBudget
import OrthoDFA.Proofs.Freedman

/-!
# The round's records that are not true

A probe's record is not true with chance at most `κ` times its undecided strings plus `ρ`
(`TallyE.spurious`). The round's tests hold those strings: the edges' over the round to `θe` of
their reads past `Xe` each, at every edge out of a node of the tree, and the start's to `θs'` of
the probes past `Xs`. So `exp (η u - ν (excess))`, with `u` the records that are not true and the
excess the undecided strings less those allowances, grows by at most `ν (θe L Lmax + θs') +
(e^η - 1) ρ` a probe in mean while the tree is in the class (`race_le`), which it is while those
records are below `(S + 1) m`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

/-- The paths to the tree's nodes, leaves included. -/
def nodes : DTree α → List (List Bool)
  | .leaf => [[]]
  | .node _ r a => [] :: (r.nodes.map (false :: ·) ++ a.nodes.map (true :: ·))

omit [Fintype α] [DecidableEq α] in
theorem paths_sub_nodes : ∀ (T : DTree α) (q : List Bool), q ∈ T.paths → q ∈ T.nodes
  | .leaf, q, h => by simpa [paths, nodes] using h
  | .node _ r a, q, h => by
    simp only [paths, nodes, List.mem_append, List.mem_map, List.mem_cons] at h ⊢
    rcases h with ⟨q', h, rfl⟩ | ⟨q', h, rfl⟩
    · exact .inr (.inl ⟨q', paths_sub_nodes r q' h, rfl⟩)
    · exact .inr (.inr ⟨q', paths_sub_nodes a q' h, rfl⟩)

omit [Fintype α] [DecidableEq α] in
theorem nodes_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p q : List Bool), q ∈ T.nodes → q ∈ (T.splitAt d p).nodes
  | .leaf, [], q, h => by simp_all [splitAt, nodes]
  | .leaf, _ :: _, q, h => by simpa [splitAt] using h
  | .node n r a, [], q, h => by simpa [splitAt] using h
  | .node n r a, false :: p, q, h => by
    simp only [splitAt, nodes, List.mem_cons, List.mem_append, List.mem_map] at h ⊢
    rcases h with rfl | ⟨q', h, rfl⟩ | ⟨q', h, rfl⟩
    · exact .inl rfl
    · exact .inr (.inl ⟨q', nodes_splitAt d r p q' h, rfl⟩)
    · exact .inr (.inr ⟨q', h, rfl⟩)
  | .node n r a, true :: p, q, h => by
    simp only [splitAt, nodes, List.mem_cons, List.mem_append, List.mem_map] at h ⊢
    rcases h with rfl | ⟨q', h, rfl⟩ | ⟨q', h, rfl⟩
    · exact .inl rfl
    · exact .inr (.inl ⟨q', h, rfl⟩)
    · exact .inr (.inr ⟨q', nodes_splitAt d a p q' h, rfl⟩)

omit [Fintype α] [DecidableEq α] in
theorem nodes_length : ∀ T : DTree α, T.nodes.length + 1 = 2 * T.paths.length
  | .leaf => by simp [nodes, paths]
  | .node _ r a => by
    have := nodes_length r
    have := nodes_length a
    simp only [nodes, paths, List.length_cons, List.length_append, List.length_map]
    omega

end DTree

section Charges

variable (cut : FreeMonoid α → Option Bool)

omit [Fintype α] [DecidableEq α] in
theorem sum_filter_le {ι κ : Type*} [DecidableEq κ] (f : ι → Option κ) (g : ι → ℕ)
    (K : Finset κ) : ∀ l : List ι,
    ∑ e ∈ K, ((l.filter fun i => f i = some e).map g).sum ≤ (l.map g).sum
  | [] => by simp
  | i :: l => by
    have ih := sum_filter_le f g K l
    have h1 : ∑ e ∈ K, (if f i = some e then g i else 0) ≤ g i := by
      rcases hf : f i with _ | v
      · simp
      · simp only [Option.some.injEq]
        rw [Finset.sum_ite_eq]
        split_ifs <;> omega
    have h2 : ∀ e, ((List.filter (fun i => decide (f i = some e)) (i :: l)).map g).sum
        = (if f i = some e then g i else 0)
          + ((l.filter fun i => decide (f i = some e)).map g).sum := by
      intro e
      by_cases h : f i = some e <;> simp [List.filter_cons, h]
    simp only [h2, Finset.sum_add_distrib, List.map_cons, List.sum_cons]
    omega

omit [Fintype α] [DecidableEq α] in
theorem sum_filterMap_le {ι κ β : Type*} [DecidableEq κ] (f : ι → Option κ) (h : ι → Option β)
    (K : Finset κ) : ∀ l : List ι,
    ∑ e ∈ K, (l.filterMap fun i => if f i = some e then h i else none).length ≤ l.length
  | [] => by simp
  | i :: l => by
    have ih := sum_filterMap_le f h K l
    have h1 : ∑ e ∈ K, (if f i = some e ∧ (h i).isSome then 1 else 0) ≤ 1 := by
      rcases hf : f i with _ | v
      · simp
      · simp only [Option.some.injEq]
        calc _ ≤ ∑ e ∈ K, (if v = e then 1 else 0) := Finset.sum_le_sum fun e _ => by
              split_ifs <;> simp_all
          _ ≤ 1 := by rw [Finset.sum_ite_eq]; split_ifs <;> omega
    have h2 : ∀ e, (List.filterMap (fun i => if f i = some e then h i else none) (i :: l)).length
        = (if f i = some e ∧ (h i).isSome then 1 else 0)
          + (l.filterMap fun i => if f i = some e then h i else none).length := by
      intro e
      by_cases hfe : f i = some e
      · rcases hh : h i with _ | b <;> simp [List.filterMap_cons, hfe, hh, add_comm]
      · simp [List.filterMap_cons, hfe]
    simp only [h2, Finset.sum_add_distrib, List.length_cons]
    omega

/-- A probe's undecided strings charged to the edges of `K` are at most its positions. -/
theorem sum_edgeHarvBy_le (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    (K : Finset (List Bool × α)) :
    ∑ e ∈ K, (edgeHarvBy cut T edges k x e).length ≤ x.toList.length :=
  (sum_filterMap_le _ _ K _).trans (siftsBy_length_le cut)

/-- A probe's reads charged to the edges of `K` are at most its positions times the leaves. -/
theorem sum_edgeReadsBy_le (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    (K : Finset (List Bool × α)) :
    ∑ e ∈ K, edgeReadsBy cut T edges k x e ≤ x.toList.length * T.paths.length := by
  refine (sum_filter_le (posEdgeBy cut T edges k x) _ K _).trans ?_
  have h1 : ∀ n ∈ (siftsBy cut T edges k x).map fun i => (T.route cut (prefixOf x i)).1.length,
      n ≤ T.paths.length := by
    intro n hn
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hn
    exact (route_length_le cut T _).trans (depth_lt_paths T).le
  refine (List.sum_le_card_nsmul _ _ h1).trans ?_
  simp only [List.length_map, smul_eq_mul]
  exact Nat.mul_le_mul_right _ (siftsBy_length_le cut)

theorem twinsBy_le (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    twinsBy cut T edges k x ≤ x.toList.length + 1 := by
  have h1 : (startHarvBy cut T k x).length ≤ 1 := by
    unfold startHarvBy; rcases T.sift cut (prefixOf x k) with _ | b <;> simp
  have h2 := sum_edgeHarvBy_le cut T edges k x (T.paths.toFinset ×ˢ Finset.univ)
  unfold twinsBy
  omega

/-- A probe charges only edges out of leaves. -/
theorem posEdgeBy_mem {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} {i : ℕ} {e : List Bool × α} (h : posEdgeBy cut T edges k x i = some e) :
    e.1 ∈ T.paths := by
  have hat : ∀ j, edgeAtBy cut T edges k x j = some e → e.1 ∈ T.paths := by
    intro j hj
    unfold edgeAtBy at hj
    rcases hl : (walkToBy cut T edges k x j).getLast? with _ | p <;>
      rcases hc : x.toList[j]? with _ | c <;> simp [hl, hc] at hj
    subst hj
    unfold walkToBy at hl
    rcases hs : T.sift cut (prefixOf x k) with p₀ | b <;> simp only [hs] at hl
    · rcases hf : follow edges p₀ ((x.toList.drop k).take (j - k)) with ps | _ <;>
        simp only [hf, Sum.elim_inl, Sum.elim_inr, id] at hl
      · exact follow_mem_paths he _ p₀ ps (DTree.sift_mem_paths _ _ _ hs) hf p
          (List.mem_of_getLast? hl)
      · simp at hl
    · simp at hl
  unfold posEdgeBy at h
  rcases ha : edgeAtBy cut T edges k x i with _ | e'
  · rw [ha] at h; exact hat _ (by simpa using h)
  · rw [ha] at h; simp only [Option.orElse_some, Option.some.injEq] at h; subst h; exact hat _ ha

theorem edgeHarvBy_mem {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} {e : List Bool × α} (h : edgeHarvBy cut T edges k x e ≠ []) :
    e.1 ∈ T.paths := by
  obtain ⟨b, hb⟩ := List.exists_mem_of_ne_nil _ h
  unfold edgeHarvBy at hb
  obtain ⟨i, -, hi⟩ := List.mem_filterMap.1 hb
  split_ifs at hi with hp
  exact posEdgeBy_mem cut he hp

theorem edgeReadsBy_mem {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} {e : List Bool × α} (h : edgeReadsBy cut T edges k x e ≠ 0) :
    e.1 ∈ T.paths := by
  unfold edgeReadsBy at h
  obtain ⟨n, hn, -⟩ := List.exists_mem_ne_zero_of_sum_ne_zero h
  obtain ⟨i, hi, -⟩ := List.mem_map.1 hn
  exact posEdgeBy_mem cut he (by simpa using (List.mem_filter.1 hi).2)

end Charges

/-- A step that goes on: no test fired, so the start's undecided strings are not past its test and
no edge out of a leaf has its undecided strings past `θe` of its reads by `exc` of them. -/
theorem tallyLook_none (C : TallyCfg) {s : TState α} (h : tallyLook C s = none) :
    rateSide C.θs C.a C.n₀ s.probes s.startH.length ≠ some true
      ∧ ∀ e : List Bool × α, e.1 ∈ s.tree.paths →
        ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2 < C.exc (s.reads e.1 e.2) := by
  unfold tallyLook at h
  split_ifs at h with h1 h2
  refine ⟨h1, fun e he => ?_⟩
  push_neg at h2
  exact h2 e he

/-- One probe's exponentiated increment: its record not true at `η`, its undecided strings at
`-ν`, averages at most `1 + (e^η - 1) ρ`. -/
theorem race_mgf {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (E : Set X) (W : X → ℕ) {κ ρ η ν : ℝ} {L : ℕ}
    (hη : 0 ≤ η) (hν : 0 ≤ ν) (hb : ∀ᵐ x ∂D, W x ≤ L + 1)
    (hside : (Real.exp η - 1) * κ * (L + 1) ≤ 1 - Real.exp (-(ν * (L + 1))))
    (hmean : D.real E ≤ κ * ∫ x, (W x : ℝ) ∂D + ρ) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (η * E.indicator 1 x - ν * W x)) ∂D
      ≤ ENNReal.ofReal (1 + (Real.exp η - 1) * ρ) := by
  classical
  set a := Real.exp η - 1
  have ha : 0 ≤ a := by simp only [a]; linarith [Real.one_le_exp hη]
  have hL1 : (0 : ℝ) < L + 1 := by positivity
  set b := (1 - Real.exp (-(ν * (L + 1)))) / (L + 1)
  have hab : a * κ ≤ b := by
    simp only [b]; rw [le_div_iff₀ hL1]; linarith
  have hWi : Integrable (fun x => (W x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (L + 1) (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact_mod_cast hx)
  have hEi : Integrable (fun x => E.indicator (fun _ => (1 : ℝ)) x) D :=
    (integrable_const 1).indicator (Set.to_countable _).measurableSet
  set g : X → ℝ := fun x => 1 + a * E.indicator (fun _ => (1 : ℝ)) x - b * W x
  have hgi : Integrable g D := ((integrable_const 1).add (hEi.const_mul a)).sub (hWi.const_mul b)
  have hpt : ∀ᵐ x ∂D, Real.exp (η * E.indicator 1 x - ν * W x) ≤ g x := by
    filter_upwards [hb] with x hx
    have hW0 : (0 : ℝ) ≤ W x := Nat.cast_nonneg _
    have hWL : (W x : ℝ) ≤ L + 1 := by exact_mod_cast hx
    have hch : Real.exp (-ν * W x) ≤ 1 + W x * ((Real.exp (-ν * (L + 1)) - 1) / (L + 1)) :=
      exp_mul_le_chord hL1 hW0 hWL
    have hch' : Real.exp (-(ν * W x)) ≤ 1 - b * W x := by
      have : -(ν * W x) = -ν * W x := by ring
      rw [this]
      refine hch.trans (le_of_eq ?_)
      simp only [b]; rw [show -ν * (L + 1) = -(ν * (L + 1)) by ring]; field_simp; ring
    have hI : E.indicator (1 : X → ℝ) x = E.indicator (fun _ => (1 : ℝ)) x := rfl
    by_cases hx' : x ∈ E
    · rw [hI, Set.indicator_of_mem hx'] at *
      simp only [g, Set.indicator_of_mem hx', mul_one]
      rw [sub_eq_add_neg, Real.exp_add]
      have hb0 : 0 ≤ b * W x := by
        have : 0 ≤ b := by
          simp only [b]
          have : Real.exp (-(ν * (L + 1))) ≤ 1 := Real.exp_le_one_iff.2 (by
            have : 0 ≤ ν * (L + 1) := by positivity
            linarith)
          positivity
        positivity
      have he : Real.exp η = 1 + a := by simp only [a]; ring
      rw [he]
      nlinarith [Real.exp_pos (-(ν * W x))]
    · simp only [hI, g, Set.indicator_of_notMem hx', mul_zero, zero_sub, add_zero]
      exact hch'
  calc ∫⁻ x, ENNReal.ofReal (Real.exp (η * E.indicator 1 x - ν * W x)) ∂D
      ≤ ∫⁻ x, ENNReal.ofReal (g x) ∂D := lintegral_mono_ae (by
        filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx)
    _ = ENNReal.ofReal (∫ x, g x ∂D) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        filter_upwards [hpt] with x hx using (Real.exp_pos _).le.trans hx
    _ ≤ _ := by
        refine ENNReal.ofReal_le_ofReal ?_
        have hint : ∫ x, g x ∂D = 1 + a * D.real E - b * ∫ x, (W x : ℝ) ∂D := by
          have e1 := integral_sub (μ := D)
            (f := fun x => 1 + a * E.indicator (fun _ => (1 : ℝ)) x)
            (g := fun x => b * (W x : ℝ)) ((integrable_const 1).add (hEi.const_mul a))
            (hWi.const_mul b)
          have e2 := integral_add (μ := D) (f := fun _ => (1 : ℝ))
            (g := fun x => a * E.indicator (fun _ => (1 : ℝ)) x) (integrable_const 1)
            (hEi.const_mul a)
          simp only [g]
          rw [e1, e2, integral_const_mul, integral_const_mul, integral_const,
            integral_indicator_const _ (Set.to_countable E).measurableSet]
          simp
        have hW0 : 0 ≤ ∫ x, (W x : ℝ) ∂D := integral_nonneg fun _ => Nat.cast_nonneg _
        rw [hint]
        nlinarith [mul_le_mul_of_nonneg_left hmean ha, mul_le_mul_of_nonneg_right hab hW0]

section Race

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg) (rd : FreeMonoid α → ARU)
  (Xe θs' Xs : ℝ)

/-- What the round keeps along its run, having made `u` records that are not true: its splits that
are not genuine, `m` each, and any edge and target's records that are not true within `u`; edges
and records pointing at leaves; undecided strings and reads only at edges out of its nodes, those
past `θe` of the reads by at most `Xe`; and the start's past `θs'` of the probes by at most
`Xs`. -/
def RInv (s : TState α) (u : ℕ) : Prop :=
  (∃ f, G.Grown s.tree f ∧ C.m * f ≤ u ∧ ∀ pct, C.m * f + badRecs G s pct ≤ u)
    ∧ EdgesInto s.tree s.edges ∧ RecsInto s
    ∧ (∀ q c, (s.harv q c ≠ [] ∨ s.reads q c ≠ 0) → q ∈ s.tree.nodes)
    ∧ (∀ q c, ((s.harv q c).length : ℝ) - C.θe * s.reads q c ≤ Xe)
    ∧ ((s.startH.length : ℝ) ≤ θs' * s.probes + Xs) ∧ s.startH.length ≤ s.probes

/-- The round's undecided strings past their allowances: at the edges out of its nodes, `θe` of
their reads, and at the start, `θs'` of the probes. -/
noncomputable def excess (s : TState α) : ℝ :=
  ∑ e ∈ s.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α),
      (((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2)
    + ((s.startH.length : ℝ) - θs' * s.probes)

theorem excess_le {s : TState α} {u : ℕ} (hs : RInv G C Xe θs' Xs s u) (hXe : 0 ≤ Xe)
    (hp : s.tree.paths.length ≤ C.Lmax) :
    excess C θs' s ≤ 2 * C.Lmax * Fintype.card α * Xe + Xs := by
  obtain ⟨-, -, -, -, hex, hst, -⟩ := hs
  have hcard : ((s.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α)).card : ℝ)
      ≤ 2 * C.Lmax * Fintype.card α := by
    rw [Finset.card_product, Finset.card_univ]
    have h1 := List.toFinset_card_le s.tree.nodes
    have h2 := DTree.nodes_length s.tree
    have : s.tree.nodes.toFinset.card ≤ 2 * C.Lmax := by omega
    calc ((s.tree.nodes.toFinset.card * Fintype.card α : ℕ) : ℝ)
        ≤ ((2 * C.Lmax * Fintype.card α : ℕ) : ℝ) := by
          exact_mod_cast Nat.mul_le_mul_right _ this
      _ = _ := by push_cast; ring
  have hsum := Finset.sum_le_card_nsmul (s.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α))
    (fun e => ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2) Xe
    (fun e _ => hex e.1 e.2)
  rw [nsmul_eq_mul] at hsum
  unfold excess
  nlinarith

open scoped Classical in
theorem rinv_step (hm : 0 < C.m) (hθs' : 0 ≤ θs') (hXe : ∀ j, C.exc j ≤ Xe) (hXs : C.n₀ ≤ Xs)
    (hlin : ∀ n h : ℕ, C.n₀ ≤ n → θs' * n + Xs ≤ h → binomSfGe n C.θs h < C.a)
    {s s' : TState α} {x : FreeMonoid α} {u : ℕ} (hs : RInv G C Xe θs' Xs s u)
    (h : tallyStep C (fun z => (rd z).cut) s x = .inl s') :
    RInv G C Xe θs' Xs s' (u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0)
      ∧ (∀ q, q ∈ s.tree.nodes → q ∈ s'.tree.nodes) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  obtain ⟨⟨f, hf, hmf, hbad⟩, he, hr, hsup, hex, hst, hstp⟩ := hs
  have htr₁ : (tallyPre C cut s x).tree = s.tree := (tallyPre_cases cut C s x).1
  have hl : tallyLook C (tallyPre C cut s x) = none := by
    rcases tallyStep_cases cut C h with ⟨e, -, he'⟩ | ⟨hl, -⟩
    · cases he'
    · exact hl
  obtain ⟨hstart, hedge⟩ := tallyLook_none C hl
  obtain ⟨c1, c2, c3, c4⟩ := tallyStep_counts C cut h
  obtain ⟨p1, p2, p3, p4⟩ := tallyPre_charge C cut s x
  have hU : ∀ pct, x ∈ untrueAt G C cut s pct → x ∈ G.untrueAt rd C.k s.tree s.edges :=
    fun pct ⟨sp, h, hn⟩ => ⟨pct, sp, h, hn⟩
  have hind : ∀ pct, (if x ∈ untrueAt G C cut s pct then 1 else 0)
      ≤ (if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0) := by
    intro pct
    split_ifs with h₁ h₂
    · exact le_rfl
    · exact absurd (hU pct h₁) h₂
    · exact Nat.zero_le _
    · exact le_rfl
  have hspec := (tallyStep_spec cut C hm he hr).1 s' h
  have hnodes : ∀ q, q ∈ s.tree.nodes → q ∈ s'.tree.nodes := by
    rcases hspec with ⟨ht, -, -⟩ | ⟨p, c, t, t₀, -, -, hp, hT, -⟩
    · intro q hq; rwa [ht]
    · intro q hq; rw [hT]; exact DTree.nodes_splitAt _ _ _ _ hq
  have hzero : ∀ q c, q ∉ s.tree.paths →
      edgeHarvBy cut s.tree s.edges C.k x (q, c) = [] ∧
        edgeReadsBy cut s.tree s.edges C.k x (q, c) = 0 := by
    intro q c hq
    exact ⟨by_contra fun hne => hq (edgeHarvBy_mem cut he hne),
      by_contra fun hne => hq (edgeReadsBy_mem cut he hne)⟩
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hnodes⟩
  · rcases hspec with ⟨ht, -, -⟩ | ⟨p, c, t, t₀, ht, ht₀, hp, hT, -, hrec, -⟩
    · refine ⟨f, ht ▸ hf, by omega, fun pct => ?_⟩
      have h1 := badRecs_pre G C cut s x pct
      have h2 : badRecs G s' pct = badRecs G (tallyPre C cut s x) pct := by
        unfold badRecs; rw [tallyStep_recs C cut h ht, ht, htr₁]
      have := hbad pct
      have := hind pct
      omega
    · have h0 : ∀ pct, badRecs G s' pct = 0 := fun pct => by simp [badRecs, hrec]
      have htr' : s'.tree ≠ s.tree := fun h' => DTree.splitAt_ne hp (hT.symm.trans h')
      obtain ⟨p, c, t, t₀, hp, ht, ht₀, hne, hm₁, hm₂, hT⟩ := tallyStep_split C cut hm he hr h htr'
      by_cases hgen : G.GenuineSplit s.tree p (FreeMonoid.of c * s.tree.midAt (lcp t t₀))
      · refine ⟨f, hT ▸ .real hf ht ht₀ hgen, by omega, fun pct => ?_⟩
        rw [h0]
        omega
      · have hngP : ¬ G.GenuineSplit (tallyPre C cut s x).tree p
            (FreeMonoid.of c * (tallyPre C cut s x).tree.midAt (lcp t t₀)) := by rw [htr₁]; exact hgen
        have key : ∀ pct', C.m ≤ badRecs G (tallyPre C cut s x) pct' →
            C.m * (f + 1) ≤ u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0 := by
          intro pct' hb
          have h1 := badRecs_pre G C cut s x pct'
          have := hbad pct'
          have := hind pct'
          rw [Nat.mul_succ]
          omega
        have hf1 : C.m * (f + 1) ≤ u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0 := by
          rcases badRecs_of_fake G C hne hm₁ hm₂ hngP with hb | hb
          · exact key _ hb
          · exact key _ hb
        refine ⟨f + 1, hT ▸ .fake hf hp ht ht₀ hgen, hf1, fun pct => ?_⟩
        rw [h0]
        omega
  · rcases hspec with ⟨-, he', -⟩ | ⟨-, -, -, -, -, -, -, -, he', -⟩ <;> exact he'
  · rcases hspec with ⟨-, -, hr'⟩ | ⟨-, -, -, -, -, -, -, -, -, hrec, -⟩
    · exact hr'
    · intro q e r hmem; rw [hrec q e] at hmem; cases hmem
  · intro q c hqc
    rw [c1, c2, p1, p2] at hqc
    simp only at hqc
    by_cases hold : s.harv q c ≠ [] ∨ s.reads q c ≠ 0
    · exact hnodes q (hsup q c hold)
    · push_neg at hold
      have hq : q ∈ s.tree.paths := by
        by_contra hq
        obtain ⟨h1, h2⟩ := hzero q c hq
        simp [hold.1, hold.2, h1, h2] at hqc
      exact hnodes q (DTree.paths_sub_nodes _ _ hq)
  · intro q c
    rw [c1, c2]
    by_cases hq : q ∈ s.tree.paths
    · have := hedge (q, c) (by rw [htr₁]; exact hq)
      exact this.le.trans (hXe _)
    · rw [p1, p2]
      obtain ⟨h1, h2⟩ := hzero q c hq
      simp only [h1, h2, List.append_nil, add_zero]
      exact hex q c
  · rw [c3, c4]
    by_contra hc
    push_neg at hc
    have hle1 : (startHarvBy cut s.tree C.k x).length ≤ 1 := by
      unfold startHarvBy; rcases s.tree.sift cut (prefixOf x C.k) with _ | b <;> simp
    rw [p3, p4] at hc hstart
    by_cases hn : C.n₀ ≤ s.probes + 1
    · have hh : θs' * ((s.probes + 1 : ℕ) : ℝ) + Xs
          ≤ ((s.startH ++ startHarvBy cut s.tree C.k x).length : ℕ) := by
        push_cast at hc ⊢; linarith
      have := hlin _ _ hn hh
      apply hstart
      unfold rateSide
      rw [if_pos hn, if_pos this]
    · have : ((s.startH ++ startHarvBy cut s.tree C.k x).length : ℝ) ≤ Xs := by
        rw [List.length_append]
        have : s.startH.length + (startHarvBy cut s.tree C.k x).length < C.n₀ := by omega
        have : ((s.startH.length + (startHarvBy cut s.tree C.k x).length : ℕ) : ℝ) < C.n₀ := by
          exact_mod_cast this
        push_cast at this ⊢
        linarith
      have : (0 : ℝ) ≤ θs' * ((s.probes + 1 : ℕ) : ℝ) := by positivity
      push_cast at hc this
      linarith
  · rw [c3, c4, p3, p4, List.length_append]
    have : (startHarvBy cut s.tree C.k x).length ≤ 1 := by
      unfold startHarvBy; rcases s.tree.sift cut (prefixOf x C.k) with _ | b <;> simp
    omega

theorem excess_step (hθe : 0 ≤ C.θe) {s s' : TState α} {x : FreeMonoid α} {u : ℕ}
    (hs : RInv G C Xe θs' Xs s u) (h : tallyStep C (fun z => (rd z).cut) s x = .inl s')
    (hnodes : ∀ q, q ∈ s.tree.nodes → q ∈ s'.tree.nodes) :
    excess C θs' s + twinsBy (fun z => (rd z).cut) s.tree s.edges C.k x
        - C.θe * (x.toList.length * s.tree.paths.length) - θs'
      ≤ excess C θs' s' := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  obtain ⟨-, he, -, hsup, -, -, -⟩ := hs
  obtain ⟨c1, c2, c3, c4⟩ := tallyStep_counts C cut h
  obtain ⟨p1, p2, p3, p4⟩ := tallyPre_charge C cut s x
  set K := s.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α)
  set K' := s'.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α)
  set Lv := s.tree.paths.toFinset ×ˢ (Finset.univ : Finset α)
  have hKK' : K ⊆ K' := by
    intro e he'
    simp only [K, K', Finset.mem_product, List.mem_toFinset, Finset.mem_univ, and_true] at he' ⊢
    exact hnodes _ he'
  have hLK : Lv ⊆ K := by
    intro e he'
    simp only [K, Lv, Finset.mem_product, List.mem_toFinset, Finset.mem_univ, and_true] at he' ⊢
    exact DTree.paths_sub_nodes _ _ he'
  set F' : List Bool × α → ℝ := fun e =>
    ((s'.harv e.1 e.2).length : ℝ) - C.θe * s'.reads e.1 e.2
  have hF' : ∀ e, F' e = (((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2)
      + ((edgeHarvBy cut s.tree s.edges C.k x e).length
        - C.θe * edgeReadsBy cut s.tree s.edges C.k x e) := by
    intro e
    simp only [F', c1, c2, p1, p2, List.length_append]
    push_cast; ring
  have hout : ∀ e ∈ K', e ∉ K → F' e = 0 := by
    intro e _ heK
    have hq : e.1 ∉ s.tree.nodes := by
      simpa [K] using heK
    have hold : s.harv e.1 e.2 = [] ∧ s.reads e.1 e.2 = 0 := by
      by_contra hc
      exact hq (hsup e.1 e.2 (by tauto))
    have hp : e.1 ∉ s.tree.paths := fun hp => hq (DTree.paths_sub_nodes _ _ hp)
    have h1 : edgeHarvBy cut s.tree s.edges C.k x e = [] :=
      by_contra fun hne => hp (edgeHarvBy_mem cut he hne)
    have h2 : edgeReadsBy cut s.tree s.edges C.k x e = 0 :=
      by_contra fun hne => hp (edgeReadsBy_mem cut he hne)
    rw [hF', hold.1, hold.2, h1, h2]
    simp
  have hsum : ∑ e ∈ K', F' e = ∑ e ∈ K, F' e :=
    (Finset.sum_subset hKK' hout).symm
  have hH : (∑ e ∈ Lv, ((edgeHarvBy cut s.tree s.edges C.k x e).length : ℝ))
      ≤ ∑ e ∈ K, ((edgeHarvBy cut s.tree s.edges C.k x e).length : ℝ) :=
    Finset.sum_le_sum_of_subset_of_nonneg hLK fun _ _ _ => Nat.cast_nonneg _
  have hR : (∑ e ∈ K, (edgeReadsBy cut s.tree s.edges C.k x e : ℝ))
      ≤ x.toList.length * s.tree.paths.length := by
    exact_mod_cast sum_edgeReadsBy_le cut s.tree s.edges C.k x K
  have htw : (twinsBy cut s.tree s.edges C.k x : ℝ)
      = (startHarvBy cut s.tree C.k x).length
        + ∑ e ∈ Lv, ((edgeHarvBy cut s.tree s.edges C.k x e).length : ℝ) := by
    simp only [twinsBy, Lv]; push_cast; ring
  unfold excess
  rw [show (∑ e ∈ s'.tree.nodes.toFinset ×ˢ (Finset.univ : Finset α),
      (((s'.harv e.1 e.2).length : ℝ) - C.θe * s'.reads e.1 e.2)) = ∑ e ∈ K', F' e from rfl,
    hsum]
  simp only [hF', Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [c3, c4, p3, p4, List.length_append, htw]
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left hR hθe]

end Race

section RaceLe

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (D : Measure (FreeMonoid α))
  [IsProbabilityMeasure D] (C : TallyCfg) (rd : FreeMonoid α → ARU) (S L : ℕ)
  (ρ θg θgs θgpt Xe θs' Xs η ν : ℝ)

open scoped Classical in
/-- From a state of the run with `u` records that are not true made, `n` more within `j` probes
have chance at most `exp (-η n + ν (allowance - excess) + j c)`. -/
theorem race_le (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hm : 0 < C.m)
    (hLmax : Fintype.card σ + S + 3 ≤ C.Lmax) (hθe : 0 ≤ C.θe) (hρ0 : 0 ≤ ρ) (hXe0 : 0 ≤ Xe)
    (hXe : ∀ j, C.exc j ≤ Xe) (hθs' : 0 ≤ θs') (hXs : C.n₀ ≤ Xs)
    (hlin : ∀ n h : ℕ, C.n₀ ≤ n → θs' * n + Xs ≤ h → binomSfGe n C.θs h < C.a)
    (hη : 0 ≤ η) (hν : 0 ≤ ν)
    (hside : (Real.exp η - 1) * G.κ * (L + 1) ≤ 1 - Real.exp (-(ν * (L + 1))))
    (hE : TallyE G D C S ρ θg θgs θgpt rd) :
    ∀ (T : ℕ) (s : TState α) (u n j : ℕ), RInv G C Xe θs' Xs s u → u + n = (S + 1) * C.m →
      (Measure.pi fun _ : Fin T => D) {xs | UntrueHit G C rd s n j T xs}
        ≤ ENNReal.ofReal (Real.exp (-η * n
          + ν * (2 * C.Lmax * Fintype.card α * Xe + Xs - excess C θs' s)
          + j * (ν * (C.θe * (L * C.Lmax) + θs') + (Real.exp η - 1) * ρ))) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  set X := 2 * C.Lmax * Fintype.card α * Xe + Xs
  set c := ν * (C.θe * (L * C.Lmax) + θs') + (Real.exp η - 1) * ρ
  have ha : 0 ≤ Real.exp η - 1 := by linarith [Real.one_le_exp hη]
  have hc : 0 ≤ c := by positivity
  have hpaths : ∀ (s : TState α) (u : ℕ), RInv G C Xe θs' Xs s u → u ≤ (S + 1) * C.m →
      s.tree.paths.length ≤ C.Lmax := by
    intro s u hs hu
    obtain ⟨f, hf, hmf, -⟩ := hs.1
    have hfS : C.m * f ≤ C.m * (S + 1) := by rw [mul_comm C.m (S + 1)]; omega
    have := Nat.le_of_mul_le_mul_left hfS hm
    have := G.grown_paths hf
    omega
  have hone : ∀ (s : TState α) (u j : ℕ), RInv G C Xe θs' Xs s u → u ≤ (S + 1) * C.m →
      (1 : ENNReal) ≤ ENNReal.ofReal (Real.exp (-η * (0 : ℕ) + ν * (X - excess C θs' s)
        + j * c)) := by
    intro s u j hs hu
    have hX := excess_le G C Xe θs' Xs hs hXe0 (hpaths s u hs hu)
    rw [← ENNReal.ofReal_one]
    refine ENNReal.ofReal_le_ofReal (Real.one_le_exp ?_)
    have : 0 ≤ ν * (X - excess C θs' s) := mul_nonneg hν (by simp only [X]; linarith)
    have : 0 ≤ (j : ℝ) * c := by positivity
    push_cast
    linarith
  intro T
  induction T with
  | zero =>
    intro s u n j hs hu
    rcases n with _ | n
    · exact prob_le_one.trans (hone s u j hs (by omega))
    · have : {xs : Fin 0 → FreeMonoid α | UntrueHit G C rd s (n + 1) j 0 xs} = ∅ := by
        ext xs; rcases j with _ | j <;> simp [UntrueHit]
      rw [this, measure_empty]; exact zero_le
  | succ T ih =>
    intro s u n j hs hu
    rcases n with _ | n
    · exact prob_le_one.trans (hone s u j hs (by omega))
    rcases j with _ | j
    · have : {xs : Fin (T + 1) → FreeMonoid α | UntrueHit G C rd s (n + 1) 0 (T + 1) xs} = ∅ := by
        ext xs; simp [UntrueHit]
      rw [this, measure_empty]; exact zero_le
    obtain ⟨f, hf, hmf, -⟩ := hs.1
    have he := hs.2.1
    have hclass : G.InClass S s.tree := by
      refine ⟨f, ?_, hf⟩
      have hfS : C.m * f < C.m * (S + 1) := by rw [mul_comm C.m (S + 1)]; omega
      have := Nat.lt_of_mul_lt_mul_left hfS
      omega
    have hp := hpaths s u hs (by omega)
    set U := G.untrueAt rd C.k s.tree s.edges
    set W := twinsBy cut s.tree s.edges C.k
    set base := -η * ((n + 1 : ℕ) : ℝ) + ν * (X - excess C θs' s) + j * c
      + ν * (C.θe * (L * C.Lmax) + θs')
    rw [pi_succ_apply]
    have hsec : ∀ᵐ x ∂D, (Measure.pi fun _ : Fin T => D)
        {xs | UntrueHit G C rd s (n + 1) (j + 1) (T + 1) (Fin.cons x xs)}
        ≤ ENNReal.ofReal (Real.exp base)
          * ENNReal.ofReal (Real.exp (η * U.indicator 1 x - ν * W x)) := by
      filter_upwards [hlen] with x hx
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      rcases hst : tallyStep C cut s x with s' | ⟨e, s'⟩
      · have hset : {xs | UntrueHit G C rd s (n + 1) (j + 1) (T + 1) (Fin.cons x xs)}
            = {xs | UntrueHit G C rd s' (if x ∈ U then n else n + 1) j T xs} := by
          ext xs
          simp only [Set.mem_ofPred_eq, UntrueHit, Fin.cons_zero, Fin.tail_cons]
          constructor
          · rintro ⟨s'', h1, h2⟩
            rw [hst] at h1; cases h1; exact h2
          · exact fun h2 => ⟨s', hst, h2⟩
        rw [hset]
        obtain ⟨hs', hnodes⟩ := rinv_step G C rd Xe θs' Xs hm hθs' hXe hXs hlin hs hst
        have hex := excess_step G C rd Xe θs' Xs hθe hs hst hnodes
        refine (ih s' _ _ j hs' (by split_ifs <;> omega)).trans (ENNReal.ofReal_le_ofReal ?_)
        refine Real.exp_le_exp.2 ?_
        have hxL : (x.toList.length : ℝ) * s.tree.paths.length ≤ L * C.Lmax := by
          have h1 : (x.toList.length : ℝ) ≤ L := by exact_mod_cast hx
          have h2 : (s.tree.paths.length : ℝ) ≤ C.Lmax := by exact_mod_cast hp
          exact mul_le_mul h1 h2 (Nat.cast_nonneg _) (Nat.cast_nonneg _)
        have hθx := mul_le_mul_of_nonneg_left hxL hθe
        simp only [base, Set.indicator_apply, Pi.one_apply]
        split_ifs with hU'
        · push_cast; nlinarith [mul_le_mul_of_nonneg_left hex hν]
        · push_cast; nlinarith [mul_le_mul_of_nonneg_left hex hν]
      · have hset : {xs | UntrueHit G C rd s (n + 1) (j + 1) (T + 1) (Fin.cons x xs)}
            = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, UntrueHit, Fin.cons_zero, Set.mem_empty_iff_false,
            iff_false, not_exists, not_and]
          intro s'' h1; rw [hst] at h1; cases h1
        rw [hset, measure_empty]; exact zero_le
    refine (lintegral_mono_ae hsec).trans ?_
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    have hmgf := race_mgf D U W hη hν (by
      filter_upwards [hlen] with x hx
      exact (twinsBy_le cut s.tree s.edges C.k x).trans (by omega)) hside
      (hE.spurious s.tree s.edges hclass he)
    refine (mul_le_mul_of_nonneg_left hmgf zero_le).trans ?_
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le]
    refine ENNReal.ofReal_le_ofReal ?_
    have h1 : 1 + (Real.exp η - 1) * ρ ≤ Real.exp ((Real.exp η - 1) * ρ) := by
      linarith [Real.add_one_le_exp ((Real.exp η - 1) * ρ)]
    calc Real.exp base * (1 + (Real.exp η - 1) * ρ)
        ≤ Real.exp base * Real.exp ((Real.exp η - 1) * ρ) :=
          mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
      _ = _ := by
          rw [← Real.exp_add]
          congr 1
          simp only [base, c]
          push_cast
          ring

end RaceLe

theorem fake_race_holds : FakeRace := by
  intro α _ _ σ _ G rd D _ C S L W ρ θg θgs θgpt Xe θs' Xs η ν hlen hm hLmax hθe hρ0 hXe0 hXe hθs'
    hXs hlin hη hν hside hE T
  have h0 : RInv G C Xe θs' Xs (tallyStart : TState α) 0 := by
    refine ⟨⟨0, .start, by simp, fun pct => by simp [badRecs, tallyStart]⟩,
      fun _ _ _ _ h => by simp [tallyStart] at h, fun _ _ _ h => by simp [tallyStart] at h,
      fun q c h => by simp [tallyStart] at h, fun q c => by simp [tallyStart, hXe0], ?_,
      by simp [tallyStart]⟩
    have : (0 : ℝ) ≤ Xs := le_trans (Nat.cast_nonneg _) hXs
    simp [tallyStart, this]
  have hex0 : excess C θs' (tallyStart : TState α) = 0 := by simp [excess, tallyStart]
  refine (race_le G D C rd S L ρ θg θgs θgpt Xe θs' Xs η ν hlen hm hLmax hθe hρ0 hXe0 hXe hθs'
    hXs hlin hη hν hside hE T tallyStart 0 ((S + 1) * C.m) W h0 (by omega)).trans
    (le_of_eq ?_)
  rw [hex0]
  unfold fakeRun
  congr 2
  push_cast
  ring

end OrthoDFA
