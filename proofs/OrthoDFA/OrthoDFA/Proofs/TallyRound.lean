import OrthoDFA.Proofs.TallyInv

/-!
# One round of the tally loop

Fix the reads and suppose `TallyE`. While fewer than `m` probes have made records that are not
true, every state is genuine, so `TallyE` applies to it: a state with a path-good transition whose
boundary has mass `q₀` gets clean records at rate `c₀q₀` at an edge not pointing at their target,
so its hypothesis changes within `n` probes unless `tally_fix_in_time`'s event happens; every
change lowers `pot`, so within `T` probes the round reaches a state with no such transition, or
ends. It ends only in success or a harvest, since genuine trees stay within `Lmax` leaves and
`pot` within `fuel`; and a harvest its stretch's hypothesis does not call for is a stretch's test
misfiring. Over the reads, `TallyE` failing adds its chance.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

variable (G : ReadModel α σ) (C : TallyCfg) (rd : FreeMonoid α → ARU)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {q₀ c₀ ρ θg θgs : ℝ}

theorem recordBy_mem {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} {p : List Bool} {c : α} {t : List Bool} {sp : FreeMonoid α}
    (h : recordBy (rdCut rd) k (T, edges) x = some ((p, c, t), sp)) :
    p ∈ T.paths ∧ t ∈ T.paths := by
  obtain ⟨p₀, ps, hk, hf, hp, ht, -⟩ := recordBy_spec (rdCut rd) h
  exact ⟨follow_mem_paths he _ _ _ (DTree.sift_mem_paths _ _ _ hk) hf p hp,
    DTree.sift_mem_paths _ _ _ ht⟩

/-- Records that are not true are rare: at most `ρ` at each edge and target, over at most
`(|Q| + 2)² |Σ|` of them. -/
theorem spurSet_le (hρ : 0 ≤ ρ) (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd) (s : TState α) :
    D.real (spurSet G C rd s) ≤ (Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ := by
  classical
  unfold spurSet
  split_ifs with h
  swap
  · simp only [measureReal_empty]; positivity
  obtain ⟨hg, he⟩ := h
  set P := s.tree.paths.toFinset
  have hsub : {x | ∃ pct sp, recordBy (rdCut rd) C.k (s.tree, s.edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec s.tree pct sp}
      ⊆ ⋃ pct ∈ P ×ˢ (Finset.univ : Finset α) ×ˢ P,
        {x | ∃ sp, recordBy (rdCut rd) C.k (s.tree, s.edges) x = some (pct, sp)
          ∧ ¬ G.TrueRec s.tree pct sp} := by
    rintro x ⟨⟨p, c, t⟩, sp, hr, hn⟩
    obtain ⟨hp, ht⟩ := recordBy_mem rd he hr
    simp only [Set.mem_iUnion, Finset.mem_product, Finset.mem_univ, true_and, exists_prop]
    exact ⟨(p, c, t), ⟨List.mem_toFinset.2 hp, List.mem_toFinset.2 ht⟩, sp, hr, hn⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans ?_)
  refine (Finset.sum_le_sum fun pct _ => hE.spurious s.tree s.edges hg he pct).trans ?_
  rw [Finset.sum_const, nsmul_eq_mul, Finset.card_product, Finset.card_product,
    Finset.card_univ]
  have hP : P.card ≤ Fintype.card σ + 2 :=
    (List.toFinset_card_le _).trans (G.genuine_paths hg)
  have : ((P.card * (Fintype.card α * P.card) : ℕ) : ℝ)
      ≤ (Fintype.card σ + 2) ^ 2 * Fintype.card α := by
    have : P.card * (Fintype.card α * P.card) ≤ (Fintype.card σ + 2) ^ 2 * Fintype.card α :=
      calc P.card * (Fintype.card α * P.card) = P.card ^ 2 * Fintype.card α := by ring
        _ ≤ _ := Nat.mul_le_mul_right _ (Nat.pow_le_pow_left hP 2)
    exact_mod_cast this
  exact mul_le_mul_of_nonneg_right this hρ

/-- An unfixed path-good transition with a heavy boundary makes the hypothesis significant. -/
theorem sig_of_not_fixed (hq₀ : 0 < q₀) (hcq : 0 < c₀ * q₀)
    (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd) {s : TState α} (hg : G.Genuine s.tree)
    (he : EdgesInto s.tree s.edges) (hnf : ¬ AllFixed G D C.k q₀ s) :
    Significant C (rdCut rd) D (c₀ * q₀) (tallyKey s) := by
  simp only [AllFixed, not_forall, not_lt] at hnf
  obtain ⟨q, c, hq, hqc, hb⟩ := hnf
  have hc₀ : 0 < c₀ := pos_of_mul_pos_left hcq hq₀.le
  have hclean := hE.clean s.tree s.edges hg he q c hq hqc hb
  have hrate : c₀ * q₀ ≤ D.real {x | (recordBy (rdCut rd) C.k (s.tree, s.edges) x).map Prod.fst
      = some (G.leafOf s.tree q, c, G.leafOf s.tree (G.M.step q c))} :=
    (mul_le_mul_of_nonneg_left hb hc₀.le).trans hclean
  refine ⟨(G.leafOf s.tree q, c, G.leafOf s.tree (G.M.step q c)), G.leafOf_mem_paths _ _, ?_,
    hrate⟩
  have hne : {x | (recordBy (rdCut rd) C.k (s.tree, s.edges) x).map Prod.fst
      = some (G.leafOf s.tree q, c, G.leafOf s.tree (G.M.step q c))}.Nonempty := by
    by_contra h
    rw [Set.not_nonempty_iff_eq_empty] at h
    rw [h, measureReal_empty] at hrate
    linarith
  obtain ⟨x, hx⟩ := hne
  simp only [Set.mem_ofPred_eq, Option.map_eq_some_iff] at hx
  obtain ⟨⟨pct, sp⟩, hr, rfl⟩ := hx
  obtain ⟨-, -, -, -, -, -, hne'⟩ := recordBy_spec (rdCut rd) hr
  exact hne'

theorem filter_split (l : List ℕ) (a b c d : ℕ → Bool) (h : ∀ i, b i = (c i || d i))
    (hx : ∀ i, ¬ (c i = true ∧ d i = true)) :
    (l.filter fun i => a i && b i).length
      = (l.filter fun i => a i && c i).length + (l.filter fun i => a i && d i).length := by
  induction l with
  | nil => simp
  | cons i l ih =>
    have hcd := hx i
    simp only [List.filter_cons, h i]
    cases hai : a i <;> cases hci : c i <;> cases hdi : d i <;>
      simp only [hai, hci, hdi, Bool.and_false, Bool.and_true, Bool.false_and, Bool.or_false,
        Bool.or_true, Bool.false_or, if_true, Bool.false_eq_true, if_false, List.length_cons,
        and_self, not_true_eq_false] at hcd ⊢ <;> omega

open scoped Classical in
theorem undec_split {T : DTree α} {edges : Edges α} {k : ℕ} (x : FreeMonoid α)
    (e : List Bool × α) :
    edgeUndecBy (rdCut rd) T edges k x e (fun _ => True)
      = G.undecAt rd k T edges e G.Good x + G.undecAt rd k T edges e (fun r => ¬ G.Good r) x := by
  unfold ReadModel.undecAt edgeUndecBy
  apply filter_split
  · intro i
    rcases (T.sift (rdCut rd) (prefixOf x i)) with p | b
    · simp
    · by_cases hb : G.Good (G.M.eval b.toList) <;> simp [hb]
  · intro i
    rcases (T.sift (rdCut rd) (prefixOf x i)) with p | b
    · simp
    · by_cases hb : G.Good (G.M.eval b.toList) <;> simp [hb]

/-- A start harvest whose undecided rate exceeds `θs` is triggered by bad read-states. -/
theorem goodEnd_start (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd) {T : DTree α} (hg : G.Genuine T)
    (h : C.θs < D.real (startSet (rdCut rd) T C.k)) :
    C.θs - θgs < D.real (G.startAt rd C.k T fun r => ¬ G.Good r) := by
  have hsplit : startSet (rdCut rd) T C.k
      = G.startAt rd C.k T G.Good ∪ G.startAt rd C.k T fun r => ¬ G.Good r := by
    ext x
    simp only [startSet, ReadModel.startAt, Set.mem_ofPred_eq, Set.mem_union]
    constructor
    · rintro ⟨b, hb⟩
      by_cases hgb : G.Good (G.M.eval b.toList)
      · exact .inl ⟨b, hb, hgb⟩
      · exact .inr ⟨b, hb, hgb⟩
    · rintro (⟨b, hb, -⟩ | ⟨b, hb, -⟩) <;> exact ⟨b, hb⟩
  have hdis : Disjoint (G.startAt rd C.k T G.Good) (G.startAt rd C.k T fun r => ¬ G.Good r) := by
    rw [Set.disjoint_left]
    rintro x ⟨b, hb, hgb⟩ ⟨b', hb', hgb'⟩
    rw [hb] at hb'
    obtain rfl := Sum.inr.inj hb'
    exact hgb' hgb
  rw [hsplit, measureReal_union hdis (Set.to_countable _).measurableSet] at h
  have := hE.goodStart T hg
  linarith

/-- An edge harvest whose undecided reads exceed `θe` of its reads is triggered by bad
read-states. -/
theorem goodEnd_edge (L : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd) {T : DTree α} {edges : Edges α} (hg : G.Genuine T)
    (he : EdgesInto T edges) (e : List Bool × α)
    (h : C.θe * ∫ x, (edgeReadsBy (rdCut rd) T edges C.k x e : ℝ) ∂D
      < ∫ x, (edgeUndecBy (rdCut rd) T edges C.k x e (fun _ => True) : ℝ) ∂D) :
    (C.θe - θg) * ∫ x, (edgeReadsBy (rdCut rd) T edges C.k x e : ℝ) ∂D
      < ∫ x, (G.undecAt rd C.k T edges e (fun r => ¬ G.Good r) x : ℝ) ∂D := by
  classical
  have hint : ∀ P : σ → Prop, Integrable (fun x => (G.undecAt rd C.k T edges e P x : ℝ)) D := by
    intro P
    refine Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable L ?_
    filter_upwards [hlen] with x hx
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    exact_mod_cast (edgeUndecBy_le (rdCut rd) e _).trans hx
  have hsum : ∫ x, (edgeUndecBy (rdCut rd) T edges C.k x e (fun _ => True) : ℝ) ∂D
      = ∫ x, (G.undecAt rd C.k T edges e G.Good x : ℝ) ∂D
        + ∫ x, (G.undecAt rd C.k T edges e (fun r => ¬ G.Good r) x : ℝ) ∂D := by
    rw [← integral_add (hint _) (hint _)]
    congr 1
    ext x
    rw [undec_split G rd x e]
    push_cast; ring
  have := hE.goodEdge T edges hg he e
  rw [hsum] at h
  nlinarith

/-- From a fresh state within `Lmax` leaves, the stretch ends in a harvest its hypothesis does
not call for. -/
def stretchEvent (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (D : Measure (FreeMonoid α))
    (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α) : Prop :=
  s.Fresh ∧ s.tree.paths.length ≤ C.Lmax ∧ StretchBad C cut D s T xs

open scoped Classical in
/-- The induction: a state keeping the invariant, with an obligation to change its hypothesis
within `r` probes unless no path-good transition's boundary is heavy, and `pot · n + r` probes
left, reaches such a state or ends in success or a harvest bad read-states trigger, unless one
of the events happens. -/
theorem main_ind (L : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hq₀ : 0 < q₀)
    (hcq : 0 < c₀ * q₀) (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd)
    (hLmax : Fintype.card σ + 2 ≤ C.Lmax) (hfuel : versionCap C (Fintype.card α) ≤ C.fuel)
    (hn₀ : 1 ≤ C.n₀) {n : ℕ} (hn : 1 ≤ n) :
    ∀ (T : ℕ) (s : TState α) (xs : Fin T → FreeMonoid α) (K r : ℕ),
      TallyInv G C K s →
      K + hitsAlong (tallyStep' C (rdCut rd)) (spurSet G C rd) s T xs < C.m →
      ¬ Lingers (tallyStep' C (rdCut rd)) tallyKey (Significant C (rdCut rd) D (c₀ * q₀)) n
        s T xs →
      ¬ Somewhere (tallyStep' C (rdCut rd)) (stretchEvent C (rdCut rd) D) s T xs →
      ¬ StretchBad C (rdCut rd) D s T xs →
      (AllFixed G D C.k q₀ s ∨ ¬ Stays (tallyStep' C (rdCut rd)) tallyKey r s T xs) →
      s.pot C.m C.Lmax * n + r ≤ T →
      RunReaches (tallyStep C (rdCut rd)) (AllFixed G D C.k q₀) (GoodEnd G D C rd θg θgs) s
        (List.ofFn xs) := by
  intro T
  induction T with
  | zero =>
    intro s xs K r hI hK hL hS hB hob hbud
    simp only [List.ofFn_zero, RunReaches]
    rcases hob with h | h
    · exact h
    · exfalso
      obtain rfl : r = 0 := by omega
      exact h trivial
  | succ T ih =>
    intro s xs K r hI hK hL hS hB hob hbud
    rw [List.ofFn_succ]
    by_cases hfix : AllFixed G D C.k q₀ s
    · exact .inl hfix
    right
    have hst := hob.resolve_left hfix
    obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := by
      rcases r with _ | r'
      · exact absurd trivial hst
      · exact ⟨r', rfl⟩
    have hK1 : K + (if xs 0 ∈ spurSet G C rd s then 1 else 0) < C.m := by
      simp only [hitsAlong] at hK; omega
    rcases step_spec G C rd (xs 0) hK1 hLmax hfuel hn₀ hI with
      ⟨s', hs', hI', hvle, hpot, hkey, hfr⟩ | ⟨e, s₁, hs₁, he, hkey⟩
    · rw [hs']
      have hstep' : tallyStep' C (rdCut rd) s (xs 0) = some s' := by simp [tallyStep', hs']
      have hK' : (K + if xs 0 ∈ spurSet G C rd s then 1 else 0)
          + hitsAlong (tallyStep' C (rdCut rd)) (spurSet G C rd) s' T (Fin.tail xs) < C.m := by
        simp only [hitsAlong, hstep'] at hK; omega
      have hL' : ¬ Lingers (tallyStep' C (rdCut rd)) tallyKey
          (Significant C (rdCut rd) D (c₀ * q₀)) n s' T (Fin.tail xs) :=
        fun h => hL (.inr ⟨s', hstep', h⟩)
      have hS' : ¬ Somewhere (tallyStep' C (rdCut rd)) (stretchEvent C (rdCut rd) D) s' T
          (Fin.tail xs) := fun h => hS (.inr ⟨s', hstep', h⟩)
      by_cases hv : s'.version = s.version
      · have hk := hkey hv
        have hB' : ¬ StretchBad C (rdCut rd) D s' T (Fin.tail xs) := fun h => hB (by
          simp only [StretchBad, hs']; exact ⟨hk, h⟩)
        have hob' : ¬ Stays (tallyStep' C (rdCut rd)) tallyKey r' s' T (Fin.tail xs) :=
          fun h => hst ⟨s', hstep', hk, h⟩
        have hpm := Nat.mul_le_mul_right n (show s'.pot C.m C.Lmax ≤ s.pot C.m C.Lmax by omega)
        exact ih s' (Fin.tail xs) _ r' hI' hK' hL' hS' hB' (.inr hob') (by omega)
      · have hfresh := hfr hv
        have hpm := Nat.mul_le_mul_right n
          (show s'.pot C.m C.Lmax + 1 ≤ s.pot C.m C.Lmax by omega)
        rw [add_mul, one_mul] at hpm
        have hpaths : s'.tree.paths.length ≤ C.Lmax := (G.genuine_paths hI'.1).trans hLmax
        have hB' : ¬ StretchBad C (rdCut rd) D s' T (Fin.tail xs) := by
          rcases T with _ | T
          · exact id
          · exact fun h => hS' (.inl ⟨hfresh, hpaths, h⟩)
        by_cases hfix' : AllFixed G D C.k q₀ s'
        · exact ih s' (Fin.tail xs) _ n hI' hK' hL' hS' hB' (.inl hfix') (by omega)
        · have hsig := sig_of_not_fixed G C rd D hq₀ hcq hE hI'.1 hI'.2.1 hfix'
          obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
          have hob' : ¬ Stays (tallyStep' C (rdCut rd)) tallyKey n s' (T' + 1) (Fin.tail xs) :=
            fun h => hL' (.inl ⟨hsig, h⟩)
          exact ih s' (Fin.tail xs) _ n hI' hK' hL' hS' hB' (.inr hob') (by omega)
    · rw [hs₁]
      have hkey' := hkey
      simp only [tallyKey, Prod.mk.injEq] at hkey'
      obtain ⟨ht, hed, -⟩ := hkey'
      rcases he with rfl | rfl | ⟨e', rfl⟩
      · trivial
      · show C.θs - θgs < D.real (G.startAt rd C.k s₁.tree fun r => ¬ G.Good r)
        rw [ht]
        apply goodEnd_start G C rd D hE hI.1
        by_contra hle
        push_neg at hle
        exact hB (by simp only [StretchBad, hs₁]; exact ⟨hkey, hle⟩)
      · show (C.θe - θg) * ∫ x, (edgeReadsBy (rdCut rd) s₁.tree s₁.edges C.k x e' : ℝ) ∂D
          < ∫ x, (G.undecAt rd C.k s₁.tree s₁.edges e' (fun r => ¬ G.Good r) x : ℝ) ∂D
        rw [ht, hed]
        apply goodEnd_edge G C rd D L hlen hE hI.1 hI.2.1 e'
        by_contra hle
        push_neg at hle
        exact hB (by simp only [StretchBad, hs₁]; exact ⟨hkey, hle⟩)

open scoped Classical in
/-- Given `TallyE`, the round fails with chance at most the probe tails. -/
theorem round_given_E (L n T : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hq₀ : 0 < q₀)
    (hcq : 0 < c₀ * q₀) (hcq1 : c₀ * q₀ ≤ 1) (hρ : 0 ≤ ρ)
    (hKρ : (Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ ≤ 1) (hθs : C.θs ≤ 1)
    (hθe : 0 < C.θe) (hm : 0 < C.m) (hn₀ : 1 ≤ C.n₀) (hLmax : Fintype.card σ + 2 ≤ C.Lmax)
    (hfuel : versionCap C (Fintype.card α) ≤ C.fuel) (hn : 1 ≤ n)
    (hT : (versionCap C (Fintype.card α) + 1) * n ≤ T)
    (hexc : ∀ j, 1 ≤ j → j ≤ T → 0 ≤ C.exc j ∧ edgeLevel C ((L + 1) * C.Lmax) j ≤ C.a)
    (hE : TallyE G D C.k q₀ c₀ ρ θg θgs rd) :
    (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunReaches (tallyStep C (rdCut rd)) (AllFixed G D C.k q₀)
          (GoodEnd G D C rd θg θgs) tallyStart (List.ofFn xs)}
      ≤ T * ENNReal.ofReal (1 - binomSfGe n (c₀ * q₀) C.m)
        + ENNReal.ofReal (binomSfGe T ((Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ) C.m)
        + T * T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a := by
  set step' := tallyStep' C (rdCut rd)
  have hI₀ := tallyStart_inv G C (α := α) (by omega)
  have hT1 : 1 ≤ T := le_trans (by nlinarith) hT
  have hsub : {xs : Fin T → FreeMonoid α | ¬ RunReaches (tallyStep C (rdCut rd))
      (AllFixed G D C.k q₀) (GoodEnd G D C rd θg θgs) tallyStart (List.ofFn xs)}
      ⊆ {xs | Lingers step' tallyKey (Significant C (rdCut rd) D (c₀ * q₀)) n tallyStart T xs}
        ∪ {xs | C.m ≤ hitsAlong step' (spurSet G C rd) tallyStart T xs}
        ∪ {xs | Somewhere step' (stretchEvent C (rdCut rd) D) tallyStart T xs} := by
    intro xs hxs
    by_contra hno
    simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, not_le] at hno
    obtain ⟨⟨hL, hH⟩, hS⟩ := hno
    apply hxs
    obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
    have hB : ¬ StretchBad C (rdCut rd) D tallyStart (T' + 1) xs := fun h =>
      hS (.inl ⟨⟨rfl, rfl, rfl, rfl, rfl⟩, by simp [tallyStart, DTree.paths]; omega, h⟩)
    have hob : AllFixed G D C.k q₀ tallyStart
        ∨ ¬ Stays step' tallyKey n tallyStart (T' + 1) xs := by
      by_cases hfix : AllFixed G D C.k q₀ (tallyStart : TState α)
      · exact .inl hfix
      · exact .inr fun h => hL (.inl ⟨sig_of_not_fixed G C rd D hq₀ hcq hE hI₀.1 hI₀.2.1 hfix, h⟩)
    have hpm := Nat.mul_le_mul_right n (show (tallyStart : TState α).pot C.m C.Lmax + 1
      ≤ versionCap C (Fintype.card α) + 1 by have := hI₀.2.2.2.2; omega)
    rw [add_mul, one_mul] at hpm
    exact main_ind G C rd D L hlen hq₀ hcq hE hLmax hfuel hn₀ hn (T' + 1) tallyStart xs 0 n hI₀
      (by rw [zero_add]; exact hH) hL hS hB hob (by omega)
  refine (measure_mono hsub).trans ?_
  refine (measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
    (add_le_add ?_ ?_)) ?_)
  · exact tally_fix_in_time C (rdCut rd) D (c₀ * q₀) hcq.le hcq1 hm n T tallyStart
  · exact hits_le D step' (spurSet G C rd) (by positivity) hKρ (spurSet_le G C rd D hρ hE)
      T tallyStart C.m
  · have := somewhere_le D step' (stretchEvent C (rdCut rd) D)
      (T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a) T (fun s T'' hT'' => by
        by_cases hs : s.Fresh ∧ s.tree.paths.length ≤ C.Lmax
        · have : {xs : Fin T'' → FreeMonoid α | stretchEvent C (rdCut rd) D s T'' xs}
              = {xs | StretchBad C (rdCut rd) D s T'' xs} := by
            ext xs; simp [stretchEvent, hs.1, hs.2]
          rw [this]
          refine (stretchBad_le C (rdCut rd) D L hlen hθs hθe (by omega) T''
            (fun j hj hjT => hexc j hj (hjT.trans hT'')) s hs.1 hs.2).trans ?_
          gcongr
        · have : {xs : Fin T'' → FreeMonoid α | stretchEvent C (rdCut rd) D s T'' xs} = ∅ := by
            ext xs
            simp only [stretchEvent, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
            exact fun h => hs ⟨h.1, h.2.1⟩
          rw [this, measure_empty]; exact zero_le) T le_rfl tallyStart
    refine this.trans (le_of_eq ?_)
    ring

/-- `TallyRound`. -/
theorem tally_round_holds : TallyRound := by
  intro α _ _ σ _ Ω _ μ _ G read D _ C q₀ c₀ ρ θg θgs L n T hlen hq₀ hcq hcq1 hρ hKρ hθs0 hθs1
    hθe hm hn₀ hLmax hfuel hn hT hexc
  set c := T * ENNReal.ofReal (1 - binomSfGe n (c₀ * q₀) C.m)
    + ENNReal.ofReal (binomSfGe T ((Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ) C.m)
    + T * T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a
  set M := toMeasurable μ {ω | ¬ TallyE G D C.k q₀ c₀ ρ θg θgs (read · ω)}
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunReaches (tallyStep C fun z => (read z ω).cut) (AllFixed G D C.k q₀)
        (GoodEnd G D C (read · ω) θg θgs) tallyStart (List.ofFn xs)}
      ≤ M.indicator 1 ω + c := by
    intro ω
    by_cases hω : TallyE G D C.k q₀ c₀ ρ θg θgs (read · ω)
    · exact (round_given_E G C (read · ω) D L n T hlen hq₀ hcq hcq1 hρ hKρ hθs1 hθe hm hn₀
        hLmax hfuel hn hT hexc hω).trans le_add_self
    · have hM : ω ∈ M := subset_toMeasurable _ _ hω
      rw [Set.indicator_of_mem hM, Pi.one_apply]
      exact prob_le_one.trans le_self_add
  calc ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunReaches (tallyStep C fun z => (read z ω).cut) (AllFixed G D C.k q₀)
          (GoodEnd G D C (read · ω) θg θgs) tallyStart (List.ofFn xs)} ∂μ
      ≤ ∫⁻ ω, (M.indicator 1 ω + c) ∂μ := lintegral_mono hpt
    _ = μ {ω | ¬ TallyE G D C.k q₀ c₀ ρ θg θgs (read · ω)} + c := by
      rw [lintegral_add_right _ measurable_const, lintegral_indicator_one
        (measurableSet_toMeasurable _ _), measure_toMeasurable, lintegral_const, measure_univ,
        mul_one]
    _ = _ := by simp only [c]; ring

end OrthoDFA
