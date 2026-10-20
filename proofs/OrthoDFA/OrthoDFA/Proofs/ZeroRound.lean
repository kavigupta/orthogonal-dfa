import OrthoDFA.Proofs.LoopE2
import OrthoDFA.Proofs.ZeroFail
import OrthoDFA.Proofs.ZeroRuns

/-!
# A round walked from before its avoidance depth

The round's probes avoid the clustering's prefixes' first `a` letters, and are walked from
`k ≤ a`. A probe reads strings `u m` with `u` a prefix of it and `m` a midfix; where the tree keeps
at most `|Q| + 2` leaves, `u m` is either `KeyOff` at `a` or at most `a + |Q|` long. With every
string of the clustering's prefixes longer, re-noising the rest leaves level 5's guarantee, read
at the depth `kd`, on the probes conditioned to avoid; where no state reads undecided more than
`1.5θ` of the time the round cannot end well in a harvest, so it ends consistent
(`round_bad_zero`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

lemma prefixMax_avoidLaw_le_at (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k kd : ℕ)
    (K₀ : Finset (FreeMonoid α)) (hm : D.real (avoiding k ↑K₀)ᶜ < 1) :
    prefixMax (avoidLaw D k K₀) kd ≤ prefixMax D kd / (1 - D.real (avoiding k ↑K₀)ᶜ) := by
  set A := avoiding (α := α) k ↑K₀
  have hA : D.real A = 1 - D.real Aᶜ := by
    rw [probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have h0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  refine ciSup_le fun w => ?_
  rw [avoidLaw_eq, popLaw_real h0 (measure_ne_top _ _), measureReal_restrict_apply_univ,
    ← hA, measureReal_restrict_apply (measurableSet_of_countable _)]
  exact div_le_div_of_nonneg_right ((measureReal_mono Set.inter_subset_left
    (measure_ne_top _ _)).trans (le_prefixMax D kd w)) hApos.le

/-- A set of `D` holding no more of the avoiding probes than `S'` does is at most `S'` under the
conditioned law, plus the mass avoided. -/
lemma real_le_avoidLaw (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (hm : D.real (avoiding k ↑K₀)ᶜ < 1) {S S' : Set (FreeMonoid α)}
    (hS : ∀ x ∈ avoiding k ↑K₀, x ∈ S → x ∈ S') :
    D.real S ≤ (avoidLaw D k K₀).real S' + D.real (avoiding k ↑K₀)ᶜ := by
  set A := avoiding (α := α) k ↑K₀
  have hA : D.real A = 1 - D.real Aᶜ := by
    rw [probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have h0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := S)
    (measurableSet_of_countable A).nullMeasurableSet (measure_ne_top _ _)
  have hinter : D.real (S ∩ A) ≤ D.real (S' ∩ A) :=
    measureReal_mono (fun x ⟨h1, h2⟩ => ⟨hS x h2 h1, h2⟩) (measure_ne_top _ _)
  have hdiff : D.real (S \ A) ≤ D.real Aᶜ := measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
  have hcond : (avoidLaw D k K₀).real S' = D.real (S' ∩ A) / D.real A := by
    rw [avoidLaw_eq, popLaw_real h0 (measure_ne_top _ _), measureReal_restrict_apply_univ,
      measureReal_restrict_apply (measurableSet_of_countable _)]
  have hle : D.real (S' ∩ A) ≤ (avoidLaw D k K₀).real S' := by
    rw [hcond, le_div_iff₀ hApos]
    have : D.real A ≤ 1 := measureReal_le_one
    have : 0 ≤ D.real (S' ∩ A) := measureReal_nonneg
    nlinarith
  linarith

/-- The strings off the clustering a probe avoiding `K₀` at `a` reads: a string `KeyOff` at `a`, or
at most `a + |Q|` long, followed by a member of `F`. -/
def zeroReads (a nQ : ℕ) (K₀ F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, (KeyOff a ↑K₀ z ∨ z.toList.length ≤ a + nQ) ∧ ∃ v ∈ F, w = z * v}

open Classical in
/-- Of them, those at strings at most `N'` long. -/
def zeroZone (N' a nQ : ℕ) (K₀ F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter (· ∈ zeroReads a nQ K₀ F)

/-- The round ended consistent, at most `ε` of `D` disagrees against its family's read
distributions, and its tree is in `classSet |Q|`. -/
def ZeroSpec {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (ε : ℝ) (k : ℕ) (o : LoopRound Ω α K) : Prop :=
  o.result.ending = .consistent
    ∧ P.D.real (disagreeSet M (familyReadDist O M o.family o.state.lo (o.state.hi + 1)) o.read k
      o.result.hypothesis) ≤ ε
    ∧ o.result.hypothesis.tree ∈ Random.classSet (Fintype.card σ)

section Bad

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

include hL in
/-- At a state and family where no state reads undecided more than `1.5θ` of the time, the round's
bad draws: where level 3 fails at re-noised reads. Its chance is at most level 5's at the depth
`kd`, plus `δ/8 + #(strings of at most |Q| letters) cross / u`; it is fixed by the answers at the
strings level 3 reads off the clustering; and off it the actual round ends consistent, with at most
`ε + u + m` of `D` disagreeing, and its tree in `classSet |Q|`. -/
theorem round_bad_zero (θ m u crossLimit : ℝ) (kd : ℕ) (ha : 1 ≤ settings.avoidDepth)
    (hka : settings.level3.walkStart ≤ settings.avoidDepth)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : RoundPremises σ α P.L settings.level3 θ (2 * prefixMax P.D kd) crossLimit)
    (hu : 0 < u) (hm1 : m ≤ 1 / 2)
    (B : State) (F K₀ : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
      BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hK₀ : K₀.card * prefixMax P.D settings.avoidDepth ≤ m)
    (hgood : ∀ q, familyReadDist O M F B.lo (B.hi + 1) q .undecided ≤ 3 / 2 * θ) :
    ∃ Bd : Set (Ω × (Fin settings.level3.probeBudget → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd
        ≤ ENNReal.ofReal (settings.level3.δ
          + (if settings.level3.walkStart < kd then 3 / 2 * θ * (settings.level3.probeBudget
            * (P.L + 1) * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2
              * Fintype.card α + (∑ i ∈ Finset.Ico settings.level3.walkStart kd,
                (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)) else 0)
          + settings.level3.δ / 8
          + (shortStrings (α := α) (Fintype.card σ)).card * crossLimit / u)
      ∧ (∀ v ω ω', (∀ w ∈ zeroZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D K₀ v) + 1 + settings.level3.probeBudget + 2))
          settings.avoidDepth (Fintype.card σ) K₀ F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ x₀ Pr Sf ω v, (ω, v) ∉ Bd →
        ZeroSpec M O P (settings.level3.disagreementRate + u + m) settings.level3.walkStart
          (mkRound settings O P.D x₀ B F K₀ Pr Sf (ω, v)) := by
  classical
  obtain ⟨hθ0, hδ0, hδ1, hθL, hGy, hyρ, hρ1, hP, hcr0, hcr⟩ := hprem
  set C := settings.level3 with hCdef
  set k := C.walkStart
  set a := settings.avoidDepth
  set nQ := Fintype.card σ
  set T := C.probeBudget
  set Oc := O.clamped
  have hηIn : 0 ≤ 1 - Oc.ηIn ∧ 1 - Oc.ηIn ≤ 1 := by
    have h1 : O.ηIn ≤ O.η := le_max_left _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; have := le_max_left O.ηIn 0
              exact sub_nonneg.2 (max_le (by linarith) (by norm_num)),
      by simp only [Oc, Oracle.clamped]; linarith [le_max_right O.ηIn 0]⟩
  have hηOut : 0 ≤ Oc.ηOut ∧ Oc.ηOut ≤ 1 := by
    have h1 : O.ηOut ≤ O.η := le_max_right _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; exact le_max_right _ _,
      by simp only [Oc, Oracle.clamped]; exact max_le (by linarith) (by norm_num)⟩
  have hb := hband F.card hcard (1 - Oc.ηIn) Oc.ηOut 0 0 hηIn.1 hηIn.2 hηOut.1 hηOut.2 le_rfl
  let fam : SuffixFamily μ α σ := ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, 0, 0, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (zeroReads a nQ K₀ F)ᶜ
  let fam' := fam.renoise R
  -- the conditioned probes
  set D' := avoidLaw P.D a K₀
  set m' := P.D.real (avoiding a ↑K₀)ᶜ
  have hm' : m' ≤ m := (real_compl_avoiding_le P.D a K₀).trans hK₀
  have hm'1 : m' < 1 := by linarith
  have hD'len := avoidLaw_len P.D P.length_le a K₀
  let P' : PrefixDistribution α := ⟨D', inferInstance, P.L, hD'len⟩
  have hpm0 : 0 ≤ prefixMax P.D kd := (measureReal_nonneg).trans (le_prefixMax P.D kd 1)
  have hpm : prefixMax D' kd ≤ 2 * prefixMax P.D kd := by
    refine (prefixMax_avoidLaw_le_at P.D a kd K₀ hm'1).trans ?_
    rw [div_le_iff₀ (by linarith)]
    have : 0 ≤ m' := measureReal_nonneg
    nlinarith
  have hpre' : ∀ v, P'.D.real {x | prefixOf x kd = v} ≤ 2 * prefixMax P.D kd :=
    fun v => (le_prefixMax D' kd v).trans hpm
  have h4 := level3Fail_le_at fam' (show (0 : ℝ) ≤ crossLimit from hcross.le) le_rfl P' C θ
    (2 * prefixMax P.D kd) u kd hu hpre' hθ0 hδ0 hδ1 hθL hGy hyρ hρ1 hP (by
      show crossLimit * _ + 0 * _ ≤ C.δ / 8
      rw [zero_mul, add_zero]
      exact hcr)
  set Sf := level3Fail fam' C D' θ u
  have hSm : MeasurableSet Sf := measurableSet_level3Fail fam' C D' θ u
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) (Measure.pi fun _ : Fin T => D') Sf hSm
  have hpr : MeasurePreserving (probesOf settings P.D K₀)
      (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) (Measure.pi fun _ => D') :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  have hmp : MeasurePreserving (fun q : Ω × (Fin T → unitInterval) =>
      (q.1, probesOf settings P.D K₀ q.2)) (μ.prod (Measure.pi fun _ => volume))
      (μ.prod (Measure.pi fun _ => D')) := (MeasurePreserving.id μ).prod hpr
  have hSω : MeasurableSet {q : Ω × (Fin T → FreeMonoid α) | ((q.1, ω₀), q.2) ∈ Sf} :=
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hSm
  refine ⟨(fun q : Ω × (Fin T → unitInterval) => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
    {q | ((q.1, ω₀), q.2) ∈ Sf}, hmp.measurable hSω, ?_, ?_, ?_⟩
  · rw [hmp.measure_preimage hSω.nullMeasurableSet]
    refine hω₀.trans (h4.trans (le_of_eq ?_))
    have e : fam'.ε + fam'.κ = crossLimit := by
      show crossLimit + 0 = crossLimit
      rw [add_zero]
    rw [e]
  · -- the bad draws read only the zone
    intro v ω ω' h hω
    have hread : ∀ z : FreeMonoid α, z.toList.length ≤ max (P.L + T + 2)
        (probesLen (probesOf settings P.D K₀ v) + 1 + T + 2) →
        fam'.read (ω, ω₀) z = fam'.read (ω', ω₀) z := by
      intro z hz
      unfold SuffixFamily.read familyRead voteCount
      congr 2
      refine Finset.filter_congr fun v' hv => ?_
      by_cases hzv : z * v' ∈ R
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_mem Oc hzv, renoise_mq_of_mem Oc hzv]
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_notMem Oc hzv, renoise_mq_of_notMem Oc hzv]
        refine h _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z, v'), Finset.mem_product.2
          ⟨mem_shortStrings.2 hz, hv⟩, rfl⟩, not_not.1 hzv⟩)
    exact level3Fail_congr_up fam' C D' hD'len θ u (probesOf settings P.D K₀ v) hread hω
  · -- off the bad draws, the actual round ends consistent
    intro x₀ Pr Sf' ω v hnot
    set xs := probesOf settings P.D K₀ v
    set read' : FreeMonoid α → ARU := fun z => fam'.read (ω, ω₀) z
    have hgood' : ¬ ((ω, ω₀), xs) ∈ Sf := hnot
    obtain ⟨hE, hcls⟩ := not_not.1 hgood'
    set read : FreeMonoid α → ARU := fun z => familyRead O.mq F B.lo (B.hi + 1) z ω
    have hoff : ∀ z, (KeyOff a ↑K₀ z ∨ z.toList.length ≤ a + nQ) → read z = read' z :=
      fun z hz => (renoise_read_of_notMem fam R (fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩)
        (ω, ω₀)).symm
    have hxs : ∀ i, Avoids a ↑K₀ (xs i) := fun i => drawFrom_avoids P.D ha K₀ (v i)
    -- a probe's reads past a short midfix
    have hpre : ∀ x, Avoids a ↑K₀ x → ∀ j, j ≤ x.toList.length → ∀ m : FreeMonoid α,
        m.toList.length ≤ nQ + 1 → read (prefixOf x j * m) = read' (prefixOf x j * m) := by
      intro x hx j hj m hm
      refine hoff _ ?_
      by_cases hja : a ≤ j
      · exact .inl (keyOff_of_prefix hx hja hj m)
      · right
        have : (prefixOf x j).toList.length ≤ j := by simp [prefixOf]
        simp only [FreeMonoid.toList_mul, List.length_append]
        omega
    have hleaves : (Random.indecisiveLStarLevel3 C read' xs).hypothesis.tree.leaves.length
        ≤ nQ + 2 := (Random.classSet_props _ _ hcls).2
    have hres : Random.indecisiveLStarLevel3 C read xs = Random.indecisiveLStarLevel3 C read' xs :=
      (Random.level3_congr_small C xs (n := nQ)
        (fun i j _ hjx m hm => (hpre (xs i) (hxs i) j hjx m hm).symm) hleaves).symm
    have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
      rw [SuffixFamily.renoise_readDist]
      rfl
    set H := (Random.indecisiveLStarLevel3 C read xs).hypothesis with hHd
    have hH : H = (Random.indecisiveLStarLevel3 C read' xs).hypothesis := by rw [hHd, hres]
    have hmidH : Random.MidsLe nQ H.tree := by
      rw [hH]; exact (Random.classSet_props _ _ hcls).1
    -- the renoised round cannot harvest well
    have hcons : (Random.indecisiveLStarLevel3 C read' xs).ending = .consistent := by
      rcases he : (Random.indecisiveLStarLevel3 C read' xs).ending with _ | site | _
      · rfl
      · exfalso
        rw [he] at hE
        obtain ⟨hlt, -, -⟩ := hE
        have hempty : {z : FreeMonoid α | ¬ fam'.readDist (fam'.M.eval z.toList) .undecided
            ≤ 3 / 2 * θ} = ∅ := by
          ext z
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_not]
          rw [hdist]
          exact hgood _
        rw [hempty, measureReal_empty] at hlt
        have : 0 ≤ (Random.harvestLaw read' D' k
            (Random.indecisiveLStarLevel3 C read' xs).hypothesis site).real Set.univ :=
          measureReal_nonneg
        linarith
      · rw [he] at hE
        exact hE.elim
    have hdis' : D'.real (disagreeSet M (familyReadDist O M F B.lo (B.hi + 1)) read' k H)
        ≤ C.disagreementRate + u := by
      have h' := hE
      rw [hcons] at h'
      rw [hH, ← hdist]
      exact h'
    refine ⟨?_, ?_, ?_⟩
    · show (Random.indecisiveLStarLevel3 C read xs).ending = .consistent
      rw [hres]; exact hcons
    · show P.D.real (disagreeSet M (familyReadDist O M F B.lo (B.hi + 1)) read k H)
        ≤ C.disagreementRate + u + m
      refine (real_le_avoidLaw P.D a K₀ hm'1 (S' := disagreeSet M
        (familyReadDist O M F B.lo (B.hi + 1)) read' k H) fun x hx h => ?_).trans ?_
      · have hp : probeOrAgree read H.tree H.edges k x = probeOrAgree read' H.tree H.edges k x :=
          Random.probeOrAgree_congr_zone H.edges k fun j _ hjx m hm =>
            hpre x hx j hjx m ((hmidH m hm).trans (Nat.le_succ _))
        have e1 : x ∈ Random.searchSet read k H ↔ x ∈ Random.searchSet read' k H := by
          simp only [Random.searchSet, Set.mem_ofPred_eq]; rw [hp]
        have e2 : x ∈ Random.endsSet read k H ↔ x ∈ Random.endsSet read' k H := by
          simp only [Random.endsSet, Set.mem_ofPred_eq]; rw [hp]
        rcases h with (h | h) | ⟨hk', m, hm, hw⟩
        · exact .inl (.inl (e1.1 h))
        · exact .inl (.inr (e2.1 h))
        · refine .inr ⟨hk', m, hm, ?_⟩
          have := hpre x hx x.toList.length le_rfl m ((hmidH m hm).trans (Nat.le_succ _))
          rw [prefixOf_length] at this
          beta_reduce at hw ⊢
          rw [← this]
          exact hw
      · linarith
    · show (Random.indecisiveLStarLevel3 C read xs).hypothesis.tree ∈ _
      rw [hres]
      exact hcls

end Bad

end OrthoDFA

end
