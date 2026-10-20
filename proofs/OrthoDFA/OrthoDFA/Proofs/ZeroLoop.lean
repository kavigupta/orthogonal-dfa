import OrthoDFA.Proofs.LoopE2Round
import OrthoDFA.Proofs.ZeroRound

/-!
# The first round of a loop walked from before its avoidance depth

The first round's level 3 fails to end consistent, where its clustering returned and no state
reads undecided more than `1.5θ` of the time, with chance at most level 5's at the depth `kd`
(`loop_round_zero`); its strings are none the clustering read, the clustering's prefixes being
longer than the short strings it reads.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Round

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (settings : LoopSettings K)

include hL in
/-- The first round's level 3, walked from before its avoidance depth, where its clustering
returned, no state reads undecided more than `1.5θ` of the time, its prefixes are longer than
the short strings it reads, and nothing collides, fails to end consistent with chance at most
level 5's at the depth `kd`, plus `δ/8 + #(strings of at most |Q| letters) cross/u`. -/
theorem loop_round_zero (Ls NP N ℓ₀ kd : ℕ) (θ m u crossLimit : ℝ)
    (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (ha : 1 ≤ settings.avoidDepth) (hka : settings.level3.walkStart ≤ settings.avoidDepth)
    (hlong : settings.avoidDepth + Fintype.card σ < ℓ₀)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : RoundPremises σ α P.L settings.level3 θ (2 * prefixMax P.D kd) crossLimit)
    (hu : 0 < u) (hm : 6 * (K + 1) * NP * prefixMax P.D settings.avoidDepth ≤ m)
    (hm1 : m ≤ 1 / 2) (hr : 0 < K) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf N p.1 p.2 0
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 0).length = 0
        ∧ Returns settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, (p.2 ⟨0, hr⟩).1)
        ∧ (∀ q, familyReadDist O M (loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).family (loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).state.lo ((loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).state.hi + 1) q .undecided ≤ 3 / 2 * θ)
        ∧ ¬ ZeroSpec M O P (settings.level3.disagreementRate + u + m)
            settings.level3.walkStart
            (loopRound settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
              (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, p.2 ⟨0, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 0) p.2 0 hr
        ∧ ∀ q ∈ prefsOf settings P.D Dsf (histAt settings P.D
            (histCodes settings O P.D Dsf N p.1 p.2 0) 0) (p.2 ⟨0, hr⟩).1,
          ℓ₀ ≤ q.toList.length}
      ≤ settings.level3.δ
          + (if settings.level3.walkStart < kd then 3 / 2 * θ * (settings.level3.probeBudget
            * (P.L + 1) * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2
              * Fintype.card α + (∑ i ∈ Finset.Ico settings.level3.walkStart kd,
                (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)) else 0)
          + settings.level3.δ / 8
          + (shortStrings (α := α) (Fintype.card σ)).card * crossLimit / u := by
  set r : ℕ := 0
  set δb := settings.level3.δ
          + (if settings.level3.walkStart < kd then 3 / 2 * θ * (settings.level3.probeBudget
            * (P.L + 1) * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2
              * Fintype.card α + (∑ i ∈ Finset.Ico settings.level3.walkStart kd,
                (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)) else 0)
          + settings.level3.δ / 8
          + (shortStrings (α := α) (Fintype.card σ)).card * crossLimit / u with hδb
  have hδb0 : 0 ≤ δb := by
    obtain ⟨hθ0, hδ0, -⟩ := hprem
    have : 0 ≤ (shortStrings (α := α) (Fintype.card σ)).card * crossLimit / u := by positivity
    have : 0 ≤ (if settings.level3.walkStart < kd then 3 / 2 * θ * (settings.level3.probeBudget
            * (P.L + 1) * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2
              * Fintype.card α + (∑ i ∈ Finset.Ico settings.level3.walkStart kd,
                (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)) else 0) := by
      split_ifs <;> positivity
    linarith
  classical
  set T := settings.level3.probeBudget
  set k := settings.level3.walkStart
  set a := settings.avoidDepth
  set dec := decodeHist P.D k with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  set ν := roundLaw α K T
  have hσ : MeasurePreserving (id : RoundDraws α K T → RoundDraws α K T) ν
      ((restLaw α K).prod (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))) :=
    MeasurePreserving.id _
  set Y := Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set Good : List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α) → Prop :=
    fun t => t.1.length = r ∧ t.2.1 ∈ settings.schedule (loopPopulations K (dec t.1).pops.length)
      ∧ SuffixFree t.2.2.1 ∧ t.2.2.1.card + 1 ≤ t.2.1.k ∧ t.2.2.2.card ≤ 6 * (K + 1) * NP
      ∧ ∀ q, familyReadDist O M t.2.2.1 t.2.1.lo (t.2.1.hi + 1) q .undecided ≤ 3 / 2 * θ
    with hGood
  have hpm0 : 0 ≤ prefixMax P.D a := (measureReal_nonneg).trans (le_prefixMax P.D a 1)
  have hex : ∀ t, Good t → ∃ Bd : Set (Ω × (Fin T → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd ≤ ENNReal.ofReal δb
      ∧ (∀ u ω ω', (∀ w ∈ zeroZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level3.probeBudget + 2))
          a (Fintype.card σ) t.2.2.2 t.2.2.1, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, u) ∈ Bd → (ω', u) ∈ Bd)
      ∧ ∀ x₀ Pr Sf ω v, (ω, v) ∉ Bd →
        ZeroSpec M O P (settings.level3.disagreementRate + u + m) k
          (mkRound settings O P.D x₀ t.2.1 t.2.2.1 t.2.2.2 Pr Sf (ω, v)) :=
    fun t ht => round_bad_zero M O hL P settings θ m u crossLimit kd ha hka hη1 hcross hprem hu
      hm1 t.2.1 t.2.2.1 t.2.2.2 ht.2.2.1 ht.2.2.2.1
      ((hband _ _ ht.2.1).2.2) (by
        have h1 : (t.2.2.2.card : ℝ) ≤ 6 * (K + 1) * NP := by exact_mod_cast ht.2.2.2.2.1
        exact (mul_le_mul_of_nonneg_right h1 hpm0).trans hm) ht.2.2.2.2.2
  set Bad : Y → Set (Ω × (Fin T → unitInterval)) := fun y => match y with
    | some t => if h : Good t then Classical.choose (hex t h) else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some t => histReads settings P.D Dsf N t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf P.D Dsf (dec t.1) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K)))
    | none => ∅
  set G : Y → Prop := fun y => ∃ t, y = some t ∧ t.1.length = r
  set Z : Y → (Fin T → unitInterval) → Finset (FreeMonoid α) := fun y u => match y with
    | some t => zeroZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level3.probeBudget + 2))
          a (Fintype.card σ) t.2.2.2 t.2.2.1
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some t => {x | NoColl settings P.D Dsf Ls t.1 x r hr
        ∧ t.2.2.1 ⊆ suffsOf settings P.D Dsf (histAt settings P.D t.1 r) (x rK).1
        ∧ t.2.2.2 = (prefsOf settings P.D Dsf (histAt settings P.D t.1 r) (x rK).1).image
          (fun p => prefixOf p a)
        ∧ ∀ q ∈ prefsOf settings P.D Dsf (histAt settings P.D t.1 r) (x rK).1,
          ℓ₀ ≤ q.toList.length}
    | none => Set.univ
  have hmain := persist_stage_le O ν (restLaw α K)
    (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) id hσ id measurable_id
    (fun w => rfl) rK (stage4 settings O P.D Dsf N r hr)
    (fun ω x x' hxx' h1 => stage4_congr_draws settings O P.D Dsf N r hr ω x x'
      (fun i hi => hxx' i hi) h1)
    (measurableSet_stage4 settings O P.D Dsf N P.length_le hN r hr) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, -⟩ := hG
      exact stage4_stop settings O P.D Dsf N P.length_le hN r hr c B F K₀ x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | t
      · rfl
      · show (if h : Good t then Classical.choose (hex t h) else ∅) = ∅
        rw [dif_neg fun h => hy ⟨t, rfl, h.1⟩])
    (fun y => by
      rcases y with _ | t
      · exact MeasurableSet.empty
      · show MeasurableSet (if h : Good t then Classical.choose (hex t h) else ∅)
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).1
        · exact MeasurableSet.empty)
    Z (fun y u ω ω' h hq => by
      rcases y with _ | t
      · exact hq
      · change (ω, u) ∈ (if h : Good t then Classical.choose (hex t h) else ∅) at hq
        show (ω', u) ∈ (if h : Good t then Classical.choose (hex t h) else ∅)
        split_ifs at hq ⊢ with hg
        · exact (Classical.choose_spec (hex t hg)).2.2.1 u ω ω' h hq
        · exact hq)
    Dj (fun y => by
      rcases y with _ | ⟨c, B, F, K₀⟩
      · exact MeasurableSet.univ
      · set h := histAt settings P.D c r
        have hx : Measurable fun x : Fin K → RoundDraws α K T => (x rK).1 :=
          measurable_fst.comp (measurable_pi_apply rK)
        have h2 : MeasurableSet {x : Fin K → RoundDraws α K T |
            F ⊆ suffsOf settings P.D Dsf h (x rK).1} := by
          have := measurableSet_of_data (fun x : Fin K → RoundDraws α K T =>
              suffsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_suffsOf settings P.D Dsf h Q))
            (fun Q => if F ⊆ Q then Set.univ else ∅) (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        have h3 : MeasurableSet {x : Fin K → RoundDraws α K T |
            K₀ = (prefsOf settings P.D Dsf h (x rK).1).image fun p => prefixOf p a} := by
          have := measurableSet_of_data (fun x : Fin K → RoundDraws α K T =>
              prefsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_prefsOf settings P.D Dsf h Q))
            (fun Q => if K₀ = Q.image fun p => prefixOf p a then Set.univ else ∅)
            (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        have h4 : MeasurableSet {x : Fin K → RoundDraws α K T |
            ∀ q ∈ prefsOf settings P.D Dsf h (x rK).1, ℓ₀ ≤ q.toList.length} := by
          have := measurableSet_of_data (fun x : Fin K → RoundDraws α K T =>
              prefsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_prefsOf settings P.D Dsf h Q))
            (fun Q => if ∀ q ∈ Q, ℓ₀ ≤ q.toList.length then Set.univ else ∅)
            (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        exact (measurableSet_noColl settings P.D Dsf Ls c r hr).inter (h2.inter (h3.inter h4)))
    (fun y x hG hx => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, hcr⟩ := hG
      obtain ⟨hnc, hF, hK, hlp⟩ := hx
      have hpop : histAt settings P.D c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hcr), hdec] <;> rfl
      have hdis := disjoint_hist_family settings P.D Dsf N Ls
        (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D K₀ (x rK).2) + 1 + settings.level3.probeBudget + 2))
        c x r hr ω₀ hnc F hF
      rw [hpop] at hdis
      refine hdis.mono_right fun w hw => ?_
      obtain ⟨hrs, z, hz, v, hv, rfl⟩ := Finset.mem_filter.1 hw
      refine Finset.mem_sdiff.2 ⟨hrs, fun hw' => ?_⟩
      obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
      obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
      have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu =>
        hnc.1 rK le_rfl u (hF hu)
      obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he.symm ((hlenF v hv).trans (hlenF v' hv').symm)
      rcases hz with hz | hz
      · apply hz.2
        rw [hK]
        dsimp only
        rw [hpop, hzp]
        exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨p, hp, rfl⟩)
      · have h1 := hlp p (by rw [hpop]; exact hp)
        have h2 : z = p := hzp
        rw [← h2] at h1
        omega)
    (ENNReal.ofReal δb) (fun y => by
      rcases y with _ | t
      · show (μ.prod _) ∅ ≤ _
        simp
      · show (μ.prod _) (if h : Good t then Classical.choose (hex t h) else ∅) ≤ _
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).2.1
        · simp)
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hret, hnob, hnot, hnc, hlp⟩
  set c := histCodes settings O P.D Dsf N p.1 p.2 r
  set H := dec c
  set cd : ClusterDraws Ω α K := (p.1, (p.2 rK).1)
  set B := roundState settings O P.D Dsf H cd
  set F := clusterAt O.mq (loopPopulations K H.pops.length) (clusterRun P.D Dsf H cd) B
  set K₀ := roundKeys settings P.D Dsf H cd
  have hst : stage4 settings O P.D Dsf N r hr p.1 p.2 = some (c, B, F, K₀) :=
    (stage4_eq_some settings O P.D Dsf N r hr p.1 p.2 c B F K₀).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have hBs : B ∈ settings.schedule (loopPopulations K H.pops.length) :=
    (Classical.epsilon_spec hret).1
  have hpop : histAt settings P.D c r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen), H, hdec] <;> rfl
  have hFs : F ⊆ suffsOf settings P.D Dsf (histAt settings P.D c r) (p.2 rK).1 := by
    rw [hpop]
    exact clusterAt_subset_suffsOf settings O P.D Dsf H p.1 (p.2 rK).1 hBs
  have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu => hnc.1 rK le_rfl u (hFs hu)
  have hK₀ : K₀ = (prefsOf settings P.D Dsf H (p.2 rK).1).image fun q => prefixOf q a :=
    roundKeys_eq_prefsOf settings P.D Dsf H p.1 (p.2 rK).1
  have hgood : Good (c, B, F, K₀) := by
    refine ⟨hlen, hBs, ?_, ?_, ?_, hnob⟩
    · intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    · obtain ⟨-, hk2, -⟩ := hband _ B hBs
      have h1 := clusterAt_card_add_one (rule := lloydClusterer) O
        (loopPopulations K H.pops.length) B (clusterRun P.D Dsf H cd)
      have h2 := lloydClusterer.card_le (fun w => O.mq w (oracleNoise (clusterRun P.D Dsf H cd)) = 1)
        (prefixesAt (loopPopulations K H.pops.length) B.npref (clusterRun P.D Dsf H cd))
        (screenedAt O.mq (loopPopulations K H.pops.length) B (clusterRun P.D Dsf H cd)) B.k
        (by omega)
      show F.card + 1 ≤ B.k
      simp only [F]
      rw [clusterAt_eq]
      unfold clusterFull at h1
      omega
    · show K₀.card ≤ 6 * (K + 1) * NP
      rw [hK₀]
      refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
      have hpops : (loopPopulations K H.pops.length).card ≤ K + 1 := by
        have := Finset.card_le_univ (loopPopulations K H.pops.length)
        simpa using this
      have := hNP (loopPopulations K H.pops.length)
      calc 3 * (loopPopulations K H.pops.length).card
            * ∑ B ∈ settings.schedule (loopPopulations K H.pops.length), B.npref
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
        _ = 6 * (K + 1) * NP := by ring
  show (p.1, (id (p.2 rK)).2) ∈ Bad (stage4 settings O P.D Dsf N r hr p.1 p.2)
    ∧ p.2 ∈ Dj (stage4 settings O P.D Dsf N r hr p.1 p.2)
  rw [hst]
  refine ⟨?_, hnc, hFs, by rw [hpop]; exact hK₀, hlp⟩
  show (p.1, (p.2 rK).2) ∈ (if h : Good (c, B, F, K₀) then Classical.choose (hex _ h) else ∅)
  rw [dif_pos hgood]
  by_contra hnb
  exact hnot ((Classical.choose_spec (hex _ hgood)).2.2.2 (clusterRun P.D Dsf H cd)
    (roundSets settings P.D Dsf H cd).1 (roundSets settings P.D Dsf H cd).2 p.1 (p.2 rK).2 hnb)

end Round

end OrthoDFA

end
