import OrthoDFA.Proofs.Level5Event
import OrthoDFA.Proofs.Level5DeepTail
import OrthoDFA.Proofs.Level5Head
import OrthoDFA.Proofs.Level5Harvest
import OrthoDFA.Proofs.Level5Rarer

/-!
# Level 5's risk

On the noise event, a round none of whose steps reads a string on its state's rarer side ends,
within `versionBound · nEnd` probes, by a test firing at a hypothesis whose tree is grown by
genuine splits alone. Where that ending is not well, the test fired at a hypothesis whose
sampler is not good, so one of the tests' weights passed `1/a` within that version. The round's
failure is at most the noise fields' tails (`noiseEvent_le`), the chance some probe reads on the
rarer side (`pi_wrong_le`), the success test passing with the rarer probes heavy (`rarer_le`), the
undecided probes' Markov term (`und_le_mass`), and every test's level at every version
(`Level5RoundRisk`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | prefixOf x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | prefixOf x k = u}) ⟨1, by
    rintro _ ⟨v, rfl⟩; exact measureReal_le_one⟩ u

theorem prefixMax_pos (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) :
    0 < prefixMax D k := by
  by_contra h
  push Not at h
  have h0 : ∀ u : FreeMonoid α, D {x | prefixOf x k = u} = 0 := by
    intro u
    have h1 := (le_prefixMax D k u).trans h
    have h2 : D.real {x | prefixOf x k = u} = 0 := le_antisymm h1 measureReal_nonneg
    rwa [measureReal_eq_zero_iff] at h2
  have : (Set.univ : Set (FreeMonoid α)) = ⋃ u, {x | prefixOf x k = u} := by
    ext x; simp
  have hu := measure_iUnion_null (μ := D) h0
  rw [← this, measure_univ] at hu
  exact one_ne_zero hu

section Sites

variable {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (C : Level5Settings) (rd : FreeMonoid α → ARU)

open scoped Classical in
theorem startHarvest_indicator (k : ℕ) (T : DTree α) (P : σ → Prop) (x : FreeMonoid α) :
    ((((startHarvestBy (fun z => (rd z).cut) T k x).filter
        fun b => decide (P (G.M.eval b.toList))).length : ℕ) : ℝ)
      = (G.startAt rd k T P).indicator 1 x := by
  classical
  unfold startHarvestBy ReadModel.startAt
  by_cases hk : x.toList.length < k
  · simp [hk, Set.indicator, not_le.2 hk]
  · rcases hs : T.siftCut (fun z => (rd z).cut) (prefixOf x k) with p | b
    · simp [hk, hs, Set.indicator]
    · by_cases hg : P (G.M.eval b.toList) <;> simp [hk, hs, hg, Set.indicator, not_lt.1 hk]

open scoped Classical in
theorem siteMean_start (k : ℕ) (v : DTree α × EdgeTable α) (P : σ → Prop) :
    siteMean G D rd k .start P v = D.real (G.startAt rd k v.1 P) := by
  unfold siteMean
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  exact integral_congr_ae (ae_of_all _ fun x => startHarvest_indicator G rd k v.1 P x)

open scoped Classical in
theorem middleHarvest_indicator (k : ℕ) (T : DTree α) (E : EdgeTable α) (P : σ → Prop)
    (x : FreeMonoid α) :
    ((((middleHarvestBy (fun z => (rd z).cut) T E k x).filter
        fun b => decide (P (G.M.eval b.toList))).length : ℕ) : ℝ)
      = (G.ptAt rd k T E P).indicator 1 x := by
  classical
  rcases middleHarvestBy_cases (fun z => (rd z).cut) T E k x with h | ⟨b, h⟩
  · have hx : x ∉ G.ptAt rd k T E P := by
      rintro ⟨b, hb, -⟩
      rw [h] at hb
      cases hb
    rw [Set.indicator_of_notMem hx, h]
    simp
  · by_cases hg : P (G.M.eval b.toList)
    · have hx : x ∈ G.ptAt rd k T E P := ⟨b, h, hg⟩
      rw [Set.indicator_of_mem hx, h]
      simp [hg]
    · have hx : x ∉ G.ptAt rd k T E P := by
        rintro ⟨b', hb', hg'⟩
        rw [h, List.cons.injEq] at hb'
        exact hg (hb'.1 ▸ hg')
      rw [Set.indicator_of_notMem hx, h]
      simp [hg]

open scoped Classical in
theorem siteMean_middles (k : ℕ) (v : DTree α × EdgeTable α) (P : σ → Prop) :
    siteMean G D rd k .middles P v = D.real (G.ptAt rd k v.1 v.2 P) := by
  unfold siteMean
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  exact integral_congr_ae (ae_of_all _ fun x => middleHarvest_indicator G rd k v.1 v.2 P x)

omit [IsProbabilityMeasure D] in
open scoped Classical in
theorem siteTotal_middles (v : DTree α × EdgeTable α) :
    siteTotal D rd C.walkStart .middles v = D.real (ptSet C rd v.1 v.2) := by
  unfold siteTotal
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  refine integral_congr_ae (ae_of_all _ fun x => ?_)
  change ((middleHarvestBy (fun z => (rd z).cut) v.1 v.2 C.walkStart x).length : ℝ) = _
  rcases middleHarvestBy_cases (fun z => (rd z).cut) v.1 v.2 C.walkStart x with h | ⟨b, h⟩ <;>
    simp [h, ptSet, Set.indicator]

omit [IsProbabilityMeasure D] in
open scoped Classical in
theorem siteTotal_start (v : DTree α × EdgeTable α) :
    siteTotal D rd C.walkStart .start v = D.real (startSet C rd v.1) := by
  unfold siteTotal
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  refine integral_congr_ae (ae_of_all _ fun x => ?_)
  change ((startHarvestBy (fun z => (rd z).cut) v.1 C.walkStart x).length : ℝ) = _
  rw [startHarvestBy_length]
  by_cases hx : x ∈ startSet C rd v.1 <;> simp [hx, Set.indicator]

end Sites

section Ends

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (D : Measure (FreeMonoid α))
  [IsProbabilityMeasure D] (C : Level5Settings) (rd : FreeMonoid α → ARU)

/-- A clean round's end: a test fired at a true hypothesis. -/
def CleanEnd (e : Level5Ending α) (s : Level5State α) : Prop :=
  e ≠ .tooBig ∧ runTests C s = some e ∧ TruthInv G C.maxLeaves s ∧ CountInv s

theorem runEnds_clean (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) (hSq : C.SeqOK)
    (hn₀ : 1 ≤ C.firstLook) (hexc : ∀ j, 0 < C.edgeExcess j) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), TruthInv G C.maxLeaves s → CountInv s →
      ¬ AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) s l →
      RunEnds (applyProbe C fun z => (rd z).cut) (fun _ _ => True) s l →
      RunEnds (applyProbe C fun z => (rd z).cut) (CleanEnd G C) s l
  | [], _, _, _, _, h => h.elim
  | x :: xs, s, ht, hc, hx, h => by
    have hx0 : ¬ WrongStep G rd C s x := fun h => hx (Or.inl h)
    obtain ⟨hinl, hinr⟩ := truthInv_step G rd C hσ ht hx0
    simp only [RunEnds] at h ⊢
    rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | ⟨e, s'⟩ <;> rw [hst] at h
    · exact runEnds_clean hσ hSq hn₀ hexc xs s' (hinl s' hst)
        (countInv_step C _ hc hst) (fun h' => hx (Or.inr ⟨s', hst, h'⟩)) h
    · obtain ⟨hne, htr⟩ := hinr e s' hst
      rcases applyProbe_inr_shape C _ hSq.a_pos hSq.a_lt hn₀ hexc hst with rfl | ⟨rfl, hr⟩
      · exact absurd rfl hne
      · exact ⟨hne, hr, htr, countInv_keep C _ hc x⟩

/-- An edge test that fired with both its potentials below `1/a` certifies its rates. -/
theorem edge_rates {θe φe a K₁ K₂ μU μR n R U exc : ℝ} (ha : 0 < a) (ha1 : a < 1)
    (hθe : 0 ≤ θe) (hn : 0 ≤ n) (hK₁ : 0 < K₁) (hK₂ : 0 < K₂) (hfire1 : φe * n ≤ R)
    (hfire2 : exc ≤ U - θe * R)
    (hexc : K₁ * Real.log (1 / a) + θe * (K₂ * Real.log (1 / a)) < exc)
    (h1 : ¬ 1 ≤ a * Real.exp ((U - 2 * n * μU) / K₁))
    (h2 : ¬ 1 ≤ a * Real.exp ((n * μR / 2 - R) / K₂)) :
    θe * φe / 2 < μU ∧ θe * μR / 4 < μU := by
  have hlg : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha]; linarith)
  have hlt : ∀ z : ℝ, ¬ 1 ≤ a * Real.exp z → z < Real.log (1 / a) := by
    intro z hz
    push Not at hz
    rw [← Real.exp_lt_exp, Real.exp_log (by positivity), lt_div_iff₀ ha]
    linarith
  have e1 := hlt _ h1
  have e2 := hlt _ h2
  rw [div_lt_iff₀ hK₁] at e1
  rw [div_lt_iff₀ hK₂] at e2
  have key : θe * R + θe * (K₂ * Real.log (1 / a)) < 2 * n * μU := by
    have : U - 2 * n * μU < Real.log (1 / a) * K₁ := e1
    nlinarith
  constructor
  · have h3 : θe * (φe * n) ≤ θe * R := mul_le_mul_of_nonneg_left hfire1 hθe
    have h4 : 0 ≤ θe * (K₂ * Real.log (1 / a)) := by positivity
    by_contra hc
    push Not at hc
    nlinarith [mul_le_mul_of_nonneg_left hc hn]
  · have h3 : θe * (n * μR / 2) ≤ θe * (R + K₂ * Real.log (1 / a)) :=
      mul_le_mul_of_nonneg_left (by linarith) hθe
    by_contra hc
    push Not at hc
    nlinarith [mul_le_mul_of_nonneg_left hc hn]


/-- The ways a clean end can fail to end well on the noise event: a test erring at its
hypothesis, one of them for each leaf and letter at an edge. -/
def AnyBad (L : ℕ) (e : Level5Ending α) (s : Level5State α) : Prop :=
  SuccessBad C D rd e s ∨ StartBad C D rd e s ∨ GateBad C D rd e s ∨ MiddlesBad C D rd e s
    ∨ ∃ i < C.maxLeaves, ∃ c : α, UndBad C D rd L i c e s ∨ ReadBad C D rd L i c e s

/-- A consistent end that is not well has success's test erring, or passing with the rarer probes
heavy. -/
theorem successBad_of_clean {J : ℕ} {ū : ℝ} (hSq : C.SeqOK) {s : Level5State α}
    (hund : undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū →
      D.real (undecidedProbes rd C.walkStart C.maxLeaves J) ≤ C.disagreementRate / 32)
    (hn : ¬ Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū rd .success s) :
    SuccessBad C D rd .success s ∨ RarerBad G D C rd .success s := by
  set H : HypothesisDFA α := ⟨s.tree, Function.uncurry s.edges⟩
  set p := C.disagreementRate
  have hp := hSq.εd_pos
  unfold Level5EndsWell at hn
  simp only [toRoundEnd, EndsWell, not_le] at hn
  by_cases hp3 : p < 2 / 3
  swap
  · exact absurd hn (not_lt.2 (measureReal_le_one.trans (by push Not at hp3; linarith)))
  by_cases hR : p / (8 - 7 * p) ≤ D.real (rarerSet G C rd s)
  · exact .inr ⟨rfl, hp3, hR⟩
  left
  refine ⟨rfl, ?_⟩
  by_contra hlt
  push Not at hlt
  have hu : D.real {x | undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū
      ∧ x ∈ undecidedProbes rd C.walkStart C.maxLeaves J} ≤ p / 32 := by
    by_cases hcu : undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū
    · simpa [hcu] using hund hcu
    · simp only [hcu, false_and, Set.setOf_false, measureReal_empty]
      linarith
  have hr' : D.real (rarerProbes G.M G.readDist rd C.walkStart H) ≤ 3 * p / 10 := by
    push Not at hR
    refine hR.le.trans ?_
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have h1 := measureReal_union_le (μ := D) (searchingProbes rd C.walkStart H
    ∪ rarerProbes G.M G.readDist rd C.walkStart H)
    {x | undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū
      ∧ x ∈ undecidedProbes rd C.walkStart C.maxLeaves J}
  have h2 := measureReal_union_le (μ := D) (searchingProbes rd C.walkStart H)
    (rarerProbes G.M G.readDist rd C.walkStart H)
  have h3 : D.real (searchingProbes rd C.walkStart ⟨s.tree, Function.uncurry s.edges⟩)
      < p := hlt
  unfold checkedProbes at hn
  linarith

/-- On the noise event, a clean end that is not well has a test erring at its hypothesis. -/
theorem bad_of_clean {L J : ℕ} {ū θg θgs θgpt : ℝ} (hSq : C.SeqOK) (hθg0 : 0 ≤ θg) (hθe : 8 * θg ≤ C.edgeUndecidedRate)
    (hφe : 0 ≤ C.edgeTraffic) (hθgs : 2 * θgs ≤ C.startUndecidedRate) (hθgpt0 : 0 ≤ θgpt)
    (hθgpt : 4 * θgpt ≤ C.middleUndecidedRate)
    (hexc : ∀ j, 2 * (L + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves)
      * Real.log (1 / C.testLevel) < C.edgeExcess j)
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {e : Level5Ending α} {s : Level5State α}
    (hc : CleanEnd G C e s) (hE : NoiseAt G D C θg θgs θgpt rd s.tree s.edges)
    (hund : undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū →
      D.real (undecidedProbes rd C.walkStart C.maxLeaves J) ≤ C.disagreementRate / 32)
    (hn : ¬ Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū rd e s) :
    AnyBad D C rd L e s ∨ RarerBad G D C rd e s := by
  obtain ⟨hne, hr, ⟨-, -, -, hL⟩, -⟩ := hc
  have hy : C.harvestYield ≤ C.startUndecidedRate ∧ C.harvestYield
      ≤ C.edgeTraffic * C.edgeUndecidedRate / 2 ∧ C.harvestYield
        ≤ C.searchShare * C.middleUndecidedRate / 2 :=
    ⟨min_le_left _ _, (min_le_right _ _).trans (min_le_left _ _),
      (min_le_right _ _).trans (min_le_right _ _)⟩
  have hwell : ∀ site, SiteGood G D rd C.walkStart C.harvestYield site (s.tree, s.edges) →
      toRoundEnd e = .harvest site → False := by
    intro site hgood hsite
    apply hn
    unfold Level5EndsWell
    rw [hsite]
    exact endsWell_of_siteGood (D := D) (k := C.walkStart) G hlen rd _ _ _
      ⟨s.tree, Function.uncurry s.edges⟩ site hgood
  refine (?_ : AnyBad D C rd L e s ∨ RarerBad G D C rd e s)
  cases e with
  | success =>
    rcases successBad_of_clean G D C rd hSq hund hn with h | h
    · exact .inl (.inl h)
    · exact .inr h
  | harvestStart =>
    left; right; left
    refine ⟨rfl, ?_⟩
    by_contra hlt
    push Not at hlt
    refine hwell .start ⟨?_, ?_⟩ rfl
    · rw [siteMean_start, siteTotal_start]
      have := hE.goodStart
      dsimp only
      linarith
    · rw [siteTotal_start]
      dsimp only
      linarith [hy.1]
  | harvestPT =>
    left; right; right
    by_contra hno
    simp only [not_or, GateBad, MiddlesBad, true_and, not_le] at hno
    obtain ⟨hdis, hpt⟩ := hno
    have hgp := hE.goodPT
    refine hwell .middles ⟨?_, ?_⟩ rfl
    · rw [siteMean_middles, siteTotal_middles]
      dsimp only
      have h1 : θgpt * (C.searchShare / 2)
          ≤ θgpt * D.real (searchAt rd C.walkStart s.tree s.edges) :=
        mul_le_mul_of_nonneg_left hdis.le hθgpt0
      have h2 : 0 ≤ D.real (searchAt rd C.walkStart s.tree s.edges) := measureReal_nonneg
      nlinarith [mul_le_mul_of_nonneg_right hθgpt h2]
    · rw [siteTotal_middles]
      dsimp only
      have h2 := hSq.θpt_pos
      nlinarith [hy.2.2, mul_lt_mul_of_pos_left hdis h2]
  | harvest e' =>
    left; right; right; right; right
    obtain ⟨hmem, hf1, hf2⟩ := runTests_harvest' C hr
    obtain ⟨i, hi, hie⟩ := List.getElem_of_mem hmem
    have hedge : e' = edgeOf s.tree i e'.2 := by
      simp only [edgeOf, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some,
        hie]
    refine ⟨i, by omega, e'.2, ?_⟩
    by_contra hno
    simp only [not_or, UndBad, ReadBad, ← hedge, true_and] at hno
    obtain ⟨h1, h2⟩ := hno
    have hK₂ : (0 : ℝ) < 2 * ((L + 1) * C.maxLeaves) := by
      have : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
      positivity
    have hrates := edge_rates hSq.a_pos hSq.a_lt (by linarith) (Nat.cast_nonneg s.probes)
      (by positivity : (0 : ℝ) < 2 * (L + 1)) hK₂ hf1 hf2
      (by have := hexc (s.edgeReads e'.1 e'.2); nlinarith) (by
        simp only [undW, ← hedge] at h1; exact h1)
      (by simp only [readW, ← hedge] at h2; exact fun h => h2 ⟨hL, h⟩)
    have hgd := hE.goodEdge e'
    refine hwell (.edge e') ⟨?_, ?_⟩ rfl
    · change 2 * (∫ x, (G.undecAt rd C.walkStart s.tree s.edges e' G.Good x : ℝ) ∂D)
        < undMean C D rd s.tree s.edges e'
      change _ ≤ θg * readMean C D rd s.tree s.edges e' + _ at hgd
      obtain ⟨hr1, hr2⟩ := hrates
      set g := ∫ x, (G.undecAt rd C.walkStart s.tree s.edges e' G.Good x : ℝ) ∂D
      set μU := undMean C D rd s.tree s.edges e'
      set μR := readMean C D rd s.tree s.edges e'
      set θe := C.edgeUndecidedRate
      set φe := C.edgeTraffic
      rcases (show (0 : ℝ) ≤ θe by linarith).lt_or_eq with hθe0 | hθe0
      · have h3 : θe * (2 * g) ≤ 8 * θg * (θe * μR / 4) + (θe - 8 * θg) * (θe * φe / 2) := by
          have := mul_le_mul_of_nonneg_left hgd hθe0.le
          nlinarith
        have h4 : 8 * θg * (θe * μR / 4) ≤ 8 * θg * μU :=
          mul_le_mul_of_nonneg_left hr2.le (by linarith)
        have h5 : (θe - 8 * θg) * (θe * φe / 2) ≤ (θe - 8 * θg) * μU :=
          mul_le_mul_of_nonneg_left hr1.le (by linarith)
        have h6 : θe * (2 * g) < θe * μU := by
          rcases (show (0 : ℝ) ≤ θe - 8 * θg by linarith).lt_or_eq with h7 | h7
          · have := mul_lt_mul_of_pos_left hr1 h7
            nlinarith
          · have h8 : 0 < 8 * θg := by linarith
            have := mul_lt_mul_of_pos_left hr2 h8
            nlinarith
        exact lt_of_mul_lt_mul_left h6 hθe0.le
      · have hθg : θg = 0 := by linarith
        rw [← hθe0, hθg] at hgd
        rw [← hθe0] at hr1
        simp at hgd hr1
        linarith
    · change C.harvestYield ≤ undMean C D rd s.tree s.edges e'
      linarith [hy.2.1, hrates.1]
  | tooBig => exact absurd rfl hne

end Ends

section Runs

variable {S X E : Type*} {step : S → X → S ⊕ (E × S)}

theorem runEnds_mono {P Q : E → S → Prop} (h : ∀ e s, P e s → Q e s) :
    ∀ (s : S) (l : List X), RunEnds step P s l → RunEnds step Q s l
  | _, [] => id
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_mono h s' xs
    · exact h e s'

theorem runEnds_or {P Q : E → S → Prop} :
    ∀ (s : S) (l : List X), RunEnds step P s l →
      RunEnds step Q s l ∨ RunEnds step (fun e s' => P e s' ∧ ¬ Q e s') s l
  | _, [] => False.elim
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_or s' xs
    · intro hp
      by_cases hq : Q e s'
      · exact .inl hq
      · exact .inr ⟨hp, hq⟩

theorem runEnds_split {P Q : E → S → Prop} :
    ∀ (s : S) (l : List X), RunEnds step (fun e s' => P e s' ∨ Q e s') s l →
      RunEnds step P s l ∨ RunEnds step Q s l
  | _, [] => False.elim
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_split s' xs
    · exact id

theorem runEnds_const_and {p : Prop} {P : E → S → Prop} :
    ∀ (s : S) (l : List X), RunEnds step (fun e s' => p ∧ P e s') s l → p ∧ RunEnds step P s l
  | _, [] => False.elim
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_const_and s' xs
    · exact id

theorem runEnds_exists {ι : Type*} {P : ι → E → S → Prop} :
    ∀ (s : S) (l : List X), RunEnds step (fun e s' => ∃ i, P i e s') s l →
      ∃ i, RunEnds step (P i) s l
  | _, [] => False.elim
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_exists s' xs
    · exact id

end Runs

section Undecided

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

omit [DecidableEq α] [IsProbabilityMeasure μ] in
/-- A measure's mass on a set the reads move, where each point's membership is measurable. -/
theorem measurable_measure_of_mem' (ν : Measure (FreeMonoid α)) (S : Ω → Set (FreeMonoid α))
    (hS : ∀ x, MeasurableSet {ω | x ∈ S ω}) : Measurable fun ω => ν (S ω) := by
  have hfun : (fun ω => ν (S ω))
      = fun ω => ∑' x, {ω | x ∈ S ω}.indicator (fun _ => ν {x}) ω := by
    funext ω
    rw [← Measure.tsum_indicator_apply_singleton ν (S ω) ((Set.to_countable _).measurableSet)]
    exact tsum_congr fun x => by simp only [Set.indicator]; rfl
  rw [hfun]
  exact Measurable.ennreal_tsum fun x => measurable_const.indicator (hS x)

theorem und_set_eq (k n J : ℕ) (x : FreeMonoid α) :
    {ω | x ∈ undecidedProbes (read · ω) k n J}
      = if x.toList.length ≤ J then ⋃ j ∈ Finset.Icc k (min x.toList.length J),
        ⋃ m ∈ shortWords n, {ω | read (prefixOf x j * m) ω = .undecided} else ∅ := by
  ext ω
  split_ifs with hJ
  · simp only [undecidedProbes, Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_Icc, exists_prop]
    constructor
    · rintro ⟨-, j, hkj, hjx, m, hm, hu⟩
      exact ⟨j, ⟨hkj, le_min hjx (hjx.trans hJ)⟩, m, hm, hu⟩
    · rintro ⟨j, ⟨hkj, hj⟩, m, hm, hu⟩
      exact ⟨hJ, j, hkj, hj.trans (min_le_left _ _), m, hm, hu⟩
  · simp only [undecidedProbes, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    exact fun h => hJ h.1

theorem und_measurable (hmeas : ∀ w, Measurable (read w)) (k n J : ℕ) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ undecidedProbes (read · ω) k n J} := by
  rw [und_set_eq]
  split_ifs
  · exact Finset.measurableSet_biUnion _ fun j _ => Finset.measurableSet_biUnion _ fun m _ =>
      hmeas _ (measurableSet_singleton _)
  · exact MeasurableSet.empty

theorem und_point_le
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (k n J : ℕ) (x : FreeMonoid α) :
    μ {ω | x ∈ undecidedProbes (read · ω) k n J}
      ≤ ENNReal.ofReal (∑ j ∈ Finset.Icc k (min x.toList.length J), ∑ m ∈ shortWords n,
          G.readDist (G.M.eval (prefixOf x j * m).toList) .undecided) := by
  rw [und_set_eq]
  split_ifs
  · refine (measure_biUnion_finset_le _ _).trans ?_
    refine (Finset.sum_le_sum fun j _ => measure_biUnion_finset_le _ _).trans (le_of_eq ?_)
    rw [ENNReal.ofReal_sum_of_nonneg fun j _ => Finset.sum_nonneg fun m _ => G.nonneg _ _]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ENNReal.ofReal_sum_of_nonneg fun m _ => G.nonneg _ _]
    exact Finset.sum_congr rfl fun m _ => measure_read G hlaw _ _
  · simp

/-- The probes reading undecided past their start, in expectation over the reads, are at most
`undecidedMass`. -/
theorem und_le_mass (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k n J : ℕ) :
    ∫⁻ ω, D {x | x ∈ undecidedProbes (read · ω) k n J} ∂μ
      ≤ ENNReal.ofReal (undecidedMass G.M G.readDist D k n J) := by
  classical
  set f : FreeMonoid α → ℝ := fun x => ∑ j ∈ Finset.Icc k (min x.toList.length J),
    ∑ m ∈ shortWords n, G.readDist (G.M.eval (prefixOf x j * m).toList) .undecided with hf
  have hf0 : ∀ x, 0 ≤ f x := fun x => Finset.sum_nonneg fun j _ =>
    Finset.sum_nonneg fun m _ => G.nonneg _ _
  have hd1 : ∀ w : FreeMonoid α, G.readDist (G.M.eval w.toList) .undecided ≤ 1 := fun w => by
    rw [← hlaw w .undecided]; exact measureReal_le_one
  have hf1 : ∀ x, f x ≤ (J + 1) * (shortWords (α := α) n).card := fun x => by
    refine (Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun m _ => hd1 _).trans ?_
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one, Nat.card_Icc]
    have : (min x.toList.length J + 1 - k : ℕ) ≤ J + 1 := by omega
    have h' : ((min x.toList.length J + 1 - k : ℕ) : ℝ) ≤ J + 1 := by exact_mod_cast this
    exact mul_le_mul_of_nonneg_right h' (Nat.cast_nonneg _)
  have hint : MeasureTheory.Integrable f D :=
    MeasureTheory.Integrable.of_bound (measurable_of_countable f).aestronglyMeasurable
      ((J + 1) * (shortWords (α := α) n).card)
      (ae_of_all _ fun x => by rw [Real.norm_eq_abs, abs_of_nonneg (hf0 x)]; exact hf1 x)
  have hD : ∀ ω, D {x | x ∈ undecidedProbes (read · ω) k n J}
      = ∫⁻ x, {x | x ∈ undecidedProbes (read · ω) k n J}.indicator 1 x ∂D := fun ω =>
    (lintegral_indicator_one (measurableSet_of_countable _)).symm
  simp_rw [hD]
  rw [lintegral_lintegral_swap]
  · calc ∫⁻ x, ∫⁻ ω, {x | x ∈ undecidedProbes (read · ω) k n J}.indicator 1 x ∂μ ∂D
        = ∫⁻ x, μ {ω | x ∈ undecidedProbes (read · ω) k n J} ∂D := by
          congr 1; funext x
          rw [← lintegral_indicator_one (und_measurable hmeas k n J x)]
          rfl
      _ ≤ ∫⁻ x, ENNReal.ofReal (f x) ∂D := lintegral_mono fun x => und_point_le G hlaw k n J x
      _ = ENNReal.ofReal (undecidedMass G.M G.readDist D k n J) :=
          (ofReal_integral_eq_lintegral_ofReal hint (ae_of_all _ hf0)).symm
  · refine (measurable_from_prod_countable_left fun x => ?_).aemeasurable
    exact measurable_one.indicator (und_measurable hmeas k n J x)

end Undecided

section Model

open scoped Classical in
/-- `Level5RoundRisk` over the proofs' view of the oracle. -/
theorem level5_model [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ) (read : FreeMonoid α → Ω → ARU)
    (hmeas : ∀ z, Measurable (read z))
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r)
    (hind : iIndepFun read μ) (P : PrefixDistribution α) (C : Level5Settings) (J : ℕ) (ū : ℝ)
    (hū : 0 ≤ ū) (hT : 32 ≤ C.disagreementRate * C.probeBudget) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
          (Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist P.D C J ū (read · ω))
          initialState (List.ofFn xs)} ∂μ
      ≤ roundRisk σ G.κ G.ε G.θ P.D P.L C + ENNReal.ofReal (32 * ū / C.disagreementRate) := by
  obtain ⟨D, _, L, hlen⟩ := P
  simp only [roundRisk]
  rw [ENNReal.iInf_add]
  refine le_iInf fun kn => ?_
  rw [ENNReal.iInf_add]
  refine le_iInf fun hA => ?_
  have hSq := hA.seq
  have hL : 2 ≤ C.maxLeaves := by have := hA.Lmax_ge; omega
  have hlg : 0 < Real.log (1 / C.testLevel) :=
    Real.log_pos (by rw [lt_div_iff₀ hSq.a_pos]; linarith [hSq.a_lt])
  have hn₀ : 1 ≤ C.firstLook := by
    have := hA.gate_test
    have : (0 : ℝ) < C.firstLook := by linarith
    exact_mod_cast this
  have hθg0 : 0 ≤ kn.θg := le_trans (by linarith [hA.θ_nonneg]) hA.θg_ge
  have hθe0 : 0 ≤ C.edgeUndecidedRate := by linarith [hA.θe_ge]
  have hexc0 : ∀ j, 0 < C.edgeExcess j := fun j => by
    have := hA.exc_gt j
    have : 0 ≤ 2 * ((L : ℝ) + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves)
      * Real.log (1 / C.testLevel) := by positivity
    linarith
  have hnEnd : 1 ≤ kn.nEnd := by
    rcases Nat.eq_zero_or_pos kn.nEnd with h0 | h0
    · have hs := hA.success_level
      have hS0 : kn.hS = 0 := by have := hA.hS_le; omega
      rw [h0, hS0] at hs
      exact absurd hs (not_seqRate_zero hSq.a_pos hSq.a_lt)
    · exact h0
  set T := C.probeBudget
  set V := versionBound C (Fintype.card α)
  set c : ENNReal := ENNReal.ofReal (V * (2 * C.maxLeaves * Fintype.card α + 5) * C.testLevel)
  set M := toMeasurable μ {ω | ¬ DeepEvent G D C 0 kn.depth kn.θg kn.θgs kn.θgpt (read · ω)}
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin T => D)
    {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C) initialState
        (List.ofFn xs)
      ∨ ¬ G.Decided (read · ω) (headStrings
        (runProbes C (fun z => (read z ω).cut) initialState (List.ofFn xs)).1.tree C.walkStart
          kn.depth)}
  have hεd := hSq.εd_pos
  set cu := undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū
  set ext : Ω → ENNReal := fun ω => ENNReal.ofReal (32 / C.disagreementRate)
    * if cu then D {x | x ∈ undecidedProbes (read · ω) C.walkStart C.maxLeaves J} else 0
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
        (Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū (read · ω)) initialState
          (List.ofFn xs)}
      ≤ M.indicator 1 ω + g ω + c + ext ω := by
    intro ω
    by_cases hω : DeepEvent G D C 0 kn.depth kn.θg kn.θgs kn.θgpt (read · ω)
    swap
    · have hM : ω ∈ M := subset_toMeasurable _ _ hω
      rw [Set.indicator_of_mem hM, Pi.one_apply]
      exact prob_le_one.trans (le_self_add.trans (le_self_add.trans le_self_add))
    by_cases hcu : cu ∧ ¬ D.real (undecidedProbes (read · ω) C.walkStart C.maxLeaves J)
        ≤ C.disagreementRate / 32
    · refine prob_le_one.trans (le_trans ?_ le_add_self)
      simp only [ext, if_pos hcu.1]
      rw [← ofReal_measureReal (measure_ne_top _ _), ← ENNReal.ofReal_mul (by positivity),
        ← ENNReal.ofReal_one]
      refine ENNReal.ofReal_le_ofReal ?_
      have h2 := hcu.2
      push Not at h2
      simp only [Set.setOf_mem_eq]
      rw [div_mul_eq_mul_div, le_div_iff₀ hεd]
      linarith
    have hund : cu → D.real (undecidedProbes (read · ω) C.walkStart C.maxLeaves J)
        ≤ C.disagreementRate / 32 := fun h => by
      by_contra h'
      exact hcu ⟨h, h'⟩
    refine le_trans ?_ (le_add_right (add_le_add (le_add_self (a := g ω)) le_rfl))
    set rd : FreeMonoid α → ARU := (read · ω)
    set Q : Level5Ending α → Level5State α → Prop := fun e s =>
      ((SuccessBad C D rd e s ∨ StartBad C D rd e s) ∨ GateBad C D rd e s ∨ MiddlesBad C D rd e s)
        ∨ ∃ ic : ℕ × α, ic.1 < C.maxLeaves
          ∧ (UndBad C D rd L ic.1 ic.2 e s ∨ ReadBad C D rd L ic.1 ic.2 e s)
    have hsub : {xs : Fin T → FreeMonoid α | ¬ RunEnds (applyProbe C fun z => (rd z).cut)
        (Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū rd) initialState
          (List.ofFn xs)}
        ⊆ {xs | AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
            (List.ofFn xs)
            ∨ ¬ G.Decided rd (headStrings
              (runProbes C (fun z => (rd z).cut) initialState (List.ofFn xs)).1.tree C.walkStart
                kn.depth)}
          ∪ ({xs | RunEnds (applyProbe C fun z => (rd z).cut) Q initialState (List.ofFn xs)}
            ∪ {xs | ¬ AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
                (List.ofFn xs)
              ∧ RunEnds (applyProbe C fun z => (rd z).cut) (RarerBad G D C rd) initialState
                (List.ofFn xs)}) := by
      intro xs hxs
      by_cases hall' : AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
          (List.ofFn xs)
          ∨ ¬ G.Decided rd (headStrings
            (runProbes C (fun z => (rd z).cut) initialState (List.ofFn xs)).1.tree C.walkStart
              kn.depth)
      · exact .inl hall'
      have hall := (not_or.1 hall').1
      have hdec := not_not.1 (not_or.1 hall').2
      right
      have hrun := runEnds_of_budget C (fun z => (rd z).cut) hSq hnEnd hA.hS_le hSq.φ_pos.le
        hA.φpt_nEnd hA.n₀_le_hS hA.pt_level hA.success_level hL (List.ofFn xs) initialState
        (verInv_start C hL) (stretchOK_start hnEnd) (by
          simp only [List.length_ofFn]
          have := hA.budget_ge
          simp [initialState, freshState]
          exact this)
      have hcl := runEnds_clean G C rd hA.Lmax_ge hSq hn₀ hexc0 _ _ (truthInv_start G hL)
        (countInv_fresh _ _ _) hall hrun
      rcases runEnds_or (Q := Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū rd)
        _ _ hcl with h | h
      · exact absurd h hxs
      have h' : RunEnds (applyProbe C fun z => (rd z).cut)
          (fun e s => Q e s ∨ RarerBad G D C rd e s) initialState (List.ofFn xs) := by
        refine runEnds_mono (fun e s ⟨⟨hc, hn⟩, hs⟩ => ?_) _ _ (runEnds_end C _ _ h)
        have hNA : NoiseAt G D C kn.θg kn.θgs kn.θgpt rd s.tree s.edges := by
          obtain ⟨-, -, ⟨hg, he, -, -⟩, -⟩ := hc
          refine noiseAt_of_deep G hω ⟨0, le_rfl, hg⟩ he ?_
          rw [hs]; exact hdec
        have := bad_of_clean G D C rd hSq hθg0 hA.θe_ge hA.φe_nonneg hA.θgs_le
          (le_trans (by have := hA.θ_nonneg; positivity) hA.θgpt_ge) hA.θgpt_le hA.exc_gt hlen hc
          hNA hund hn
        rcases this with (h1 | h1 | h1 | h1 | ⟨i, hi, cc, h1⟩) | h1
        · exact .inl (.inl (.inl (.inl h1)))
        · exact .inl (.inl (.inl (.inr h1)))
        · exact .inl (.inl (.inr (.inl h1)))
        · exact .inl (.inl (.inr (.inr h1)))
        · exact .inl (.inr ⟨(i, cc), hi, h1⟩)
        · exact .inr h1
      rcases runEnds_split _ _ h' with h'' | h''
      · exact .inl h''
      · exact .inr ⟨hall, h''⟩
    refine (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add le_rfl ?_))
    refine (measure_union_le _ _).trans ?_
    have hR := rarer_le G D C rd hSq hL hn₀ hexc0 hA.alt_eq T
    have hva0 : 0 ≤ (V : ℝ) * C.testLevel := mul_nonneg (Nat.cast_nonneg _) hSq.a_pos.le
    have hc4 : c = ENNReal.ofReal (V * (2 * C.maxLeaves * Fintype.card α + 4) * C.testLevel)
        + ENNReal.ofReal (V * C.testLevel) := by
      rw [← ENNReal.ofReal_add (mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (by positivity))
        hSq.a_pos.le) hva0]
      simp only [c]
      congr 1
      ring
    rw [hc4]
    refine add_le_add ?_ hR
    set step := applyProbe C fun z => (rd z).cut
    set w := ENNReal.ofReal (V * C.testLevel)
    set K := Finset.range C.maxLeaves ×ˢ (Finset.univ : Finset α)
    have hQsub : {xs : Fin T → FreeMonoid α | RunEnds step Q initialState (List.ofFn xs)}
        ⊆ (({xs | RunEnds step (SuccessBad C D rd) initialState (List.ofFn xs)}
            ∪ {xs | RunEnds step (StartBad C D rd) initialState (List.ofFn xs)})
          ∪ ({xs | RunEnds step (GateBad C D rd) initialState (List.ofFn xs)}
            ∪ {xs | RunEnds step (MiddlesBad C D rd) initialState (List.ofFn xs)}))
          ∪ ⋃ ic ∈ K, ({xs | RunEnds step (UndBad C D rd L ic.1 ic.2) initialState (List.ofFn xs)}
            ∪ {xs | RunEnds step (ReadBad C D rd L ic.1 ic.2) initialState (List.ofFn xs)}) := by
      intro xs hxs
      rcases runEnds_split _ _ hxs with h | h
      · left
        rcases runEnds_split _ _ h with h | h
        · left; exact runEnds_split _ _ h
        · right; exact runEnds_split _ _ h
      · right
        obtain ⟨ic, h⟩ := runEnds_exists _ _ h
        obtain ⟨hi, h⟩ := runEnds_const_and _ _ h
        exact Set.mem_biUnion (Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_univ _⟩)
          (runEnds_split _ _ h)
    have h1 := success_le C D rd hSq hL hn₀ hexc0 T
    have h2 := start_le C D rd hSq hL hn₀ hexc0 T
    have h3 := gate_le C D rd hSq hL hn₀ hexc0 hA.gate_test T
    have h4 := middles_le C D rd hSq hL hn₀ hexc0 T
    have h5 : ∀ ic ∈ K, (Measure.pi fun _ : Fin T => D)
        ({xs | RunEnds step (UndBad C D rd L ic.1 ic.2) initialState (List.ofFn xs)}
          ∪ {xs | RunEnds step (ReadBad C D rd L ic.1 ic.2) initialState (List.ofFn xs)})
        ≤ w + w := fun ic _ => (measure_union_le _ _).trans
          (add_le_add (und_le C D rd hSq hL hn₀ hexc0 hlen ic.1 ic.2 T)
            (read_le C D rd hSq hL hn₀ hexc0 hlen ic.1 ic.2 T))
    refine (measure_mono hQsub).trans ((measure_union_le _ _).trans ?_)
    refine (add_le_add ((measure_union_le _ _).trans (add_le_add
      ((measure_union_le _ _).trans (add_le_add h1 h2))
      ((measure_union_le _ _).trans (add_le_add h3 h4))))
      ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h5))).trans (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul]
    simp only [K, Finset.card_product, Finset.card_range, Finset.card_univ]
    have hva : 0 ≤ (V : ℝ) * C.testLevel := mul_nonneg (Nat.cast_nonneg _) hSq.a_pos.le
    have e1 : w + w = ENNReal.ofReal (2 * (V * C.testLevel)) := by
      rw [← ENNReal.ofReal_add hva hva]; ring_nf
    have h2v : 0 ≤ 2 * ((V : ℝ) * C.testLevel) := by linarith
    rw [e1, ← ENNReal.ofReal_add h2v h2v, ← ENNReal.ofReal_natCast,
      ← ENNReal.ofReal_mul (Nat.cast_nonneg _),
      ← ENNReal.ofReal_add (by linarith) (mul_nonneg (Nat.cast_nonneg _) h2v)]
    congr 1
    push_cast
    ring
  have hp₀ := prefixMax_pos D kn.depth
  have hpmax := le_prefixMax D kn.depth
  have hE := deepEvent_le (θgs := kn.θgs) G read hmeas hind hlaw hA.θ_nonneg D C 0 kn.depth L hlen
    hp₀ hpmax hA.l_nonneg hA.φe_nonneg hSq.φ_pos.le hA.θe_ge hA.θg_ge (by simpa using hA.θgpt_ge)
    hA.θgs_nonneg
  have hcard : ((classSet (Fintype.card σ + 0) : Finset (DTree α)).card : ℝ)
      ≤ classCount (α := α) σ 0 := by
    unfold classCount
    exact_mod_cast classSet_card (Fintype.card σ + 0)
  have hD : ∫⁻ ω, g ω ∂μ
      ≤ wrongTerm (α := α) G.κ G.ε L C + headTerm (α := α) G.κ G.ε G.θ L C kn.depth := by
    by_cases hk : C.walkStart < kn.depth
    · refine (pi_wrongHead_le G C hA.Lmax_ge hmeas hlaw hind hA.κ_nonneg hA.ε_nonneg hA.θ_nonneg
        kn.depth D hlen).trans ?_
      unfold wrongTerm headTerm
      rw [if_pos hk, ← ENNReal.ofReal_add (by have := hA.κ_nonneg; have := hA.ε_nonneg; positivity)
        (by have := hA.κ_nonneg; have := hA.ε_nonneg; have := hA.θ_nonneg; positivity)]
      exact ENNReal.ofReal_le_ofReal (le_of_eq (by push_cast; ring))
    · have hg : ∀ ω, g ω = (Measure.pi fun _ : Fin T => D)
          {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C)
            initialState (List.ofFn xs)} := by
        intro ω
        simp only [g, headStrings_empty (not_lt.1 hk), ReadModel.Decided, Finset.notMem_empty,
          IsEmpty.forall_iff, implies_true, not_true_eq_false, or_false]
      simp_rw [hg]
      refine (pi_wrong_le G C hA.Lmax_ge hmeas hlaw hind hA.κ_nonneg hA.ε_nonneg D hlen).trans ?_
      unfold wrongTerm headTerm
      rw [if_neg hk, ENNReal.ofReal_zero, add_zero]
  have hM : Measurable (M.indicator (1 : Ω → ENNReal)) :=
    measurable_one.indicator (measurableSet_toMeasurable _ _)
  have hextm : Measurable ext := by
    refine Measurable.const_mul ?_ _
    split_ifs
    · exact measurable_measure_of_mem' D _ fun x => und_measurable hmeas _ _ _ x
    · exact measurable_const
  have hext : ∫⁻ ω, ext ω ∂μ ≤ ENNReal.ofReal (32 * ū / C.disagreementRate) := by
    rw [lintegral_const_mul _ (by
      split_ifs
      · exact measurable_measure_of_mem' D _ fun x => und_measurable hmeas _ _ _ x
      · exact measurable_const)]
    have e : ENNReal.ofReal (32 * ū / C.disagreementRate)
        = ENNReal.ofReal (32 / C.disagreementRate) * ENNReal.ofReal ū := by
      rw [← ENNReal.ofReal_mul (by positivity)]; ring_nf
    rw [e]
    gcongr
    split_ifs with hc
    · exact (und_le_mass G hmeas hlaw D _ _ _).trans (ENNReal.ofReal_le_ofReal hc)
    · simp
  refine (lintegral_mono hpt).trans ?_
  rw [lintegral_add_right _ hextm]
  refine add_le_add ?_ hext
  calc ∫⁻ ω, (M.indicator 1 ω + g ω + c) ∂μ
      = μ M + ∫⁻ ω, g ω ∂μ + c := by
        rw [lintegral_add_right _ measurable_const, lintegral_add_left hM,
          lintegral_indicator_one (measurableSet_toMeasurable _ _), lintegral_const, measure_univ,
          mul_one]
    _ ≤ (edgeTail σ G.θ D L C kn + startTail σ G.θ D kn + middlesTail σ G.θ D L C kn)
        + (wrongTerm (α := α) G.κ G.ε L C + headTerm (α := α) G.κ G.ε G.θ L C kn.depth)
        + testLevels (α := α) C := by
        refine add_le_add (add_le_add ?_ hD) le_rfl
        · rw [measure_toMeasurable]
          refine hE.trans ?_
          unfold edgeTail startTail middlesTail edgeMapCount
          simp only [Nat.cast_zero, add_zero] at hcard ⊢
          refine add_le_add (add_le_add ?_ ?_) ?_ <;> refine ENNReal.ofReal_le_ofReal ?_
          · gcongr ?_ * _ * _
          · gcongr ?_ * _
          · simp only [mul_one]
            gcongr ?_ * _ * _
    _ = roundBound σ G.κ G.ε G.θ D L C kn := by unfold roundBound baseBound; ring

open scoped Classical in
/-- `Level5ConsistentRisk` over the proofs' view of the oracle: away from a rarer-side read and
the undecided probes' Markov term, a clean end that is neither well nor a harvest has success's
test erring, so the noise fields are not needed. -/
theorem level5_model_consistent [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r)
    (hind : iIndepFun read μ) (P : PrefixDistribution α) (C : Level5Settings) (J : ℕ) (ū : ℝ)
    (hū : 0 ≤ ū) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
          (fun e s => Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist P.D C J ū (read · ω) e s
            ∨ ∃ h, toRoundEnd e = .harvest h)
          initialState (List.ofFn xs)} ∂μ
      ≤ baseRisk σ G.κ G.ε G.θ P.D P.L C + ENNReal.ofReal (32 * ū / C.disagreementRate) := by
  obtain ⟨D, _, L, hlen⟩ := P
  simp only [baseRisk]
  rw [ENNReal.iInf_add]
  refine le_iInf fun kn => ?_
  rw [ENNReal.iInf_add]
  refine le_iInf fun hA => ?_
  have hSq := hA.seq
  have hL : 2 ≤ C.maxLeaves := by have := hA.Lmax_ge; omega
  have hn₀ : 1 ≤ C.firstLook := by
    have := hA.gate_test
    have hlg : 0 < Real.log (1 / C.testLevel) :=
      Real.log_pos (by rw [lt_div_iff₀ hSq.a_pos]; linarith [hSq.a_lt])
    have : (0 : ℝ) < C.firstLook := by linarith
    exact_mod_cast this
  have hexc0 : ∀ j, 0 < C.edgeExcess j := fun j => by
    have := hA.exc_gt j
    have : 0 ≤ 2 * ((L : ℝ) + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves)
      * Real.log (1 / C.testLevel) := by
      have hlg : 0 < Real.log (1 / C.testLevel) :=
        Real.log_pos (by rw [lt_div_iff₀ hSq.a_pos]; linarith [hSq.a_lt])
      have : 0 ≤ C.edgeUndecidedRate := by linarith [hA.θe_ge, hA.θ_nonneg, hA.θg_ge]
      positivity
    linarith
  have hnEnd : 1 ≤ kn.nEnd := by
    rcases Nat.eq_zero_or_pos kn.nEnd with h0 | h0
    · have hs := hA.success_level
      have hS0 : kn.hS = 0 := by have := hA.hS_le; omega
      rw [h0, hS0] at hs
      exact absurd hs (not_seqRate_zero hSq.a_pos hSq.a_lt)
    · exact h0
  set T := C.probeBudget
  set V := versionBound C (Fintype.card α)
  set c : ENNReal := ENNReal.ofReal (V * (2 * C.maxLeaves * Fintype.card α + 5) * C.testLevel)
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin T => D)
    {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C) initialState
      (List.ofFn xs)}
  have hεd := hSq.εd_pos
  set cu := undecidedMass G.M G.readDist D C.walkStart C.maxLeaves J ≤ ū
  set ext : Ω → ENNReal := fun ω => ENNReal.ofReal (32 / C.disagreementRate)
    * if cu then D {x | x ∈ undecidedProbes (read · ω) C.walkStart C.maxLeaves J} else 0
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
        (fun e s => Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū (read · ω) e s
          ∨ ∃ h, toRoundEnd e = .harvest h) initialState (List.ofFn xs)}
      ≤ g ω + c + ext ω := by
    intro ω
    by_cases hcu : cu ∧ ¬ D.real (undecidedProbes (read · ω) C.walkStart C.maxLeaves J)
        ≤ C.disagreementRate / 32
    · refine prob_le_one.trans (le_trans ?_ le_add_self)
      simp only [ext, if_pos hcu.1]
      rw [← ofReal_measureReal (measure_ne_top _ _), ← ENNReal.ofReal_mul (by positivity),
        ← ENNReal.ofReal_one]
      refine ENNReal.ofReal_le_ofReal ?_
      have h2 := hcu.2
      push Not at h2
      simp only [Set.setOf_mem_eq]
      rw [div_mul_eq_mul_div, le_div_iff₀ hεd]
      linarith
    have hund : cu → D.real (undecidedProbes (read · ω) C.walkStart C.maxLeaves J)
        ≤ C.disagreementRate / 32 := fun h => by
      by_contra h'
      exact hcu ⟨h, h'⟩
    refine le_trans ?_ (le_add_right le_rfl)
    set rd : FreeMonoid α → ARU := (read · ω)
    have hsub : {xs : Fin T → FreeMonoid α | ¬ RunEnds (applyProbe C fun z => (rd z).cut)
        (fun e s => Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C J ū rd e s
          ∨ ∃ h, toRoundEnd e = .harvest h) initialState (List.ofFn xs)}
        ⊆ {xs | AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
            (List.ofFn xs)}
          ∪ ({xs | RunEnds (applyProbe C fun z => (rd z).cut) (SuccessBad C D rd) initialState
              (List.ofFn xs)}
            ∪ {xs | ¬ AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
                (List.ofFn xs)
              ∧ RunEnds (applyProbe C fun z => (rd z).cut) (RarerBad G D C rd) initialState
                (List.ofFn xs)}) := by
      intro xs hxs
      by_cases hall : AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
          (List.ofFn xs)
      · exact .inl hall
      right
      have hrun := runEnds_of_budget C (fun z => (rd z).cut) hSq hnEnd hA.hS_le hSq.φ_pos.le
        hA.φpt_nEnd hA.n₀_le_hS hA.pt_level hA.success_level hL (List.ofFn xs) initialState
        (verInv_start C hL) (stretchOK_start hnEnd) (by
          simp only [List.length_ofFn]
          have := hA.budget_ge
          simp [initialState, freshState]
          exact this)
      have hcl := runEnds_clean G C rd hA.Lmax_ge hSq hn₀ hexc0 _ _ (truthInv_start G hL)
        (countInv_fresh _ _ _) hall hrun
      rcases runEnds_or (Q := fun e s => Level5EndsWell G.M (fun q => ¬ G.Good q) G.readDist D C
          J ū rd e s ∨ ∃ h, toRoundEnd e = .harvest h) _ _ hcl with h | h
      · exact absurd h hxs
      have h' : RunEnds (applyProbe C fun z => (rd z).cut)
          (fun e s => SuccessBad C D rd e s ∨ RarerBad G D C rd e s) initialState
          (List.ofFn xs) := by
        refine runEnds_mono (fun e s ⟨hc, hn⟩ => ?_) _ _ h
        obtain ⟨hn, hh⟩ := not_or.1 hn
        cases e with
        | success => exact successBad_of_clean G D C rd hSq hund hn
        | tooBig => exact absurd rfl hc.1
        | harvestStart => exact absurd ⟨_, rfl⟩ hh
        | harvestPT => exact absurd ⟨_, rfl⟩ hh
        | harvest e' => exact absurd ⟨_, rfl⟩ hh
      rcases runEnds_split _ _ h' with h'' | h''
      · exact .inl h''
      · exact .inr ⟨hall, h''⟩
    refine (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add le_rfl ?_))
    refine (measure_union_le _ _).trans ((add_le_add (success_le C D rd hSq hL hn₀ hexc0 T)
      (rarer_le G D C rd hSq hL hn₀ hexc0 hA.alt_eq T)).trans ?_)
    have hva : 0 ≤ (V : ℝ) * C.testLevel := mul_nonneg (Nat.cast_nonneg _) hSq.a_pos.le
    rw [← ENNReal.ofReal_add hva hva]
    refine ENNReal.ofReal_le_ofReal ?_
    have : (2 : ℝ) ≤ 2 * C.maxLeaves * Fintype.card α + 5 := by
      have : (0 : ℝ) ≤ 2 * C.maxLeaves * Fintype.card α := by positivity
      linarith
    have e : (V : ℝ) * (2 * C.maxLeaves * Fintype.card α + 5) * C.testLevel
        = (2 * C.maxLeaves * Fintype.card α + 5) * (V * C.testLevel) := by ring
    rw [e]
    nlinarith
  have hextm : Measurable ext := by
    refine Measurable.const_mul ?_ _
    split_ifs
    · exact measurable_measure_of_mem' D _ fun x => und_measurable hmeas _ _ _ x
    · exact measurable_const
  have hext : ∫⁻ ω, ext ω ∂μ ≤ ENNReal.ofReal (32 * ū / C.disagreementRate) := by
    rw [lintegral_const_mul _ (by
      split_ifs
      · exact measurable_measure_of_mem' D _ fun x => und_measurable hmeas _ _ _ x
      · exact measurable_const)]
    have e : ENNReal.ofReal (32 * ū / C.disagreementRate)
        = ENNReal.ofReal (32 / C.disagreementRate) * ENNReal.ofReal ū := by
      rw [← ENNReal.ofReal_mul (by positivity)]; ring_nf
    rw [e]
    gcongr
    split_ifs with hc
    · exact (und_le_mass G hmeas hlaw D _ _ _).trans (ENNReal.ofReal_le_ofReal hc)
    · simp
  refine (lintegral_mono hpt).trans ?_
  rw [lintegral_add_right _ hextm]
  refine add_le_add ?_ hext
  rw [lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one]
  calc ∫⁻ ω, g ω ∂μ + c ≤ wrongTerm (α := α) G.κ G.ε L C + testLevels (α := α) C :=
        add_le_add (pi_wrong_le G C hA.Lmax_ge hmeas hlaw hind hA.κ_nonneg hA.ε_nonneg D hlen)
          le_rfl
    _ ≤ baseBound σ G.κ G.ε G.θ D L C kn := by
        unfold baseBound
        rw [add_assoc _ (wrongTerm _ _ _ _)]
        exact le_add_self

end Model
theorem level5_round_risk_holds : Level5RoundRisk := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ P C J ū hū hT
  exact level5_model ⟨M, O.readDist, θ, O.κ, O.ε, O.readDist_nonneg, O.rarer_side_le⟩ O.read
    O.measurable O.reads_by_state O.indep P C J ū hū hT

theorem level5_consistent_risk_holds : Level5ConsistentRisk := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ P C J ū hū
  exact level5_model_consistent ⟨M, O.readDist, θ, O.κ, O.ε, O.readDist_nonneg, O.rarer_side_le⟩
    O.read O.measurable O.reads_by_state O.indep P C J ū hū

end OrthoDFA
