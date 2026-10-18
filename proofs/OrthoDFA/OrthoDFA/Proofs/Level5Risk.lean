import OrthoDFA.Proofs.Level5Event
import OrthoDFA.Proofs.Level5Harvest
import OrthoDFA.Proofs.Level5Reads

/-!
# Level 5's risk

On the noise event, a round none of whose steps reads a string on its state's rarer side ends,
within `versionBound · nEnd` probes, by a test firing at a hypothesis whose tree is grown by
genuine splits alone. Where that ending is not well, the test fired at a hypothesis whose
sampler is not good, so one of the tests' weights passed `1/a` within that version. The round's
failure is at most the noise fields' tails (`noiseEvent_le`), the chance some probe reads on the
rarer side (`pi_wrong_le`), and every test's level at every version (`Level5RoundRisk`).
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

/-- On the noise event, a clean end that is not well has a test erring at its hypothesis. -/
theorem bad_of_clean {L : ℕ} {θg θgs θgpt : ℝ} (hE : NoiseEvent G D C 0 θg θgs θgpt rd)
    (hSq : C.SeqOK) (hθg0 : 0 ≤ θg) (hθe : 8 * θg ≤ C.edgeUndecidedRate)
    (hφe : 0 ≤ C.edgeTraffic) (hθgs : 2 * θgs ≤ C.startUndecidedRate) (hθgpt0 : 0 ≤ θgpt)
    (hθgpt : 4 * θgpt ≤ C.middleUndecidedRate)
    (hexc : ∀ j, 2 * (L + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves)
      * Real.log (1 / C.testLevel) < C.edgeExcess j)
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {e : Level5Ending α} {s : Level5State α}
    (hc : CleanEnd G C e s) (hn : ¬ Level5EndsWell G.M (fun q => ¬ G.Good q) D C rd e s) :
    AnyBad D C rd L e s := by
  obtain ⟨hne, hr, ⟨hg, he, -, hL⟩, -⟩ := hc
  have hcl : G.InClass 0 s.tree := ⟨0, le_rfl, hg⟩
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
  cases e with
  | success =>
    left
    refine ⟨rfl, ?_⟩
    unfold Level5EndsWell at hn
    simp only [toRoundEnd, EndsWell, not_le] at hn
    exact hn.le
  | harvestStart =>
    right; left
    refine ⟨rfl, ?_⟩
    by_contra hlt
    push Not at hlt
    refine hwell .start ⟨?_, ?_⟩ rfl
    · rw [siteMean_start, siteTotal_start]
      have := hE.goodStart s.tree hcl
      dsimp only
      linarith
    · rw [siteTotal_start]
      dsimp only
      linarith [hy.1]
  | harvestPT =>
    right; right
    by_contra hno
    simp only [not_or, GateBad, MiddlesBad, true_and, not_le] at hno
    obtain ⟨hdis, hpt⟩ := hno
    have hgp := hE.goodPT s.tree s.edges hcl he
    refine hwell .middles ⟨?_, ?_⟩ rfl
    · rw [siteMean_middles, siteTotal_middles]
      dsimp only
      have h1 : θgpt * (C.searchShare / 2) ≤ θgpt * D.real (searchAt rd C.walkStart s.tree s.edges) :=
        mul_le_mul_of_nonneg_left hdis.le hθgpt0
      have h2 : 0 ≤ D.real (searchAt rd C.walkStart s.tree s.edges) := measureReal_nonneg
      nlinarith [mul_le_mul_of_nonneg_right hθgpt h2]
    · rw [siteTotal_middles]
      dsimp only
      have h2 := hSq.θpt_pos
      nlinarith [hy.2.2, mul_lt_mul_of_pos_left hdis h2]
  | harvest e' =>
    right; right; right; right
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
        simp only [undW, ← hedge] at h1; exact h1) (by simp only [readW, ← hedge] at h2; exact fun h => h2 ⟨hL, h⟩)
    have hgd := hE.goodEdge s.tree s.edges hcl he e'
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

section Model

open scoped Classical in
/-- `Level5RoundRisk` over the proofs' view of the oracle. -/
theorem level5_model [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ) (read : FreeMonoid α → Ω → ARU)
    (hmeas : ∀ z, Measurable (read z))
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r)
    (hind : iIndepFun read μ) (P : PrefixDistribution α) (C : Level5Settings) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
          (Level5EndsWell G.M (fun q => ¬ G.Good q) P.D C (read · ω))
          initialState (List.ofFn xs)} ∂μ
      ≤ roundRisk σ G.κ G.ε G.θ P.D P.L C := by
  obtain ⟨D, _, L, hlen⟩ := P
  simp only [roundRisk]
  refine le_iInf₂ fun kn hA => ?_
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
  set c : ENNReal := ENNReal.ofReal (V * (2 * C.maxLeaves * Fintype.card α + 4) * C.testLevel)
  set M := toMeasurable μ {ω | ¬ NoiseEvent G D C 0 kn.θg kn.θgs kn.θgpt (read · ω)}
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin T => D)
    {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C) initialState
      (List.ofFn xs)}
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
        (Level5EndsWell G.M (fun q => ¬ G.Good q) D C (read · ω)) initialState (List.ofFn xs)}
      ≤ M.indicator 1 ω + g ω + c := by
    intro ω
    by_cases hω : NoiseEvent G D C 0 kn.θg kn.θgs kn.θgpt (read · ω)
    swap
    · have hM : ω ∈ M := subset_toMeasurable _ _ hω
      rw [Set.indicator_of_mem hM, Pi.one_apply]
      exact prob_le_one.trans (le_self_add.trans le_self_add)
    refine le_trans ?_ (add_le_add (le_add_self (a := g ω)) le_rfl)
    set rd : FreeMonoid α → ARU := (read · ω)
    set Q : Level5Ending α → Level5State α → Prop := fun e s =>
      ((SuccessBad C D rd e s ∨ StartBad C D rd e s) ∨ GateBad C D rd e s ∨ MiddlesBad C D rd e s)
        ∨ ∃ ic : ℕ × α, ic.1 < C.maxLeaves
          ∧ (UndBad C D rd L ic.1 ic.2 e s ∨ ReadBad C D rd L ic.1 ic.2 e s)
    have hsub : {xs : Fin T → FreeMonoid α | ¬ RunEnds (applyProbe C fun z => (rd z).cut)
        (Level5EndsWell G.M (fun q => ¬ G.Good q) D C rd) initialState (List.ofFn xs)}
        ⊆ {xs | AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
            (List.ofFn xs)}
          ∪ {xs | RunEnds (applyProbe C fun z => (rd z).cut) Q initialState (List.ofFn xs)} := by
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
      rcases runEnds_or (Q := Level5EndsWell G.M (fun q => ¬ G.Good q) D C rd) _ _ hcl with h | h
      · exact absurd h hxs
      refine runEnds_mono (fun e s ⟨hc, hn⟩ => ?_) _ _ h
      have := bad_of_clean G D C rd hω hSq hθg0 hA.θe_ge hA.φe_nonneg hA.θgs_le
        (le_trans (by have := hA.θ_nonneg; positivity) hA.θgpt_ge) hA.θgpt_le hA.exc_gt hlen hc hn
      rcases this with h1 | h1 | h1 | h1 | ⟨i, hi, cc, h1⟩
      · exact .inl (.inl (.inl h1))
      · exact .inl (.inl (.inr h1))
      · exact .inl (.inr (.inl h1))
      · exact .inl (.inr (.inr h1))
      · exact .inr ⟨(i, cc), hi, h1⟩
    refine (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add le_rfl ?_))
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
  have hp₀ := prefixMax_pos D C.walkStart
  have hpmax := le_prefixMax D C.walkStart
  have hE := noiseEvent_le (θgs := kn.θgs) G read hmeas hind hlaw hA.θ_nonneg D C 0 L hlen hp₀
    hpmax hA.l_nonneg hA.φe_nonneg hSq.φ_pos.le hA.θe_ge hA.θg_ge (by simpa using hA.θgpt_ge)
  have hcard : ((classSet (Fintype.card σ + 0) : Finset (DTree α)).card : ℝ)
      ≤ classCount (α := α) σ 0 := by
    unfold classCount
    exact_mod_cast classSet_card (Fintype.card σ + 0)
  have hD := pi_wrong_le G C hA.Lmax_ge hmeas hlaw hind hA.κ_nonneg hA.ε_nonneg D hlen
  have hM : Measurable (M.indicator (1 : Ω → ENNReal)) :=
    measurable_one.indicator (measurableSet_toMeasurable _ _)
  calc ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunEnds (applyProbe C fun z => (read z ω).cut)
          (Level5EndsWell G.M (fun q => ¬ G.Good q) D C (read · ω))
          initialState (List.ofFn xs)} ∂μ
      ≤ ∫⁻ ω, (M.indicator 1 ω + g ω + c) ∂μ := lintegral_mono hpt
    _ = μ M + ∫⁻ ω, g ω ∂μ + c := by
        rw [lintegral_add_right _ measurable_const, lintegral_add_left hM,
          lintegral_indicator_one (measurableSet_toMeasurable _ _), lintegral_const, measure_univ,
          mul_one]
    _ ≤ (edgeTail σ G.θ D L C kn + startTail σ G.θ D C kn + middlesTail σ G.θ D L C kn)
        + wrongTerm (α := α) G.κ G.ε L C + testLevels (α := α) C := by
        refine add_le_add (add_le_add ?_ ?_) le_rfl
        · rw [measure_toMeasurable]
          refine hE.trans ?_
          unfold edgeTail startTail middlesTail edgeMapCount
          simp only [Nat.cast_zero, add_zero] at hcard ⊢
          refine add_le_add (add_le_add ?_ ?_) ?_ <;> refine ENNReal.ofReal_le_ofReal ?_
          · gcongr ?_ * _ * _
          · gcongr ?_ * _
          · simp only [mul_one]
            gcongr ?_ * _ * _
        · unfold wrongTerm
          refine hD.trans (ENNReal.ofReal_le_ofReal (le_of_eq ?_))
          ring
    _ = roundBound σ G.κ G.ε G.θ D L C kn := by unfold roundBound; rfl

end Model
theorem level5_round_risk_holds : Level5RoundRisk := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ P C
  exact level5_model ⟨M, O.readDist, θ, O.κ, O.ε, O.readDist_nonneg, O.rarer_side_le⟩ O.read
    O.measurable O.reads_by_state O.indep P C

end OrthoDFA
