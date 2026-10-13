import OrthoDFA.Proofs.LoopE2Pre
import OrthoDFA.Proofs.Level4Correct
import OrthoDFA.Proofs.BandWide

/-!
# A round's level 4 on one noise

Level 4 reads, at probes avoiding the clustering's prefixes, only strings the clustering did not.
Where the clustering's reads are re-noised it reads a fresh family on probes from `D` conditioned
to avoid them, so its guarantee holds there (`level4_bad_exists`), and the actual run is the same
run. Transferred back to `D` it ends well, and off a collision it does so with the chance it has on
fresh noise (`loop_level4_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

instance : Countable State :=
  Function.Injective.countable (f := fun B : State => (B.nsuff, B.npref, B.k, B.lo, B.hi, B.sc,
    B.scd)) fun a b h => by
    cases a; cases b
    simp only [Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
    subst h1 h2 h3 h4 h5 h6 h7
    rfl

/-- `O` with a rate no string has set to zero. -/
def Oracle.clamped (O : Oracle μ (FreeMonoid α)) : Oracle μ (FreeMonoid α) :=
  { O with
    ηIn := max O.ηIn 0
    ηOut := max O.ηOut 0
    noise_mean_in := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_pos hw] at h
      rw [O.noise_mean_in w hw, max_eq_left h]
    noise_mean_out := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_neg hw] at h
      rw [O.noise_mean_out w hw, max_eq_left h] }

omit [IsProbabilityMeasure μ] in
lemma popLaw_null {ν : Measure (FreeMonoid α)} {A : Set (FreeMonoid α)} (hA : ν A = 0)
    (h1 : (1 : FreeMonoid α) ∉ A) : popLaw ν A = 0 := by
  unfold popLaw
  split_ifs
  · rw [Measure.dirac_apply' _ (measurableSet_of_countable A), Set.indicator_of_notMem h1]
  · rw [Measure.smul_apply, hA, smul_zero]

lemma renoise_mq_of_mem (O : Oracle μ (FreeMonoid α)) {R : Set (FreeMonoid α)}
    {w : FreeMonoid α} (hw : w ∈ R) (p : Ω × Ω) : (O.renoise R).mq w p = O.mq w p.2 := by
  classical
  simp only [Oracle.mq, Oracle.label, Oracle.renoise, renoiseNoise, if_pos hw]

omit [DecidableEq α] in
lemma unionLog_nonneg {σ : Type*} [Fintype σ] [Nonempty α] (S : ℕ) {δ : ℝ} (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) : 0 ≤ unionLog (α := α) σ S δ := by
  unfold unionLog
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  refine mul_nonneg (by positivity) (Real.log_nonneg ?_)
  rw [le_div_iff₀ hδ]
  nlinarith

/-- The strings level 4 reads at a probe whose first `k` letters avoid `K₀`. -/
def offReads (k : ℕ) (K₀ F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, KeyOff k ↑K₀ z ∧ ∃ v ∈ F, w = z * v}

open Classical in
/-- Of them, those at strings at most `N'` long. -/
def l4Zone (N' k : ℕ) (K₀ F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter (· ∈ offReads k K₀ F)

lemma avoidLaw_len (D : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k : ℕ) (K₀ : Finset (FreeMonoid α)) : ∀ᵐ x ∂(avoidLaw D k K₀), x.toList.length ≤ L := by
  refine ae_iff.2 (popLaw_null (measure_mono_null (fun x hx => hx)
    (Measure.restrict_le_self.absolutelyContinuous (ae_iff.1 hlen))) ?_)
  simp

section Bad

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

include hL in
/-- At a state and family, level 4's bad draws: where it fails at re-noised clustering reads.
Its chance is at most `δ`, it is fixed by the answers at the strings level 4 reads off the
clustering's prefixes, and off it the actual round meets its claim. -/
theorem level4_bad_exists (S : ℕ) (εd δ crossLimit θ y m : ℝ)
    (hmax : settings.level4.maxLeaves = Fintype.card σ + S + 3)
    (hk : 1 ≤ settings.level4.walkStart) (hrec : 1 ≤ settings.level4.recordsToMove)
    (hdis : settings.level4.disagreementRate = εd / 2) (hη1 : O.η ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hy : 0 < y) (hcross : 0 < crossLimit) (hθ0 : 0 ≤ θ)
    (hθ : 384 * (P.L + 1) * (Fintype.card σ + S + 1) * θ ≤ 1)
    (hε : (P.L + 1) * (Fintype.card σ + S + 1) * crossLimit
      + 3 * unionLog (α := α) σ S δ * (2 * prefixMax P.D settings.level4.walkStart) ≤ y)
    (hm1 : m ≤ 1 / 2) (hmε : m ≤ εd / 2) (hmc : (P.L + 1) * m ≤ y / 2)
    (hlevel4 : ∀ (P' : PrefixDistribution α), P'.L = P.L → ∀ (O' : TriOracle (μ.prod μ) M) (θ : ℝ),
      let n : ℝ := Fintype.card σ + S
      let A : ℝ := Fintype.card α
      let bad := fun q => ¬ O'.Good θ q
      0 ≤ θ → 384 * (P'.L + 1) * (n + 1) * θ ≤ 1 →
      0 < O'.κ → 4 * O'.κ * (8 * (n + 3) * (P'.L + 1) * Real.sqrt (A * y / 3) + 1) ≤ y →
      0 ≤ O'.ε → (P'.L + 1) * (n + 1) * O'.ε
        + 3 * unionLog (α := α) σ S δ * prefixMax P'.D settings.level4.walkStart ≤ y →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.level4.probeBudget => P'.D)
          {draws | let result := indecisiveLStarLevel4 settings.level4 (O'.read · ω) draws
            ¬ EndsWell M bad P'.D settings.level4.disagreementRate
              (searchingProbes (O'.read · ω) settings.level4.walkStart result.hypothesis)
              settings.level4.walkStart
              (harvestLaw P'.D (fun z => (O'.read z ω).cut) settings.level4.walkStart)
              y result.ending} ∂(μ.prod μ)
        ≤ ENNReal.ofReal δ)
    (B : State) (F K₀ : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
      BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hK₀ : K₀.card * prefixMax P.D settings.level4.walkStart ≤ m) :
    ∃ Bd : Set (Ω × (Fin settings.level4.probeBudget → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd ≤ ENNReal.ofReal δ
      ∧ (∀ u ω ω', (∀ w ∈ l4Zone (max (P.L + settings.level4.maxLeaves)
          (probesLen (probesOf settings P.D K₀ u) + 1 + settings.level4.maxLeaves))
          settings.level4.walkStart K₀ F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, u) ∈ Bd → (ω', u) ∈ Bd)
      ∧ ∀ x₀ ω u, (ω, u) ∉ Bd →
        Level4Spec M O P θ εd (y / 2) settings.level4.walkStart
          (mkRound settings O P.D x₀ B F K₀ (ω, u)) := by
  classical
  set C := settings.level4 with hCdef
  set k := C.walkStart
  set T := C.probeBudget
  have hC : 2 ≤ C.maxLeaves := by rw [hmax]; omega
  -- the band's `κ`
  set n : ℝ := (Fintype.card σ : ℝ) + S
  set A : ℝ := (Fintype.card α : ℝ)
  set W : ℝ := 8 * (n + 3) * (P.L + 1) * Real.sqrt (A * y / 3) + 1
  have hW : 0 < W := by positivity
  set κ₀ : ℝ := y / (4 * W)
  have hκ₀ : 0 < κ₀ := by positivity
  have hκ₀W : 4 * κ₀ * W ≤ y := by
    simp only [κ₀]
    rw [show 4 * (y / (4 * W)) * W = y by field_simp]
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
  have hb := hband F.card hcard (1 - Oc.ηIn) Oc.ηOut 0 κ₀ hηIn.1 hηIn.2 hηOut.1 hηOut.2 hκ₀.le
  let fam : SuffixFamily μ α σ := ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, 0, κ₀, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (offReads k K₀ F)ᶜ
  let fam' := fam.renoise R
  -- the conditioned probes
  set D' := avoidLaw P.D k K₀
  set m' := P.D.real (avoiding k ↑K₀)ᶜ
  have hm' : m' ≤ m := (real_compl_avoiding_le P.D k K₀).trans hK₀
  have hm'0 : 0 ≤ m' := measureReal_nonneg
  have hD'len := avoidLaw_len P.D P.length_le k K₀
  let P' : PrefixDistribution α := ⟨D', inferInstance, P.L, hD'len⟩
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hpm : prefixMax D' k ≤ 2 * prefixMax P.D k := by
    refine (prefixMax_avoidLaw_le P.D k K₀ (by linarith)).trans ?_
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hU := unionLog_nonneg (α := α) (σ := σ) S hδ hδ1
  have hε' : (P'.L + 1) * (n + 1) * fam'.oracle.ε
      + 3 * unionLog (α := α) σ S δ * prefixMax P'.D k ≤ y := by
    refine le_trans ?_ hε
    show (P.L + 1) * (n + 1) * crossLimit + 3 * unionLog (α := α) σ S δ * prefixMax D' k ≤ _
    have := mul_le_mul_of_nonneg_left hpm (by positivity : (0 : ℝ) ≤ 3 * unionLog (α := α) σ S δ)
    simp only [n] at this ⊢
    linarith
  have h4 := hlevel4 P' rfl fam'.oracle θ hθ0 hθ hκ₀ hκ₀W hcross.le hε'
  set Sf := l4Fail fam' C D' θ y
  have hSm : MeasurableSet Sf := measurableSet_l4Fail fam' C hC D' θ y
  have hS : ((μ.prod μ).prod (Measure.pi fun _ : Fin T => D')) Sf ≤ ENNReal.ofReal δ := by
    rw [Measure.prod_apply hSm]
    exact h4
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
    exact hω₀.trans hS
  · -- the bad draws read only the zone
    intro u ω ω' h hω
    have hread : ∀ z : FreeMonoid α, z.toList.length ≤ max (P.L + C.maxLeaves)
        (probesLen (probesOf settings P.D K₀ u) + 1 + C.maxLeaves) →
        fam'.read (ω, ω₀) z = fam'.read (ω', ω₀) z := by
      intro z hz
      unfold SuffixFamily.read familyRead voteCount
      congr 2
      refine Finset.filter_congr fun v hv => ?_
      by_cases hzv : z * v ∈ R
      · show (Oc.renoise R).mq (z * v) (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v) (ω', ω₀) = 1
        rw [renoise_mq_of_mem Oc hzv, renoise_mq_of_mem Oc hzv]
      · show (Oc.renoise R).mq (z * v) (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v) (ω', ω₀) = 1
        rw [renoise_mq_of_notMem Oc hzv, renoise_mq_of_notMem Oc hzv]
        refine h _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z, v), Finset.mem_product.2
          ⟨mem_shortStrings.2 hz, hv⟩, rfl⟩, not_not.1 hzv⟩)
    have := endsWell_congr_reads C hC M (fun q => ¬ fam'.oracle.Good θ q) D' hD'len y
      (probesOf settings P.D K₀ u) hread
    exact fun hE => hω (this.2 hE)
  · -- off the bad draws, the actual round meets its claim
    intro x₀ ω u hnot
    have hE : EndsWell M (fun q => ¬ fam'.oracle.Good θ q) D' C.disagreementRate
        (searchingProbes (fun z => fam'.read (ω, ω₀) z) k
          (indecisiveLStarLevel4 C (fun z => fam'.read (ω, ω₀) z)
            (probesOf settings P.D K₀ u)).hypothesis) k
        (harvestLaw D' (fun z => (fam'.read (ω, ω₀) z).cut) k) y
        (indecisiveLStarLevel4 C (fun z => fam'.read (ω, ω₀) z)
          (probesOf settings P.D K₀ u)).ending := by
      by_contra hc
      exact hnot hc
    set read : FreeMonoid α → ARU := fun z => familyRead O.mq F B.lo (B.hi + 1) z ω
    have hoff : ∀ z, KeyOff k ↑K₀ z → read z = fam'.read (ω, ω₀) z := fun z hz =>
      (renoise_read_of_notMem fam R (fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩) (ω, ω₀)).symm
    have hres := level4_congr_off C hrec (probesOf settings P.D K₀ u)
      (fun i => drawFrom_avoids P.D hk K₀ (u i)) hoff
    have hmc' : (P.L + 1) * m' ≤ (1 - m') * y := by
      have := mul_le_mul_of_nonneg_left hm' (by positivity : (0 : ℝ) ≤ P.L + 1)
      nlinarith
    have hq := quarterEndsWell_of_endsWell P.D P.length_le M (fun q => ¬ fam'.oracle.Good θ q)
      (k := k) (K₀ := ↑K₀) (read := read) (read' := fun z => fam'.read (ω, ω₀) z)
      (fun z hz => by dsimp only; rw [hoff z hz])
      (indecisiveLStarLevel4 C read (probesOf settings P.D K₀ u)).hypothesis (εd / 2) y hy.le
      (indecisiveLStarLevel4 C read (probesOf settings P.D K₀ u)).ending (by linarith) hmc'
      (by rw [hres]; rw [← hdis]; exact hE)
    have hq' := hq.mono (ε' := εd) (y' := y / 2) (by linarith) (by nlinarith)
    have hbad : (fun q => ¬ fam'.oracle.Good θ q)
        = fun q => ¬ familyReadDist O M F B.lo (B.hi + 1) q .undecided ≤ 3 / 2 * θ := by
      funext q
      rw [SuffixFamily.renoise_good]
      rfl
    rw [hbad] at hq'
    exact hq'

end Bad

end OrthoDFA
