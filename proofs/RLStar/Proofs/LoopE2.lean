import RLStar.Proofs.LoopE2Pre
import RLStar.Proofs.BandKappa
import RLStar.Proofs.HeavyTrie

/-!
# A round's lazy round on one noise

The lazy round reads, at probes avoiding the clustering's prefixes, only strings the clustering
did not. Where the clustering's reads are re-noised it reads a fresh family on probes from `D`
conditioned to avoid them, so `lazyFail_le` holds there wherever the round's trie holds every
heavy prefix (`lazy_round_bad_exists`), and the actual run is the same run. Transferred back to
`D` it ends well, and off a collision it does so with the chance it has on fresh noise
(`loop_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

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

omit [DecidableEq α] in
lemma renoise_mq_of_mem (O : Oracle μ (FreeMonoid α)) {R : Set (FreeMonoid α)}
    {w : FreeMonoid α} (hw : w ∈ R) (p : Ω × Ω) : (O.renoise R).mq w p = O.mq w p.2 := by
  classical
  simp only [Oracle.mq, Oracle.label, Oracle.renoise, renoiseNoise, if_pos hw]

/-- The strings a lazy round from `CP`'s trie reads through `F` at probes avoiding `CP`'s first `k`
letters. -/
def lazyOffReads (k : ℕ) (CP F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, LazyKeyOff k CP z ∧ ∃ v ∈ F, w = z * v}

lemma avoidLaw_len (D : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k : ℕ) (K₀ : Finset (FreeMonoid α)) : ∀ᵐ x ∂(avoidLaw D k K₀), x.toList.length ≤ L := by
  refine ae_iff.2 (popLaw_null (measure_mono_null (fun x hx => hx)
    (Measure.restrict_le_self.absolutelyContinuous (ae_iff.1 hlen))) ?_)
  simp

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma LooseEndsWell.add_mass {σ : Type*} {M : _root_.DFA α σ} {U : σ → ℝ} {A sl m₀ pc y : ℝ}
    {D : Measure (FreeMonoid α)} {ε : ℝ} {dis : Set (FreeMonoid α)}
    {law : Random.Level3Site → Measure (FreeMonoid α)} {m m' : ℝ} (hm : m' ≤ m) (hm1 : m ≤ 1)
    {e : Random.LazyEnd α}
    (h : LooseEndsWell M U A sl (m₀ + m') pc ((1 - m') * y) D (ε + m') dis law e) :
    LooseEndsWell M U A sl (m₀ + m) pc ((1 - m) * y) D (ε + m) dis law e := by
  cases e with
  | consistent => exact le_trans h (by linarith)
  | harvest s _ =>
    obtain ⟨h1, h2, h3⟩ := h
    refine ⟨fun T hT => (h1 T hT).trans (by linarith), h2, ?_⟩
    have h0 : 0 ≤ (law s).real Set.univ := measureReal_nonneg
    rcases le_or_gt 0 y with hy | hy
    · nlinarith
    · nlinarith
  | failed => exact (show False from h).elim

section Bad

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

include hL in
/-- At a state, family and prefixes `CP`, the round's bad draws: where the trie draws miss a heavy
prefix, or the lazy round fails at re-noised clustering reads. Its chance is at most
`lazyFail_le`'s bound and `heavy_miss_le`'s, it is fixed by the answers at the strings the lazy
round reads off `CP`, and off it the actual round meets its claim. -/
theorem lazy_round_bad_exists (m crossLimit θ tw : ℝ) (Nr : ℕ)
    (hk : 1 ≤ settings.level3.walkStart)
    (hKw : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : LazyPremises σ α settings.level3 settings.epochs settings.blocks (2 * θ))
    (hnh : settings.listSize = 1) (hθ : 0 < θ)
    (hpm : prefixMax P.D settings.level3.walkStart ≤ θ)
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize P.L
      (Fintype.card σ) (Fintype.card α) ≤ Nr)
    (htw : 0 < tw) (hm1 : m ≤ 1 / 2)
    (B : State) (F CP : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      RarerBounded N B.lo (B.hi + 1) p r crossLimit)
    (hK₀ : (CP.image fun p => prefixOf p settings.level3.walkStart).card
      * prefixMax P.D settings.level3.walkStart ≤ m) :
    ∃ Bd : Set (Ω × settings.Uniforms), MeasurableSet Bd
      ∧ (μ.prod settings.uniformLaw) Bd
        ≤ ENNReal.ofReal (lazyRoundBound σ settings.level3 settings.epochs settings.blocks
          settings.blockSize crossLimit Nr P.L tw
          + settings.level3.walkStart / θ * (1 - θ) ^ settings.heavyDraws)
      ∧ (∀ v ω ω', (∀ w ∈ lazyZone (max (P.L + settings.level3.probeBudget + 2)
          (lazyProbesLen (probesOf settings P.D
            (CP.image fun p => prefixOf p settings.level3.walkStart) v)
            + 1 + settings.level3.probeBudget + 2)) settings.level3.walkStart CP F,
          (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (lazySlope σ crossLimit settings.level3 P.L)
          (lazySlack σ α settings.level3 settings.epochs (2 * θ)) (2 * tw + m) θ
          ((1 - m) * (settings.level3.harvestRate - tw))
          (settings.level3.disagreementRate + tw + m) settings.level3.walkStart
          (mkRound settings O P.D B F (CP.image fun p => prefixOf p settings.level3.walkStart)
            CP Sf (ω, v)) := by
  classical
  set C := settings.level3 with hCdef
  set k := C.walkStart
  set Nh := settings.heavyDraws
  set K₀ := CP.image fun p => prefixOf p k
  set Oc := O.clamped
  have hprem' := hprem
  obtain ⟨hy0, hp8, hyd, hδ0, hδ1, hd0, hd1, -, hE, -⟩ := hprem
  have hθ1 : θ ≤ 1 := by linarith
  have hηIn : 0 ≤ 1 - Oc.ηIn ∧ 1 - Oc.ηIn ≤ 1 := by
    have h1 : O.ηIn ≤ O.η := le_max_left _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; have := le_max_left O.ηIn 0
              exact sub_nonneg.2 (max_le (by linarith) (by norm_num)),
      by simp only [Oc, Oracle.clamped]; linarith [le_max_right O.ηIn 0]⟩
  have hηOut : 0 ≤ Oc.ηOut ∧ Oc.ηOut ≤ 1 := by
    have h1 : O.ηOut ≤ O.η := le_max_right _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; exact le_max_right _ _,
      by simp only [Oc, Oracle.clamped]; exact max_le (by linarith) (by norm_num)⟩
  have hb := hband F.card hcard (1 - Oc.ηIn) Oc.ηOut hηIn.1 hηIn.2 hηOut.1 hηOut.2
  let fam : SuffixFamily μ α σ :=
    ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (lazyOffReads k CP F)ᶜ
  let fam' := fam.renoise R
  set D' := avoidLaw P.D k K₀
  set A := avoiding k (↑K₀ : Set (FreeMonoid α))
  set m' := P.D.real Aᶜ
  have hm' : m' ≤ m := (real_compl_avoiding_le P.D k K₀).trans hK₀
  have hm'0 : 0 ≤ m' := measureReal_nonneg
  have hA : P.D.real A = 1 - m' := by
    rw [show m' = P.D.real Aᶜ from rfl, probReal_compl_eq_one_sub (measurableSet_of_countable A)]
    ring
  have hApos : 0 < P.D.real A := by rw [hA]; linarith
  have hA0 : P.D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hD'eq : D' = (P.D.restrict A Set.univ)⁻¹ • P.D.restrict A := by
    show popLaw (P.D.restrict A) = _
    unfold popLaw
    rw [if_neg (not_or.2 ⟨hA0, measure_ne_top _ _⟩)]
  have hAv : ∀ᵐ x ∂D', x ∈ A := by
    rw [hD'eq]
    exact Measure.smul_absolutelyContinuous.ae_le (ae_restrict_mem (measurableSet_of_countable A))
  have hD'len : ∀ᵐ x ∂D', k ≤ x.toList.length ∧ x.toList.length ≤ P.L := by
    have hac : D' ≪ P.D := by
      rw [hD'eq]
      exact Measure.smul_absolutelyContinuous.trans Measure.restrict_le_self.absolutelyContinuous
    exact hac.ae_le (hKw.and P.length_le)
  have hlenA : ∀ᵐ x ∂D', x.toList.length ≤ P.L ∧ k ≤ x.toList.length ∧ Avoids k ↑K₀ x :=
    (hD'len.and hAv).mono fun x hx => ⟨hx.1.2, hx.1.1, hx.2⟩
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hp₁ : ∀ u, D'.real {x | prefixOf x k = u} ≤ 2 * θ := by
    have hpm' : prefixMax D' k ≤ 2 * θ := by
      refine (prefixMax_avoidLaw_le P.D k K₀ (by linarith)).trans ?_
      rw [div_le_iff₀ (by linarith)]
      nlinarith
    exact fun u => (le_prefixMax D' k u).trans hpm'
  have hcyl : ∀ c : FreeMonoid α, D'.real {x | prefixOf x c.toList.length = c}
      ≤ 2 * P.D.real {x | prefixOf x c.toList.length = c} := by
    intro c
    rw [show D' = popLaw (P.D.restrict A) from rfl, popLaw_real hA0 (measure_ne_top _ _),
      measureReal_restrict_apply_univ, measureReal_restrict_apply (measurableSet_of_countable _),
      div_le_iff₀ hApos]
    have h1 := measureReal_mono (μ := P.D)
      (Set.inter_subset_left : {x | prefixOf x c.toList.length = c} ∩ A ⊆ _) (measure_ne_top _ _)
    have h2 : (0 : ℝ) ≤ P.D.real {x | prefixOf x c.toList.length = c} := measureReal_nonneg
    rw [hA]
    nlinarith
  -- the trie draws missing a heavy prefix
  set HM : Set (Fin Nh → unitInterval) := {u | ∃ c : FreeMonoid α, c.toList.length < k
    ∧ θ < P.D.real {x | prefixOf x c.toList.length = c}
    ∧ c ∉ Random.prefixTrie k (heavySet P.D u)} with hHM
  have hHMm : MeasurableSet HM := measurableSet_heavyMiss P.D k Nh θ
  set X₀ : (Fin Nh → unitInterval) → Finset (FreeMonoid α) :=
    fun u => Random.prefixTrie k (CP ∪ heavySet P.D u) with hX₀
  have hX₀CP : ∀ u, Random.prefixTrie k CP ⊆ X₀ u :=
    fun u => prefixTrie_mono Finset.subset_union_left
  have hlightD : ∀ u ∉ HM, ∀ c : FreeMonoid α, c.toList.length < k → c ∉ X₀ u →
      P.D.real {x | prefixOf x c.toList.length = c} ≤ θ := by
    intro u hu c hc hcX
    by_contra hlt
    exact hu ⟨c, hc, lt_of_not_ge hlt,
      fun hm => hcX (prefixTrie_mono Finset.subset_union_right hm)⟩
  have hlight : ∀ u ∉ HM, ∀ c : FreeMonoid α, c.toList.length < k → c ∉ X₀ u →
      D'.real {x | prefixOf x c.toList.length = c} ≤ 2 * θ :=
    fun u hu c hc hcX => (hcyl c).trans (by linarith [hlightD u hu c hc hcX])
  set δb := lazyRoundBound σ C settings.epochs settings.blocks settings.blockSize crossLimit Nr
    P.L tw with hδb
  have hδb0 : 0 ≤ δb := by
    have h1 : 0 ≤ 1 - C.harvestRate / 4 := by linarith
    have h2 : (0 : ℝ) ≤ settings.epochs * settings.blocks
        * (1 - C.harvestRate / 4) ^ settings.blockSize := by positivity
    have h3 : (0 : ℝ) ≤ crossLimit * Nr := by positivity
    have h4 : (0 : ℝ) ≤ (P.L + 1) * (Fintype.card σ + 2) * crossLimit / tw := by positivity
    simp only [hδb, lazyRoundBound]
    linarith
  set Sf : Finset (FreeMonoid α) → Set _ := fun X => lazyFail fam' settings X D'
    (lazySlope σ crossLimit C P.L) (lazySlack σ α C settings.epochs (max (2 * θ) (2 * θ)))
    (2 * tw) (max (2 * θ) (2 * θ)) (C.harvestRate - tw) tw with hSf
  have h4 : ∀ u ∉ HM, ((μ.prod μ).prod (lazyDrawLaw settings D')) (Sf (X₀ u))
      ≤ ENNReal.ofReal δb := fun u hu =>
    lazyFail_le fam' hcross.le settings hnh (X₀ u) (trieOK_prefixTrie k _) D' (2 * θ) (2 * θ) tw
      Nr P.L hD'len hp₁ (hlight u hu) hprem' hNr htw
  have hSm : ∀ X, MeasurableSet (Sf X) := fun X => measurableSet_lazyFail fam' settings X D' _ _ _
    _ _ tw
  choose ω₀ hω₀ using fun X => exists_noise_le (μ := μ) (lazyDrawLaw settings D') (Sf X) (hSm X)
  set ν := lazyLaw settings.epochs C.probeBudget settings.blocks settings.blockSize
  set ρ := Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)
  have hpr : MeasurePreserving (lazyProbes (E := settings.epochs) (P := C.probeBudget)
      (J := settings.blocks) (b := settings.blockSize) D') ν (lazyDrawLaw settings D') :=
    measurePreserving_pi _ _ fun _ =>
      (measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩).prod
        (measurePreserving_pi _ _ fun _ => measurePreserving_pi _ _ fun _ =>
          ⟨measurable_drawFrom _, map_drawFrom _⟩)
  have hmp : MeasurePreserving (fun q : Ω × LazyUniforms settings.epochs C.probeBudget
      settings.blocks settings.blockSize => (q.1, lazyProbes D' q.2)) (μ.prod ν)
      (μ.prod (lazyDrawLaw settings D')) :=
    (MeasurePreserving.id μ).prod hpr
  have hSω : ∀ X, MeasurableSet {q : Ω × (Fin settings.epochs →
      Random.EpochDraws α C.probeBudget settings.blocks settings.blockSize) |
      ((q.1, ω₀ X), q.2) ∈ Sf X} := fun X =>
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) (hSm X)
  -- the draws, null under `ν`, with a probe shorter than `k`
  set Sh : Set (Ω × settings.Uniforms) :=
    {q | ¬ LazyDrawsAll (fun x => k ≤ x.toList.length) (probesOf settings P.D K₀ q.2)}
  have hShm : MeasurableSet Sh :=
    ((measurable_probesOf settings P.D K₀).comp measurable_snd)
      (Set.to_countable {d | ¬ LazyDrawsAll (fun x => k ≤ x.toList.length) d}).measurableSet
  have hSh0 : (μ.prod settings.uniformLaw) Sh = 0 := by
    have hk' : ∀ᵐ x ∂D', k ≤ x.toList.length := hD'len.mono fun x hx => hx.1
    have hall : ∀ᵐ d ∂(lazyDrawLaw settings D'),
        LazyDrawsAll (fun x => k ≤ x.toList.length) d := by
      set ρ' := (Measure.pi fun _ : Fin C.probeBudget => D').prod
        (Measure.pi fun _ : Fin settings.blocks => Measure.pi fun _ : Fin settings.blockSize => D')
      have he : ∀ e, MeasurePreserving (fun d : Fin settings.epochs →
          Random.EpochDraws α C.probeBudget settings.blocks settings.blockSize => d e)
          (lazyDrawLaw settings D') ρ' := fun e => measurePreserving_eval (fun _ => ρ') e
      have h1 : ∀ i, MeasurePreserving (fun p : Random.EpochDraws α C.probeBudget settings.blocks
          settings.blockSize => p.1 i) ρ' D' := fun i =>
        (measurePreserving_eval (fun _ => D') i).comp measurePreserving_fst
      have h2 : ∀ j t, MeasurePreserving (fun p : Random.EpochDraws α C.probeBudget settings.blocks
          settings.blockSize => p.2 j t) ρ' D' := fun j t =>
        (measurePreserving_eval (fun _ => D') t).comp
          ((measurePreserving_eval (fun _ => Measure.pi fun _ => D') j).comp measurePreserving_snd)
      unfold LazyDrawsAll
      refine ae_all_iff.2 fun e => (ae_all_iff.2 fun i => ?_).and
        (ae_all_iff.2 fun j => ae_all_iff.2 fun t => ?_)
      · exact ((h1 i).comp (he e)).quasiMeasurePreserving.ae hk'
      · exact ((h2 j t).comp (he e)).quasiMeasurePreserving.ae hk'
    have hU : MeasurePreserving (fun q : Ω × settings.Uniforms => q.2.2)
        (μ.prod settings.uniformLaw) ν :=
      (measurePreserving_snd (μ := ρ) (ν := ν)).comp (measurePreserving_snd (μ := μ)
        (ν := settings.uniformLaw))
    exact ae_iff.1 (hU.quasiMeasurePreserving.ae (hpr.quasiMeasurePreserving.ae hall))
  have hdraws : ∀ v, LazyDrawsAll (fun x => k ≤ x.toList.length) (probesOf settings P.D K₀ v) →
      LazyDrawsAll (fun x => k ≤ x.toList.length ∧ Avoids k ↑K₀ x)
        (probesOf settings P.D K₀ v) := fun v hv e =>
    ⟨fun i => ⟨(hv e).1 i, drawFrom_avoids P.D hk K₀ _⟩,
      fun j t => ⟨(hv e).2 j t, drawFrom_avoids P.D hk K₀ _⟩⟩
  have hoff : ∀ z, LazyKeyOff k CP z → ∀ p : Ω × Ω, fam'.read p z = fam.read p.1 z :=
    fun z hz => renoise_read_of_notMem fam R fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩
  set Bf : Set (Ω × settings.Uniforms) := {q | q.2.1 ∉ HM
    ∧ ((q.1, ω₀ (X₀ q.2.1)), probesOf settings P.D K₀ q.2) ∈ Sf (X₀ q.2.1)} with hBf
  have hBfm : MeasurableSet Bf := by
    have e : Bf = ⋃ X : Finset (FreeMonoid α), ({q : Ω × settings.Uniforms | X₀ q.2.1 = X}
        ∩ {q | q.2.1 ∉ HM}) ∩ {q | ((q.1, ω₀ X), probesOf settings P.D K₀ q.2) ∈ Sf X} := by
      ext q
      simp only [hBf, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff]
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨_, ⟨rfl, h1⟩, h2⟩
      · rintro ⟨X, ⟨rfl, h1⟩, h2⟩; exact ⟨h1, h2⟩
    rw [e]
    refine MeasurableSet.iUnion fun X => MeasurableSet.inter (MeasurableSet.inter ?_ ?_) ?_
    · have : {q : Ω × settings.Uniforms | X₀ q.2.1 = X} = (fun q : Ω × settings.Uniforms =>
          heavyDrawsOf settings P.D q.2) ⁻¹'
          {v | Random.prefixTrie k (CP ∪ Finset.univ.image v) = X} := rfl
      rw [this]
      exact ((measurable_heavyDrawsOf settings P.D).comp measurable_snd)
        (Set.to_countable _).measurableSet
    · exact (measurable_fst.comp measurable_snd) hHMm.compl
    · exact ((measurable_fst.prodMk measurable_const).prodMk
        ((measurable_probesOf settings P.D K₀).comp measurable_snd)) (hSm X)
  have hHMq : MeasurableSet {q : Ω × settings.Uniforms | q.2.1 ∈ HM} :=
    (measurable_fst.comp measurable_snd) hHMm
  refine ⟨({q | q.2.1 ∈ HM} ∪ Bf) ∪ Sh, (hHMq.union hBfm).union hShm, ?_, ?_, ?_⟩
  · -- the chance
    have hHMb : (μ.prod settings.uniformLaw) {q | q.2.1 ∈ HM}
        ≤ ENNReal.ofReal (k / θ * (1 - θ) ^ Nh) := by
      have e : {q : Ω × settings.Uniforms | q.2.1 ∈ HM} = Set.univ ×ˢ (HM ×ˢ Set.univ) := by
        ext q; simp
      rw [e, Measure.prod_prod, show settings.uniformLaw = ρ.prod ν from rfl, Measure.prod_prod,
        measure_univ, measure_univ, one_mul, mul_one, ← ofReal_measureReal]
      exact ENNReal.ofReal_le_ofReal (heavy_miss_le P.D k Nh hθ hθ1)
    have hBfb : (μ.prod settings.uniformLaw) Bf ≤ ENNReal.ofReal δb := by
      set e : (Fin Nh → unitInterval) × (Ω × LazyUniforms settings.epochs C.probeBudget
          settings.blocks settings.blockSize) → Ω × settings.Uniforms :=
        fun t => (t.2.1, (t.1, t.2.2))
      have he : MeasurePreserving e (ρ.prod (μ.prod ν)) (μ.prod settings.uniformLaw) := by
        have h1 : MeasurePreserving (fun t : (Fin Nh → unitInterval) × (Ω × LazyUniforms
            settings.epochs C.probeBudget settings.blocks settings.blockSize) =>
            ((t.1, t.2.1), t.2.2)) (ρ.prod (μ.prod ν)) ((ρ.prod μ).prod ν) :=
          (measurePreserving_prodAssoc ρ μ ν).symm
        have h2 : MeasurePreserving (fun t : ((Fin Nh → unitInterval) × Ω) × LazyUniforms
            settings.epochs C.probeBudget settings.blocks settings.blockSize =>
            ((t.1.2, t.1.1), t.2)) ((ρ.prod μ).prod ν) ((μ.prod ρ).prod ν) :=
          (Measure.measurePreserving_swap (μ := ρ) (ν := μ)).prod (MeasurePreserving.id ν)
        have h3 : MeasurePreserving (fun t : (Ω × (Fin Nh → unitInterval)) × LazyUniforms
            settings.epochs C.probeBudget settings.blocks settings.blockSize =>
            (t.1.1, (t.1.2, t.2))) ((μ.prod ρ).prod ν) (μ.prod (ρ.prod ν)) :=
          measurePreserving_prodAssoc μ ρ ν
        exact h3.comp (h2.comp h1)
      rw [← he.measure_preimage hBfm.nullMeasurableSet, Measure.prod_apply (he.measurable hBfm)]
      calc ∫⁻ u, (μ.prod ν) (Prod.mk u ⁻¹' (e ⁻¹' Bf)) ∂ρ ≤ ∫⁻ _u, ENNReal.ofReal δb ∂ρ := by
            refine lintegral_mono fun u => ?_
            by_cases hu : u ∈ HM
            · have : Prod.mk u ⁻¹' (e ⁻¹' Bf) = ∅ := by
                ext q; simp [e, hBf, hu]
              rw [this, measure_empty]
              exact bot_le
            have hsub : Prod.mk u ⁻¹' (e ⁻¹' Bf) = (fun q : Ω × LazyUniforms settings.epochs
                C.probeBudget settings.blocks settings.blockSize => (q.1, lazyProbes D' q.2)) ⁻¹'
                {q | ((q.1, ω₀ (X₀ u)), q.2) ∈ Sf (X₀ u)} := by
              ext q
              simp only [Set.mem_preimage, e, hBf, Set.mem_ofPred_eq, hu, not_false_eq_true,
                true_and]
              exact Iff.rfl
            rw [hsub, hmp.measure_preimage (hSω _).nullMeasurableSet]
            exact (hω₀ _).trans (h4 u hu)
        _ = ENNReal.ofReal δb := by rw [lintegral_const, measure_univ, mul_one]
    have hK0 : (0 : ℝ) ≤ k / θ * (1 - θ) ^ Nh := by
      have : 0 ≤ 1 - θ := by linarith
      positivity
    calc (μ.prod settings.uniformLaw) (({q | q.2.1 ∈ HM} ∪ Bf) ∪ Sh)
        ≤ (μ.prod settings.uniformLaw) ({q | q.2.1 ∈ HM} ∪ Bf)
          + (μ.prod settings.uniformLaw) Sh := measure_union_le _ _
      _ ≤ ((μ.prod settings.uniformLaw) {q | q.2.1 ∈ HM} + (μ.prod settings.uniformLaw) Bf)
          + 0 := by rw [hSh0]; exact add_le_add (measure_union_le _ _) le_rfl
      _ ≤ (ENNReal.ofReal (k / θ * (1 - θ) ^ Nh) + ENNReal.ofReal δb) + 0 := by gcongr
      _ = ENNReal.ofReal (δb + k / θ * (1 - θ) ^ Nh) := by
        rw [add_zero, add_comm, ENNReal.ofReal_add hδb0 hK0]
  · -- the bad draws read only the zone
    intro v ω ω' h hω
    rcases hω with (hω | hω) | hω
    · exact Or.inl (Or.inl hω)
    · by_cases hsh : (ω', v) ∈ Sh
      · exact Or.inr hsh
      refine Or.inl (Or.inr ⟨hω.1, ?_⟩)
      have hω2 := hω.2
      change ((ω, ω₀ (X₀ v.1)), probesOf settings P.D K₀ v) ∈ Sf (X₀ v.1) at hω2
      show ((ω', ω₀ (X₀ v.1)), probesOf settings P.D K₀ v) ∈ Sf (X₀ v.1)
      refine lazyFail_congr_up fam' settings CP (X₀ v.1) (hX₀CP v.1) D' hlenA _ _ _ _ _ tw
        (probesOf settings P.D K₀ v) (hdraws v (not_not.1 hsh)) (fun z hz hzl => ?_) hω2
      rw [hoff z hz, hoff z hz]
      refine familyRead_congr Oc (Z := {z}) (fun w hw => ?_) (Finset.mem_singleton_self z)
      obtain ⟨⟨z', v'⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hz', hv'⟩ := Finset.mem_product.1 hzv
      rw [Finset.mem_singleton] at hz'
      subst hz'
      exact h _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z', v'),
        Finset.mem_product.2 ⟨mem_shortStrings.2 hzl, hv'⟩, rfl⟩, z', hz, hzl, v', hv', rfl⟩)
    · exact Or.inr hω
  · intro Sf' ω v hnot
    simp only [Set.mem_union, Set.mem_ofPred_eq, not_or] at hnot
    obtain ⟨⟨hHMv, hBfv⟩, hlong⟩ := hnot
    set X := X₀ v.1
    have hgood : ((ω, ω₀ X), probesOf settings P.D K₀ v) ∉ Sf X := fun h' => hBfv ⟨hHMv, h'⟩
    simp only [hSf, lazyFail, Set.mem_ofPred_eq, not_not] at hgood
    obtain ⟨hE', hpk⟩ := hgood
    have hd := hdraws v (not_not.1 hlong)
    set r := lazyRun fam' settings X ((ω, ω₀ X), probesOf settings P.D K₀ v)
    have hres : (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result = r := by
      have hreads := reads_lazyRoundQ_keyOff_of_sub C k settings.listSize CP X (hX₀CP v.1)
        (probesOf settings P.D K₀ v) (fun z => fam'.read (ω, ω₀ X) z) rfl hd
      exact (Query.eval_congr (fun z => fam'.read (ω, ω₀ X) z) _ fun z hz =>
        (hoff z (hreads z hz).1 (ω, ω₀ X)).symm).1
    have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
      rw [SuffixFamily.renoise_readDist]
      rfl
    obtain ⟨hsub, hmid⟩ := lazyRoundQ_expanded_mids (fun z => fam'.read (ω, ω₀ X) z) C k
      settings.listSize X (probesOf settings P.D K₀ v)
    have hshape := lazyRoundShape_holds (fun z => fam'.read (ω, ω₀ X) z) C k settings.listSize
      settings.epochs settings.blocks settings.blockSize X (probesOf settings P.D K₀ v)
      (by omega) (trieOK_prefixTrie k _)
    have hXr : Random.prefixTrie k CP ⊆ r.expanded := (hX₀CP v.1).trans hsub
    set rd : FreeMonoid α → ARU := fun z => familyRead O.mq F B.lo (B.hi + 1) z ω
    have hrd : ∀ z, LazyKeyOff k CP z → fam'.read (ω, ω₀ X) z = rd z :=
      fun z hz => hoff z hz (ω, ω₀ X)
    set S := Random.lazyDisagree rd k r.expanded r.hypothesis
    set S' := Random.lazyDisagree (fun z => fam'.read (ω, ω₀ X) z) k r.expanded r.hypothesis
    have hSS : D'.real S' = D'.real S := by
      refine measureReal_congr ?_
      filter_upwards [hlenA] with x hx
      exact propext (mem_lazyDisagree_congr hXr hmid hx.2.1 hx.2.2 fun z hz _ => hrd z hz)
    have hlaw : (fun site => Random.lazyHarvestLaw (fun z => fam'.read (ω, ω₀ X) z) D' k
        r.expanded r.hypothesis site) = fun site =>
        Random.lazyHarvestLaw rd D' k r.expanded r.hypothesis site :=
      funext fun site => lazyHarvestLaw_congr_keyOff hXr hmid hlenA (fun z hz _ => hrd z hz) site
    have hU : (fun q => fam'.readDist q .undecided)
        = fun q => familyReadDist O M F B.lo (B.hi + 1) q .undecided := by rw [hdist]
    rw [max_self] at hE'
    have h1 := hE'.of_real (D := D') (S := S) (ε := C.disagreementRate + tw)
      (fun h' => hSS ▸ h') hlaw
    rw [hU] at h1
    have hslope : 0 ≤ lazySlope σ crossLimit C P.L := by
      unfold lazySlope; positivity
    have hslack : 0 ≤ lazySlack σ α C settings.epochs (2 * θ) := by
      unfold lazySlack
      refine div_nonneg (mul_nonneg (by linarith) (Real.log_nonneg ?_)) hy0.le
      have hE1 : (1 : ℝ) ≤ settings.epochs := by
        have : (0 : ℝ) ≤ 2 * (C.walkStart : ℝ) / C.harvestRate := by positivity
        linarith
      rw [le_div_iff₀ hδ0]
      have hKt : (1 : ℝ) ≤ ((1 + (Fintype.card σ + 1) ^ 3 * (Fintype.card α + 1))
          ^ Fintype.card σ * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2)
            * (Fintype.card α + 1)) : ℕ) := by
        exact_mod_cast Nat.one_le_iff_ne_zero.2 (by positivity)
      have hn : (1 : ℝ) ≤ 8 * (Fintype.card σ + 1) := by
        have : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
        linarith
      have hEE : (1 : ℝ) ≤ settings.epochs * (settings.epochs + 1) := by nlinarith
      calc 1 * C.δ ≤ 1 := by linarith
        _ ≤ 8 * (Fintype.card σ + 1) * _ * (settings.epochs * (settings.epochs + 1)) := by
          have := mul_le_mul hn hKt zero_le_one (by linarith)
          nlinarith
    have hpc : ∀ site t, (Random.lazyHarvestLaw rd P.D k r.expanded r.hypothesis site).real {t}
        ≤ θ := fun site t =>
      (lazyHarvestLaw_atom_le rd P.D k r.expanded r.hypothesis site θ (fun c hc => hshape.1.1 c hc)
        hKw (fun c hc hcX => hlightD v.1 hHMv c hc fun hcX0 => hcX (hsub hcX0)) t).trans
        (max_le le_rfl hpm)
    have h2 := looseEndsWell_of_avoiding P.D M (fun q => familyReadDist O M F B.lo (B.hi + 1) q
        .undecided) hslope hslack (by linarith : (0 : ℝ) ≤ 2 * tw) (read := rd) (read' := rd) k
      r.expanded r.hypothesis (fun _ _ _ => rfl) S S (fun _ _ h => h) (by linarith) hpc r.ending h1
    have h3 := h2.add_mass hm' (by linarith)
    refine ⟨?_, ?_⟩
    · rw [hres]
      exact h3
    · rw [hres]
      exact hpk

end Bad

end OrthoDFA
