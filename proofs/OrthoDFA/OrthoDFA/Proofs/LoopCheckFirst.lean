import OrthoDFA.Proofs.LoopCheckBad
import OrthoDFA.Proofs.LoopColl
import OrthoDFA.LoopCheck

/-!
# The checked loop's first round, given what it inherits

What reaches the first round's level 3 is its clustering's state, family and keys (`stage0`),
fixed by the answers at the strings the clustering read. Where every suffix the clustering read is
`Ls` long and every prefix at least `k` and shorter than `Ls` (`LensOK`), the round reads none of
those strings off the clustering's prefixes, so it fails to end consistent with at most the chance
its re-noised round has (`check_round0_le`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

lemma avoidLaw_real_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (hm : D.real (avoiding k ↑K₀)ᶜ ≤ 1 / 2) (S : Set (FreeMonoid α)) :
    (avoidLaw D k K₀).real S ≤ 2 * D.real S := by
  set A := avoiding (α := α) k ↑K₀
  have hA : D.real A = 1 - D.real Aᶜ := by
    rw [probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have h0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  rw [avoidLaw_eq, popLaw_real h0 (measure_ne_top _ _), measureReal_restrict_apply_univ,
    measureReal_restrict_apply (measurableSet_of_countable _), div_le_iff₀ hApos]
  have h1 : D.real (S ∩ A) ≤ D.real S := measureReal_mono Set.inter_subset_left (measure_ne_top _ _)
  have h2 : 0 ≤ D.real S := measureReal_nonneg
  nlinarith

section Stage

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (h : LoopHist α) (a : ℕ)

open Classical in
/-- What reaches the round's level 3 where its clustering returned: its state, family, and the
first `avoidDepth` and `a` letters of the prefixes its clustering read. -/
def stage0 (c : ClusterDraws Ω α K) :
    Option (State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :=
  if Returns settings O D Dsf h c then
    some (roundState settings O D Dsf h c,
      clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h c)
        (roundState settings O D Dsf h c),
      roundKeys settings D Dsf h c,
      clusterKeys (settings.schedule (loopPopulations K h.pops.length))
        (loopPopulations K h.pops.length) a (clusterRun D Dsf h c))
  else none

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma stage0_eq_some (c : ClusterDraws Ω α K) (B : State) (F K₀ Kc : Finset (FreeMonoid α)) :
    stage0 settings O D Dsf h a c = some (B, F, K₀, Kc) ↔
      Returns settings O D Dsf h c ∧ roundState settings O D Dsf h c = B
        ∧ clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h c) B = F
        ∧ roundKeys settings D Dsf h c = K₀
        ∧ clusterKeys (settings.schedule (loopPopulations K h.pops.length))
          (loopPopulations K h.pops.length) a (clusterRun D Dsf h c) = Kc := by
  unfold stage0
  split_ifs with hr
  · simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨rfl, rfl, rfl, rfl⟩
      exact ⟨hr, rfl, rfl, rfl, rfl⟩
    · rintro ⟨-, rfl, rfl, rfl, rfl⟩
      exact ⟨rfl, rfl, rfl, rfl⟩
  · simp only [false_iff]
    exact fun h' => hr h'.1

lemma measurableSet_returns' :
    MeasurableSet {c : ClusterDraws Ω α K | Returns settings O D Dsf h c} := by
  classical
  have hset : {c : ClusterDraws Ω α K | Returns settings O D Dsf h c}
      = ⋃ B ∈ settings.schedule (loopPopulations K h.pops.length), clusterRun D Dsf h ⁻¹'
        ret O.mq (loopPopulations K h.pops.length) none
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B := by
    ext c
    simp [Returns]
  rw [hset]
  exact Finset.measurableSet_biUnion _ fun B _ =>
    measurable_clusterRun D Dsf h (measurableSet_ret O _ none _ _ _ _ B)

lemma measurableSet_stage0
    (y : Option (State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α))) :
    MeasurableSet {c : ClusterDraws Ω α K | stage0 settings O D Dsf h a c = y} := by
  classical
  rcases y with _ | ⟨B, F, K₀, Kc⟩
  · have : {c : ClusterDraws Ω α K | stage0 settings O D Dsf h a c = none}
        = {c | Returns settings O D Dsf h c}ᶜ := by
      ext c
      simp only [stage0, Set.mem_ofPred_eq, Set.mem_compl_iff]
      split_ifs with hr <;> simp [hr]
    rw [this]
    exact (measurableSet_returns' settings O D Dsf h).compl
  · have hKc := measurableSet_roundKeys (Ω := Ω) { settings with avoidDepth := a } D Dsf h Kc
    have : {c : ClusterDraws Ω α K | stage0 settings O D Dsf h a c = some (B, F, K₀, Kc)}
        = {c | Returns settings O D Dsf h c}
          ∩ {c | roundState settings O D Dsf h c = B
            ∧ clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h c)
              (roundState settings O D Dsf h c) = F}
          ∩ {c | roundKeys settings D Dsf h c = K₀}
          ∩ {c | roundKeys { settings with avoidDepth := a } D Dsf h c = Kc} := by
      ext c
      simp only [Set.mem_ofPred_eq, stage0_eq_some, Set.mem_inter_iff]
      constructor
      · rintro ⟨h1, h2, h3, h4, h5⟩
        exact ⟨⟨⟨h1, h2, by rw [h2]; exact h3⟩, h4⟩, h5⟩
      · rintro ⟨⟨⟨h1, h2, h3⟩, h4⟩, h5⟩
        exact ⟨h1, h2, by rw [← h2]; exact h3, h4, h5⟩
    rw [this]
    exact (((measurableSet_returns' settings O D Dsf h).inter
      (measurableSet_roundPick settings O D Dsf h B F)).inter
      (measurableSet_roundKeys settings D Dsf h K₀)).inter hKc

/-- The answers at the strings the clustering read fix what reaches level 3. -/
lemma stage0_stop (c : ClusterRest α K) (ω ω' : Ω)
    (t : State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α))
    (hs : stage0 settings O D Dsf h a (ω, c) = some t)
    (hw : ∀ w ∈ schedReads (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (clusterRun D Dsf h (ω, c)),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stage0 settings O D Dsf h a (ω', c) = some t := by
  classical
  obtain ⟨B, F, K₀, Kc⟩ := t
  obtain ⟨h1, h2, h3, h4, h5⟩ := (stage0_eq_some settings O D Dsf h a _ B F K₀ Kc).1 hs
  set sched := settings.schedule (loopPopulations K h.pops.length)
  have hret' : Returns settings O D Dsf h (ω', c) := by
    obtain ⟨B', hB', hret⟩ := h1
    exact ⟨B', hB', (ret_congr O sched (loopPopulations K h.pops.length) none
      (Finset.mem_insert_self _ _) _ _ _ _ hB' _ hw).1 hret⟩
  have hrs : roundState settings O D Dsf h (ω', c) = B := by
    rw [← h2]
    exact (roundState_congr settings O D Dsf h c hw).symm
  have hBs : B ∈ sched := by
    rw [← h2]
    exact (Classical.epsilon_spec h1).1
  refine (stage0_eq_some settings O D Dsf h a _ B F K₀ Kc).2 ⟨hret', hrs, ?_, h4, h5⟩
  rw [← h3]
  exact (clusterAt_congr_sched O sched (loopPopulations K h.pops.length) hBs _ hw).symm

end Stage

section Lens

variable (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)

/-- Every suffix the clustering read is `Ls` long, and every prefix at least `k` and shorter than
`Ls`. -/
def LensOK (k Ls : ℕ) (c : ClusterRest α K) : Prop :=
  (∀ v ∈ suffsOf settings D Dsf h c, v.toList.length = Ls)
    ∧ ∀ p ∈ prefsOf settings D Dsf h c, k ≤ p.toList.length ∧ p.toList.length < Ls

/-- Every uniform the clustering draws by is below `1`. -/
def RestLt (c : ClusterRest α K) : Prop :=
  (∀ n, (c.1 n : ℝ) < 1) ∧ (∀ j n, (c.2.1 j n : ℝ) < 1) ∧ ∀ j n, (c.2.2.1 j n : ℝ) < 1

omit [DecidableEq α] in
lemma restLt_ae : (restLaw α K) {c | ¬ RestLt c} = 0 := by
  have hev : ∀ n, MeasurePreserving (fun f : ℕ → unitInterval => f n)
      (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun n =>
    ⟨measurable_pi_apply n, Measure.infinitePi_map_eval _ n⟩
  have hev2 : ∀ j n, MeasurePreserving (fun f : Option (Fin K) → ℕ → unitInterval => f j n)
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun j n =>
    (hev n).comp (measurePreserving_eval _ j)
  have h1 : MeasurableSet {u : unitInterval | ¬ (u : ℝ) < 1} :=
    (measurableSet_lt continuous_subtype_val.measurable measurable_const).compl
  have hnull : ∀ {f : ClusterRest α K → unitInterval},
      MeasurePreserving f (restLaw α K) volume → (restLaw α K) {x | ¬ (f x : ℝ) < 1} = 0 :=
    fun hf => by
      rw [show {x | ¬ (_ : ℝ) < 1} = _ ⁻¹' {u : unitInterval | ¬ (u : ℝ) < 1} from rfl,
        hf.measure_preimage h1.nullMeasurableSet, volume_eq_one_unit]
  have hset : {c : ClusterRest α K | ¬ RestLt c}
      ⊆ ⋃ n : ℕ, ({c | ¬ ((c.1 n : unitInterval) : ℝ) < 1}
        ∪ ⋃ j : Option (Fin K), ({c | ¬ ((c.2.1 j n : unitInterval) : ℝ) < 1}
          ∪ {c | ¬ ((c.2.2.1 j n : unitInterval) : ℝ) < 1})) := by
    intro c hc
    simp only [RestLt, not_and_or, not_forall] at hc
    simp only [Set.mem_iUnion, Set.mem_union, Set.mem_ofPred_eq]
    rcases hc with ⟨n, hn⟩ | ⟨j, n, hn⟩ | ⟨j, n, hn⟩
    · exact ⟨n, Or.inl hn⟩
    · exact ⟨n, Or.inr ⟨j, Or.inl hn⟩⟩
    · exact ⟨n, Or.inr ⟨j, Or.inr hn⟩⟩
  refine measure_mono_null hset (measure_iUnion_null fun n =>
    measure_union_null ?_ (measure_iUnion_null fun j => measure_union_null ?_ ?_))
  · exact hnull ((hev n).comp measurePreserving_fst)
  · exact hnull ((hev2 j n).comp (measurePreserving_fst.comp measurePreserving_snd))
  · exact hnull ((hev2 j n).comp (measurePreserving_fst.comp
      (measurePreserving_snd.comp measurePreserving_snd)))

/-- Where every uniform is below `1`, every string the clustering reads has a length its
distribution gives mass. -/
theorem lensOK_of_restLt [IsProbabilityMeasure D] [IsProbabilityMeasure Dsf] (k Ls : ℕ)
    (hD : D {q | ¬ (k ≤ q.toList.length ∧ q.toList.length < Ls)} = 0)
    (hpops : ∀ ν ∈ h.pops, IsProbabilityMeasure ν
      ∧ ν {q | ¬ (k ≤ q.toList.length ∧ q.toList.length < Ls)} = 0)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpm : D.real ↑h.prefs ≤ 1 / 2) (hsm : Dsf.real ↑h.suffs ≤ 1 / 2)
    (c : ClusterRest α K) (hlt : RestLt c) : LensOK settings D Dsf h k Ls c := by
  classical
  set A := {q : FreeMonoid α | k ≤ q.toList.length ∧ q.toList.length < Ls}
  have hprob : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) D h.pops j) :=
    isProbabilityMeasure_loopDistributions D h.pops fun ν hν => (hpops ν hν).1
  have hshort : ∀ j ∈ loopPopulations K h.pops.length, loopDistributions D h.pops j Aᶜ = 0 := by
    intro j hj
    cases j with
    | none => exact hD
    | some i' =>
      have hi' : i'.val < h.pops.length := lt_of_some_mem_loopPops hj
      show (h.pops.getD i' (Measure.dirac 1)) _ = 0
      rw [List.getD_eq_getElem (hn := hi')]
      exact (hpops _ (List.getElem_mem _)).2
  refine ⟨fun v hv => ?_, fun p hp => ?_⟩
  · obtain ⟨n, rfl⟩ := mem_suffsOf_draw settings D Dsf hv
    obtain ⟨-, hpos⟩ := drawFrom_offLaw (compl_ne_zero hsm) (hlt.1 n)
    have := mem_of_pos (A := {q : FreeMonoid α | q.toList.length = Ls})
      (by refine measure_mono_null (fun x hx => ?_) hDsf; simpa using hx) hpos
    exact this
  · obtain ⟨j, hj, n, hpn⟩ := mem_prefsOf_draw settings D Dsf hp
    obtain ⟨u, hu, rfl⟩ : ∃ u : unitInterval, (u : ℝ) < 1 ∧
        p = drawFrom (popOff D h.pops j h.prefs) u := by
      rcases hpn with h' | h'
      · exact ⟨_, hlt.2.1 j n, h'⟩
      · exact ⟨_, hlt.2.2 j n, h'⟩
    unfold popOff
    split_ifs with hdeg
    · obtain ⟨-, hpos⟩ := drawFrom_offLaw (compl_ne_zero hpm) hu
      exact mem_of_pos (A := A) hD hpos
    · obtain ⟨-, hpos⟩ := drawFrom_offLaw hdeg hu
      exact mem_of_pos (A := A) (hshort j hj) hpos

end Lens

/-- Where the lengths are as `LensOK` says, the strings the clustering read and the checked round's
zone through a family of its suffixes are disjoint. -/
theorem disjoint_sched_checkZone (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (h : LoopHist α) (k a Ls N' : ℕ) (c : ClusterRest α K) (ω₀ : Ω)
    (hl : LensOK settings D Dsf h k Ls c) (F K₀ Kc : Finset (FreeMonoid α))
    (hF : F ⊆ suffsOf settings D Dsf h c)
    (hK₀ : K₀ = (prefsOf settings D Dsf h c).image fun p => prefixOf p k)
    (hKc : Kc = (prefsOf settings D Dsf h c).image fun p => prefixOf p a) :
    Disjoint (schedReads (settings.schedule (loopPopulations K h.pops.length))
        (loopPopulations K h.pops.length)
        ((ω₀, runOf D Dsf h c) : Run Ω (FreeMonoid α) (Option (Fin K))))
      (checkZone N' k a K₀ Kc F) := by
  classical
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨-, z, hz, v, hv, rfl⟩ := Finset.mem_filter.1 hw'
  have hvl := hl.1 v (hF hv)
  obtain ⟨p, hp, hpw⟩ := mem_schedReads_shape hw
  have hpl := hl.2 p hp
  rcases hpw with he | ⟨v', hv', he⟩
  · have := congrArg (fun t : FreeMonoid α => t.toList.length) he
    simp only [length_mul] at this
    omega
  · obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he (hvl.trans (hl.1 v' hv').symm)
    subst hzp
    rcases hz with hk | hk | hk
    · exact hk.2 (by rw [hK₀]; exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨_, hp, rfl⟩))
    · exact hk.2 (by rw [hKc]; exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨_, hp, rfl⟩))
    · omega

instance (T n : ℕ) : IsProbabilityMeasure (checkRoundLaw α K T n) := by
  unfold checkRoundLaw; infer_instance

omit [IsProbabilityMeasure μ] in
lemma measurableSet_sets (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (h : LoopHist α) (Φ : Finset (FreeMonoid α) → Finset (FreeMonoid α) → Prop) :
    MeasurableSet {c : ClusterRest α K | Φ (prefsOf settings D Dsf h c) (suffsOf settings D Dsf h c)} := by
  classical
  have : {c : ClusterRest α K | Φ (prefsOf settings D Dsf h c) (suffsOf settings D Dsf h c)}
      = ⋃ Q : Finset (FreeMonoid α) × Finset (FreeMonoid α), ⋃ (_ : Φ Q.1 Q.2),
        {c | prefsOf settings D Dsf h c = Q.1} ∩ {c | suffsOf settings D Dsf h c = Q.2} := by
    ext c
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, exists_prop, Prod.exists]
    exact ⟨fun hc => ⟨_, _, hc, rfl, rfl⟩, fun ⟨_, _, hc, e1, e2⟩ => e1 ▸ e2 ▸ hc⟩
  rw [this]
  exact MeasurableSet.iUnion fun Q => MeasurableSet.iUnion fun _ =>
    (measurableSet_prefsOf settings D Dsf h Q.1).inter (measurableSet_suffsOf settings D Dsf h Q.2)

section Round

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)

/-- A stage the round's claim is proved at: its state is scheduled, its family suffix-free and
smaller than the state's count, every state read undecided at most `u` of the time, and at most
`NK` keys. -/
def GoodStage (sched : Finset State) (u : ℝ) (NK : ℕ)
    (t : State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α)) : Prop :=
  t.1 ∈ sched ∧ SuffixFree t.2.1 ∧ t.2.1.card + 1 ≤ t.1.k
    ∧ (∀ z : FreeMonoid α,
      familyReadDist O M t.2.1 t.1.lo (t.1.hi + 1) (M.eval z.toList) .undecided ≤ u)
    ∧ t.2.2.1.card ≤ NK ∧ t.2.2.2.card ≤ NK

include hL in
/-- The first round, where its clustering returned at a good stage and read strings of the
lengths `LensOK` says, fails to end consistent with at most the chance its re-noised round has. -/
theorem check_round0_le (settings : CheckLoopSettings K) (P Pc : PrefixDistribution α)
    (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (h₀ : LoopHist α) (hK : 0 < K)
    (Ls NK h : ℕ) (crossLimit p₀ u ph ηc : ℝ)
    (hz : settings.avoidDepth = settings.level3.walkStart)
    (hk : 1 ≤ settings.level3.walkStart) (ha : 1 ≤ settings.checkDepth)
    (hak : settings.checkDepth + Fintype.card σ + 1 ≤ settings.level3.walkStart)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
      0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ → BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hNKk : NK * prefixMax P.D settings.level3.walkStart ≤ 1 / 2)
    (hp₀ : 2 * prefixMax P.D settings.level3.walkStart ≤ p₀)
    (hNKa : NK * prefixMax Pc.D settings.checkDepth ≤ 1 / 2)
    (hph : 2 * prefixMax Pc.D h ≤ ph) :
    let C := settings.level3
    let Kk := settings.check
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let Kt : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (P.L + 1))
    let η := p₀ * Real.log (16 * (nQ + 1) * Kt / δ)
    let Ms : ℝ := (nQ + 1) * nα ^ nQ
    let θ₀ : ℝ := (Pc.L + 1) * Ms * u
    let q : ℝ := 2 * θ₀ + ηc
    0 < δ → 0 < y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ u → 0 < ph → h ≤ Pc.L →
    0 ≤ ηc → 8 * θ₀ ≤ 1 → q ≤ 1 → 0 < Kk.rate → Kk.rate < 1 → 0 < Kk.δ →
    crossLimit * C.probeBudget * ((P.L + 1) * nQ + Kk.size * ((Pc.L + 1) * nQ)
      + (Kk.size + 1) * ((nQ + 1) ^ 2 * nα)) ≤ δ / 8 →
    A * u + η / y < 1 →
    (checkLoopMeasure μ α K settings.level3.probeBudget settings.check.size)
      {p | ∃ t, stage0 settings.toLoopSettings O P.D Dsf h₀ settings.checkDepth
            (p.1, (p.2 ⟨0, hK⟩).1) = some t
          ∧ GoodStage M O (settings.schedule (loopPopulations K h₀.pops.length)) u NK t
          ∧ LensOK settings.toLoopSettings P.D Dsf h₀ settings.level3.walkStart Ls (p.2 ⟨0, hK⟩).1
          ∧ (checkRound settings O P.D Dsf Pc.D h₀ (p.1, p.2 ⟨0, hK⟩)).result.ending
            ≠ .consistent}
      ≤ ENNReal.ofReal (9 * δ / 8 + 2 * (∑ j ∈ Finset.range S, Kk.silentBase q ^ Kk.size
            / Kk.level j + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph)))) := by
  intro C Kk y ε δ nQ nα S Kt A η Ms θ₀ q hδ hy0 hyρ hρ1 hP hu0 hph0 hhL hηc h8 hq1 hr0 hr1 hKδ
    hw hAu
  classical
  set T := C.probeBudget
  set n := Kk.size
  set k := C.walkStart
  set a := settings.checkDepth
  set ls := settings.toLoopSettings
  set pops := loopPopulations K h₀.pops.length
  set sched := settings.schedule pops
  set r0 : Fin K := ⟨0, hK⟩
  set bound := 9 * δ / 8 + 2 * (∑ j ∈ Finset.range S, Kk.silentBase q ^ n / Kk.level j
    + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph)))
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set Y := Option (State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set X₂ := Fin T → unitInterval × (Fin n → unitInterval)
  set ν₂ : Measure X₂ := Measure.pi fun _ : Fin T => (volume : Measure unitInterval).prod
    (Measure.pi fun _ : Fin n => (volume : Measure unitInterval))
  have hσ : MeasurePreserving (id : CheckRoundDraws α K T n → CheckRoundDraws α K T n)
      (checkRoundLaw α K T n) ((restLaw α K).prod ν₂) := MeasurePreserving.id _
  have hpmk : 0 ≤ prefixMax P.D k := measureReal_nonneg.trans (le_prefixMax P.D k 1)
  have hpma : 0 ≤ prefixMax Pc.D a := measureReal_nonneg.trans (le_prefixMax Pc.D a 1)
  have hex : ∀ t, GoodStage M O sched u NK t → ∃ Bd : Set (Ω × X₂), MeasurableSet Bd
      ∧ (μ.prod ν₂) Bd ≤ ENNReal.ofReal bound
      ∧ (∀ v ω ω', (∀ w ∈ checkZone (drawsLen Kk (checkDrawsOf Kk P.D Pc.D k a t.2.2.1 t.2.2.2 v)
            + T + 1) k a t.2.2.1 t.2.2.2 t.2.1, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ ω v, (ω, v) ∉ Bd →
        (indecisiveLStarCheck C Kk (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z ω)
          (checkDrawsOf Kk P.D Pc.D k a t.2.2.1 t.2.2.2 v)).ending = .consistent := by
    intro t ht
    obtain ⟨hBs, hsf, hcard, hU, hK₀, hKc⟩ := ht
    have hm0 : P.D.real (avoiding k ↑t.2.2.1)ᶜ ≤ 1 / 2 :=
      (real_compl_avoiding_le P.D k _).trans (le_trans (by gcongr) hNKk)
    have hmc : Pc.D.real (avoiding a ↑t.2.2.2)ᶜ ≤ 1 / 2 :=
      (real_compl_avoiding_le Pc.D a _).trans (le_trans (by gcongr) hNKa)
    exact check_round_bad_exists M O hL C Kk P Pc a h crossLimit p₀ u ph ηc hk ha hak hη1 hcross
      t.1 t.2.1 t.2.2.1 t.2.2.2 hsf hcard (hband pops t.1 hBs) hU
      (fun v => (avoidLaw_real_le P.D k _ hm0 _).trans (by
        have := le_prefixMax P.D k v; linarith)) hδ hy0 hyρ hρ1 hP hu0
      (fun v => (avoidLaw_real_le Pc.D a _ hmc _).trans (by
        have := le_prefixMax Pc.D h v; linarith)) hph0 hhL hηc h8 hq1 hr0 hr1 hKδ hw hAu
  set Bad : Y → Set (Ω × X₂) := fun y => match y with
    | some t => if hg : GoodStage M O sched u NK t then Classical.choose (hex t hg) else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → CheckRoundDraws α K T n) → Finset (FreeMonoid α) := fun _ x =>
    schedReads sched pops ((ω₀, runOf P.D Dsf h₀ (x r0).1) : Run Ω (FreeMonoid α) (Option (Fin K)))
  set Z : Y → X₂ → Finset (FreeMonoid α) := fun y v => match y with
    | some t => checkZone (drawsLen Kk (checkDrawsOf Kk P.D Pc.D k a t.2.2.1 t.2.2.2 v) + T + 1)
        k a t.2.2.1 t.2.2.2 t.2.1
    | none => ∅
  set Dj : Y → Set (Fin K → CheckRoundDraws α K T n) := fun y => match y with
    | some t => {x | LensOK ls P.D Dsf h₀ k Ls (x r0).1
        ∧ t.2.1 ⊆ suffsOf ls P.D Dsf h₀ (x r0).1
        ∧ t.2.2.1 = (prefsOf ls P.D Dsf h₀ (x r0).1).image (fun p => prefixOf p k)
        ∧ t.2.2.2 = (prefsOf ls P.D Dsf h₀ (x r0).1).image (fun p => prefixOf p a)}
    | none => Set.univ
  have hx0 : Measurable fun x : Fin K → CheckRoundDraws α K T n => (x r0).1 :=
    measurable_fst.comp (measurable_pi_apply r0)
  have hmain := persist_stage_le O (checkRoundLaw α K T n) (restLaw α K) ν₂ id hσ id measurable_id
    (fun w => rfl) r0 (fun ω x => stage0 ls O P.D Dsf h₀ a (ω, (x r0).1))
    (fun ω x x' _ h1 => by
      change (x r0).1 = (x' r0).1 at h1
      simp only [h1])
    (fun y => (measurable_fst.prodMk (hx0.comp measurable_snd))
      (measurableSet_stage0 ls O P.D Dsf h₀ a y))
    W (fun y => ∃ t, y = some t)
    (fun y x ω ω' hG hs h' => by
      obtain ⟨t, rfl⟩ := hG
      exact stage0_stop ls O P.D Dsf h₀ a (x r0).1 ω ω' t hs h')
    Bad (fun y hy => by
      rcases y with _ | t
      · rfl
      · exact absurd ⟨t, rfl⟩ hy)
    (fun y => by
      rcases y with _ | t
      · exact MeasurableSet.empty
      · show MeasurableSet (if hg : GoodStage M O sched u NK t then Classical.choose (hex t hg) else ∅)
        split_ifs with hg
        · exact (Classical.choose_spec (hex t hg)).1
        · exact MeasurableSet.empty)
    Z (fun y v ω ω' h' hq => by
      rcases y with _ | t
      · exact hq
      · change (ω, v) ∈ (if hg : GoodStage M O sched u NK t then Classical.choose (hex t hg)
          else ∅) at hq
        show (ω', v) ∈ (if hg : GoodStage M O sched u NK t then Classical.choose (hex t hg) else ∅)
        split_ifs at hq ⊢ with hg
        · exact (Classical.choose_spec (hex t hg)).2.2.1 v ω ω' h' hq
        · exact hq)
    Dj (fun y => by
      rcases y with _ | ⟨B, F, K₀, Kc⟩
      · exact MeasurableSet.univ
      · show MeasurableSet {x : Fin K → CheckRoundDraws α K T n |
          LensOK ls P.D Dsf h₀ k Ls (x r0).1 ∧ F ⊆ suffsOf ls P.D Dsf h₀ (x r0).1
            ∧ K₀ = (prefsOf ls P.D Dsf h₀ (x r0).1).image (fun p => prefixOf p k)
            ∧ Kc = (prefsOf ls P.D Dsf h₀ (x r0).1).image (fun p => prefixOf p a)}
        exact hx0 (measurableSet_sets ls P.D Dsf h₀ fun Pr Sf =>
          ((∀ v ∈ Sf, v.toList.length = Ls) ∧ ∀ p ∈ Pr, k ≤ p.toList.length ∧ p.toList.length < Ls)
            ∧ F ⊆ Sf ∧ K₀ = Pr.image (fun p => prefixOf p k)
            ∧ Kc = Pr.image (fun p => prefixOf p a)))
    (fun y x hG hx => by
      obtain ⟨⟨B, F, K₀, Kc⟩, rfl⟩ := hG
      obtain ⟨hl, hF, hK₀, hKc⟩ := hx
      exact disjoint_sched_checkZone ls P.D Dsf h₀ k a Ls _ (x r0).1 ω₀ hl F K₀ Kc hF hK₀ hKc)
    (ENNReal.ofReal bound) (fun y => by
      rcases y with _ | t
      · show (μ.prod ν₂) ∅ ≤ _
        simp
      · show (μ.prod ν₂) (if hg : GoodStage M O sched u NK t then Classical.choose (hex t hg)
          else ∅) ≤ _
        split_ifs with hg
        · exact (Classical.choose_spec (hex t hg)).2.1
        · simp)
  refine (measure_mono ?_).trans hmain
  rintro p ⟨⟨B, F, K₀, Kc⟩, hst, hgood, hl, hnc⟩
  obtain ⟨h1, h2, h3, h4, h5⟩ := (stage0_eq_some ls O P.D Dsf h₀ a _ B F K₀ Kc).1 hst
  have hBs : B ∈ sched := hgood.1
  show (p.1, (p.2 r0).2) ∈ Bad (stage0 ls O P.D Dsf h₀ a (p.1, (p.2 r0).1))
    ∧ p.2 ∈ Dj (stage0 ls O P.D Dsf h₀ a (p.1, (p.2 r0).1))
  rw [hst]
  refine ⟨?_, hl, ?_, ?_, ?_⟩
  · show (p.1, (p.2 r0).2) ∈ (if hg : GoodStage M O sched u NK (B, F, K₀, Kc) then
      Classical.choose (hex _ hg) else ∅)
    rw [dif_pos hgood]
    by_contra hnb
    apply hnc
    have hres := (Classical.choose_spec (hex _ hgood)).2.2.2 p.1 (p.2 r0).2 hnb
    have e : (checkRound settings O P.D Dsf Pc.D h₀ (p.1, p.2 r0)).result
        = indecisiveLStarCheck C Kk (fun z => familyRead O.mq
            (clusterAt O.mq pops (clusterRun P.D Dsf h₀ (p.1, (p.2 r0).1))
              (roundState ls O P.D Dsf h₀ (p.1, (p.2 r0).1)))
            (roundState ls O P.D Dsf h₀ (p.1, (p.2 r0).1)).lo
            ((roundState ls O P.D Dsf h₀ (p.1, (p.2 r0).1)).hi + 1) z p.1)
          (checkDrawsOf Kk P.D Pc.D settings.avoidDepth a (roundKeys ls P.D Dsf h₀ (p.1, (p.2 r0).1))
            (clusterKeys sched pops a (clusterRun P.D Dsf h₀ (p.1, (p.2 r0).1))) (p.2 r0).2) := rfl
    rw [e, h2, h3, h4, h5, hz]
    exact hres
  · rw [← h3]
    exact clusterAt_subset_suffsOf ls O P.D Dsf h₀ p.1 (p.2 r0).1 (h2 ▸ hBs)
  · rw [← h4, roundKeys_eq_prefsOf, hz]
  · rw [← h5]
    rfl

end Round

end OrthoDFA
