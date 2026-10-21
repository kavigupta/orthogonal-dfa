import OrthoDFA.Proofs.LoopCheckRenoise
import OrthoDFA.Proofs.LoopE2

/-!
# The checked round on one noise

The checked round reads, at probes whose first `k` letters avoid the clustering's prefixes' and
check strings whose first `a` letters do, only strings `z v`, `v` in the family, with `z` off
those prefixes' first `k` or `a` letters or shorter than `k` (`OffC`), once its midfixes are at
most `|Q|` long and `a + |Q| + 1 ≤ k`. Re-noised everywhere else, the family is fresh, so the
round's claim (`Level4CheckRoundCorrect`) holds of it, and the round reads the same there. Where
every state reads undecided at most `u` of the time and `A u + η/y < 1`, a harvest puts all its
law on states read undecided at most `u` of the time and so cannot yield: the round ends
consistent but for that claim's chance (`check_round_bad_exists`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `z` starts off the keys `K₀` at `k` or `Kc` at `a`, or is shorter than `k`. -/
def OffC (k a : ℕ) (K₀ Kc : Finset (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  KeyOff k ↑K₀ z ∨ KeyOff a ↑Kc z ∨ z.toList.length < k

/-- The strings the checked round reads off the clustering's prefixes. -/
def offReadsC (k a : ℕ) (K₀ Kc F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, OffC k a K₀ Kc z ∧ ∃ v ∈ F, w = z * v}

open Classical in
/-- Of them, those at strings at most `N'` long followed by the family. -/
def checkZone (N' k a : ℕ) (K₀ Kc F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter (· ∈ offReadsC k a K₀ Kc F)

/-- The checked round's draws from its uniforms: probes from `D` avoiding `K₀` at `k`, check
strings from `Dc` avoiding `Kc` at `a`. -/
def checkDrawsOf {T : ℕ} (K : CheckSettings) (D Dc : Measure (FreeMonoid α)) (k a : ℕ)
    (K₀ Kc : Finset (FreeMonoid α)) (v : Fin T → unitInterval × (Fin K.size → unitInterval)) :
    Fin T → CheckDraw α K :=
  fun i => (drawFrom (avoidLaw D k K₀) (v i).1, fun j => drawFrom (avoidLaw Dc a Kc) ((v i).2 j))

theorem measurable_checkDrawsOf {T : ℕ} (K : CheckSettings) (D Dc : Measure (FreeMonoid α))
    (k a : ℕ) (K₀ Kc : Finset (FreeMonoid α)) :
    Measurable (checkDrawsOf (T := T) K D Dc k a K₀ Kc) := by
  refine measurable_pi_lambda _ fun i => ?_
  refine Measurable.prodMk ((measurable_drawFrom _).comp (measurable_fst.comp
    (measurable_pi_apply i))) (measurable_pi_lambda _ fun j => ?_)
  exact (measurable_drawFrom _).comp ((measurable_pi_apply j).comp
    (measurable_snd.comp (measurable_pi_apply i)))

theorem measurePreserving_checkDrawsOf {T : ℕ} (K : CheckSettings) (D Dc : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] [IsProbabilityMeasure Dc] (k a : ℕ) (K₀ Kc : Finset (FreeMonoid α)) :
    MeasurePreserving (checkDrawsOf (T := T) K D Dc k a K₀ Kc)
      (Measure.pi fun _ : Fin T => (volume : Measure unitInterval).prod
        (Measure.pi fun _ : Fin K.size => (volume : Measure unitInterval)))
      (Measure.pi fun _ : Fin T => (avoidLaw D k K₀).prod
        (Measure.pi fun _ : Fin K.size => avoidLaw Dc a Kc)) := by
  exact measurePreserving_pi (fun _ : Fin T => (volume : Measure unitInterval).prod
      (Measure.pi fun _ : Fin K.size => (volume : Measure unitInterval)))
    (fun _ : Fin T => (avoidLaw D k K₀).prod (Measure.pi fun _ : Fin K.size => avoidLaw Dc a Kc))
    (f := fun _ => Prod.map (drawFrom (avoidLaw D k K₀))
      (fun w j => drawFrom (avoidLaw Dc a Kc) (w j)))
    fun _ => (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving
      (drawFrom (avoidLaw D k K₀)) volume (avoidLaw D k K₀)).prod
      (measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩)

namespace SuffixFamily

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}
  (fam : SuffixFamily μ α σ)

/-- A family the band check holds to `κ = 0`, as level 4's oracle. -/
def epsOracle (hε : 0 ≤ fam.ε) (hκ : fam.κ = 0) :
    Eps.EpsReadOracle μ fam.M (fun q => readSide (fam.readDist q))
      (fun q => fam.readDist q .undecided) fam.ε where
  read := fun z ω => fam.read ω z
  measurable := fam.read_measurable
  indep := fam.read_indep
  undecided := fun z => fam.reads_by_state z .undecided
  wrong := fun z => by
    rw [fam.reads_by_state]
    have := fam.readDist_wrong hε (le_of_eq hκ.symm) (fam.M.eval z.toList)
    rwa [hκ, zero_mul, add_zero] at this

end SuffixFamily

omit [Fintype α] [DecidableEq α] in
theorem mids_short_of_midsShort {n : ℕ} :
    ∀ T : DTree α, T.MidsShort n → ∀ m ∈ mids T, m.toList.length ≤ n
  | .leaf, _ => by simp [mids]
  | .node m r a, ⟨h1, h2, h3⟩ => by
    classical
    intro m' hm
    simp only [mids, Finset.mem_insert, Finset.mem_union] at hm
    rcases hm with rfl | hm | hm
    · exact h1
    · exact mids_short_of_midsShort r h2 m' hm
    · exact mids_short_of_midsShort a h3 m' hm

omit [DecidableEq α] in
theorem not_endsWell_failed {σ H : Type*} (s : RoundOutcome α σ H) (he : s.ending = .failed) :
    ¬ EndsWell s := by
  unfold EndsWell
  rw [he]
  exact id

omit [DecidableEq α] in
/-- A harvest whose law sits on states read undecided at most `u` of the time, with
`A u + sl < 1`, does not yield. -/
theorem not_endsWell_harvest {σ H : Type*} (s : RoundOutcome α σ H) (h : H)
    (he : s.ending = .harvest h) {u : ℝ} (hu0 : 0 ≤ u)
    (hU : ∀ z : FreeMonoid α, s.undecided (s.target.eval z.toList) ≤ u)
    (hA : s.harvestSlope * u + s.harvestSlack < 1) (hy : 0 < s.yield) : ¬ EndsWell s := by
  intro hw
  unfold EndsWell at hw
  rw [he] at hw
  obtain ⟨hlaw, -, hyield⟩ := hw
  have h1 := hlaw u hu0
  have hall : {z : FreeMonoid α | s.undecided (s.target.eval z.toList) ≤ u} = Set.univ :=
    Set.eq_univ_of_forall hU
  rw [hall] at h1
  have h0 : 0 ≤ (s.harvestLaw h).real Set.univ := measureReal_nonneg
  unfold Yields at hyield
  nlinarith

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Whether the checked round over a family's reads ends somewhere is measurable. -/
theorem measurableSet_of_checkResult {σ : Type*} (fam : SuffixFamily μ α σ) (C : Level3Settings)
    (K : CheckSettings) (Φ : Level3Result α → Prop) :
    MeasurableSet {d : Ω × (Fin C.probeBudget → CheckDraw α K) |
      Φ (indecisiveLStarCheck C K (fun z => fam.read d.1 z) d.2)} := by
  classical
  refine measurableSet_prod_countable_right _ fun ds => ?_
  set A := readStrings (shortStrings (drawsLen K ds + C.probeBudget + 1)) fam.F
  refine measurableSet_of_reads fam.O A _ fun ω ω' h hω => ?_
  have e : indecisiveLStarCheck C K (fun z => fam.read ω z) ds
      = indecisiveLStarCheck C K (fun z => fam.read ω' z) ds :=
    checkResult_congr_short C K ds fun z hz => familyRead_congr fam.O h (mem_shortStrings.2 hz)
  show Φ _
  rw [← e]
  exact hω

section Bad

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)

include hL in
/-- At a state and family, the checked round's bad draws: where it fails to end consistent with
short midfixes at re-noised clustering reads. Its chance is at most `Level4CheckRoundCorrect`'s
bound, it is fixed by the answers at the strings the round reads off the clustering's prefixes,
and off it the actual round ends consistent. -/
theorem check_round_bad_exists (C : Level3Settings) (Kk : CheckSettings)
    (P Pc : PrefixDistribution α) (a h : ℕ) (crossLimit p₀ u ph ηc : ℝ)
    (hk : 1 ≤ C.walkStart) (ha : 1 ≤ a) (hak : a + Fintype.card σ + 1 ≤ C.walkStart)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (B : State) (F K₀ Kc : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
      BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hU : ∀ z : FreeMonoid α,
      familyReadDist O M F B.lo (B.hi + 1) (M.eval z.toList) .undecided ≤ u) :
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
    (∀ v, (avoidLaw P.D C.walkStart K₀).real {x | prefixOf x C.walkStart = v} ≤ p₀) →
    0 < δ → 0 < y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ u → (∀ v, (avoidLaw Pc.D a Kc).real {x | prefixOf x h = v} ≤ ph) → 0 < ph → h ≤ Pc.L →
    0 ≤ ηc → 8 * θ₀ ≤ 1 → q ≤ 1 → 0 < Kk.rate → Kk.rate < 1 → 0 < Kk.δ →
    crossLimit * C.probeBudget * ((P.L + 1) * nQ + Kk.size * ((Pc.L + 1) * nQ)
      + (Kk.size + 1) * ((nQ + 1) ^ 2 * nα)) ≤ δ / 8 →
    A * u + η / y < 1 →
    ∃ Bd : Set (Ω × (Fin C.probeBudget → unitInterval × (Fin Kk.size → unitInterval))),
      MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => (volume : Measure unitInterval).prod
          (Measure.pi fun _ => volume))) Bd
        ≤ ENNReal.ofReal (9 * δ / 8 + 2 * (∑ j ∈ Finset.range S, Kk.silentBase q ^ Kk.size
            / Kk.level j + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph))))
      ∧ (∀ v ω ω', (∀ w ∈ checkZone (drawsLen Kk (checkDrawsOf Kk P.D Pc.D C.walkStart a K₀ Kc v)
            + C.probeBudget + 1) C.walkStart a K₀ Kc F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ ω v, (ω, v) ∉ Bd →
        (indecisiveLStarCheck C Kk (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω)
          (checkDrawsOf Kk P.D Pc.D C.walkStart a K₀ Kc v)).ending = .consistent := by
  intro y ε δ nQ nα S Kt A η Ms θ₀ q hpre hδ hy0 hyρ hρ1 hP hu0 hph hph0 hhL hηc h8 hq1 hr0 hr1
    hKδ hw hAu
  classical
  set k := C.walkStart
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
  set R : Set (FreeMonoid α) := (offReadsC k a K₀ Kc F)ᶜ
  let fam' := fam.renoise R
  have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
    rw [SuffixFamily.renoise_readDist]
    rfl
  set D' := avoidLaw P.D k K₀
  set Dc' := avoidLaw Pc.D a Kc
  have hD'len := avoidLaw_len P.D P.length_le k K₀
  have hDc'len := avoidLaw_len Pc.D Pc.length_le a Kc
  let P' : PrefixDistribution α := ⟨D', inferInstance, P.L, hD'len⟩
  set U : σ → ℝ := fun q => fam'.readDist q .undecided
  have hUz : ∀ z : FreeMonoid α, U (M.eval z.toList) ≤ u := fun z => by
    show fam'.readDist _ _ ≤ u
    rw [hdist]
    exact hU z
  have hL4 := level4CheckRoundCorrect_holds M (fun q => readSide (fam'.readDist q)) U crossLimit
    (μ.prod μ) (fam'.epsOracle (show (0 : ℝ) ≤ crossLimit from hcross.le) rfl) P' Dc' Pc.L h C Kk p₀ u ph ηc hpre hδ hy0 hyρ hρ1 hP hDc'len
    (fun z _ => hUz z) hu0 hph hph0 hhL hηc h8 hq1 hr0 hr1 hKδ (by
      have : crossLimit * T * ((P.L + 1) * nQ + Kk.size * ((Pc.L + 1) * nQ)
          + (Kk.size + 1) * ((nQ + 1) ^ 2 * nα)) ≤ δ / 8 := hw
      convert this using 1)
  set π := Measure.pi fun _ : Fin T => D'.prod (Measure.pi fun _ : Fin Kk.size => Dc')
  set Sf := {d : (Ω × Ω) × (Fin T → CheckDraw α Kk) |
    ¬ ((indecisiveLStarCheck C Kk (fun z => fam'.read d.1 z) d.2).ending = .consistent
      ∧ (indecisiveLStarCheck C Kk (fun z => fam'.read d.1 z) d.2).hypothesis.tree.MidsShort nQ)}
  have hSm : MeasurableSet Sf := measurableSet_of_checkResult fam' C Kk
    fun res => ¬ (res.ending = .consistent ∧ res.hypothesis.tree.MidsShort nQ)
  -- the round over the re-noised family fails its claim at most as often as level 4's bound
  have hSf : ((μ.prod μ).prod π) Sf ≤ ENNReal.ofReal (9 * δ / 8 + 2 * (∑ j ∈ Finset.range S,
      Kk.silentBase q ^ Kk.size / Kk.level j + (h + 1) * nα ^ h * Ms * u
        + Real.exp (-(ηc / ph)))) := by
    rw [Measure.prod_apply hSm]
    refine le_trans (lintegral_mono fun ω' => measure_mono fun ds hds => ?_) hL4
    simp only [Set.mem_preimage, Sf, Set.mem_ofPred_eq] at hds ⊢
    rintro ⟨hE, hms⟩
    apply hds
    refine ⟨?_, hms⟩
    rcases hend : (indecisiveLStarCheck C Kk (fun z => fam'.read ω' z) ds).ending with
      _ | site | _
    · rfl
    · exfalso
      refine not_endsWell_harvest _ site (by exact hend) hu0 hUz ?_ hy0 hE
      exact hAu
    · exact absurd hE (not_endsWell_failed _ (by exact hend))
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) π Sf hSm
  have hdm := measurePreserving_checkDrawsOf (T := T) Kk P.D Pc.D k a K₀ Kc
  have hmp : MeasurePreserving (fun q : Ω × (Fin T → unitInterval × (Fin Kk.size → unitInterval)) =>
      (q.1, checkDrawsOf Kk P.D Pc.D k a K₀ Kc q.2))
      (μ.prod (Measure.pi fun _ => (volume : Measure unitInterval).prod
        (Measure.pi fun _ => volume))) (μ.prod π) := (MeasurePreserving.id μ).prod hdm
  have hSω : MeasurableSet {q : Ω × (Fin T → CheckDraw α Kk) | ((q.1, ω₀), q.2) ∈ Sf} :=
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hSm
  refine ⟨(fun q : Ω × (Fin T → unitInterval × (Fin Kk.size → unitInterval)) =>
    (q.1, checkDrawsOf Kk P.D Pc.D k a K₀ Kc q.2)) ⁻¹' {q | ((q.1, ω₀), q.2) ∈ Sf},
    hmp.measurable hSω, ?_, ?_, ?_⟩
  · rw [hmp.measure_preimage hSω.nullMeasurableSet]
    exact hω₀.trans hSf
  · -- the bad draws read only the zone
    intro v ω ω' hz hω
    set ds := checkDrawsOf Kk P.D Pc.D k a K₀ Kc v
    have hread : ∀ z : FreeMonoid α, z.toList.length ≤ drawsLen Kk ds + T + 1 →
        fam'.read (ω, ω₀) z = fam'.read (ω', ω₀) z := by
      intro z hzl
      unfold SuffixFamily.read familyRead voteCount
      congr 2
      refine Finset.filter_congr fun v' hv => ?_
      by_cases hzv : z * v' ∈ R
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_mem Oc hzv, renoise_mq_of_mem Oc hzv]
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_notMem Oc hzv, renoise_mq_of_notMem Oc hzv]
        exact hz _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z, v'), Finset.mem_product.2
          ⟨mem_shortStrings.2 hzl, hv⟩, rfl⟩, not_not.1 hzv⟩)
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, Sf] at hω ⊢
    rwa [checkResult_congr_short C Kk ds (r := fun z => fam'.read (ω', ω₀) z)
      (r' := fun z => fam'.read (ω, ω₀) z) fun z hzl => (hread z hzl).symm]
  · -- off the bad draws, the actual round is the re-noised one, which ends consistent
    intro ω v hnot
    set ds := checkDrawsOf Kk P.D Pc.D k a K₀ Kc v
    obtain ⟨hcons, hms⟩ : (indecisiveLStarCheck C Kk (fun z => fam'.read (ω, ω₀) z) ds).ending
        = .consistent ∧ (indecisiveLStarCheck C Kk (fun z => fam'.read (ω, ω₀) z)
          ds).hypothesis.tree.MidsShort nQ := not_not.1 hnot
    set read' : FreeMonoid α → ARU := fun z => fam'.read (ω, ω₀) z
    set read : FreeMonoid α → ARU := fun z => familyRead O.mq F B.lo (B.hi + 1) z ω
    have hoff : ∀ z, OffC k a K₀ Kc z → read' z = read z := fun z hz =>
      renoise_read_of_notMem fam R (fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩) (ω, ω₀)
    have hrun := runChecked_congr_zone C Kk (r := read') (r' := read)
      (X := (List.ofFn ds).map Prod.fst) (Y := (List.ofFn ds).flatMap fun d => List.ofFn d.2)
      (n := nQ)
      (fun x hx j hkj hj m _ => by
        obtain ⟨d, hd, rfl⟩ := List.mem_map.1 hx
        obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hd
        exact hoff _ (Or.inl (keyOff_of_prefix (drawFrom_avoids P.D hk K₀ (v i).1) hkj hj m)))
      (fun y' hy j hj m hm => by
        obtain ⟨d, hd, hy⟩ := List.mem_flatMap.1 hy
        obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hd
        obtain ⟨j', rfl⟩ := List.mem_ofFn.1 hy
        refine hoff _ ?_
        by_cases haj : a ≤ j
        · exact Or.inr (Or.inl (keyOff_of_prefix (drawFrom_avoids Pc.D ha Kc ((v i).2 j')) haj hj m))
        · refine Or.inr (Or.inr ?_)
          have : (prefixOf ((ds i).2 j') j).toList.length ≤ j := by simp [prefixOf]
          simp only [FreeMonoid.toList_mul, List.length_append]
          omega)
      (List.ofFn ds) initialState
      (fun d hd => ⟨List.mem_map_of_mem hd, fun i => List.mem_flatMap.2 ⟨d, hd, by simp⟩⟩)
      (checkRunInv_start read' C _ _)
      (mids_short_of_midsShort _ hms)
    have hres : indecisiveLStarCheck C Kk read ds = indecisiveLStarCheck C Kk read' ds := by
      simp only [indecisiveLStarCheck, hrun]
    rw [hres]
    exact hcons

end Bad

end OrthoDFA
