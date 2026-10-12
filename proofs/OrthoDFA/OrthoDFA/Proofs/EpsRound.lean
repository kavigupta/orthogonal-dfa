import OrthoDFA.Proofs.EpsTree
import OrthoDFA.Proofs.RandomRound
import OrthoDFA.Proofs.Yield

/-!
# The random round with rare wrong reads: the proofs

A record at a target that is not real comes from a probe that can read a string wrong: its two
sifts end at leaves, and with every read on their way right they would end at the true leaves of
a state and of its successor, making the target real. So a fake fix needs `recordsToMove` such
records at one edge and target within a stretch, which `fakeRisk` bounds given the read field's
second tail. Otherwise the proof is the random round's, with `GInv` in place of the reads being
right.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (searchOutcome siftAgrees probe Outcome length_walk_le pre_of_le pre_succ
  agrees_inl walk_cons EdgesOK OutcomeOK)

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

/-! ## Where a record is made -/

theorem searchOutcome_edge (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)
    (un : Option (α × List Bool)) (lo hi : ℕ) (hle : lo ≤ hi) {p : List Bool} {c : α}
    {u : FreeMonoid α} {t : List Bool}
    (h : searchOutcome read T x st un hi (tripartiteSearch true (siftAgrees read T x st) lo hi).1
      = .wrongEdge p c u t) :
    ∃ q, lo < q ∧ q ≤ hi ∧ u = prefixOf x (q - 1) ∧ x.toList[q - 1]? = some c := by
  have hs := tripartiteSearch_ok (recover := true) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch true (siftAgrees read T x st) lo hi).1 = r at hs h
  cases r with
  | edge q =>
    obtain ⟨h1, h2, -, -⟩ := hs
    simp only [searchOutcome] at h
    split at h
    · rename_i c' t' hc ht
      simp only [Outcome.wrongEdge.injEq] at h
      obtain ⟨-, rfl, rfl, -⟩ := h
      exact ⟨q, h1, h2, rfl, hc⟩
    · simp at h
  | allAgree => simp only [searchOutcome] at h; split at h <;> simp at h
  | _ => simp [searchOutcome] at h

/-- A record's prefix and letter are the probe's, past its first `k` letters. -/
theorem probe_edge (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) {p : List Bool} {c : α}
    {u : FreeMonoid α} {t : List Bool} (h : probe read T E k x = .wrongEdge p c u t) :
    ∃ i, k ≤ i ∧ i < x.toList.length ∧ u = prefixOf x i ∧ x.toList[i]? = some c := by
  unfold probe at h
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · rw [ha] at h
    simp only at h
    have hpos := Ideal.length_walk_pos E a (x.toList.drop k)
    have hlt : ∀ q, x.toList[q - 1]? = some c → q - 1 < x.toList.length := fun q hc => by
      by_contra hcon
      rw [List.getElem?_eq_none (by omega)] at hc
      simp at hc
    split at h
    · split at h
      · simp at h
      · obtain ⟨q, h1, -, h3, h4⟩ := searchOutcome_edge read T x _ _ k _ (by omega) h
        exact ⟨q - 1, by omega, hlt q h4, h3, h4⟩
    · obtain ⟨q, h1, -, h3, h4⟩ := searchOutcome_edge read T x _ _ k _ (by omega) h
      exact ⟨q - 1, by omega, hlt q h4, h3, h4⟩
  · rw [ha] at h
    simp at h

/-- The probes that can read a string wrong. -/
def PotWrong (k : ℕ) (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, read z = if side (M.eval z.toList) then .reject else .accept}

/-- A record at a target that is not real comes from a probe that can read a string wrong. -/
theorem untrue_wrong {T : DTree α} {E : Edges α} (hE : EdgesOK read T E)
    {τ : List Bool × α × List Bool} {x : FreeMonoid α} (hx : fR read C T E τ x = true)
    (hr : ¬ RealT M side T τ.1 τ.2.1 τ.2.2) : x ∈ PotWrong read M side C.walkStart T := by
  classical
  obtain ⟨p, c, t⟩ := τ
  simp only [fR, decide_eq_true_eq] at hx
  have hOK := probeOrAgree_ok read hE C.walkStart x
  generalize ho : probeOrAgree read T E C.walkStart x = o at hx hOK
  cases o with
  | wrongEdge p' c' u t' =>
    simp only [recOf, Option.some.injEq, Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    obtain ⟨-, hu, huc, -⟩ := hOK
    have hk : C.walkStart ≤ x.toList.length := by
      by_contra hc
      simp [probeOrAgree, show x.toList.length < C.walkStart by omega] at ho
    have ho' : probe read T E C.walkStart x = .wrongEdge p' c' u t' := by
      simpa [probeOrAgree, show ¬ x.toList.length < C.walkStart by omega] using ho
    obtain ⟨i, hi1, hi2, rfl, hc⟩ := probe_edge read T E C.walkStart x ho'
    by_contra hw
    simp only [PotWrong, Set.mem_ofPred_eq, not_exists, not_and] at hw
    have hp := sift_true M side read T _ p' hu fun m hm =>
      hw _ (mem_pot hi1 (by omega) hm)
    have hsucc := pre_succ hc
    rw [← hsucc] at huc
    have ht := sift_true M side read T _ t' huc fun m hm =>
      hw _ (mem_pot (by omega) (by omega) hm)
    apply hr
    refine ⟨M.eval (prefixOf x i).toList, hp.symm, ?_⟩
    rw [ht, hsucc, eval_mul]
    rfl
  | agree => simp [recOf] at hx
  | undecided k zs cl => simp [recOf] at hx
  | unlearnedEdge p' c' u t' => simp [recOf] at hx

/-! ## Fake fixes in a stretch -/

/-- The edges and targets of records. -/
noncomputable def recSet (T : DTree α) : Finset (List Bool × α × List Bool) :=
  T.leaves.toFinset ×ˢ ((Finset.univ : Finset α) ×ˢ T.leaves.toFinset)

/-- `recordsToMove` records over the stretch at an edge and a target that is not real. -/
def FakeEv (T : DTree α) (E : Edges α) (zs : List (FreeMonoid α)) : Prop :=
  ∃ τ ∈ recSet T, ¬ RealT M side T τ.1 τ.2.1 τ.2.2 ∧ C.recordsToMove ≤ zs.countP (fR read C T E τ)

theorem fakeEv_of_fix (U : σ → ℝ) (θ : ℝ) (L : ℕ) {T : DTree α} {E : Edges α}
    {ys : List (FreeMonoid α)} {s : Level3State α} (hs : Inv read C s)
    (hst : StInv read C M U θ T E L ys s) {x : FreeMonoid α} (hf : FakeFix M side read C s x) :
    FakeEv read C M side T E (ys ++ [x]) := by
  obtain ⟨p, c, u, t, ho, hc, hr⟩ := hf
  have hOK := probeOrAgree_ok read hs.edges C.walkStart x
  rw [ho] at hOK
  obtain ⟨hp, -, huc, -⟩ := hOK
  have ht := DTree.sift_inl_mem read _ _ t huc
  rw [hst.tree] at hp ht hr
  refine ⟨(p, c, t), by simp [recSet, hp, ht], hr, ?_⟩
  rw [countP_snoc, ← hst.records, ← hst.tree, ← hst.edges]
  simp only [fR, ho, recOf, decide_true, if_true]
  simpa [List.count_append] using hc

theorem fake_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {Gw : ℝ} (Ns : ℕ)
    {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) (hsz : T.leaves.length ≤ C.maxLeaves)
    (hW : D.real (PotWrong read M side C.walkStart T) ≤ Gw) (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1)
    (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ FakeEv read C M side T E ((List.ofFn xs).take n)}
      ≤ fakeRisk C (Fintype.card α) Ns Gw := by
  classical
  set K := min N Ns
  set R := (recSet T).filter fun τ => ¬ RealT M side T τ.1 τ.2.1 τ.2.2
  have hsub : {xs : Fin N → FreeMonoid α |
      ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ FakeEv read C M side T E ((List.ofFn xs).take n)}
      ⊆ ⋃ τ ∈ R, {xs | C.recordsToMove ≤ cnt (fR read C T E τ) K (List.ofFn xs)} := by
    rintro xs ⟨n, hn, hnN, τ, hτ, hr, hc⟩
    refine Set.mem_biUnion (Finset.mem_coe.2 (Finset.mem_filter.2 ⟨hτ, hr⟩)) ?_
    simp only [Set.mem_ofPred_eq, cnt]
    refine hc.trans (List.Sublist.countP_le ?_)
    have : n ≤ K := le_min hnN (Finset.mem_Icc.1 hn).2
    rw [show (List.ofFn xs).take n = ((List.ofFn xs).take K).take n by
      rw [List.take_take, min_eq_left this]]
    exact List.take_sublist _ _
  have hτ : ∀ τ ∈ R, (Measure.pi fun _ : Fin N => D).real
      {xs | C.recordsToMove ≤ cnt (fR read C T E τ) K (List.ofFn xs)}
          ≤ binomSfGe Ns Gw C.recordsToMove := by
    intro τ hτ
    have hr := (Finset.mem_filter.1 hτ).2
    have hle : D.real {x | fR read C T E τ x} ≤ Gw :=
      (measureReal_mono fun x hx => untrue_wrong read C M side hE hx hr).trans hW
    exact (pi_cnt_up D _ (min_le_left N Ns) hle hGw1 (fun j => C.recordsToMove ≤ j)
      (fun j j' h hj => hj.trans h) C.recordsToMove fun j hj => hj).trans
      (binomSfGe_mono_n hGw0 hGw1 (min_le_right N Ns) _)
  have hR : (R.card : ℝ) ≤ (C.maxLeaves ^ 2 * Fintype.card α : ℕ) := by
    have h1 : T.leaves.toFinset.card ≤ C.maxLeaves := (List.toFinset_card_le _).trans hsz
    have : R.card ≤ C.maxLeaves ^ 2 * Fintype.card α := by
      refine (Finset.card_filter_le _ _).trans ?_
      simp only [recSet, Finset.card_product, Finset.card_univ]
      calc _ ≤ C.maxLeaves * (Fintype.card α * C.maxLeaves) := by gcongr
        _ = _ := by ring
    exact_mod_cast this
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ τ ∈ R, (Measure.pi fun _ : Fin N => D).real
          {xs | C.recordsToMove ≤ cnt (fR read C T E τ) K (List.ofFn xs)} :=
              measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _τ ∈ R, binomSfGe Ns Gw C.recordsToMove := Finset.sum_le_sum hτ
    _ = R.card * binomSfGe Ns Gw C.recordsToMove := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ _ := by
        unfold fakeRisk
        exact mul_le_mul_of_nonneg_right hR (binomSfGe_nonneg hGw0 hGw1 _ _)

/-! ## The round -/

variable (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α)) (ε : ℝ) (Ns : ℕ)

theorem gInv_same {s s' : Level3State α} (hg : GInv M side s) (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) (hm : s'.redirected = s.redirected) : GInv M side s' := by
  obtain ⟨h1, h2⟩ := hg
  refine ⟨by rw [ht]; exact h1, fun p c t₀ w hE hmv => ?_⟩
  rw [ht]
  rw [he] at hE; rw [hm] at hmv
  exact h2 p c t₀ w hE hmv

theorem probe_level_eps [IsProbabilityMeasure D] (L P N₁ : ℕ) (G Gw θr εd' θpt' : ℝ)
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hNw : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotWrong read M side C.walkStart T) ≤ Gw) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    (hm : 1 ≤ C.recordsToMove) (hn₀ : 1 ≤ C.firstLook) (hN₁ : C.firstLook ≤ N₁) (hN₁s : N₁ ≤ Ns)
    (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1) (ha : 0 ≤ C.testLevel)
    (hθs0 : 0 ≤ C.startRate) (hθs1 : C.startRate ≤ 1) (hθe : 0 ≤ C.edgeRate)
    (hθpt0 : 0 ≤ C.middleRate) (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate)
    (hεd1 : C.disagreementRate ≤ 1) (hεd : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    (hP : stretches C (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := runProbes read C initialState (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees read ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
            0 0 (fun _ _ => []) 0 r.2}
      ≤ ENNReal.ofReal (stretches C (Fintype.card α)
        * (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw)) := by
  have h2 : 2 ≤ C.maxLeaves := by omega
  have hNs : 0 < Ns := by omega
  have hst := inv_start read C hm h2 (α := α)
  have hΨ := Ψr_start C h2 (α := α)
  set X := FakeFix M side read C
  set b := ENNReal.ofReal (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw)
  set I : Level3State α → Prop := fun s => Inv read C s ∧ GInv M side s ∧ s.probeCount < Ns
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges s.redirected → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns X) Same s (List.ofFn xs)}
            ≤ b := by
    rintro s ⟨hs, hg, -⟩ hf N
    have hT := gInv_cls M side read C hs hg
    exact seg_le read C M U θ D ε Ns X (FakeEv read C M side s.tree s.edges)
      (fakeRisk C (Fintype.card α) Ns Gw) L N₁ G θr εd' θpt' hL hs hf hT hN
      (fun ys s' x hs' hst' hx => fakeEv_of_fix read C M side U θ L hs' hst' hx)
      (fun ys s' x hs' hst' hmv hx => step_ne_tooBig_g M side read C hcap hs'
        (gInv_same M side hg hst'.tree hst'.edges hmv) hx)
      (fun N => fake_le read C M side D Ns hs.edges hs.size (hNw _ hT) hGw0 hGw1 N)
      hm hn₀ hN₁ hN₁s hG0 hG1 ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0 hεd1 hεd hθr0 hθr1 hεd'0
      hεd'1 hθpt'0 hθpt'1 hsep N
  have hI0 : I initialState := ⟨hst, gInv_start M side,
      by simp only [initialState, newHypothesis]; omega⟩
  have hany := anyB_le (applyProbe read C) (BadStep read C M U θ D ε Ns X) Same D I
    (fun s => s = newHypothesis s.tree s.edges s.redirected) (Ψr C) b
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_same read C hs h hsm.1 hsm.2
      obtain ⟨rfl, -, -⟩ := step_same_chg read C hs h hsm
      refine ⟨⟨h1, gInv_same M side hg rfl rfl rfl, ?_⟩, h2⟩
      have : (chg read C s x).probeCount ≠ Ns := fun hc => hb (Or.inr ⟨_, h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_change read C hm hs h hsm
      have hnf : ¬ X s x := fun hx => hb (Or.inl ⟨hn, Or.inr hx⟩)
      refine ⟨⟨h1, gInv_change M side read C hs hg h hsm hnf, ?_⟩, h2, h3⟩
      rw [h2]; simp only [newHypothesis]; omega)
    hseg P initialState hI0
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns X) initialState
            (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns X hm _ initialState hst
          (by rw [hn0]; exact hNs) ?_ hb)
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr C (initialState : Level3State α) + 1) * Ns ≤ stretches C (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b + Ψr C (initialState : Level3State α) * b :=
        hany.trans (add_le_add (hseg _ hI0 rfl P) le_rfl)
    _ = ((Ψr C (initialState : Level3State α) + 1 : ℕ) : ENNReal) * b := by push_cast; ring
    _ ≤ (stretches C (Fintype.card α) : ENNReal) * b := by gcongr
    _ = _ := by
        rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_natCast]

theorem epsRoundCorrect_holds : EpsRoundCorrect := by
  intro α _ _ σ _ M side U θ εw Ω _ μ _ R P C ε
  obtain ⟨D, _, L, st, hst, hL, q, hq, hpreq⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  have hεw : 0 ≤ εw := measureReal_nonneg.trans (hwrong 1)
  simp only [epsRoundRisk, ENNReal.iInf_add]
  refine le_iInf₂ fun kn hA => ?_
  obtain ⟨⟨Ns, N₁, G, θr, εd', θpt'⟩, Gw⟩ := kn
  obtain ⟨⟨hcap, hm, hn₀, hN₁, hN₁s, hθ, hG0, hG1, ha, hθs0, hθs1, hθe, hθpt0, hθpt1, hεd0, hεd1,
    hεd, hθr0, hθr1, hεd'0, hεd'1, hθpt'0, hθpt'1, hsep, hP⟩, hGw0, hGw1⟩ := hA
  simp only at hN₁ hN₁s hG0 hG1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP hGw0 hGw1
  unfold epsRoundBound roundBound
  simp only
  set P := C.probeBudget
  set p₀ := prefixMax D C.walkStart
  have hp₀ : 0 < p₀ := prefixMax_pos D C.walkStart
  have hpre : ∀ u : FreeMonoid α, u.toList.length = C.walkStart
      → D.real {x | prefixOf x C.walkStart = u} ≤ p₀ :=
    fun u _ => le_prefixMax D C.walkStart u
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.walkStart (read · ω) T) ≤ G}
  set Nw := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotWrong (read · ω) M side C.walkStart T) ≤ Gw}
  have hNn := noise_le M U θ μ read D C.walkStart L p₀ G hmeas hind hU hL hp₀ hpre hθ
  set m₀ := 3 / 2 * θ * coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ)
  set Ny := {ω | (D.prod (uniformAnchor st L)).real {r | r.2 ≤ r.1.toList.length ∧
      read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided} < m₀ / 2}
  have hm₀ : m₀ ≤ 3 / 2 * θ * (D.prod (uniformAnchor st L)).real {r | r.2 ≤ r.1.toList.length ∧
      BadAt U θ (M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} := by
    have hcov : coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ)
        ≤ (D.prod (uniformAnchor st L)).real {r | r.2 ≤ r.1.toList.length ∧
          BadAt U θ (M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} :=
      coverage_le_anchor D st L M (BadAt U θ)
    simp only [m₀, mul_div_assoc]
    exact mul_le_mul_of_nonneg_left hcov (by linarith)
  have hNy : μ Ny ≤ ENNReal.ofReal
      (Real.exp (-(3 / 2 * θ * coverage D st L M (BadAt U θ)) / (16 * q))) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ Ny)]
    refine ENNReal.ofReal_le_ofReal ((anchor_reads read hmeas hind D hst
      (fun u => BadAt U θ (M.eval u.toList)) (3 / 2 * θ) (fun u hu => (hU u).symm ▸ hu.le) m₀ q
      hm₀ hq hpreq).trans_eq ?_)
    have hn : ((L + 1 - st : ℕ) : ℝ) ≠ 0 := by exact_mod_cast (anchorCount_pos hst).ne'
    simp only [m₀]
    congr 1
    field_simp
  have hNw : μ Nw ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw) :=
    noise_hit μ read D C.walkStart L (Fintype.card σ) p₀ Gw εw
      (fun z r => r = if side (M.eval z.toList) then .reject else .accept) hmeas hind hwrong hεw
      hL hp₀ hpre
  set c := ENNReal.ofReal (stretches C (Fintype.card α)
    * (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw))
  set B := toMeasurable μ (Nn ∪ Nw ∪ Ny)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
          st L (Ideal.replay (read · ω) ⟨r.1.tree, r.1.edges⟩)
          (3 / 4 * θ * coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ))
          r.2}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ Nn ∪ Nw ∪ Ny := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨⟨hωN, hωW⟩, hωY⟩ := hω'
      simp only [Nn, Nw, Set.mem_ofPred_eq, not_not] at hωN hωW
      simp only [Ny, Set.mem_ofPred_eq, not_lt] at hωY
      refine le_trans (measure_mono ?_) (probe_level_eps (read · ω) C M side U θ D ε Ns L P N₁ G
        Gw θr εd' θpt' hL hωN hωW hcap hm hn₀ hN₁ hN₁s hG0 hG1 hGw0 hGw1 ha hθs0 hθs1 hθe
        hθpt0 hθpt1 hεd0 hεd1 hεd hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP)
      intro xs hxs h
      refine hxs (h.respread (Ideal.replay_spread (read · ω) _ _ D st L) ?_)
      have hroot := round_root (read · ω) C (List.ofFn xs) initialState ⟨_, _, rfl⟩
      refine le_trans (le_of_eq ?_) (hωY.trans (measureReal_mono ?_ (measure_ne_top _ _)))
      · simp only [m₀]; ring
      · rintro ⟨w, e⟩ ⟨he, hu⟩
        exact Ideal.replay_root (read · ω) hroot _ he hu
  have hf0 : 0 ≤ fakeRisk C (Fintype.card α) Ns Gw :=
    mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _)
  have hnn : 0 ≤ stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw :=
    add_nonneg (stretchRisk_nonneg C L Ns N₁ G θr εd' θpt' hG0 hG1 ha hθr0 hθr1 hεd'0 hεd'1
      hθpt'0 hθpt'1) hf0
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ Nn + μ Nw + μ Ny) := by
        gcongr
        rw [measure_toMeasurable]
        exact (measure_union_le _ _).trans (by gcongr; exact measure_union_le _ _)
    _ ≤ c + (ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G)
          + ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw)
          + ENNReal.ofReal (Real.exp (-(3 / 2 * θ * coverage D st L M (BadAt U θ))
            / (16 * q)))) := by
        gcongr
    _ = _ := by
        rw [← add_assoc _ _ (ENNReal.ofReal (Real.exp _)), ← add_assoc c]
        congr 1
        rw [add_assoc, add_comm, ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by unfold noiseRisk; positivity), ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by positivity)]
        congr 1
        ring

end Random

end OrthoDFA
