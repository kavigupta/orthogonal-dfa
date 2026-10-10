import OrthoDFA.Proofs.EpsTree
import OrthoDFA.Proofs.RandomRound

/-!
# The random round with rare wrong reads: the proofs

A record at a target that is not real comes from a probe that can read a string wrong: its two
sifts end at leaves, and with every read on their way right they would end at the true leaves of
a state and of its successor, making the target real. So a fake fix needs `m` such records at
one edge and target within a stretch, which `fakeRisk` bounds given the read field's second
tail. Otherwise the proof is the random round's, with `GInv` in place of the reads being right.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree Edges Disagrees pre located search agrees probe Outcome walk
  length_walk_le pre_of_le pre_succ agrees_inl walk_cons EdgesOK OutcomeOK)

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

/-! ## Where a record is made -/

theorem located_edge (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool) (lo hi : ℕ)
    (hle : lo ≤ hi) (hlo : agrees read T x st lo = some true)
    (hhi : agrees read T x st hi = some false) {p : List Bool} {c : α} {u : FreeMonoid α}
    {t : List Bool} (h : located read T x st lo hi = .edge p c u t) :
    ∃ q, lo < q ∧ q ≤ hi ∧ u = pre x (q - 1) ∧ x.toList[q - 1]? = some c := by
  have hf := Ideal.search_ok (agrees read T x st) _ lo hi rfl hle hlo hhi
  unfold located at h
  rcases hsr : search (agrees read T x st) lo hi with q | q | q
  · rw [hsr] at hf h
    have h1 := hf.1
    have h2 := hf.2.1
    simp only at h
    rcases hc : x.toList[q - 1]? with _ | c'
    · rw [hc] at h; simp at h
    · rcases ht : T.sift read (pre x q) with t' | z
      · rw [hc, ht] at h
        simp only [Outcome.edge.injEq] at h
        obtain ⟨-, rfl, rfl, -⟩ := h
        exact ⟨q, h1, h2, rfl, hc⟩
      · rw [hc, ht] at h; simp at h
  · rw [hsr] at h; simp at h
  · rw [hsr] at h; simp at h

/-- A record's prefix and letter are the probe's, past its first `k` letters. -/
theorem probe_edge (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)
    (hk : k ≤ x.toList.length) {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probe read T E k x = .edge p c u t) :
    ∃ i, k ≤ i ∧ i < x.toList.length ∧ u = pre x i ∧ x.toList[i]? = some c := by
  unfold probe at h
  rcases ha : T.sift read (pre x k) with a | z₀
  · rw [ha] at h
    simp only at h
    revert h
    generalize hcs : x.toList.drop k = cs
    generalize hss : walk E a cs = ss
    intro h
    have hle : ss.length ≤ cs.length + 1 := hss ▸ length_walk_le E a cs
    have hpos : 0 < ss.length := hss ▸ Ideal.length_walk_pos E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    have hst0 : ss.getD (k - k) [] = a := by
      rw [Nat.sub_self, ← hss]
      obtain ⟨l, hl⟩ := walk_cons E a cs
      simp [hl]
    have hagk : agrees read T x (fun p => ss.getD (p - k) []) k = some true := by
      rw [agrees_inl read T x _ ha]
      simp only [hst0, decide_true]
    split at h
    · rename_i c' hc
      have hj : k + ss.length - 1 < x.toList.length := by
        by_contra hcon
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      split at h
      · simp at h
      · simp at h
      · rename_i t' s' hs'
        split_ifs at h with hseq
        obtain ⟨q, h1, h2, h3, h4⟩ := located_edge read T x _ k _ (by omega) hagk
          (by rw [agrees_inl read T x _ hs']; simpa using hseq) h
        exact ⟨q - 1, by omega, by omega, h3, h4⟩
    · rename_i hc
      have hjx : x.toList.length ≤ k + ss.length - 1 := by
        by_contra hcon
        rw [List.getElem?_eq_getElem (by omega)] at hc
        simp at hc
      split at h
      · simp at h
      · rename_i e he
        split_ifs at h with heq
        have hj : k + ss.length - 1 = x.toList.length := by omega
        obtain ⟨q, h1, h2, h3, h4⟩ := located_edge read T x _ k _ (by omega) hagk
          (by rw [agrees_inl read T x _ (by rw [hj, pre_of_le le_rfl]; exact he)]
              simpa using heq) h
        exact ⟨q - 1, by omega, by omega, h3, h4⟩
  · rw [ha] at h
    simp at h

/-- The probes that can read a string wrong. -/
def PotWrong (k : ℕ) (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, read z = if side (M.eval z.toList) then .reject else .accept}

/-- A record at a target that is not real comes from a probe that can read a string wrong. -/
theorem untrue_wrong {T : DTree α} {E : Edges α} (hE : EdgesOK read T E)
    {τ : List Bool × α × List Bool} {x : FreeMonoid α} (hx : fR read C T E τ x = true)
    (hr : ¬ RealT M side T τ.1 τ.2.1 τ.2.2) : x ∈ PotWrong read M side C.k T := by
  classical
  obtain ⟨p, c, t⟩ := τ
  simp only [fR, decide_eq_true_eq] at hx
  have hOK := probeR_ok read hE C.k x
  generalize ho : probeR read T E C.k x = o at hx hOK
  cases o with
  | edge p' c' u t' =>
    simp only [recOf, Option.some.injEq, Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    obtain ⟨-, hu, huc, -⟩ := hOK
    have hk : C.k ≤ x.toList.length := by
      by_contra hc
      simp [probeR, show x.toList.length < C.k by omega] at ho
    have ho' : probe read T E C.k x = .edge p' c' u t' := by
      simpa [probeR, show ¬ x.toList.length < C.k by omega] using ho
    obtain ⟨i, hi1, hi2, rfl, hc⟩ := probe_edge read T E C.k x hk ho'
    by_contra hw
    simp only [PotWrong, Set.mem_ofPred_eq, not_exists, not_and] at hw
    have hp := sift_true M side read T _ p' hu fun m hm =>
      hw _ (mem_pot hi1 (by omega) hm)
    have hsucc := pre_succ hc
    rw [← hsucc] at huc
    have ht := sift_true M side read T _ t' huc fun m hm =>
      hw _ (mem_pot (by omega) (by omega) hm)
    apply hr
    refine ⟨M.eval (pre x i).toList, hp.symm, ?_⟩
    rw [ht, hsucc, eval_mul]
    rfl
  | agree => simp [recOf] at hx
  | startU z => simp [recOf] at hx
  | endU z => simp [recOf] at hx
  | triple k zs => simp [recOf] at hx
  | pair k zs => simp [recOf] at hx
  | member p' c' u t' => simp [recOf] at hx

/-! ## Fake fixes in a stretch -/

/-- The edges and targets of records. -/
noncomputable def recSet (T : DTree α) : Finset (List Bool × α × List Bool) :=
  T.leaves.toFinset ×ˢ ((Finset.univ : Finset α) ×ˢ T.leaves.toFinset)

/-- `m` records over the stretch at an edge and a target that is not real. -/
def FakeEv (T : DTree α) (E : Edges α) (zs : List (FreeMonoid α)) : Prop :=
  ∃ τ ∈ recSet T, ¬ RealT M side T τ.1 τ.2.1 τ.2.2 ∧ C.m ≤ zs.countP (fR read C T E τ)

theorem fakeEv_of_fix (U : σ → ℝ) (θ : ℝ) (L : ℕ) {T : DTree α} {E : Edges α}
    {ys : List (FreeMonoid α)} {s : RState α} (hs : Inv read C s)
    (hst : StInv read C M U θ T E L ys s) {x : FreeMonoid α} (hf : FakeFix M side read C s x) :
    FakeEv read C M side T E (ys ++ [x]) := by
  obtain ⟨p, c, u, t, ho, hc, hr⟩ := hf
  have hOK := probeR_ok read hs.edges C.k x
  rw [ho] at hOK
  obtain ⟨hp, -, huc, -⟩ := hOK
  have ht := DTree.sift_inl_mem read _ _ t huc
  rw [hst.tree] at hp ht hr
  refine ⟨(p, c, t), by simp [recSet, hp, ht], hr, ?_⟩
  rw [countP_snoc, ← hst.recs, ← hst.tree, ← hst.edges]
  simp only [fR, ho, recOf, decide_true, if_true]
  simpa [List.count_append] using hc

theorem fake_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {Gw : ℝ} (Ns : ℕ)
    {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) (hsz : T.leaves.length ≤ C.Lmax)
    (hW : D.real (PotWrong read M side C.k T) ≤ Gw) (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ FakeEv read C M side T E ((List.ofFn xs).take n)}
      ≤ fakeRisk C (Fintype.card α) Ns Gw := by
  classical
  set K := min N Ns
  set R := (recSet T).filter fun τ => ¬ RealT M side T τ.1 τ.2.1 τ.2.2
  have hsub : {xs : Fin N → FreeMonoid α |
      ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ FakeEv read C M side T E ((List.ofFn xs).take n)}
      ⊆ ⋃ τ ∈ R, {xs | C.m ≤ cnt (fR read C T E τ) K (List.ofFn xs)} := by
    rintro xs ⟨n, hn, hnN, τ, hτ, hr, hc⟩
    refine Set.mem_biUnion (Finset.mem_coe.2 (Finset.mem_filter.2 ⟨hτ, hr⟩)) ?_
    simp only [Set.mem_ofPred_eq, cnt]
    refine hc.trans (List.Sublist.countP_le ?_)
    have : n ≤ K := le_min hnN (Finset.mem_Icc.1 hn).2
    rw [show (List.ofFn xs).take n = ((List.ofFn xs).take K).take n by
      rw [List.take_take, min_eq_left this]]
    exact List.take_sublist _ _
  have hτ : ∀ τ ∈ R, (Measure.pi fun _ : Fin N => D).real
      {xs | C.m ≤ cnt (fR read C T E τ) K (List.ofFn xs)} ≤ binomSfGe Ns Gw C.m := by
    intro τ hτ
    have hr := (Finset.mem_filter.1 hτ).2
    have hle : D.real {x | fR read C T E τ x} ≤ Gw :=
      (measureReal_mono fun x hx => untrue_wrong read C M side hE hx hr).trans hW
    exact (pi_cnt_up D _ (min_le_left N Ns) hle hGw1 (fun j => C.m ≤ j)
      (fun j j' h hj => hj.trans h) C.m fun j hj => hj).trans
      (binomSfGe_mono_n hGw0 hGw1 (min_le_right N Ns) _)
  have hR : (R.card : ℝ) ≤ (C.Lmax ^ 2 * Fintype.card α : ℕ) := by
    have h1 : T.leaves.toFinset.card ≤ C.Lmax := (List.toFinset_card_le _).trans hsz
    have : R.card ≤ C.Lmax ^ 2 * Fintype.card α := by
      refine (Finset.card_filter_le _ _).trans ?_
      simp only [recSet, Finset.card_product, Finset.card_univ]
      calc _ ≤ C.Lmax * (Fintype.card α * C.Lmax) := by gcongr
        _ = _ := by ring
    exact_mod_cast this
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ τ ∈ R, (Measure.pi fun _ : Fin N => D).real
          {xs | C.m ≤ cnt (fR read C T E τ) K (List.ofFn xs)} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _τ ∈ R, binomSfGe Ns Gw C.m := Finset.sum_le_sum hτ
    _ = R.card * binomSfGe Ns Gw C.m := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ _ := by
        unfold fakeRisk
        exact mul_le_mul_of_nonneg_right hR (binomSfGe_nonneg hGw0 hGw1 _ _)

/-! ## The round -/

variable (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α)) (ε : ℝ) (Ns : ℕ)

theorem gInv_same {s s' : RState α} (hg : GInv M side s) (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) (hm : s'.moved = s.moved) : GInv M side s' := by
  obtain ⟨h1, h2⟩ := hg
  refine ⟨by rw [ht]; exact h1, fun p c t₀ w hE hmv => ?_⟩
  rw [ht]
  rw [he] at hE; rw [hm] at hmv
  exact h2 p c t₀ w hE hmv

theorem probe_level_eps [IsProbabilityMeasure D] (L P N₁ : ℕ) (G Gw θr εd' θpt' : ℝ)
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.k read T) ≤ G)
    (hNw : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotWrong read M side C.k T) ≤ Gw)
    (hcap : Fintype.card σ + 2 ≤ C.Lmax) (hm : 1 ≤ C.m) (hn₀ : 1 ≤ C.n₀) (hN₁ : C.n₀ ≤ N₁)
    (hN₁s : N₁ ≤ Ns) (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1)
    (ha : 0 ≤ C.a) (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) (hεd : C.εd ≤ ε) (hθr0 : 0 ≤ θr)
    (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    (hP : stretches C (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := round read C start (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees read r.1.tree r.1.edges C.k x}
            (toRoundEnd r.2)}
      ≤ ENNReal.ofReal (stretches C (Fintype.card α)
        * (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw)) := by
  have h2 : 2 ≤ C.Lmax := by omega
  have hNs : 0 < Ns := by omega
  have hst := inv_start read C hm h2 (α := α)
  have hΨ := Ψr_start C h2 (α := α)
  set X := FakeFix M side read C
  set b := ENNReal.ofReal (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw)
  set I : RState α → Prop := fun s => Inv read C s ∧ GInv M side s ∧ s.n < Ns
  have hseg : ∀ s, I s → s = fresh s.tree s.edges s.moved → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegB (step read C) (BadStep read C M U θ D ε Ns X) Same s (List.ofFn xs)} ≤ b := by
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
  have hI0 : I start := ⟨hst, gInv_start M side, by simp only [start, fresh]; omega⟩
  have hany := anyB_le (step read C) (BadStep read C M U θ D ε Ns X) Same D I
    (fun s => s = fresh s.tree s.edges s.moved) (Ψr C) b
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_same read C hs h hsm.1 hsm.2
      obtain ⟨rfl, -, -⟩ := step_same_chg read C hs h hsm
      refine ⟨⟨h1, gInv_same M side hg rfl rfl rfl, ?_⟩, h2⟩
      have : (chg read C s x).n ≠ Ns := fun hc => hb (Or.inr ⟨_, h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_change read C hm hs h hsm
      have hnf : ¬ X s x := fun hx => hb (Or.inl ⟨hn, Or.inr hx⟩)
      refine ⟨⟨h1, gInv_change M side read C hs hg h hsm hnf, ?_⟩, h2, h3⟩
      rw [h2]; simp only [fresh]; omega)
    hseg P start hI0
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (step read C) (BadStep read C M U θ D ε Ns X) start (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (start : RState α).n = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns X hm _ start hst
          (by rw [hn0]; exact hNs) ?_ hb)
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr C (start : RState α) + 1) * Ns ≤ stretches C (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b + Ψr C (start : RState α) * b :=
        hany.trans (add_le_add (hseg _ hI0 rfl P) le_rfl)
    _ = ((Ψr C (start : RState α) + 1 : ℕ) : ENNReal) * b := by push_cast; ring
    _ ≤ (stretches C (Fintype.card α) : ENNReal) * b := by gcongr
    _ = _ := by
        rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_natCast]

theorem epsRoundCorrect_holds : EpsRoundCorrect := by
  intro α _ _ σ _ M side U θ εw Ω _ μ _ read D _ C L P Ns N₁ p₀ ε G Gw θr εd' θpt' hmeas hind
    hU hwrong hεw hL hp₀ hpre hcap hm hn₀ hN₁ hN₁s hθ hG0 hG1 hGw0 hGw1 ha hθs0 hθs1 hθe hθpt0
    hθpt1 hεd0 hεd1 hεd hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.k (read · ω) T) ≤ G}
  set Nw := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotWrong (read · ω) M side C.k T) ≤ Gw}
  have hNn := noise_le M U θ μ read D C.k L p₀ G hmeas hind hU hL hp₀ hpre hθ
  have hNw : μ Nw ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw) :=
    noise_hit μ read D C.k L (Fintype.card σ) p₀ Gw εw
      (fun z r => r = if side (M.eval z.toList) then .reject else .accept) hmeas hind hwrong hεw
      hL hp₀ hpre
  set c := ENNReal.ofReal (stretches C (Fintype.card α)
    * (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw))
  set B := toMeasurable μ (Nn ∪ Nw)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := round (read · ω) C start (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) r.1.tree r.1.edges C.k x}
          (toRoundEnd r.2)} ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ Nn ∪ Nw := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or, Nn, Nw, Set.mem_ofPred_eq, not_not] at hω'
      exact probe_level_eps (read · ω) C M side U θ D ε Ns L P N₁ G Gw θr εd' θpt' hL hω'.1
        hω'.2 hcap hm hn₀ hN₁ hN₁s hG0 hG1 hGw0 hGw1 ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0 hεd1 hεd
        hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP
  have hf0 : 0 ≤ fakeRisk C (Fintype.card α) Ns Gw :=
    mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _)
  have hnn : 0 ≤ stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw :=
    add_nonneg (stretchRisk_nonneg C L Ns N₁ G θr εd' θpt' hG0 hG1 ha hθr0 hθr1 hεd'0 hεd'1
      hθpt'0 hθpt'1) hf0
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ Nn + μ Nw) := by
        gcongr
        rw [measure_toMeasurable]
        exact measure_union_le _ _
    _ ≤ c + (ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G)
          + ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw)) := by
        gcongr
    _ = _ := by
        rw [add_comm, ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by unfold noiseRisk; positivity), ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by positivity)]

end Random

end OrthoDFA
