import OrthoDFA.Proofs.TallyKeep

/-!
# Success is sound

Within a stretch the hypothesis is fixed, so its searches are independent draws at its
disagreement rate `d`. The success test fires at `n` probes only on a count of searches at most
`H n`, the largest one whose lower tail under `Bin(n, εd)` is below `a`. Where `d ≥ εd`, the
stretch's first `n` probes have at most `H n` searches with chance at most that tail, below `a`;
over the stretches' starts and lengths, a round ends in success at a hypothesis disagreeing on at
least `εd` of the probes with chance at most `T² a`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Success

variable (C : TallyCfg) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The hypothesis's disagreement rate: the probes that search. -/
noncomputable def disRate (s : TState α) : ℝ := D.real (ReadModel.searchAt rd C.k s.tree s.edges)

/-- The searches a probe adds to the stretch's count. -/
noncomputable def dt (s : TState α) (x : FreeMonoid α) : ℕ :=
  (tallyPre C (fun z => (rd z).cut) s x).dis - s.dis

/-- From `s`, at a hypothesis disagreeing on at least `εd` of the probes, the next `N + 1` probes
stay in the stretch and search at most `h` times. -/
def EvC : ℕ → ℕ → TState α → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | 0, h, s, x :: _ => C.εd ≤ disRate C D rd s ∧ dt C rd s x ≤ h
  | N + 1, h, s, x :: xs => C.εd ≤ disRate C D rd s ∧ dt C rd s x ≤ h ∧
    match tallyStep C (fun z => (rd z).cut) s x with
    | .inl s' => s'.n ≠ 0 ∧ EvC N (h - dt C rd s x) s' xs
    | .inr _ => False

/-- After `i` probes, the run is at a state from which `Q` holds of the remaining draws. -/
def SkipC (Q : TState α → List (FreeMonoid α) → Prop) : ℕ → TState α → List (FreeMonoid α) → Prop
  | 0, s, l => Q s l
  | _ + 1, _, [] => False
  | i + 1, s, x :: xs => match tallyStep C (fun z => (rd z).cut) s x with
    | .inl s' => SkipC Q i s' xs
    | .inr _ => False

theorem skipC_le [IsProbabilityMeasure D] (Q : TState α → List (FreeMonoid α) → Prop)
    {c : ENNReal} (hQ : ∀ (T : ℕ) (s : TState α),
      (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)} ≤ c) :
    ∀ (T i : ℕ) (s : TState α),
      (Measure.pi fun _ : Fin T => D) {xs | SkipC C rd Q i s (List.ofFn xs)} ≤ c := by
  intro T
  induction T with
  | zero =>
    intro i s
    rcases i with _ | i
    · exact hQ 0 s
    · simp [SkipC]
  | succ T ih =>
    intro i s
    rcases i with _ | i
    · exact hQ _ s
    rw [pi_succ_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | SkipC C rd Q (i + 1) s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ _, c ∂D := lintegral_mono fun x => by
          rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipC C rd Q (i + 1) s (List.ofFn xs)}} = {xs | SkipC C rd Q i s' (List.ofFn xs)} := by
              ext xs; simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipC, hst]
            rw [this]; exact ih i s'
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipC C rd Q (i + 1) s (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipC, hst, Set.mem_empty_iff_false]
            rw [this, measure_empty]; exact zero_le
      _ = c := by rw [lintegral_const, measure_univ, mul_one]

open scoped Classical in
theorem tallyPre_dis_search (s : TState α) (x : FreeMonoid α) :
    dt C rd s x = if x ∈ ReadModel.searchAt rd C.k s.tree s.edges then 1 else 0 :=
  tallyPre_dis_sub C rd s x

/-- A stretch at a hypothesis disagreeing on at least `εd` of the probes searches at most `h`
times in its next `N + 1` probes with chance at most `P(Bin(N + 1, εd) ≤ h)`. -/
theorem evC_le [IsProbabilityMeasure D] (hε0 : 0 ≤ C.εd) (hε1 : C.εd ≤ 1) :
    ∀ (T N h : ℕ) (s : TState α),
      (Measure.pi fun _ : Fin T => D) {xs | EvC C D rd N h s (List.ofFn xs)}
        ≤ ENNReal.ofReal (1 - binomSfGe (N + 1) C.εd (h + 1)) := by
  classical
  intro T
  induction T with
  | zero => intro N h s; simp [EvC]
  | succ T ih =>
    intro N h s
    by_cases hd : C.εd ≤ disRate C D rd s
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α | EvC C D rd N h s (List.ofFn xs)} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, List.ofFn_succ, Set.mem_empty_iff_false, iff_false]
        rcases N with _ | N
        · exact fun h' => hd h'.1
        · exact fun h' => hd h'.1
      rw [this, measure_empty]; exact zero_le
    set B := ReadModel.searchAt rd C.k s.tree s.edges
    have hB1 : D.real B ≤ 1 := by
      have := measureReal_mono (μ := D) (Set.subset_univ B)
      rwa [probReal_univ] at this
    set A : ℝ := 1 - binomSfGe N C.εd h
    set Cc : ℝ := 1 - binomSfGe N C.εd (h + 1)
    have hA0 : 0 ≤ A := sub_nonneg.2 (binomSfGe_le_one hε0 hε1 _)
    have hAC : A ≤ Cc := by
      have := binomSfGe_antitone hε0 hε1 (n := N) h
      simp only [A, Cc]; linarith
    have hC0 : 0 ≤ Cc := hA0.trans hAC
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvC C D rd N h s (List.ofFn xs)}}
        ≤ B.indicator (fun _ => ENNReal.ofReal A) x
          + Bᶜ.indicator (fun _ => ENNReal.ofReal Cc) x := by
      intro x
      have hdt := tallyPre_dis_search C rd s x
      by_cases hx : x ∈ B
      · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (Set.notMem_compl_iff.2 hx), add_zero]
        rw [if_pos hx] at hdt
        rcases N with _ | N
        · rcases h with _ | h
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC] at hxs
            have := hxs.2
            omega
          · simp only [A, binomSfGe_zero_left, sub_zero, ENNReal.ofReal_one]
            exact prob_le_one
        · rcases h with _ | h
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC] at hxs
            have := hxs.2.1
            omega
          rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
          · refine le_trans (measure_mono fun xs hxs => ?_) (ih N h s')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC, hst] at hxs
            rw [hdt] at hxs
            exact hxs.2.2.2
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC, hst] at hxs
            exact hxs.2.2
      · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (Set.mem_compl hx), zero_add]
        rw [if_neg hx] at hdt
        rcases N with _ | N
        · have : Cc = 1 := by simp [Cc, binomSfGe_zero_left]
          rw [this, ENNReal.ofReal_one]; exact prob_le_one
        · rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
          · refine le_trans (measure_mono fun xs hxs => ?_) (ih N h s')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC, hst] at hxs
            rw [hdt, Nat.sub_zero] at hxs
            exact hxs.2.2.2
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvC, hst] at hxs
            exact hxs.2.2
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    have hB0 : 0 ≤ D.real B := measureReal_nonneg
    rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
        (Set.to_countable _).measurableSet, lintegral_indicator
        (Set.to_countable _).measurableSet,
      setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
      ← ENNReal.ofReal_mul hA0, ← ENNReal.ofReal_mul hC0,
      ← ENNReal.ofReal_add (mul_nonneg hA0 hB0) (mul_nonneg hC0 (by linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    have hrec : 1 - binomSfGe (N + 1) C.εd (h + 1) = C.εd * A + (1 - C.εd) * Cc := by
      simp only [A, Cc, binomSfGe_succ]; ring
    rw [hrec]
    have hdB : C.εd ≤ D.real B := hd
    nlinarith

theorem tallyLook_success' {s : TState α} (h : tallyLook C s = some .success) :
    C.n₀ ≤ s.n ∧ 1 - binomSfGe s.n C.εd (s.dis + 1) < C.a := by
  unfold tallyLook at h
  split_ifs at h with h1 h2 h3 h4 <;> simp at h
  unfold rateSide at h4
  split_ifs at h4 with h5 h6 h7 <;> simp_all

open scoped Classical in
/-- A round ends in success at a hypothesis disagreeing on at least `εd` of the probes with chance
at most `T² a`. -/
theorem success_sound [IsProbabilityMeasure D] (hε0 : 0 ≤ C.εd) (hε1 : C.εd ≤ 1)
    (ha1 : C.a ≤ 1) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (tallyStep C fun z => (rd z).cut)
          (fun e s' => e = .success ∧ C.εd ≤ disRate C D rd s') tallyStart (List.ofFn xs)}
      ≤ ENNReal.ofReal (T * T * C.a) := by
  set P : ℕ → ℕ → Prop := fun n h => 1 - binomSfGe n C.εd (h + 1) < C.a
  set H : ℕ → ℕ := fun n => Nat.findGreatest (P n) n
  set Q : ℕ → TState α → List (FreeMonoid α) → Prop := fun N s l =>
    P (N + 1) (H (N + 1)) ∧ EvC C D rd N (H (N + 1)) s l
  have hfire : ∀ n d, P n d → d ≤ H n ∧ P n (H n) := by
    intro n d hP
    have hdn : d ≤ n := by
      by_contra hc
      simp only [P] at hP
      rw [binomSfGe_gt _ (by omega), sub_zero] at hP
      linarith
    exact ⟨Nat.le_findGreatest hdn hP, Nat.findGreatest_spec hdn hP⟩
  have hev_ne : ∀ N h (s : TState α) (l : List (FreeMonoid α)), EvC C D rd N h s l →
      C.εd ≤ disRate C D rd s ∧ l ≠ [] := by
    intro N h s l hl
    rcases l with _ | ⟨y, ys⟩
    · cases N <;> exact hl.elim
    · rcases N with _ | N
      · exact ⟨hl.1, by simp⟩
      · exact ⟨hl.1, by simp⟩
  have hincl : ∀ (l : List (FreeMonoid α)) (s : TState α),
      RunEnds (tallyStep C fun z => (rd z).cut)
        (fun e s' => e = .success ∧ C.εd ≤ disRate C D rd s') s l →
      (∃ i N, i < l.length ∧ N < l.length ∧ SkipC C rd (Q N) i s l)
        ∨ ∃ N, N < l.length ∧ P (s.n + N + 1) (H (s.n + N + 1)) ∧ s.dis ≤ H (s.n + N + 1)
          ∧ EvC C D rd N (H (s.n + N + 1) - s.dis) s l := by
    intro l
    induction l with
    | nil => intro s h; exact h.elim
    | cons x xs ih =>
      intro s h
      simp only [RunEnds] at h
      rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' h with ⟨i, N, hi, hN, hk⟩ | ⟨N, hN, hP, hdis, hev⟩
        · refine .inl ⟨i + 1, N, by simp; omega, by simp; omega, ?_⟩
          simp only [SkipC, hst]
          exact hk
        · have hxs := (hev_ne _ _ _ _ hev).2
          have hlen : 1 ≤ xs.length := by
            rcases xs with _ | _
            · exact absurd rfl hxs
            · simp
          rcases tallyStep_fresh C _ hst with ⟨hn0, hd0, -⟩ | ⟨hpre, hn, htr, hed⟩
          · refine .inl ⟨1, N, by simp; omega, by simp; omega, ?_⟩
            simp only [SkipC, hst, Q]
            rw [hn0, zero_add] at hP hdis hev
            rw [hd0, Nat.sub_zero] at hev
            exact ⟨hP, hev⟩
          · subst hpre
            have hdR : disRate C D rd (tallyPre C (fun z => (rd z).cut) s x)
                = disRate C D rd s := by
              simp only [disRate, htr, hed]
            have hdtv : (tallyPre C (fun z => (rd z).cut) s x).dis = s.dis + dt C rd s x := by
              rcases (tallyPre_cases (fun z => (rd z).cut) C s x).2 with
                ⟨-, -, -, -, hdis'⟩ | ⟨-, hn', -⟩
              · simp only [dt]; rw [hdis']; split_ifs <;> omega
              · omega
            have hεs : C.εd ≤ disRate C D rd s := hdR ▸ (hev_ne _ _ _ _ hev).1
            have he : s.n + (N + 1) + 1 = (tallyPre C (fun z => (rd z).cut) s x).n + N + 1 := by
              omega
            refine .inr ⟨N + 1, by simp; omega, by rw [he]; exact hP, by rw [he]; omega, ?_⟩
            simp only [EvC]
            refine ⟨hεs, by rw [he]; omega, ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rwa [show H (s.n + (N + 1) + 1) - s.dis - dt C rd s x
              = H ((tallyPre C (fun z => (rd z).cut) s x).n + N + 1)
                - (tallyPre C (fun z => (rd z).cut) s x).dis by rw [he]; omega]
      · obtain ⟨rfl, hd⟩ := h
        obtain ⟨rfl, hl⟩ := tallyStep_inr_pre C _ hst (by simp)
        obtain ⟨hn₀, hlt⟩ := tallyLook_success' C hl
        have htr := (tallyPre_cases (fun z => (rd z).cut) C s x).1
        rcases (tallyPre_cases (fun z => (rd z).cut) C s x).2 with
          ⟨-, hed, hn, -, hdis'⟩ | ⟨-, hn0, hd0, -⟩
        · have hdR : disRate C D rd (tallyPre C (fun z => (rd z).cut) s x)
              = disRate C D rd s := by
            simp only [disRate, htr, hed]
          have hdtv : (tallyPre C (fun z => (rd z).cut) s x).dis = s.dis + dt C rd s x := by
            simp only [dt]; rw [hdis']; split_ifs <;> omega
          rw [hn] at hlt
          obtain ⟨hle, hPH⟩ := hfire _ _ hlt
          refine .inr ⟨0, by simp, by simpa using hPH, by simp only [add_zero]; omega, ?_⟩
          simp only [EvC]
          exact ⟨hdR ▸ hd, by simp only [add_zero]; omega⟩
        · rw [hn0, hd0, binomSfGe_zero_left, sub_zero] at hlt
          linarith
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (tallyStep C fun z => (rd z).cut)
        (fun e s' => e = .success ∧ C.εd ≤ disRate C D rd s') tallyStart (List.ofFn xs)}
      ⊆ ⋃ q ∈ Finset.range T ×ˢ Finset.range T,
          {xs | SkipC C rd (Q q.2) q.1 tallyStart (List.ofFn xs)} := by
    intro xs hxs
    rcases hincl _ _ hxs with ⟨i, N, hi, hN, h⟩ | ⟨N, hN, hP, -, h⟩
    · simp only [List.length_ofFn] at hi hN
      exact Set.mem_biUnion (x := (i, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hN⟩) h
    · simp only [List.length_ofFn] at hN
      refine Set.mem_biUnion (x := (0, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 (by omega), Finset.mem_range.2 hN⟩) ?_
      simp only [Set.mem_ofPred_eq, SkipC, Q]
      simp only [tallyStart, zero_add, Nat.sub_zero] at hP h
      exact ⟨hP, h⟩
  have hterm : ∀ q ∈ Finset.range T ×ˢ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | SkipC C rd (Q q.2) q.1 tallyStart (List.ofFn xs)} ≤ ENNReal.ofReal C.a := by
    intro q _
    refine skipC_le C D rd (Q q.2) (fun T' s => ?_) T q.1 tallyStart
    by_cases hP : P (q.2 + 1) (H (q.2 + 1))
    · exact (measure_mono fun xs hxs => hxs.2).trans ((evC_le C D rd hε0 hε1 T' q.2 _ s).trans
        (ENNReal.ofReal_le_ofReal hP.le))
    · exact (measure_mono (t := ∅) fun xs hxs => hP hxs.1).trans (by simp)
  calc _ ≤ ∑ q ∈ Finset.range T ×ˢ Finset.range T, (Measure.pi fun _ : Fin T => D)
        {xs | SkipC C rd (Q q.2) q.1 tallyStart (List.ofFn xs)} :=
        (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
    _ ≤ ∑ _q ∈ Finset.range T ×ˢ Finset.range T, ENNReal.ofReal C.a := Finset.sum_le_sum hterm
    _ = _ := by
        rw [Finset.sum_const, Finset.card_product, Finset.card_range, nsmul_eq_mul,
          ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_mul (by positivity)]
        push_cast
        rw [ENNReal.ofReal_natCast]

end Success

theorem success_sound_holds : SuccessSound := by
  intro α _ _ rd D _ C T hε0 hε1 ha1
  exact success_sound C D rd hε0 hε1 ha1 T

end OrthoDFA
