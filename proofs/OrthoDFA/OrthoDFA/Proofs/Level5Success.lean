import OrthoDFA.Proofs.Level5Keep

/-!
# Success is sound

Within a stretch the hypothesis is fixed, so its searches are independent draws at its
disagreement rate `d`. Where `d ≥ εd`, the success test's likelihood ratio times `a` is a
supermartingale over the stretch, so by Ville's inequality the test fires in it with chance at
most `a`; over the stretches' starts, a round ends in success at a hypothesis disagreeing on at
least `εd` of the probes with chance at most `T a`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Success

variable (C : Level5Settings) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- After `i` probes, the run is at a state from which `Q` holds of the remaining draws. -/
def SkipC (Q : Level5State α → List (FreeMonoid α) → Prop) : ℕ → Level5State α
    → List (FreeMonoid α) → Prop
  | 0, s, l => Q s l
  | _ + 1, _, [] => False
  | i + 1, s, x :: xs => match applyProbe C (fun z => (rd z).cut) s x with
    | .inl s' => SkipC Q i s' xs
    | .inr _ => False

theorem skipC_le [IsProbabilityMeasure D] (Q : Level5State α → List (FreeMonoid α) → Prop)
    {c : ENNReal}
    (hQ : ∀ (T : ℕ) (s : Level5State α), (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)}
      ≤ c) :
    ∀ (T i : ℕ) (s : Level5State α),
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
          rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | _
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipC C rd Q (i + 1) s (List.ofFn xs)}}
                    = {xs | SkipC C rd Q i s' (List.ofFn xs)} := by
              ext xs; simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipC, hst]
            rw [this]; exact ih i s'
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipC C rd Q (i + 1) s (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipC, hst, Set.mem_empty_iff_false]
            rw [this, measure_empty]; exact zero_le
      _ = c := by rw [lintegral_const, measure_univ, mul_one]

open scoped Classical in
theorem runTests_success' {s : Level5State α} (h : runTests C s = some .success) :
    seqRate C.disagreementRate C.disagreementAltRate C.testLevel s.stretchProbes
      s.stretchDisagreements := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4 <;> first | exact h4 | simp at h

open scoped Classical in
/-- A round ends in success at a hypothesis disagreeing on at least `εd` of the probes with chance
at most `T a`: within a stretch the hypothesis is fixed, so the success test's ratio times `a` is
a supermartingale there, and it errs at most `a` a stretch. -/
theorem success_sound [IsProbabilityMeasure D] (hSq : C.SeqOK) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut)
          (fun e s' => e = .success ∧ C.disagreementRate ≤ disRate C D rd s') initialState
              (List.ofFn xs)}
      ≤ ENNReal.ofReal (T * C.testLevel) := by
  have ha := hSq.a_pos
  have ha1 := hSq.a_lt
  set ok : Level5State α → Prop := fun s => C.disagreementRate ≤ disRate C D rd s
  set f := searchFactor C rd C.disagreementRate C.disagreementAltRate
  set LR : Level5State α → ℝ := fun s => likelihoodRatio C.disagreementRate C.disagreementAltRate
    s.stretchProbes s.stretchDisagreements
  set Q : Level5State α → List (FreeMonoid α) → Prop := fun s l =>
    s.stretchProbes = 0 ∧ s.stretchDisagreements = 0
      ∧ EvL C (fun z => (rd z).cut) ok f C.testLevel s l
  have hdr : ∀ s s' : Level5State α, s'.tree = s.tree → s'.edges = s.edges →
      disRate C D rd s' = disRate C D rd s := by
    intro s s' h1 h2; simp only [disRate, h1, h2]
  have hincl : ∀ (l : List (FreeMonoid α)) (s : Level5State α), CountInv s →
      RunEnds (applyProbe C fun z => (rd z).cut)
        (fun e s' => e = .success ∧ C.disagreementRate ≤ disRate C D rd s') s l →
      (∃ i, i < l.length ∧ SkipC C rd Q i s l)
        ∨ EvL C (fun z => (rd z).cut) ok f (C.testLevel * LR s) s l := by
    intro l
    induction l with
    | nil => intro s _ h; exact h.elim
    | cons x xs ih =>
      intro s hci h
      simp only [RunEnds] at h
      rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' (countInv_step C _ hci hst) h with ⟨i, hi, hk⟩ | hev
        · refine .inl ⟨i + 1, by simp; omega, ?_⟩
          simp only [SkipC, hst]
          exact hk
        · have hne : xs ≠ [] := by rintro rfl; exact hev
          rcases applyProbe_newStretch C _ hst with ⟨hn0, hd0, -⟩ | ⟨hpre, hn, htr, hed⟩
          · refine .inl ⟨1, by
              rcases xs with _ | _
              · exact absurd rfl hne
              · simp, ?_⟩
            simp only [SkipC, hst, Q]
            refine ⟨hn0, hd0, ?_⟩
            simpa [LR, hn0, hd0, likelihoodRatio_zero] using hev
          · subst hpre
            have hok : ok s := by
              rcases xs with _ | ⟨y, ys⟩
              · exact hev.elim
              · have := hev.1
                simp only [ok] at this ⊢
                rwa [hdr s _ htr hed] at this
            refine .inr ?_
            simp only [EvL]
            refine ⟨hok, .inr ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rw [show C.testLevel * LR s * f s x
                = C.testLevel * LR (recordProbe C (fun z => (rd z).cut) s x) by
              simp only [LR, f]
              rw [lr_search_step C rd _ _ hci hn]; ring]
            exact hev
      · obtain ⟨rfl, hd⟩ := h
        obtain ⟨rfl, hl⟩ := applyProbe_inr_pre C _ hst (by simp)
        have hfire := runTests_success' C hl
        have htr := (recordProbe_cases (fun z => (rd z).cut) C s x).1
        rcases (recordProbe_cases (fun z => (rd z).cut) C s x).2 with
          ⟨-, hed, hn, -, -⟩ | ⟨-, hn0, hd0, -⟩
        · have hdR : disRate C D rd (recordProbe C (fun z => (rd z).cut) s x)
              = disRate C D rd s := by
            simp only [disRate, htr, hed]
          refine .inr ?_
          simp only [EvL]
          refine ⟨show C.disagreementRate ≤ disRate C D rd s by rw [← hdR]; exact hd, .inl ?_⟩
          have hl' : 1 / C.testLevel ≤ LR (recordProbe C (fun z => (rd z).cut) s x) := hfire
          simp only [LR] at hl'
          rw [lr_search_step C rd _ _ hci hn, div_le_iff₀ ha] at hl'
          simp only [LR, f]
          linarith
        · rw [hn0, hd0] at hfire
          exact absurd hfire (not_seqRate_zero ha ha1)
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (applyProbe C fun z => (rd z).cut)
        (fun e s' => e = .success ∧ C.disagreementRate ≤ disRate C D rd s') initialState
            (List.ofFn xs)}
      ⊆ ⋃ i ∈ Finset.range T, {xs | SkipC C rd Q i initialState (List.ofFn xs)} := by
    intro xs hxs
    rcases hincl _ _ countInv_start hxs with ⟨i, hi, h⟩ | h
    · simp only [List.length_ofFn] at hi
      exact Set.mem_biUnion (Finset.mem_range.2 hi) h
    · have hne : List.ofFn xs ≠ [] := by intro h0; rw [h0] at h; exact h
      have hT : 0 < T := by
        rcases T with _ | T
        · simp at hne
        · omega
      refine Set.mem_biUnion (x := 0) (Finset.mem_range.2 hT) ?_
      simp only [Set.mem_ofPred_eq, SkipC, Q]
      refine ⟨rfl, rfl, ?_⟩
      simpa [LR, initialState, likelihoodRatio_zero] using h
  have hterm : ∀ i ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | SkipC C rd Q i initialState (List.ofFn xs)} ≤ ENNReal.ofReal C.testLevel := by
    intro i _
    refine skipC_le C D rd Q (fun T' s => ?_) T i initialState
    refine (measure_mono fun xs hxs => hxs.2.2).trans ?_
    exact evL_le C (fun z => (rd z).cut) D ok f
      (fun s x => searchFactor_nonneg C rd hSq.εd_pos hSq.εd_lt hSq.εd_alt_pos.le
        (hSq.εd_alt.trans hSq.εd_lt).le s x)
      (fun s hs => lintegral_searchFactor_le C D rd hSq.εd_pos hSq.εd_lt hSq.εd_alt_pos.le
        (hSq.εd_alt.trans hSq.εd_lt).le (by
          have : C.disagreementAltRate - C.disagreementRate < 0 := by linarith [hSq.εd_alt]
          have hs' : C.disagreementRate ≤ disRate C D rd s := hs
          nlinarith))
      T' C.testLevel s ha.le
  calc _ ≤ ∑ i ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
        {xs | SkipC C rd Q i initialState (List.ofFn xs)} :=
        (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
    _ ≤ ∑ _i ∈ Finset.range T, ENNReal.ofReal C.testLevel := Finset.sum_le_sum hterm
    _ = _ := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, ENNReal.ofReal_mul (by positivity),
          ENNReal.ofReal_natCast]

end Success

theorem success_sound_holds : SuccessSound := by
  intro α _ _ rd D _ C T hSq
  exact success_sound C D rd hSq T

end OrthoDFA
