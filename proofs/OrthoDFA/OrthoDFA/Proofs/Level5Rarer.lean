import OrthoDFA.Proofs.Level5Harvest
import OrthoDFA.Proofs.Level5RareReads

/-!
# A successful round rarely reads its tree's midfixes on a rarer side

Off any step reading on a rarer side, no probe of the last hypothesis reads a prefix followed by
one of its tree's midfixes on a rarer side. Success sees at least
`log(1/a)/log((1 − 7εd/8)/(1 − εd))` probes, so a hypothesis whose fresh probes do so at least `εd/(8 − 7εd)` of the time is unlikely
to reach it (`rarer_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Good

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- `bad_end_skip` over runs none of whose steps satisfies `Wst`, the weight multiplying by a
probe's factor only at such steps. -/
theorem bad_end_skip_good (Wst : Level5State α → FreeMonoid α → Prop) (P : Level5State α → Prop)
    (hP : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → P s')
    (W : Level5State α → ℝ) (f : Level5State α → FreeMonoid α → ℝ) (ok : Level5State α → Prop)
    (a : ℝ) (Bad : Level5Ending α → Level5State α → Prop)
    (hnew : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → s'.version ≠ s.version → W s' = 1)
    (hkeep : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → s'.version = s.version →
      ¬ Wst s x → W s' = W s * f s x ∧ (ok s' ↔ ok s))
    (hbad : ∀ s s' x e, P s → applyProbe C cut s x = .inr (e, s') → Bad e s' → ¬ Wst s x →
      ok s ∧ 1 ≤ a * (W s * f s x)) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), P s →
      ¬ AnyStep (applyProbe C cut) Wst s l → RunEnds (applyProbe C cut) Bad s l →
        (∃ v, s.version < v ∧ SkipV C cut (EvL C cut ok f a) v s l)
          ∨ EvL C cut ok f (a * W s) s l
  | [], _, _, _, h => h.elim
  | x :: xs, s, hs, hg, h => by
    have hx : ¬ Wst s x := fun h' => hg (.inl h')
    simp only [RunEnds] at h
    rcases hst : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [hst] at h
    · have hs' := hP s s' x hs hst
      rcases bad_end_skip_good Wst P hP W f ok a Bad hnew hkeep hbad xs s' hs'
        (fun h' => hg (.inr ⟨s', hst, h'⟩)) h with ⟨v, hv, hsk⟩ | hev
      · by_cases hver : s'.version = s.version
        · refine .inl ⟨v, hver ▸ hv, ?_⟩
          simp only [SkipV, show s.version ≠ v by omega, if_false, hst]
          exact hsk
        · have hlt : s.version < s'.version :=
            lt_of_le_of_ne (applyProbe_version C cut hst) (Ne.symm hver)
          refine .inl ⟨v, by omega, ?_⟩
          simp only [SkipV, show s.version ≠ v by omega, if_false, hst]
          exact hsk
      · by_cases hver : s'.version = s.version
        · obtain ⟨hW, hok⟩ := hkeep s s' x hs hst hver hx
          refine .inr ⟨?_, .inr ?_⟩
          · rcases xs with _ | ⟨y, ys⟩
            · exact hev.elim
            · exact hok.1 hev.1
          · rw [hst]
            refine ⟨hver, ?_⟩
            rw [hW, ← mul_assoc] at hev
            exact hev
        · have hW := hnew s s' x hs hst hver
          rw [hW, mul_one] at hev
          refine .inl ⟨s'.version, lt_of_le_of_ne (applyProbe_version C cut hst) (Ne.symm hver), ?_⟩
          · simp only [SkipV, show s.version ≠ s'.version from fun h => hver h.symm, if_false, hst]
            rcases xs with _ | ⟨y, ys⟩
            · exact hev.elim
            · simp only [SkipV, if_true]
              exact hev
    · obtain ⟨hok, h1⟩ := hbad s s' x e hs hst h hx
      exact .inr ⟨hok, .inl (by rw [mul_assoc]; exact h1)⟩


/-- `site_le` over runs none of whose steps satisfies `Wst`. A test whose weight is one at a fresh
hypothesis and multiplies by a probe's factor while the hypothesis keeps, at hypotheses where the
factor averages at most one, ends the round badly with chance at most `versionBound · a`. -/
theorem site_le_good (Wst : Level5State α → FreeMonoid α → Prop) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hL : 2 ≤ C.maxLeaves)
    (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j)
    (W : Level5State α → ℝ) (f : Level5State α → FreeMonoid α → ℝ) (ok : Level5State α → Prop)
    (Bad : Level5Ending α → Level5State α → Prop)
    (hf0 : ∀ s x, 0 ≤ f s x) (hf : ∀ s, ok s → ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ 1)
    (hfresh : ∀ T E v, W (freshState T E v) = 1)
    (hkeep : ∀ s x, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      ¬ Wst s x → W (keepProbe C cut s x) = W s * f s x ∧ (ok (keepProbe C cut s x) ↔ ok s))
    (htoo : ∀ s, ¬ Bad .tooBig s)
    (hbad : ∀ s x e, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      ¬ Wst s x → Bad e (keepProbe C cut s x) →
      runTests C (keepProbe C cut s x) = some e → ok s ∧ 1 ≤ C.testLevel * W (keepProbe C cut s x))
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | ¬ AnyStep (applyProbe C cut) Wst initialState (List.ofFn xs)
          ∧ RunEnds (applyProbe C cut) Bad initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set Q := EvL C cut ok f C.testLevel
  have hsub : {xs : Fin T → FreeMonoid α | ¬ AnyStep (applyProbe C cut) Wst initialState
      (List.ofFn xs) ∧ RunEnds (applyProbe C cut) Bad initialState (List.ofFn xs)}
        ⊆ ⋃ v ∈ Finset.range (versionBound C (Fintype.card α)),
        {xs | SkipV C cut Q v initialState (List.ofFn xs)} := by
    intro xs hxs
    have key := bad_end_skip_good C cut Wst (RunInv C) (fun s s' x hs h => runInv_step C cut hs h)
      W f ok
      C.testLevel Bad
      (fun s s' x hs h hv => by
        rcases applyProbe_inl_shape C cut h with ⟨-, T, E, he⟩ | he
        · rw [he]; exact hfresh _ _ _
        · exact absurd (by rw [he]; rfl) hv)
      (fun s s' x hs h hv hw => by
        have hm := not_member_of_inl C cut h hv
        rcases applyProbe_inl_shape C cut h with ⟨hlt, -⟩ | he
        · omega
        · rw [he]; exact hkeep s x hs.2 hm hw)
      (fun s s' x e hs h hb hw => by
        have hm := not_member_of_inr C cut ha ha1 hn₀ hexc h fun he => htoo _ (he ▸ hb)
        rcases applyProbe_inr_shape C cut ha ha1 hn₀ hexc h with rfl | ⟨rfl, hr⟩
        · exact absurd hb (htoo _)
        · obtain ⟨hok, h1⟩ := hbad s x e hs.2 hm hw hb hr
          exact ⟨hok, by rw [← (hkeep s x hs.2 hm hw).1]; exact h1⟩)
      (List.ofFn xs) initialState (runInv_start C hL) hxs.1 hxs.2
    have hW : W initialState = 1 := hfresh _ _ _
    obtain ⟨v, hsk⟩ : ∃ v, SkipV C cut Q v initialState (List.ofFn xs) := by
      rcases key with ⟨v, -, hsk⟩ | hev
      · exact ⟨v, hsk⟩
      · rw [hW, mul_one] at hev
        refine ⟨0, ?_⟩
        rcases hl : List.ofFn xs with _ | ⟨y, ys⟩ <;> rw [hl] at hev
        · exact hev.elim
        · simp only [SkipV, show (initialState : Level5State α).version = 0 from rfl, if_true]
          exact hev
    obtain ⟨s', hs', hv⟩ := skipV_runInv C cut Q v _ _ (runInv_start C hL) hsk
    exact Set.mem_biUnion (Finset.mem_range.2 (hv ▸ version_lt C hs'.1 hL)) hsk
  refine (measure_mono hsub).trans ((measure_biUnion_finset_le _ _).trans ?_)
  have hQ : ∀ v ∈ Finset.range (versionBound C (Fintype.card α)),
      (Measure.pi fun _ : Fin T => D) {xs | SkipV C cut Q v initialState (List.ofFn xs)}
        ≤ ENNReal.ofReal C.testLevel := fun v _ =>
    skipV_le C cut D (RunInv C) (fun s s' x hs h => runInv_step C cut hs h) Q v
      (fun T' s _ _ => evL_le C cut D ok f hf0 hf T' C.testLevel s ha.le) T _ (runInv_start C hL)
  refine (Finset.sum_le_sum hQ).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
    ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]

end Good

/-- A success test of `p` against `7p/8` that fired has seen at least `log (1/a)` over
`log ((1 − 7p/8)/(1 − p))` draws. -/
theorem success_draws {p a : ℝ} (hp : 0 < p) (hp1 : p < 1) (ha : 0 < a) {n h : ℕ}
    (hh : h ≤ n) (hs : seqRate p (7 / 8 * p) a n h) :
    Real.log (1 / a) ≤ n * Real.log ((1 - 7 / 8 * p) / (1 - p)) := by
  rw [seqRate_iff_log hp hp1 (by linarith) (by linarith) ha hh] at hs
  have hA : Real.log (7 / 8 * p / p) ≤ 0 := by
    rw [show 7 / 8 * p / p = 7 / 8 by field_simp]
    exact (Real.log_neg (by norm_num) (by norm_num)).le
  have hB : 0 ≤ Real.log ((1 - 7 / 8 * p) / (1 - p)) :=
    Real.log_nonneg (by rw [le_div_iff₀ (by linarith)]; linarith)
  have hhn : (h : ℝ) ≤ n := by exact_mod_cast hh
  have hh0 : (0 : ℝ) ≤ h := Nat.cast_nonneg h
  have h1 : (h : ℝ) * Real.log (7 / 8 * p / p) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hh0 hA
  have h2 : (n - h : ℝ) * Real.log ((1 - 7 / 8 * p) / (1 - p))
      ≤ n * Real.log ((1 - 7 / 8 * p) / (1 - p)) :=
    mul_le_mul_of_nonneg_right (by linarith) hB
  linarith

section Rarer

variable {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (C : Level5Settings) (rd : FreeMonoid α → ARU)

/-- The probes reading a prefix followed by one of the state's tree's midfixes on a rarer side. -/
def rarerSet (s : Level5State α) : Set (FreeMonoid α) :=
  rarerProbes G.M G.readDist rd C.walkStart ⟨s.tree, Function.uncurry s.edges⟩

/-- Success at a hypothesis whose fresh probes read so at least `εd/(8 − 7εd)` of the time. -/
def RarerBad (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .success ∧ C.disagreementRate < 2 / 3
    ∧ C.disagreementRate / (8 - 7 * C.disagreementRate) ≤ D.real (rarerSet G C rd s)

theorem wrongStep_of_rarer {s : Level5State α} {x : FreeMonoid α} (h : x ∈ rarerSet G C rd s) :
    WrongStep G rd C s x := by
  obtain ⟨j, -, hjx, m, hm, hc⟩ := h
  exact ⟨_, Finset.mem_union_left _ (mem_potMid hjx hm), hc⟩

open scoped Classical in
theorem rarer_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j)
    (hdalt : C.disagreementAltRate = 7 / 8 * C.disagreementRate) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | ¬ AnyStep (applyProbe C fun z => (rd z).cut) (WrongStep G rd C) initialState
            (List.ofFn xs)
          ∧ RunEnds (applyProbe C fun z => (rd z).cut) (RarerBad G D C rd) initialState
            (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  set p := C.disagreementRate
  have hp := hSq.εd_pos
  have hp1 := hSq.εd_lt
  set u := p / (8 - 7 * p) with hud
  have h87 : 0 < 8 - 7 * p := by linarith
  have hq : 1 - u = (1 - p) / (1 - 7 / 8 * p) := by
    have h1 : (8 - 7 * p) ≠ 0 := by linarith
    have h2 : (1 - 7 / 8 * p) ≠ 0 := by linarith
    have e1 : 1 - p / (8 - 7 * p) = (8 - 8 * p) / (8 - 7 * p) := by
      rw [eq_div_iff h1, sub_mul, div_mul_cancel₀ _ h1]; ring
    have e2 : (1 - p) / (1 - 7 / 8 * p) = (8 - 8 * p) / (8 - 7 * p) := by
      rw [div_eq_div_iff h2 h1]; ring
    rw [hud, e1, e2]
  have hq0 : 0 < 1 - u := by rw [hq]; exact div_pos (by linarith) (by linarith)
  set r := 1 / (1 - u) with hrd
  have hr0 : 0 < r := by positivity
  have hr : r = (1 - 7 / 8 * p) / (1 - p) := by rw [hrd, hq, one_div_div]
  refine site_le_good C cut (WrongStep G rd C) D hL hSq.a_pos hSq.a_lt hn₀ hexc
    (fun s => r ^ s.probes) (fun s x => if x ∈ rarerSet G C rd s then 0 else r)
    (fun s => u ≤ D.real (rarerSet G C rd s)) (RarerBad G D C rd)
    (fun s x => by split <;> [exact le_rfl; exact hr0.le]) (fun s hs => ?_)
    (fun T E v => by simp [freshState]) (fun s x hs _ hw => ⟨?_, Iff.rfl⟩)
    (fun s h => by simp [RarerBad] at h) (fun s x e hs _ hw hb hres => ⟨hb.2.2, ?_⟩) T
  · -- the factor averages at most one
    have hmeas : MeasurableSet (rarerSet G C rd s) := (Set.to_countable _).measurableSet
    have hpt : ∀ x, ENNReal.ofReal (if x ∈ rarerSet G C rd s then 0 else r)
        = (rarerSet G C rd s)ᶜ.indicator (fun _ => ENNReal.ofReal r) x := by
      intro x
      by_cases hx : x ∈ rarerSet G C rd s <;> simp [hx, Set.indicator]
    simp_rw [hpt]
    rw [lintegral_indicator hmeas.compl, setLIntegral_const, ← ofReal_measureReal,
      measureReal_compl hmeas, probReal_univ, ← ENNReal.ofReal_mul hr0.le, ← ENNReal.ofReal_one]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [hrd, one_div_mul_eq_div, div_le_one hq0]
    linarith
  · have hx : x ∉ rarerSet G C rd s := fun h => hw (wrongStep_of_rarer G C rd h)
    show r ^ (s.probes + 1) = r ^ s.probes * (if x ∈ rarerSet G C rd s then 0 else r)
    rw [if_neg hx, pow_succ]
  · obtain ⟨rfl, -, -⟩ := hb
    have hsucc := runTests_success' C hres
    rw [hdalt] at hsucc
    have hci := countInv_keep C cut hs x
    have hn := success_draws hp hp1 hSq.a_pos hci.2.2.1 hsucc
    rw [← hr] at hn
    set n := (keepProbe C cut s x).probes
    show 1 ≤ C.testLevel * r ^ n
    have hlr : Real.log (1 / C.testLevel) ≤ Real.log (r ^ n) := by
      rw [Real.log_pow]; exact hn
    rw [Real.log_le_log_iff (by have := hSq.a_pos; positivity) (by positivity)] at hlr
    rw [div_le_iff₀ hSq.a_pos] at hlr
    linarith

end Rarer

end OrthoDFA
