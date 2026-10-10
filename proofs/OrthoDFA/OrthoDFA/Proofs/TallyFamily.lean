import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyEvent
import OrthoDFA.Proofs.TallySpurNoise
import OrthoDFA.Proofs.TallyKeep
import OrthoDFA.Proofs.TallyRace
import OrthoDFA.Proofs.TallySuccess
import OrthoDFA.Proofs.FamilyRead

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
  [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [Fintype σ] in
/-- The family's reads as a read model: independent, measurable, each with its state's law, and
every law nonnegative. -/
theorem readModel_of_family (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (ε ε₂ κ θ : ℝ)
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (hF : SuffixFree F)
    (hband : BandPasses F.card kl kh (1 - O.ηIn) O.ηOut ε ε₂ κ) :
    ∃ G : ReadModel α σ, G.M = M ∧ G.θ = θ ∧ G.κ = κ ∧ G.ε = ε
      ∧ (∀ w, readProb O F kl kh w = G.dist (M.eval w.toList))
      ∧ (∀ z, Measurable (familyRead O.mq F kl kh z))
      ∧ iIndepFun (fun z => familyRead O.mq F kl kh z) μ
      ∧ (∀ z r, μ.real {ω | familyRead O.mq F kl kh z ω = r} = G.dist (G.M.eval z.toList) r)
      ∧ ∀ q r, 0 ≤ G.dist q r := by
  classical
  obtain ⟨hind, hq⟩ := family_read_bound_holds O M F kl kh ε ε₂ κ hL hF hband
  choose d hd _ hrare using hq
  set dist : σ → ARU → ℝ := fun q =>
    if ∃ w : FreeMonoid α, M.eval w.toList = q then d q else fun _ => 0
  have hdist : ∀ w : FreeMonoid α, readProb O F kl kh w = dist (M.eval w.toList) := by
    intro w
    have h : ∃ w' : FreeMonoid α, M.eval w'.toList = M.eval w.toList := ⟨w, rfl⟩
    simp only [dist, if_pos h]
    exact hd _ w rfl
  have hr : ∀ q, min (dist q .accept) (dist q .reject) ≤ max ε (κ * dist q .undecided) := by
    intro q
    by_cases h : ∃ w : FreeMonoid α, M.eval w.toList = q
    · simp only [dist, if_pos h]
      exact hrare q
    · simp only [dist, if_neg h]
      simp
  refine ⟨⟨M, dist, θ, κ, ε, hr⟩, rfl, rfl, rfl, rfl, hdist,
    fun z => measurable_from_nat.comp (measurable_voteCount O F z), hind,
    fun z r => congrFun (hdist z) r, fun q r => ?_⟩
  simp only [dist]
  split_ifs with h
  · calc (0 : ℝ) ≤ readProb O F kl kh h.choose r := measureReal_nonneg
      _ = d q r := congrFun (hd q _ h.choose_spec) r
  · exact le_rfl

omit [Fintype σ] in
theorem leafOf_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided)) :
    ∀ (T : DTree α) (q : σ), ({ G with κ := κ', rare := h } : ReadModel α σ).leafOf T q
      = G.leafOf T q
  | .leaf, _ => rfl
  | .node m r a, q => by
    simp only [ReadModel.leafOf, ReadModel.side, ReadModel.at', leafOf_with G κ' h r q,
      leafOf_with G κ' h a q]
    congr

omit [Fintype σ] in
theorem grown_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    {T : DTree α} {f : ℕ} (hg : G.Grown T f) : ({ G with κ := κ', rare := h } : ReadModel α σ).Grown T f := by
  induction hg with
  | start => exact .start
  | real _ ht ht₀ hs ih =>
    refine .real ih ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 8 <;> exact decide_eq_decide.2 Iff.rfl
  | fake _ hp ht ht₀ hs ih =>
    refine .fake ih hp ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 9 <;> exact decide_eq_decide.2 Iff.rfl

omit [Fintype σ] in
theorem grown_of_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    {T : DTree α} {f : ℕ} (hg : ({ G with κ := κ', rare := h } : ReadModel α σ).Grown T f) :
    G.Grown T f := by
  induction hg with
  | start => exact .start
  | real _ ht ht₀ hs ih =>
    refine .real ih ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 8 <;> exact decide_eq_decide.2 Iff.rfl
  | fake _ hp ht ht₀ hs ih =>
    refine .fake ih hp ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 9 <;> exact decide_eq_decide.2 Iff.rfl

omit [Fintype σ] in
theorem untrueAt_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    ({ G with κ := κ', rare := h } : ReadModel α σ).untrueAt rd k T edges
      = G.untrueAt rd k T edges := by
  simp only [ReadModel.untrueAt, ReadModel.TrueRec, leafOf_with]

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ O M F kl kh ε ε₂ κ κs θ D _ C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt'
    θr εd' Xe θs' Xs η ν p₀ l hL hF hband hκ hκs hε hρ hθ hk hp₀ hpmax hl hθg hθgpt hlen hm hLmax
    hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS hhS' hφn hL1
    ha ha1 hθe hφe0 hexc hθgs0 hθgs1 hstart hθgpt1 hstartpt hφ0 hφ1 hlow hXe0 hXe hθs' hXs hlin hη
    hν hside hT
  obtain ⟨G₁, hM, hGθ, hGκ, hGε, hlawG, hmeas, hind, hlaw, hnn⟩ :=
    readModel_of_family (μ := μ) O M F kl kh ε ε₂ κ θ hL hF hband
  subst hM hGθ hGκ hGε
  have hrare : ∀ r, min (G₁.dist r .accept) (G₁.dist r .reject)
      ≤ max G₁.ε (κs * G₁.dist r .undecided) := fun r =>
    (G₁.rare r).trans (max_le_max le_rfl (mul_le_mul_of_nonneg_right (by linarith) (hnn r _)))
  set G : ReadModel α σ := { G₁ with κ := κs, rare := hrare }
  refine ⟨G, rfl, rfl, rfl, rfl, hlawG, ?_⟩
  set read : FreeMonoid α → Ω → ARU := fun z ω => familyRead O.mq F kl kh z ω
  have hρ0 : 0 ≤ ρ := le_trans (by positivity) hρ
  have hθg0 : 0 ≤ θg := le_trans (by positivity) hθg
  have hθgpt0 : 0 ≤ θgpt := le_trans (by positivity) hθgpt
  have hTR := tally_round_of sub_round_holds fake_race_holds harvest_good_holds
    success_sound_holds (μ := μ) G read D C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt' θr εd' Xe θs'
    Xs η ν hlen hρ0 hm hLmax hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1
    hcond hhP hhS hhS' hφn hL1 ha ha1 hθg0 hθe hφe0 hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1
    hstartpt hφ0 hφ1 hlow (by linarith) hXe0 hXe hθs' hXs hlin hη hν hside hT
  have hE := tallyE_le (ρ := ρ) (θgs := θgs) G read hmeas hind hlaw hθ D C S L hL1 hlen hk hp₀
    hpmax hl hφe0 hφ0 hθe hθg hθgpt
  have hSp := spurious_le G₁ read hmeas hind hlaw hκ hε D C.k L S hlen hk hp₀ hpmax hκs hρ
  have hcard : ((classSet (Fintype.card σ + S) : Finset (DTree α)).card : ℝ)
      ≤ (1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α) ^ (Fintype.card σ + S) := by
    exact_mod_cast classSet_card (Fintype.card σ + S)
  refine hTR.trans (add_le_add (add_le_add (add_le_add (hE.trans ?_) le_rfl) le_rfl) le_rfl)
  refine add_le_add (add_le_add (add_le_add ((measure_mono ?_).trans (hSp.trans ?_)) ?_) ?_) ?_
  · intro ω hω
    simp only [Set.mem_setOf_eq, not_forall] at hω ⊢
    obtain ⟨T, edges, ⟨f, hf, hg⟩, he, hlt⟩ := hω
    refine ⟨T, edges, ⟨f, hf, grown_of_with G₁ κs hrare hg⟩, he, ?_⟩
    rwa [untrueAt_with] at hlt
  all_goals
    refine ENNReal.ofReal_le_ofReal ?_
  · gcongr ?_ * _ * _
  · gcongr ?_ * _ * Real.exp ?_
    exact le_of_eq (by ring)
  · gcongr ?_ * _
  · gcongr ?_ * _ * Real.exp ?_
    exact le_of_eq (by ring)

end OrthoDFA
