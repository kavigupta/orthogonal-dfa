import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyEvent
import OrthoDFA.Proofs.TallyKeep
import OrthoDFA.Proofs.TallyRace
import OrthoDFA.Proofs.TallySuccess
import OrthoDFA.Proofs.FamilyRead

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
  [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [Fintype σ] in
/-- The family's reads as a read model: independent, measurable, each with its state's law. -/
theorem readModel_of_family (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (ε ε₂ κ θ : ℝ)
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (hF : SuffixFree F)
    (hband : BandPasses F.card kl kh (1 - O.ηIn) O.ηOut ε ε₂ κ) :
    ∃ G : ReadModel α σ, G.M = M ∧ G.θ = θ ∧ G.κ = κ ∧ G.ε = ε
      ∧ (∀ w, readProb O F kl kh w = G.dist (M.eval w.toList))
      ∧ (∀ z, Measurable (familyRead O.mq F kl kh z))
      ∧ iIndepFun (fun z => familyRead O.mq F kl kh z) μ
      ∧ ∀ z r, μ.real {ω | familyRead O.mq F kl kh z ω = r} = G.dist (G.M.eval z.toList) r := by
  obtain ⟨hind, hq⟩ := family_read_bound_holds O M F kl kh ε ε₂ κ hL hF hband
  choose dist hdist _ hrare using hq
  refine ⟨⟨M, dist, θ, κ, ε, hrare⟩, rfl, rfl, rfl, rfl, fun w => hdist _ w rfl,
    fun z => measurable_from_nat.comp (measurable_voteCount O F z), hind,
    fun z r => congrFun (hdist _ z rfl) r⟩

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ O M F kl kh ε ε₂ κ θ D _ C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt' θr
    εd' Xe θs' Xs η ν p₀ l hL hF hband hθ hk hp₀ hpmax hl hθg hθgpt hlen hρ0 hm hLmax hn₀ hn₀'
    hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS hhS' hφn hL1 ha ha1
    hθe hφe0 hexc hθgs0 hθgs1 hstart hθgpt1 hstartpt hφ0 hφ1 hlow hXe0 hXe hθs' hXs hlin hη hν
    hside hT
  obtain ⟨G, hM, hGθ, hGκ, hGε, hlawG, hmeas, hind, hlaw⟩ :=
    readModel_of_family (μ := μ) O M F kl kh ε ε₂ κ θ hL hF hband
  subst hM hGθ hGκ
  refine ⟨G, rfl, rfl, rfl, hGε, hlawG, ?_⟩
  set read : FreeMonoid α → Ω → ARU := fun z ω => familyRead O.mq F kl kh z ω
  have hθg0 : 0 ≤ θg := le_trans (by positivity) hθg
  have hθgpt0 : 0 ≤ θgpt := le_trans (by positivity) hθgpt
  have hTR := tally_round_of sub_round_holds fake_race_holds harvest_good_holds
    success_sound_holds (μ := μ) G read D C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt' θr εd' Xe θs'
    Xs η ν hlen hρ0 hm hLmax hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1
    hcond     hhP hhS hhS' hφn hL1 ha ha1 hθg0 hθe hφe0 hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hφ0
    hφ1 hlow (by linarith) hXe0 hXe hθs' hXs hlin hη hν hside hT
  have hE := tallyE_le (ρ := ρ) (θgs := θgs) G read hmeas hind hlaw hθ D C S L hL1 hlen hk hp₀
    hpmax hl hφe0 hφ0 hθe hθg hθgpt
  have hcard : ((classSet (Fintype.card σ + S) : Finset (DTree α)).card : ℝ)
      ≤ (1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α) ^ (Fintype.card σ + S) := by
    exact_mod_cast classSet_card (Fintype.card σ + S)
  refine hTR.trans (add_le_add (add_le_add (add_le_add (hE.trans ?_) le_rfl) le_rfl) le_rfl)
  refine add_le_add (add_le_add (add_le_add le_rfl ?_) ?_) ?_ <;>
    refine ENNReal.ofReal_le_ofReal ?_
  · gcongr ?_ * _ * Real.exp ?_
    exact le_of_eq (by ring)
  · gcongr ?_ * _
  · gcongr ?_ * _ * Real.exp ?_
    exact le_of_eq (by ring)

end OrthoDFA
