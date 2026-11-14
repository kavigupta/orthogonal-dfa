import RLStar.Proofs.LazyCore
import RLStar.Proofs.ClaimRLStarArith

/-!
# The lazy loop at the claim's knobs

`lazy_loop_core` at the knobs of `ClaimRLStarArith`, with `η₀ = 1/2 − s₀/2`, `c = ξc`, the
coverage's accuracy `ξc`, `t = ξc/4`, `tw = ε/12288` and `Ae = |Σ|`. A walk start of no letters
puts all of `D` on one prefix, which `p₀Tol < 1` rules out.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

lemma rlStarSucceeds_mono {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} {settings : LoopSettings K}
    {O : Oracle μ (FreeMonoid A)} {D Dsf T : Measure (FreeMonoid A)} {M : _root_.DFA A Q}
    {ε X X' R R' : ℝ} (hX : X ≤ X') (hR : R ≤ R') {p : settings.Draws Ω A}
    (h : RLStarSucceeds settings O D Dsf T M ε X R p) :
    RLStarSucceeds settings O D Dsf T M ε X' R' p := by
  obtain ⟨⟨rounds, hrun, ⟨hlen, o, ho, hc, hT, hXo⟩, hpk⟩, S, hS, hdep⟩ := h
  exact ⟨⟨rounds, hrun, ⟨hlen, o, ho, hc, hT, hXo.trans hX⟩, hpk⟩, S, hS.trans hR, hdep⟩

lemma lazyReads_mono_nX {K a b f Nr Nr' A nX nX' q : ℝ} (hK : 0 ≤ K) (hf : 0 ≤ f) (hA : 0 ≤ A)
    (hq : 0 ≤ q) (hN : Nr ≤ Nr') (h : nX ≤ nX') :
    lazyReads K a b f Nr A nX q ≤ lazyReads K a b f Nr' A nX' q := by
  unfold lazyReads
  gcongr

open Classical in
/-- `knP₀` on the region, positive off it. -/
def knTol (r : RLStarRun) : ℝ := if r ∈ rlStarRunRegion then knP₀ r else 1

lemma knTol_pos (r : RLStarRun) : 0 < knTol r := by
  unfold knTol
  split_ifs with hr
  · exact (knP₀_bounds hr).1
  · exact one_pos

set_option maxHeartbeats 2000000 in -- `lazy_loop_core`'s lets unfold against the knobs
theorem lazy_within : RLStarWithin.{u, v, w} knTol knPref knSuff knProbes knTrie knReads := by
  intro α _ _ _ σ _ Ω _ μ _ x M O L k D _ L₁ Ls hx hQ hA hLang hgap hbL hbN hlong hkL hpre hpAPu
  obtain ⟨Q, A, δ, ξ, pAP, s₀, β⟩ := x
  subst hQ hA
  set r : RLStarRun := ⟨⟨Fintype.card σ, Fintype.card α, δ, ξ, pAP, s₀, β⟩, L⟩ with hrd
  have hr : r ∈ rlStarRunRegion := hx
  obtain ⟨hQ1, hA2', hX, hQX, hLX, hδ, hδ2, hξc, hξc1, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨-, -, hξcξ, hξcβ⟩ := ξc_mem hr
  have hgap' : 2 * r.s₀ ≤ O.gap := hgap
  have hbL' : r.β ≤ (prefixLaw D L).real {p | p ∈ O.L} := hbL
  have hbN' : r.β ≤ (prefixLaw D L).real {p | p ∉ O.L} := hbN
  have hpre' : ∀ u, D.real {w | prefixOf w k = u} ≤ knP₀ r := fun u =>
    (hpre u).trans (le_of_eq (if_pos hr))
  have hkL' : k ≤ r.L := hkL
  have hA2 : 2 ≤ Fintype.card α := hx.2.1
  obtain ⟨hP0, hP1, hPcap, hPmc⟩ := knP₀_bounds hr
  -- a walk start of no letters is ruled out
  have hk1 : 1 ≤ k := by
    by_contra h0
    have hk0 : k = 0 := by omega
    have h := hpre' 1
    have hset : {w : FreeMonoid α | prefixOf w k = 1} = Set.univ := by
      ext w
      simp [hk0, prefixOf]
    rw [hset, probReal_univ] at h
    linarith
  obtain ⟨hεc0, -, -, hεc2, -⟩ := knεcov_bounds hr
  have hbal : 2 * knεcov r ≤ r.β := by
    have : r.ξc / knX r ≤ r.β := by
      rw [div_le_iff₀ (by linarith)]
      unfold knX
      linarith
    linarith
  obtain ⟨hι0, hι8, -⟩ := knι_bounds hr
  obtain ⟨hW1, hNrW, htwW, -⟩ := knW_bounds hr
  obtain ⟨-, hNh⟩ := knNh_bounds hr
  obtain ⟨-, -, hEk⟩ := knE_bounds hr
  obtain ⟨-, -, hJk⟩ := knJ_bounds hr
  obtain ⟨htw0, htwε⟩ := knTw_spec hr
  obtain ⟨settings, hwalk, hbud, -, -, hhd, hE', hJ', hb', hcnt, hprob⟩ :=
    lazy_loop_core (Ω := Ω) (μ := μ) M O L k (knη₀ r) r.ξc r.δ r.pAP r.ξc r.ξ (r.ξc / 4)
      (r.A : ℝ) (knP₀ r) (knι r) (knW r) (knTu r) (knTw r) (knE r) (knJ r) (knB r) (knNr r)
      (knNh r) D hLang (by rw [knη₀_s]; linarith) (by unfold knη₀; linarith)
      (hbal.trans hbL') (hbal.trans hbN') hξc hξc1 hδ (by linarith) hp hξc hA2 le_rfl hk1 hkL
      (by linarith) hlong hP0 hpre' hpAPu hPcap (knMcD_spec hr hP0 hPmc) hι0 hι8 hW1
      (hNh k hkL') (knSlack_spec hr) (hEk k hkL') (hJk k hkL') (knb_spec hr) (knNr_spec hr hkL')
      hNrW htwW htw0 htwε (knTu_pos hr) (knSlope_spec hr hkL') (knξT_spec hr)
  refine ⟨settings, hwalk, fun p rounds hrun => ?_, ?_⟩
  · obtain ⟨hlen, hpref, hsuff⟩ := hcnt p rounds hrun
    refine ⟨hpref, hsuff, ?_⟩
    have hlen' : (rounds.length : ℝ) ≤ r.Q + 1 := by exact_mod_cast hlen
    have hbud' : (settings.level3.probeBudget : ℝ) ≤ knB₃ r := hbud
    rw [hhd, hE', hJ', hb']
    unfold knProbes
    gcongr
  · obtain ⟨hRB0, -⟩ := knRB_bounds hr
    obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
    obtain ⟨hSC0, -⟩ := knSC_bounds hr
    obtain ⟨hHR1, -⟩ := knHR_bounds hr
    have hk : (k : ℝ) ≤ r.L := by exact_mod_cast hkL'
    have hRp : (Random.Kappa.probeSifts (L₁ + k) : ℝ) ≤ knRp r := by
      have e1 : r.L = L := rfl
      have e2 : r.Q = Fintype.card σ := rfl
      unfold knRp
      exact_mod_cast probeSifts_mono
        (show L + Fintype.card σ + 3 + k ≤ r.L + r.Q + 3 + r.L by omega)
    have hle : lazyReads (r.Q + 1) (6 * (r.Q + 2) * knRB r) (knPB r + 1) (knFV r)
        (knNr r + (r.Q + 1) * (knSC r * (knHR r * (Random.Kappa.probeSifts (L₁ + k) * (r.Q + 1)))))
        r.A (k * (6 * (r.Q + 2) * knRB r + knNh r)) r.Q ≤ knReads r := by
      unfold knReads knTrie
      refine lazyReads_mono_nX (by positivity) hFV0 (by positivity) (by positivity) ?_
        (by gcongr)
      gcongr
    refine le_trans (measureReal_mono (fun p hp => ?_) (measure_ne_top _ _)) hprob
    exact fun hs => hp (rlStarSucceeds_mono le_rfl hle hs)

end OrthoDFA

end
