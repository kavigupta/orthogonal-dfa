import RLStar.Proofs.LazyCore
import RLStar.Proofs.ClaimRLStarArith

/-!
# The lazy loop at the claim's knobs

`lazy_loop_core` at the knobs of `ClaimRLStarArith`, with `η₀ = 1/2 − s₀/2`, `c = ξc`, the
coverage's accuracy `ξc`, `t = tw = ξc/4` and `Ae = |Σ|`. A walk start of no letters puts all of
`D` on one prefix, which `p₀Tol < 1` rules out.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

lemma rlStarSucceeds_mono {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} {settings : LoopSettings K}
    {O : Oracle μ (FreeMonoid A)} {D Dsf T : Measure (FreeMonoid A)} {M : _root_.DFA A Q}
    {ε R R' : ℝ} (hR : R ≤ R') {p : settings.Draws Ω A}
    (h : RLStarSucceeds settings O D Dsf T M ε R p) :
    RLStarSucceeds settings O D Dsf T M ε R' p := by
  obtain ⟨h1, S, hS, hdep⟩ := h
  exact ⟨h1, S, hS.trans hR, hdep⟩

lemma lazyReads_mono_nX {K a b f Nr A nX nX' q : ℝ} (hf : 0 ≤ f) (hA : 0 ≤ A) (hq : 0 ≤ q)
    (h : nX ≤ nX') : lazyReads K a b f Nr A nX q ≤ lazyReads K a b f Nr A nX' q := by
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
theorem lazy_within : RLStarWithin.{u, v, w} knTol knPref knSuff knProbes knReads := by
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
  obtain ⟨hP0, hP1, hPcap, hP3, hPnh, hPmc⟩ := knP₀_bounds hr
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
  obtain ⟨hι0, hι8, -, -, -⟩ := knι_bounds hr
  obtain ⟨hW1, hNrW, htwW, -⟩ := knW_bounds hr
  obtain ⟨hnhc, -, hnh1, -⟩ := knNh_bounds hr
  have hnh0 : 0 < knNh r := by exact_mod_cast (show (0 : ℝ) < knNh r by linarith)
  obtain ⟨-, -, hEk⟩ := knE_bounds hr
  obtain ⟨-, -, -, hJk⟩ := knJ_bounds hr
  have hcapP := knCapP_ge hr
  have hcap0 : 0 < knCapInv r := by
    have := (knCapInv_bounds hr).1
    have : 0 < (r.Q : ℝ) ^ 3 * knΛ r * knιInv r := by
      have := (knΛ_bounds hr).1; have := knιInv_one hr; positivity
    linarith
  have hk1' : knP₀ r ≤ knCapP r * knε r / 1536 := by
    refine hPcap.trans ?_
    have := (knε_bounds hr).1
    gcongr
  have hnhcap : 2 ≤ (knNh r : ℝ) * knCapP r := by
    have h1 : 2 * knCapInv r * (1 / knCapInv r) ≤ (knNh r : ℝ) * knCapP r :=
      mul_le_mul hnhc hcapP (by positivity) (by linarith)
    have h2 : 2 * knCapInv r * (1 / knCapInv r) = 2 := by field_simp
    linarith
  obtain ⟨settings, hwalk, hbud, -, -, hE', hJ', hb', hcnt, hprob⟩ :=
    lazy_loop_core (Ω := Ω) (μ := μ) M O L k (knη₀ r) r.ξc r.δ r.pAP r.ξc r.ξ (r.ξc / 4)
      (r.A : ℝ) (knP₀ r) (knι r) (knW r) (knTu r) (r.ξc / 4) (knNh r) (knE r) (knJ r) (knB r)
      (knNr r) D hLang (by rw [knη₀_s]; linarith) (by unfold knη₀; linarith)
      (hbal.trans hbL') (hbal.trans hbN') hξc hξc1 hδ (by linarith) hp hξc hA2 le_rfl hk1 hkL
      (by linarith) hlong hP0 hpre' hpAPu hk1' hP3 (knMcD_spec hr hP0 hPmc) hι0 hι8 hW1 hnh0
      hnhcap hPnh (hEk k hkL') (hJk k hkL') (knb_spec hr) (knNr_spec hr hkL') hNrW htwW
      (by linarith) (knTu_pos hr) (knSlope_spec hr) (knξT_spec hr)
  refine ⟨settings, hwalk, fun p rounds hrun => ?_, ?_⟩
  · obtain ⟨hlen, hpref, hsuff⟩ := hcnt p rounds hrun
    refine ⟨hpref, hsuff, ?_⟩
    have hlen' : (rounds.length : ℝ) ≤ r.Q + 1 := by exact_mod_cast hlen
    have hbud' : (settings.level3.probeBudget : ℝ) ≤ knB₃ r := hbud
    rw [hE', hJ', hb']
    unfold knProbes
    gcongr
  · obtain ⟨hRB0, -⟩ := knRB_bounds hr
    obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
    have hk : (k : ℝ) ≤ r.L := by exact_mod_cast hkL'
    have hle : lazyReads (r.Q + 1) (6 * (r.Q + 2) * knRB r) (knPB r + 1) (knFV r) (knNr r) r.A
        (k * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) r.Q ≤ knReads r :=
      lazyReads_mono_nX hFV0 (by positivity) (by positivity) (by gcongr)
    refine le_trans (measureReal_mono (fun p hp => ?_) (measure_ne_top _ _)) hprob
    exact fun hs => hp (rlStarSucceeds_mono hle hs)

end OrthoDFA

end
