import RLStar.Claims.RLStar
import RLStar.Proofs.TrieTol
import RLStar.Proofs.ClaimRLStarUniform
import RLStar.Proofs.ClaimReadsArith

/-!
# The learner's costs, asymptotically: the proof

`trie_loop_tol` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, and its probes by
`(|Q| + 1) loopBudget`, all at `ℓ = L + |Q| + 4`, once `D` holds at most `1/(2⁴ trieTolInv⁵ ℓ)` on
any first `k` letters; but for chance `δ` the run succeeds, asking the oracle at most
`learnerReads` strings with no held-out ones.
The claim runs it at `η₀ = 1/2 − s₀/4`, so the gap `s₀` is `4 (1/2 − η₀)`, and at main's rates at
`R = 1` with edges crossed at `|Σ|` and `ξ` replaced by `ξc = min(ξ, β)`: `c = ξc/2` and `ξc/4`,
so the clustering's `2 εcov` is at most `ξc/16 ≤ β`. `ClaimRLStarArith` and `ClaimReadsArith`
bound those shapes at `s₀/4` (`quarterRun`), each within a constant of its shape at `s₀`.
`RLStarClaimUniform` and `RLStarClaimUniformSimple` follow (`ClaimRLStarUniform`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

/-- The run at a quarter of the signal. -/
def quarterRun (r : RLStarRun) : RLStarRun := { r with s₀ := r.s₀ / 4 }

lemma quarterRun_maps : Set.MapsTo quarterRun rlStarRunRegion rlStarRunRegion := by
  intro r hr
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := hr
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, by show 0 < r.s₀ / 4; linarith,
    by show r.s₀ / 4 ≤ 1 / 2; linarith, h11⟩

lemma ratioOne_maps :
    Set.MapsTo (fun r => toDistRun (ratioOneRun r)) rlStarRunRegion rlStarDistRunRegion :=
  fun _ hr => toDist_mem (ratioOne_mem hr)

lemma run_coords {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
    1 ≤ (r.Q : ℝ) ∧ 2 ≤ (r.A : ℝ) ∧ 1 ≤ (r.L : ℝ) + r.Q ∧ 0 < r.δ ∧ r.δ ≤ 1 / 2 ∧ 0 < r.ξc
      ∧ r.ξc ≤ 1 ∧ 0 < r.pAP ∧ r.pAP ≤ 1 ∧ 0 < r.s₀ ∧ r.s₀ ≤ 1 / 2 := by
  obtain ⟨hξc, hξc1, -, -⟩ := ξc_mem (x := r.toRLStarInputs) hr
  obtain ⟨hQ, hA, hδ, hδ2, -, -, hp, hp1, hs, hs1, -⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  exact ⟨hQ', by exact_mod_cast hA, by linarith, hδ, hδ2, hξc, hξc1, hp, hp1, hs, hs1⟩

/-! `RLStarClaim`'s shapes at `X = L + |Q|`, accuracy `ξ` and signal `s`. -/

def p₀Sh (Q A X δ ξ p s : ℝ) : ℝ :=
  X ^ 4 * log (X * A / (δ * ξ * p * s)) * log (Q * A / (δ * ξ * p * s)) * Q ^ 3 * A / ξ
    * (Q ^ 2 * A ^ 2 / ξ ^ 3 * (1 / s ^ 6 + Q ^ 2 * A / s ^ 5 + Q ^ 4 * A / s ^ 4)
      + 1 / (p ^ 2 * s ^ 2))

def prefSh (Q A X δ ξ p s : ℝ) : ℝ :=
  Q ^ 7 * A ^ 2 * X ^ 2 / (ξ ^ 2 * s ^ 4) * log (Q * log (X * A / (δ * ξ * s)) / (δ * p * s))
    + Q ^ 2 * X ^ 2 / ξ ^ 2 * (1 / s ^ 4 + Q ^ 4 * A ^ 2) * log (X * A / (δ * ξ * p * s))

def suffSh (Q A X δ ξ p s : ℝ) : ℝ :=
  Q * (log (X * A / (δ * ξ * s)) / (p * s ^ 2) + log (Q / δ) / p)

def probeSh (Q A X δ ξ : ℝ) : ℝ := Q ^ 3 * A ^ 2 * X / ξ * log (Q * A / δ)

def famSh (A X δ ξ s : ℝ) : ℝ := log (X * A / (δ * ξ * s)) / s ^ 2

def readsSh (Q A X δ ξ p s : ℝ) : ℝ :=
  prefSh Q A X δ ξ p s * suffSh Q A X δ ξ p s / Q
    + probeSh Q A X δ ξ * famSh A X δ ξ s * Q * log (X + 1)
    + Q * prefSh Q A X δ ξ p s * famSh A X δ ξ s * log (X + 1) * (A * X / ξ)

def p₀Of (r : RLStarRun) : ℝ := p₀Sh r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀

def prefOf (r : RLStarRun) : ℝ := prefSh r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀

def suffOf (r : RLStarRun) : ℝ := suffSh r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀

def probesOf (r : RLStarRun) : ℝ := probeSh r.Q r.A (r.L + r.Q) r.δ r.ξc

def readsOf (r : RLStarRun) : ℝ := readsSh r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀

/-! The shapes at a quarter of the signal. -/

lemma quarter_dens {δ ξ p s : ℝ} (hδ : 0 < δ) (hδ2 : δ ≤ 1 / 2) (hξ1 : ξ ≤ 1) (hξ : 0 < ξ)
    (hp1 : p ≤ 1) (hp : 0 < p) (hs : 0 < s) (hs1 : s ≤ 1 / 2) :
    4 * (δ * ξ * s) ≤ 1 ∧ 4 * (δ * ξ * p * s) ≤ 1 ∧ 4 * (δ * p * s) ≤ 1 := by
  have h1 : δ * ξ ≤ 1 / 2 := (mul_le_of_le_one_right hδ.le hξ1).trans hδ2
  have h2 : δ * ξ * p ≤ 1 / 2 := (mul_le_of_le_one_right (by positivity) hp1).trans h1
  have h3 : δ * p ≤ 1 / 2 := (mul_le_of_le_one_right hδ.le hp1).trans hδ2
  have k1 := mul_le_mul h1 hs1 hs.le (by norm_num : (0 : ℝ) ≤ 1 / 2)
  have k2 := mul_le_mul h2 hs1 hs.le (by norm_num : (0 : ℝ) ≤ 1 / 2)
  have k3 := mul_le_mul h3 hs1 hs.le (by norm_num : (0 : ℝ) ≤ 1 / 2)
  exact ⟨by linarith, by linarith, by linarith⟩

/-- `log (a/(b s))` at a quarter of `s`, once `a/(b s) ≥ 4`. -/
lemma log_quarter {a b s : ℝ} (hb : 0 < b) (hs : 0 < s) (h : 4 * (b * s) ≤ a) :
    0 ≤ log (a / (b * s)) ∧ log (a / (b * s)) ≤ log (a / (b * (s / 4)))
      ∧ log (a / (b * (s / 4))) ≤ 2 * log (a / (b * s)) := by
  have hY : 4 ≤ a / (b * s) := by rw [le_div_iff₀ (by positivity)]; linarith
  have e : a / (b * (s / 4)) = 4 * (a / (b * s)) := by ring
  rw [e]
  exact ⟨Real.log_nonneg (by linarith), Real.log_le_log (by linarith) (by linarith),
    log_four_mul_le hY⟩

lemma logZ_quarter {Q A X δ ξ p s : ℝ} (hQ : 1 ≤ Q) (hXA : 2 ≤ X * A) (hδ : 0 < δ)
    (hξ : 0 < ξ) (hp : 0 < p) (hs : 0 < s) (hD : 4 * (δ * ξ * s) ≤ 1)
    (hD' : 4 * (δ * p * s) ≤ 1) :
    0 ≤ log (Q * log (X * A / (δ * ξ * s)) / (δ * p * s))
      ∧ 0 ≤ log (Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * (s / 4)))
      ∧ log (Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * (s / 4)))
        ≤ 4 * log (Q * log (X * A / (δ * ξ * s)) / (δ * p * s)) := by
  obtain ⟨-, hll, hl⟩ := log_quarter (a := X * A) (b := δ * ξ) (by positivity) hs (by linarith)
  have hl1 : 1 ≤ log (X * A / (δ * ξ * s)) :=
    one_le_log_of_three_le (by rw [le_div_iff₀ (by positivity)]; linarith)
  have hd : 0 < δ * p * s := by positivity
  have hQl : 1 ≤ Q * log (X * A / (δ * ξ * s)) := by nlinarith
  have hZ2 : 2 ≤ Q * log (X * A / (δ * ξ * s)) / (δ * p * s) := by
    rw [le_div_iff₀ hd]; linarith
  have e : Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * (s / 4))
      = 4 * (Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * s)) := by ring
  have h1 : Q * log (X * A / (δ * ξ * s)) / (δ * p * s)
      ≤ Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * s) :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hll (by linarith)) hd.le
  have h2 : Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * s)
      ≤ Q * (2 * log (X * A / (δ * ξ * s))) / (δ * p * s) :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hl (by linarith)) hd.le
  have e2 : Q * (2 * log (X * A / (δ * ξ * s))) / (δ * p * s)
      = 2 * (Q * log (X * A / (δ * ξ * s)) / (δ * p * s)) := by ring
  have hZ' : 8 ≤ Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * (s / 4)) := by
    rw [e]; linarith
  have hZ8 : Q * log (X * A / (δ * ξ * (s / 4))) / (δ * p * (s / 4))
      ≤ 8 * (Q * log (X * A / (δ * ξ * s)) / (δ * p * s)) := by
    rw [e]; linarith
  exact ⟨Real.log_nonneg (by linarith), Real.log_nonneg (by linarith),
    (Real.log_le_log (by linarith) hZ8).trans (log_eight_mul_le hZ2)⟩

lemma p₀_core {K l1 l1' l2 l2' B B' : ℝ} (hK : 0 ≤ K) (hl1 : 0 ≤ l1) (hl1' : 0 ≤ l1')
    (hl2 : 0 ≤ l2) (hl2' : 0 ≤ l2') (h1 : l1' ≤ 2 * l1) (h2 : l2' ≤ 2 * l2) (hB : 0 ≤ B)
    (hB' : 0 ≤ B') (hBB : B' ≤ 4096 * B) :
    0 ≤ K * (l1' * l2') * B' ∧ 0 ≤ K * (l1 * l2) * B
      ∧ K * (l1' * l2') * B' ≤ 2 ^ 14 * (K * (l1 * l2) * B) := by
  have hll : l1' * l2' ≤ 2 * l1 * (2 * l2) := mul_le_mul h1 h2 hl2' (by linarith)
  have h := mul_le_mul (mul_le_mul_of_nonneg_left hll hK) hBB hB' (by positivity)
  exact ⟨by positivity, by positivity, h.trans_eq (by ring)⟩

lemma p₀Sh_quarter {Q A X δ ξ p s : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X) (hδ : 0 < δ)
    (hδ2 : δ ≤ 1 / 2) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1) (hp : 0 < p) (hp1 : p ≤ 1) (hs : 0 < s)
    (hs1 : s ≤ 1 / 2) :
    0 ≤ p₀Sh Q A X δ ξ p (s / 4) ∧ 0 ≤ p₀Sh Q A X δ ξ p s
      ∧ p₀Sh Q A X δ ξ p (s / 4) ≤ 2 ^ 14 * p₀Sh Q A X δ ξ p s := by
  obtain ⟨-, hD, -⟩ := quarter_dens hδ hδ2 hξ1 hξ hp1 hp hs hs1
  have hXA : 2 ≤ X * A := by nlinarith
  have hQA : 2 ≤ Q * A := by nlinarith
  obtain ⟨a0, a1, ha⟩ := log_quarter (a := X * A) (b := δ * ξ * p) (by positivity) hs
    (by linarith)
  obtain ⟨b0, b1, hb⟩ := log_quarter (a := Q * A) (b := δ * ξ * p) (by positivity) hs
    (by linarith)
  have e : ∀ t, p₀Sh Q A X δ ξ p t = X ^ 4 * Q ^ 3 * A / ξ
      * (log (X * A / (δ * ξ * p * t)) * log (Q * A / (δ * ξ * p * t)))
      * (Q ^ 2 * A ^ 2 / ξ ^ 3 * (1 / t ^ 6 + Q ^ 2 * A / t ^ 5 + Q ^ 4 * A / t ^ 4)
        + 1 / (p ^ 2 * t ^ 2)) := fun t => by unfold p₀Sh; ring
  have eB : Q ^ 2 * A ^ 2 / ξ ^ 3
        * (1 / (s / 4) ^ 6 + Q ^ 2 * A / (s / 4) ^ 5 + Q ^ 4 * A / (s / 4) ^ 4)
        + 1 / (p ^ 2 * (s / 4) ^ 2)
      = Q ^ 2 * A ^ 2 / ξ ^ 3 * (4096 * (1 / s ^ 6) + 1024 * (Q ^ 2 * A / s ^ 5)
        + 256 * (Q ^ 4 * A / s ^ 4)) + 16 * (1 / (p ^ 2 * s ^ 2)) := by ring
  have hc : 0 ≤ Q ^ 2 * A ^ 2 / ξ ^ 3 := by positivity
  have hb1 : 0 ≤ 1 / s ^ 6 := by positivity
  have hb2 : 0 ≤ Q ^ 2 * A / s ^ 5 := by positivity
  have hb3 : 0 ≤ Q ^ 4 * A / s ^ 4 := by positivity
  have hb4 : 0 ≤ 1 / (p ^ 2 * s ^ 2) := by positivity
  simp only [e]
  refine p₀_core (by positivity) a0 (a0.trans a1) b0 (b0.trans b1) ha hb (by positivity)
    (by positivity) ?_
  rw [eB]
  nlinarith [mul_nonneg hc hb1, mul_nonneg hc hb2, mul_nonneg hc hb3]

lemma pref_core {C1 lZ lZ' C2 B B' lY lY' : ℝ} (hC1 : 0 ≤ C1) (hC2 : 0 ≤ C2) (hlZ : 0 ≤ lZ)
    (hlZ' : 0 ≤ lZ') (hZ : lZ' ≤ 4 * lZ) (hB : 0 ≤ B) (hB' : 0 ≤ B') (hBB : B' ≤ 256 * B)
    (hlY : 0 ≤ lY) (hlY' : 0 ≤ lY') (hY : lY' ≤ 2 * lY) :
    0 ≤ 256 * C1 * lZ' + C2 * B' * lY' ∧ 0 ≤ C1 * lZ + C2 * B * lY
      ∧ 256 * C1 * lZ' + C2 * B' * lY' ≤ 1024 * (C1 * lZ + C2 * B * lY) := by
  have t1 := mul_le_mul_of_nonneg_left hZ hC1
  have t2 := mul_le_mul (mul_le_mul_of_nonneg_left hBB hC2) hY hlY' (by positivity)
  have p2 : 0 ≤ C2 * B * lY := by positivity
  refine ⟨by positivity, by positivity, ?_⟩
  nlinarith [t1, t2, p2]

lemma prefSh_quarter {Q A X δ ξ p s : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X) (hδ : 0 < δ)
    (hδ2 : δ ≤ 1 / 2) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1) (hp : 0 < p) (hp1 : p ≤ 1) (hs : 0 < s)
    (hs1 : s ≤ 1 / 2) :
    0 ≤ prefSh Q A X δ ξ p (s / 4) ∧ 0 ≤ prefSh Q A X δ ξ p s
      ∧ prefSh Q A X δ ξ p (s / 4) ≤ 1024 * prefSh Q A X δ ξ p s := by
  obtain ⟨hD1, hD2, hD3⟩ := quarter_dens hδ hδ2 hξ1 hξ hp1 hp hs hs1
  have hXA : 2 ≤ X * A := by nlinarith
  obtain ⟨z0, z0', hz⟩ := logZ_quarter hQ hXA hδ hξ hp hs hD1 hD3
  obtain ⟨y0, y1, hy⟩ := log_quarter (a := X * A) (b := δ * ξ * p) (by positivity) hs
    (by linarith)
  have eC : Q ^ 7 * A ^ 2 * X ^ 2 / (ξ ^ 2 * (s / 4) ^ 4)
      = 256 * (Q ^ 7 * A ^ 2 * X ^ 2 / (ξ ^ 2 * s ^ 4)) := by ring
  have eB : 1 / (s / 4) ^ 4 = 256 * (1 / s ^ 4) := by ring
  have hq : 0 ≤ Q ^ 4 * A ^ 2 := by positivity
  simp only [prefSh]
  rw [eC]
  refine pref_core (by positivity) (by positivity) z0 z0' hz (by positivity) (by positivity) ?_
    y0 (y0.trans y1) hy
  rw [eB]
  linarith

lemma suffSh_quarter {Q A X δ ξ p s : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X) (hδ : 0 < δ)
    (hδ2 : δ ≤ 1 / 2) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1) (hp : 0 < p) (hp1 : p ≤ 1) (hs : 0 < s)
    (hs1 : s ≤ 1 / 2) :
    0 ≤ suffSh Q A X δ ξ p (s / 4) ∧ 0 ≤ suffSh Q A X δ ξ p s
      ∧ suffSh Q A X δ ξ p (s / 4) ≤ 32 * suffSh Q A X δ ξ p s := by
  obtain ⟨hD1, -, -⟩ := quarter_dens hδ hδ2 hξ1 hξ hp1 hp hs hs1
  have hXA : 2 ≤ X * A := by nlinarith
  obtain ⟨y0, y1, hy⟩ := log_quarter (a := X * A) (b := δ * ξ) (by positivity) hs (by linarith)
  have hv : 0 ≤ log (Q / δ) / p :=
    div_nonneg (Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)) hp.le
  have hu : 0 ≤ log (X * A / (δ * ξ * s)) / (p * s ^ 2) := div_nonneg y0 (by positivity)
  have hu' : 0 ≤ log (X * A / (δ * ξ * (s / 4))) / (p * (s / 4) ^ 2) :=
    div_nonneg (y0.trans y1) (by positivity)
  have e : log (X * A / (δ * ξ * (s / 4))) / (p * (s / 4) ^ 2)
      = 16 * (log (X * A / (δ * ξ * (s / 4))) / (p * s ^ 2)) := by ring
  have h1 : log (X * A / (δ * ξ * (s / 4))) / (p * s ^ 2)
      ≤ 2 * log (X * A / (δ * ξ * s)) / (p * s ^ 2) :=
    div_le_div_of_nonneg_right hy (by positivity)
  have e2 : 2 * log (X * A / (δ * ξ * s)) / (p * s ^ 2)
      = 2 * (log (X * A / (δ * ξ * s)) / (p * s ^ 2)) := by ring
  have hw : log (X * A / (δ * ξ * (s / 4))) / (p * (s / 4) ^ 2) + log (Q / δ) / p
      ≤ 32 * (log (X * A / (δ * ξ * s)) / (p * s ^ 2) + log (Q / δ) / p) := by
    rw [e]; linarith
  have hQ0 : 0 ≤ Q := by linarith
  simp only [suffSh]
  refine ⟨mul_nonneg hQ0 (add_nonneg hu' hv), mul_nonneg hQ0 (add_nonneg hu hv), ?_⟩
  nlinarith [mul_le_mul_of_nonneg_left hw hQ0]

lemma famSh_quarter {A X δ ξ s : ℝ} (hA : 2 ≤ A) (hX : 1 ≤ X) (hδ : 0 < δ) (hδ2 : δ ≤ 1 / 2)
    (hξ : 0 < ξ) (hξ1 : ξ ≤ 1) (hs : 0 < s) (hs1 : s ≤ 1 / 2) :
    0 ≤ famSh A X δ ξ (s / 4) ∧ 0 ≤ famSh A X δ ξ s
      ∧ famSh A X δ ξ (s / 4) ≤ 32 * famSh A X δ ξ s := by
  obtain ⟨hD1, -, -⟩ := quarter_dens hδ hδ2 hξ1 hξ (le_refl 1) one_pos hs hs1
  have hXA : 2 ≤ X * A := by nlinarith
  obtain ⟨y0, y1, hy⟩ := log_quarter (a := X * A) (b := δ * ξ) (by positivity) hs (by linarith)
  have e : log (X * A / (δ * ξ * (s / 4))) / (s / 4) ^ 2
      = 16 * (log (X * A / (δ * ξ * (s / 4))) / s ^ 2) := by ring
  have h1 : log (X * A / (δ * ξ * (s / 4))) / s ^ 2 ≤ 2 * log (X * A / (δ * ξ * s)) / s ^ 2 :=
    div_le_div_of_nonneg_right hy (by positivity)
  have e2 : 2 * log (X * A / (δ * ξ * s)) / s ^ 2 = 2 * (log (X * A / (δ * ξ * s)) / s ^ 2) := by
    ring
  simp only [famSh]
  refine ⟨div_nonneg (y0.trans y1) (by positivity), div_nonneg y0 (by positivity), ?_⟩
  rw [e]; linarith

lemma reads_core {Q Pf Pf' Sf Sf' Pr Fm Fm' lg ρ : ℝ} (hQ : 0 < Q) (hPf : 0 ≤ Pf)
    (hPf' : 0 ≤ Pf') (hSf : 0 ≤ Sf) (hSf' : 0 ≤ Sf') (hPr : 0 ≤ Pr) (hFm : 0 ≤ Fm)
    (hFm' : 0 ≤ Fm') (hlg : 0 ≤ lg) (hρ : 0 ≤ ρ) (h1 : Pf' ≤ 1024 * Pf) (h2 : Sf' ≤ 32 * Sf)
    (h3 : Fm' ≤ 32 * Fm) :
    0 ≤ Pf' * Sf' / Q + Pr * Fm' * Q * lg + Q * Pf' * Fm' * lg * ρ
      ∧ 0 ≤ Pf * Sf / Q + Pr * Fm * Q * lg + Q * Pf * Fm * lg * ρ
      ∧ Pf' * Sf' / Q + Pr * Fm' * Q * lg + Q * Pf' * Fm' * lg * ρ
        ≤ 2 ^ 15 * (Pf * Sf / Q + Pr * Fm * Q * lg + Q * Pf * Fm * lg * ρ) := by
  have a : Pf' * Sf' ≤ 1024 * Pf * (32 * Sf) := mul_le_mul h1 h2 hSf' (by positivity)
  have b : Pf' * Fm' ≤ 1024 * Pf * (32 * Fm) := mul_le_mul h1 h3 hFm' (by positivity)
  have a' : Pf' * Sf' / Q ≤ 1024 * Pf * (32 * Sf) / Q := div_le_div_of_nonneg_right a hQ.le
  have e1 : 1024 * Pf * (32 * Sf) / Q = 2 ^ 15 * (Pf * Sf / Q) := by ring
  have b' : Pr * Q * lg * Fm' ≤ Pr * Q * lg * (32 * Fm) :=
    mul_le_mul_of_nonneg_left h3 (by positivity)
  have c' : Q * lg * ρ * (Pf' * Fm') ≤ Q * lg * ρ * (1024 * Pf * (32 * Fm)) :=
    mul_le_mul_of_nonneg_left b (by positivity)
  have n1 : 0 ≤ Pr * Fm * Q * lg := by positivity
  have n2 : 0 ≤ Q * Pf * Fm * lg * ρ := by positivity
  refine ⟨by positivity, by positivity, ?_⟩
  nlinarith [a', e1, b', c', n1, n2]

lemma readsSh_quarter {Q A X δ ξ p s : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X) (hδ : 0 < δ)
    (hδ2 : δ ≤ 1 / 2) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1) (hp : 0 < p) (hp1 : p ≤ 1) (hs : 0 < s)
    (hs1 : s ≤ 1 / 2) :
    0 ≤ readsSh Q A X δ ξ p (s / 4) ∧ 0 ≤ readsSh Q A X δ ξ p s
      ∧ readsSh Q A X δ ξ p (s / 4) ≤ 2 ^ 15 * readsSh Q A X δ ξ p s := by
  obtain ⟨P0', P0, hP⟩ := prefSh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1
  obtain ⟨S0', S0, hS⟩ := suffSh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1
  obtain ⟨F0', F0, hF⟩ := famSh_quarter hA hX hδ hδ2 hξ hξ1 hs hs1
  have hPr : 0 ≤ probeSh Q A X δ ξ := by
    have : 0 ≤ log (Q * A / δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
    unfold probeSh; positivity
  have hlg : 0 ≤ log (X + 1) := Real.log_nonneg (by linarith)
  simp only [readsSh]
  exact reads_core (by linarith) P0 P0' S0 S0' hPr F0 F0' hlg (by positivity) hP hS hF

lemma p₀_quarter :
    (fun r => p₀Of (quarterRun r) ^ 5) =O[𝓟 rlStarRunRegion] (fun r => p₀Of r ^ 5) := by
  refine isBigO_on ((2 ^ 14) ^ 5) fun r hr => ?_
  obtain ⟨hQ, hA, hX, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := run_coords hr
  obtain ⟨h0', h0, h⟩ := p₀Sh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1
  refine ⟨pow_nonneg h0' 5, pow_nonneg h0 5, ?_⟩
  calc p₀Of (quarterRun r) ^ 5 ≤ (2 ^ 14 * p₀Of r) ^ 5 := pow_le_pow_left₀ h0' h 5
    _ = (2 ^ 14) ^ 5 * p₀Of r ^ 5 := by rw [mul_pow]

lemma pref_quarter :
    (fun r => prefOf (quarterRun r)) =O[𝓟 rlStarRunRegion] prefOf := by
  refine isBigO_on 1024 fun r hr => ?_
  obtain ⟨hQ, hA, hX, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := run_coords hr
  exact prefSh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1

lemma suff_quarter :
    (fun r => suffOf (quarterRun r)) =O[𝓟 rlStarRunRegion] suffOf := by
  refine isBigO_on 32 fun r hr => ?_
  obtain ⟨hQ, hA, hX, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := run_coords hr
  exact suffSh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1

lemma reads_quarter :
    (fun r => readsOf (quarterRun r)) =O[𝓟 rlStarRunRegion] readsOf := by
  refine isBigO_on (2 ^ 15) fun r hr => ?_
  obtain ⟨hQ, hA, hX, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := run_coords hr
  exact readsSh_quarter hQ hA hX hδ hδ2 hξ hξ1 hp hp1 hs hs1

/-! The costs at the run's own signal. -/

lemma p₀_dist : (fun r : RLStarRun => 1 / (distP₀Tol (toDistRun (ratioOneRun r)) ^ 5 / 2 ^ 4))
    =O[𝓟 rlStarRunRegion] (fun r => p₀Of r ^ 5) := by
  refine (((isBigO_comp_on distP₀Tol_isBigO ratioOne_maps).pow 5).const_mul_left (2 ^ 4)).congr'
    (eventually_principal.2 fun r _ => ?_) (eventually_principal.2 fun r hr => ?_)
  · simp only [Function.comp_apply]
    rw [one_div_pow, mul_one_div, one_div_div]
  · simp only [Function.comp_apply, p₀Of, p₀Sh]
    rw [p₀Shape_one hr]

lemma pref_dist : (fun r : RLStarRun => distPref (toDistRun (ratioOneRun r)))
    =O[𝓟 rlStarRunRegion] prefOf := by
  refine (isBigO_comp_on distPref_isBigO ratioOne_maps).congr' EventuallyEq.rfl
    (eventually_principal.2 fun r hr => ?_)
  have hξ := (ξc_mem (x := r.toRLStarInputs) hr).1
  obtain ⟨-, hA, hδ, -, -, -, hp, -, hs, -, -⟩ := hr
  have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
  simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, one_div_one_div, prefOf,
    prefSh]
  obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
  rw [e, e']
  simp only [mul_one, one_pow]

lemma suff_dist : (fun r : RLStarRun => distSuff (toDistRun (ratioOneRun r)))
    =O[𝓟 rlStarRunRegion] suffOf := by
  refine (isBigO_comp_on distSuff_isBigO ratioOne_maps).congr' EventuallyEq.rfl
    (eventually_principal.2 fun r hr => ?_)
  have hξ := (ξc_mem (x := r.toRLStarInputs) hr).1
  obtain ⟨-, hA, hδ, -, -, -, hp, -, hs, -, -⟩ := hr
  have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
  simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, suffOf, suffSh]
  rw [(uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs).2]
  simp only [mul_one]

lemma probes_dist : (fun r : RLStarRun => distProbes (toDistRun (ratioOneRun r)))
    =O[𝓟 rlStarRunRegion] probesOf := by
  refine (isBigO_comp_on distProbes_isBigO ratioOne_maps).congr' EventuallyEq.rfl
    (eventually_principal.2 fun r _ => ?_)
  simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, one_div_one_div, probesOf,
    probeSh]
  ring

lemma reads_dist : (fun r : RLStarRun => distLoopReads (toDistRun (ratioOneRun r)))
    =O[𝓟 rlStarRunRegion] readsOf := by
  refine (isBigO_comp_on distLoopReads_isBigO ratioOne_maps).congr' EventuallyEq.rfl
    (eventually_principal.2 fun r hr => ?_)
  have hξ := (ξc_mem (x := r.toRLStarInputs) hr).1
  obtain ⟨-, hA, hδ, -, -, -, hp, -, hs, -, -⟩ := hr
  have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
  simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, shCluster, shSearch,
    shHarvest, shPref, shSuff, shProbe, shFam, shRho, shLg, one_div_one_div, readsOf, readsSh,
    prefSh, suffSh, probeSh, famSh]
  obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
  rw [e, e']
  ring

theorem rlStarClaim_holds : RLStarClaim := by
  refine ⟨fun r => distP₀Tol (toDistRun (ratioOneRun (quarterRun r))) ^ 5 / 2 ^ 4,
    fun r => distPref (toDistRun (ratioOneRun (quarterRun r))),
    fun r => distSuff (toDistRun (ratioOneRun (quarterRun r))),
    fun r => distProbes (toDistRun (ratioOneRun (quarterRun r))),
    fun r => distLoopReads (toDistRun (ratioOneRun (quarterRun r))),
    (isBigO_comp_on p₀_dist quarterRun_maps).trans p₀_quarter,
    (isBigO_comp_on pref_dist quarterRun_maps).trans pref_quarter,
    (isBigO_comp_on suff_dist quarterRun_maps).trans suff_quarter,
    isBigO_comp_on probes_dist quarterRun_maps,
    (isBigO_comp_on reads_dist quarterRun_maps).trans reads_quarter,
    fun r => div_pos (pow_pos (distP₀Tol_pos _) 5) (by norm_num), ?_⟩
  -- the run
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨Q, A, δ, ξ, pAP, s₀, β⟩ := x
  intro M O L k D _ L₁ Ls hx hQ hA hLang hgap hbL hbN hlong hkL hpre hpAPu
  subst hQ hA
  obtain ⟨hQ1, hA2, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1, hβ⟩ := hx
  have hA' : (2 : ℝ) ≤ Fintype.card α := by exact_mod_cast hA2
  set ξc : ℝ := min ξ β with hξcd
  have hξc0 : 0 < ξc := lt_min hξ hβ
  have hξc1 : ξc ≤ 1 := (min_le_left _ _).trans hξ1
  have hξcξ : ξc ≤ ξ := min_le_left _ _
  have hξcβ : ξc ≤ β := min_le_right _ _
  have hη₀ : 1 / 2 - s₀ / 4 < 1 / 2 := by linarith
  have hc0 : 0 < ξc / (2 * 1) := by positivity
  have hc1 : ξc / (2 * 1) ≤ 1 := by linarith
  set T := trieTolInv ((L : ℝ) + Fintype.card σ + 4) (Fintype.card σ)
      (Fintype.card α) (1 / (1 / (Fintype.card α : ℝ))) (Fintype.card σ) (Fintype.card α) δ
      (ξc / (2 * 1)) (ξc / 4) pAP (1 / 2 - s₀ / 4) with hTd
  have hT1 : 1 ≤ T := one_le_trieTolInv _ _
    (by have : (0 : ℝ) ≤ L := Nat.cast_nonneg _; linarith) (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by rw [one_div_one_div]; exact hA') hδ (by linarith) hc0 hc1 (by positivity) hp hp1
    (by linarith) hη₀
  have hTm : max 1 T = T := max_eq_right hT1
  -- `D`'s first `k` letters weigh at most `1/Ak`
  have hpre' : ∀ u, D.real {w | prefixOf w k = u} ≤ (2 ^ 4 * T ^ 5)⁻¹ := by
    intro u
    refine (hpre u).trans (le_of_eq ?_)
    change (1 / max 1 T) ^ 5 / 2 ^ 4 = _
    rw [hTm, one_div_pow, div_div, one_div, mul_comm]
  obtain ⟨settings, hwk, hprobe, hprob⟩ := trie_loop_tol (Ω := Ω) (μ := μ) M O L k k
    (1 / 2 - s₀ / 4) (ξc / (2 * 1)) δ pAP (ξc / 4) ξ (1 / (1 / (Fintype.card α : ℝ)))
    (2 ^ 4 * T ^ 5) D hLang (by linarith) hη₀ (by linarith) (by linarith) hc0 hc1 hδ
    (by linarith) hp (by positivity) hA2 (le_of_eq (one_div_one_div _).symm) hkL le_rfl hlong
    hpre' hpAPu le_rfl (by linarith)
  exact ⟨settings, hwk, hprobe, hprob⟩

theorem rlStarClaimUniform_holds : RLStarClaimUniform :=
  rlStarClaimUniform_of_rlStarClaim rlStarClaim_holds

theorem rlStarClaimUniformSimple_holds : RLStarClaimUniformSimple :=
  rlStarClaimUniformSimple_of_claim rlStarClaimUniform_holds

end OrthoDFA

end
