import Mathlib.Analysis.SpecialFunctions.Log.Base
import RLStar.Claims.TrainFamily
import RLStar.Proofs.Budget
import RLStar.Proofs.BigO

/-!
# The clustering's costs, asymptotically

The witnesses are `ClusteringGuarantee`'s bounds at its `δ = α = δ/2`, the prefix count's log
taken at the pool's ceiling; the tolerances are its per-state atom and collision conditions at the
ceilings, and `1` off the parameters' range. They are stated and bounded on `toVec x`, the counts
and the rates' reciprocals, and carried back to `TrainFamilyInputs`.
-/

noncomputable section

namespace OrthoDFA

open Real Asymptotics Filter MeasureTheory

private def kB (s r c : ℝ) : ℝ :=
  25 * (max (4 * log (20 / r)) (log (1 / c)) + 12) / (8 * s ^ 2) + 5 / s + 7

private def nsuffB (s r c pAP δ : ℝ) : ℝ :=
  2 * kB s r c / pAP + 60 * log (3 / δ) / pAP

private def nprefB (s r m δ α pAP εs nJ nH n : ℝ) : ℝ :=
  2 ^ 22 * nJ * log (max 1 (64 * (n + 1) / δ)) / (s ^ 4 * m ^ 2)
    + 3 * 2 ^ 25 * log (max 1 ((nJ + 2) * n / (δ * α * pAP * r))) / r ^ 2
    + 6 * log (256 * nJ / δ) / εs + 2 ^ 11 * nH / r

private lemma rs_eq {ie iu is : ℝ} (his : 0 < is) :
    rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie) = min (4 / (ie * is ^ 2)) (1 / iu) := by
  unfold rateScale
  have : 1 - 2 * (1 / 2 - 1 / is) = 2 / is := by field_simp; ring
  rw [this]
  congr 1
  field_simp
  ring

private lemma rs_pos {ie iu is : ℝ} (hie : 0 < ie) (hiu : 0 < iu) (his : 0 < is) :
    0 < rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie) := by
  rw [rs_eq his]; exact lt_min (by positivity) (by positivity)

private lemma rs_le {ie iu is : ℝ} (hie : 0 < ie) (hiu : 0 < iu) (his : 2 ≤ is) :
    1 / rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie) ≤ ie * is ^ 2 + iu := by
  rw [rs_eq (by linarith)]
  have h1 : 0 < 4 / (ie * is ^ 2) := by positivity
  refine (one_div_min_le h1 (by positivity)).trans ?_
  rw [one_div_div, one_div_one_div]
  have : 0 ≤ ie * is ^ 2 := by positivity
  linarith

/-- `(nJ, 1/δ, 1/pAP, 1/εcov, 1/ε_s, 1/ε_u, 1/κ, 1/s₀, nH)`. -/
private abbrev ClusterVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set ClusterVec := Set.Ici (1, 2, 1, 2, 2, 2, 1, 2, 0)

private lemma region {nJ iδ ip ie isl iu iw is nH : ℝ}
    (hb : (nJ, iδ, ip, ie, isl, iu, iw, is, nH) ∈ vecRegion) :
    1 ≤ nJ ∧ 2 ≤ iδ ∧ 1 ≤ ip ∧ 2 ≤ ie ∧ 2 ≤ isl ∧ 2 ≤ iu ∧ 1 ≤ iw ∧ 2 ≤ is ∧ 0 ≤ nH := by
  simpa only [vecRegion, Set.mem_Ici, Prod.mk_le_mk] using hb

private lemma prods {nJ iδ ip ie iu iw is : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip)
    (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) :
    8 ≤ ie * iu * iw * is ∧ 2 ≤ nJ * iδ ∧ 16 ≤ nJ * iδ * ip * ie * iu * iw * is := by
  have h1 : 4 ≤ ie * iu := by nlinarith
  have h2 : 4 ≤ ie * iu * iw := by nlinarith
  have h3 : 2 ≤ nJ * iδ := by nlinarith
  have h5 : 2 ≤ nJ * iδ * ip := by nlinarith
  have h6 : 4 ≤ nJ * iδ * ip * ie := by nlinarith
  have h7 : 8 ≤ nJ * iδ * ip * ie * iu := by nlinarith
  have h8 : 8 ≤ nJ * iδ * ip * ie * iu * iw := by nlinarith
  exact ⟨by nlinarith, h3, by nlinarith⟩

private lemma kB_bound {ie iu iw is r : ℝ} (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw)
    (his : 2 ≤ is) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) :
    0 ≤ kB (1 / is) r ((1 / iw) ^ 4)
      ∧ kB (1 / is) r ((1 / iw) ^ 4) ≤ 110 * (log (max (ie * is) (max iu iw)) * is ^ 2) := by
  have hieM : ie * is ≤ max (ie * is) (max iu iw) := le_max_left _ _
  have hiuM : iu ≤ max (ie * is) (max iu iw) := (le_max_left _ _).trans (le_max_right _ _)
  have hiwM : iw ≤ max (ie * is) (max iu iw) := (le_max_right _ _).trans (le_max_right _ _)
  set M := max (ie * is) (max iu iw)
  have hM : 4 ≤ M := by nlinarith
  have hisM : is ≤ M := by nlinarith
  have hL : 1 ≤ log M := one_le_log_of_three_le (by linarith)
  have h20 : log (20 / r) ≤ 5 * log M := by
    refine (log_le_mul_log_of_le_pow (P := M) 5 (by positivity) ?_).trans (by norm_num)
    have : ie * is ^ 2 ≤ M * M := by
      calc ie * is ^ 2 = (ie * is) * is := by ring
        _ ≤ M * M := mul_le_mul hieM hisM (by linarith) (by linarith)
    have h1 : 1 / r ≤ 2 * M ^ 2 := by nlinarith
    have h2 : 40 * M ^ 2 ≤ M ^ 5 := by
      have : 64 ≤ M ^ 3 := by nlinarith
      nlinarith [mul_nonneg (sub_nonneg.2 this) (sq_nonneg M)]
    rw [div_eq_mul_one_div]
    nlinarith
  have hw : log iw ≤ log M := Real.log_le_log (by linarith) hiwM
  have hw0 : 0 ≤ log iw := Real.log_nonneg hiw
  have hlw : log (1 / (1 / iw) ^ 4) = 4 * log iw := by
    rw [one_div_pow, one_div_one_div, Real.log_pow]; norm_num
  have hm : max (4 * log (20 / r)) (4 * log iw) ≤ 20 * log M :=
    max_le (by linarith) (by linarith)
  have hm0 : 0 ≤ max (4 * log (20 / r)) (4 * log iw) := le_max_of_le_right (by linarith)
  unfold kB
  rw [hlw]
  generalize max (4 * log (20 / r)) (4 * log iw) = m at *
  rw [show 25 * (m + 12) / (8 * (1 / is) ^ 2) + 5 / (1 / is) + 7
      = 25 * (m + 12) * is ^ 2 / 8 + 5 * is + 7 by field_simp]
  have h1 : 2 * is ≤ is ^ 2 * log M := by nlinarith
  have h2 : 4 ≤ is ^ 2 * log M := by nlinarith
  have h3 := mul_le_mul_of_nonneg_right hm (sq_nonneg is)
  constructor
  · positivity
  · nlinarith

private lemma nsuffB_bound {iδ ip ie iu iw is r : ℝ} (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) (hr : 0 < r)
    (hr2 : 1 / r ≤ ie * is ^ 2 + iu) :
    0 ≤ nsuffB (1 / is) r ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ))
      ∧ nsuffB (1 / is) r ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ))
      ≤ 240 * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip) := by
  obtain ⟨hk0, hk⟩ := kB_bound hie hiu hiw his hr hr2
  have hJl : log (3 / (1 / (2 * iδ))) ≤ 4 * log iδ := by
    refine (log_le_mul_log_of_le_pow (P := iδ) 4 (by positivity) ?_).trans (by norm_num)
    rw [div_one_div']
    have h3 : (2 : ℝ) ^ 3 ≤ iδ ^ 3 := pow_le_pow_left₀ (by norm_num) hiδ 3
    nlinarith [mul_le_mul_of_nonneg_left h3 (by linarith : (0 : ℝ) ≤ iδ)]
  have hJ0 : 0 ≤ log (3 / (1 / (2 * iδ))) := Real.log_nonneg (by
    rw [div_one_div']; nlinarith)
  have hLJ : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  unfold nsuffB
  generalize kB (1 / is) r ((1 / iw) ^ 4) = k at *
  generalize log (3 / (1 / (2 * iδ))) = l at *
  rw [show 2 * k / (1 / ip) + 60 * l / (1 / ip) = 2 * k * ip + 60 * l * ip by field_simp]
  have := mul_le_mul_of_nonneg_right hk (show 0 ≤ ip by linarith)
  have := mul_le_mul_of_nonneg_right hJl (show 0 ≤ ip by linarith)
  have : 0 ≤ log iδ * ip := by positivity
  constructor
  · positivity
  · nlinarith

private lemma Z_ge {ℓ iδ ip is : ℝ} (hℓ : 1 ≤ ℓ) (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    4 ≤ ℓ * iδ * ip * is := by
  have h1 : 2 ≤ ℓ * iδ := by nlinarith
  have h2 : 2 ≤ ℓ * iδ * ip := by nlinarith
  nlinarith

set_option maxHeartbeats 1000000 in
private lemma nprefB_bound {nJ iδ ip ie isl iu iw is nH r n : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hisl : 2 ≤ isl) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) (hn0 : 0 ≤ n)
    (hn : n ≤ 240 * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip)) :
    0 ≤ nprefB (1 / is) r (min (1 / ie) (1 / iu)) (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl)
      nJ nH n
    ∧ nprefB (1 / is) r (min (1 / ie) (1 / iu)) (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl)
      nJ nH n ≤ 2 ^ 32 *
      (nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is) * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu)) := by
  have hl : ∀ x ∈ [nJ, iδ, ip, ie, iu, iw, is], 1 ≤ x := fun x hx => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> linarith
  have hsub (l : List ℝ) (hs : l.Sublist [nJ, iδ, ip, ie, iu, iw, is]) :
      l.prod ≤ nJ * iδ * ip * ie * iu * iw * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  obtain ⟨-, hx, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  have hLJ : log (nJ * iδ) ≤ nJ * iδ := Real.log_le_self (by positivity)
  have hLJ0 : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have hM4 : 4 ≤ max (ie * is) (max iu iw) := le_max_of_le_left (by nlinarith)
  have hLM1 : 1 ≤ log (max (ie * is) (max iu iw)) := one_le_log_of_three_le (by linarith)
  have hPJ : nJ * iδ ≤ nJ * iδ * ip * ie * iu * iw * is := by
    simpa [mul_assoc] using hsub [nJ, iδ] (by simp)
  set P := nJ * iδ * ip * ie * iu * iw * is
  have hJP : nJ ≤ P := by simpa using hsub [nJ] (by simp)
  have hδP : iδ ≤ P := by simpa using hsub [iδ] (by simp)
  have hpP : ip ≤ P := by simpa using hsub [ip] (by simp)
  have heP : ie ≤ P := by simpa using hsub [ie] (by simp)
  have huP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hsP : is ≤ P := by simpa using hsub [is] (by simp)
  have hesP : ie * is ≤ P := by
    simpa using hsub [ie, is]
      (.cons _ <| .cons _ <| .cons _ <| .cons_cons _ <| .cons _ <| .cons _ <| .cons_cons _ .slnil)
  have hwP : iw ≤ P := by simpa using hsub [iw] (by simp)
  have hLM : log (max (ie * is) (max iu iw)) ≤ P :=
    (Real.log_le_self (by linarith)).trans (max_le hesP (max_le huP hwP))
  have hL8 : 1 ≤ log P := one_le_log_of_three_le (by linarith)
  have hn' : n ≤ 480 * P ^ 4 := by
    have : log (max (ie * is) (max iu iw)) * ip * is ^ 2 ≤ P * P * P ^ 2 := by gcongr
    have : log iδ * ip ≤ P * P := by
      gcongr; exact (Real.log_le_self (by linarith)).trans hδP
    have : P * P ≤ P * P * P ^ 2 :=
      le_mul_of_one_le_right (by positivity) (one_le_pow₀ (by linarith))
    linarith
  have hR : 1 / r ≤ 2 * P ^ 3 := by
    have : ie * is ^ 2 ≤ P * P ^ 2 := by gcongr
    have : P ≤ P * P ^ 2 := le_mul_of_one_le_right (by positivity) (one_le_pow₀ (by linarith))
    linarith
  have hX : max 1 ((nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r)) ≤ P ^ 15 := by
    have h1 : (1 : ℝ) ≤ P ^ 15 := one_le_pow₀ (by linarith)
    refine max_le h1 ?_
    rw [show (nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r)
      = (nJ + 2) * n * (2 * iδ) * (2 * iδ) * ip * (1 / r) by field_simp]
    have h3 : (16 : ℝ) ^ 4 ≤ P ^ 4 := pow_le_pow_left₀ (by norm_num) hP 4
    calc (nJ + 2) * n * (2 * iδ) * (2 * iδ) * ip * (1 / r)
        ≤ (3 * P) * (480 * P ^ 4) * (2 * P) * (2 * P) * P * (2 * P ^ 3) := by
          gcongr; linarith
      _ = 11520 * P ^ 11 := by ring
      _ ≤ P ^ 4 * P ^ 11 := by gcongr; linarith
      _ = P ^ 15 := by ring
  have hl := log_le_mul_log_of_le_pow 15 (by positivity) hX
  have hl0 : 0 ≤ log (max 1 ((nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r))) :=
    Real.log_nonneg (le_max_left _ _)
  -- the count rule's log is over the pool alone
  set ℓ := log (max (ie * is) (max iu iw))
  have hℓi : 1 ≤ ℓ * is := one_le_mul_of_one_le_of_one_le hLM1 (by linarith)
  have hℓd : 2 ≤ ℓ * iδ := by nlinarith only [hLM1, hiδ]
  have hℓdp : 2 ≤ ℓ * iδ * ip := by nlinarith only [hℓd, hip]
  obtain ⟨Z, hZ⟩ : ∃ Z, ℓ * iδ * ip * is = Z := ⟨_, rfl⟩
  have hZ4 : 4 ≤ Z := by
    rw [← hZ]; nlinarith only [hℓdp, his]
  have hZ1 : 1 ≤ Z := by linarith
  have hiδZ : iδ * ip ≤ Z := by
    rw [← hZ]
    calc iδ * ip = iδ * ip * 1 := by ring
      _ ≤ iδ * ip * (ℓ * is) := mul_le_mul_of_nonneg_left hℓi (by positivity)
      _ = ℓ * iδ * ip * is := by ring
  have hisZ : is ≤ Z := by
    rw [← hZ]
    calc is = 1 * is := by ring
      _ ≤ ℓ * iδ * ip * is := mul_le_mul_of_nonneg_right (by linarith) (by linarith)
  have hn1 : n + 1 ≤ 481 * Z ^ 2 := by
    have h1 : ℓ * ip * is ^ 2 ≤ Z * Z := by
      have h0 : 0 ≤ ℓ * ip * is := by positivity
      have : ℓ * ip * is ^ 2 ≤ Z * is := by
        rw [← hZ]
        calc ℓ * ip * is ^ 2 = (ℓ * ip * is) * is := by ring
          _ ≤ (ℓ * ip * is) * (iδ * is) :=
            mul_le_mul_of_nonneg_left (le_mul_of_one_le_left (by linarith) (by linarith)) h0
          _ = ℓ * iδ * ip * is * is := by ring
      calc ℓ * ip * is ^ 2 ≤ Z * is := this
        _ ≤ Z * Z := mul_le_mul_of_nonneg_left hisZ (by linarith)
    have h2 : log iδ * ip ≤ Z := by
      have : log iδ ≤ iδ := Real.log_le_self (by linarith)
      calc log iδ * ip ≤ iδ * ip := by gcongr
        _ ≤ Z := hiδZ
    have : Z ≤ Z ^ 2 := le_self_pow₀ hZ1 two_ne_zero
    have e : Z * Z = Z ^ 2 := by ring
    linarith
  have hY : max 1 (64 * (n + 1) / (1 / (2 * iδ))) ≤ Z ^ 11 := by
    have h1 : (1 : ℝ) ≤ Z ^ 11 := one_le_pow₀ hZ1
    refine max_le h1 ?_
    rw [div_one_div']
    have hiZ : iδ ≤ Z := le_trans (le_mul_of_one_le_right (by linarith) hip) hiδZ
    have h8 : (4 : ℝ) ^ 8 ≤ Z ^ 8 := pow_le_pow_left₀ (by norm_num) hZ4 8
    calc 64 * (n + 1) * (2 * iδ) ≤ 64 * (481 * Z ^ 2) * (2 * Z) := by gcongr
      _ = 61568 * Z ^ 3 := by ring
      _ ≤ Z ^ 8 * Z ^ 3 := by gcongr; linarith
      _ = Z ^ 11 := by ring
  have hy := log_le_mul_log_of_le_pow 11 (by positivity) hY
  have hy0 : 0 ≤ log (max 1 (64 * (n + 1) / (1 / (2 * iδ)))) := Real.log_nonneg (le_max_left _ _)
  have hlZ : 0 ≤ log Z := Real.log_nonneg hZ1
  have hs : log (256 * nJ / (1 / (2 * iδ))) ≤ 10 * log (nJ * iδ) := by
    refine (log_le_mul_log_of_le_pow (P := nJ * iδ) 10 (by positivity) ?_).trans (by norm_num)
    have : (2 : ℝ) ^ 9 ≤ (nJ * iδ) ^ 9 := pow_le_pow_left₀ (by norm_num) hx 9
    rw [div_one_div']
    nlinarith
  have hs0 : 0 ≤ log (256 * nJ / (1 / (2 * iδ))) :=
    Real.log_nonneg (by rw [div_one_div']; nlinarith)
  unfold nprefB
  generalize log (max 1 ((nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r))) = l at *
  generalize log (max 1 (64 * (n + 1) / (1 / (2 * iδ)))) = lc at *
  generalize log (256 * nJ / (1 / (2 * iδ))) = ls at *
  have hm0 : 0 < min (1 / ie) (1 / iu) := lt_min (by positivity) (by positivity)
  rw [show 2 ^ 22 * nJ * lc / ((1 / is) ^ 4 * min (1 / ie) (1 / iu) ^ 2) + 3 * 2 ^ 25 * l / r ^ 2
      + 6 * ls / (1 / isl) + 2 ^ 11 * nH / r
    = 2 ^ 22 * (nJ * lc * is ^ 4 * (1 / min (1 / ie) (1 / iu)) ^ 2)
      + 3 * 2 ^ 25 * (l * (1 / r) ^ 2) + 6 * (ls * isl) + 2 ^ 11 * (nH * (1 / r)) by field_simp]
  have h3 := mul_le_mul_of_nonneg_right hs (by linarith : (0 : ℝ) ≤ isl)
  have : 0 ≤ isl * log (nJ * iδ) := by positivity
  have hR0 : 0 ≤ 1 / r := by positivity
  have hR2 : (1 / r) ^ 2 ≤ 2 * (ie ^ 2 * is ^ 4 + iu ^ 2) := by
    have := pow_le_pow_left₀ hR0 hr2 2
    have := add_sq_le (a := ie * is ^ 2) (b := iu)
    linarith
  have hm : 1 / min (1 / ie) (1 / iu) ≤ ie + iu := by
    have := one_div_min_le (x := 1 / ie) (y := 1 / iu) (by positivity) (by positivity)
    rwa [one_div_one_div, one_div_one_div] at this
  have hM2 : (1 / min (1 / ie) (1 / iu)) ^ 2 ≤ 2 * (ie ^ 2 + iu ^ 2) :=
    (pow_le_pow_left₀ (by positivity) hm 2).trans add_sq_le
  have h1 : nJ * lc * is ^ 4 * (1 / min (1 / ie) (1 / iu)) ^ 2
      ≤ nJ * (11 * log Z) * is ^ 4 * (2 * (ie ^ 2 + iu ^ 2)) := by
    gcongr; exact_mod_cast hy
  rw [hZ]
  have h1' : l * (1 / r) ^ 2 ≤ (15 * log P) * (2 * (ie ^ 2 * is ^ 4 + iu ^ 2)) := by
    gcongr; exact_mod_cast hl
  have h2 : nH * (1 / r) ≤ nH * (ie * is ^ 2 + iu) := by gcongr
  have : 0 ≤ nJ * log Z * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4) := by positivity
  have : 0 ≤ log P * (ie ^ 2 * is ^ 4 + iu ^ 2) :=
    mul_nonneg (by linarith) (by positivity)
  have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
  constructor
  · positivity
  · linarith

private lemma gn_le_pow {nJ iδ ip ie isl iu iw is nH : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hisl : 2 ≤ isl) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) :
    2 ^ 32 * (nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
          * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu))
      ≤ (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) ^ 22 := by
  have hl : ∀ x ∈ [nJ, iδ, ip, ie, isl, iu, iw, is, nH + 1], 1 ≤ x := fun x hx => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> linarith
  have hsub (l : List ℝ) (hs : l.Sublist [nJ, iδ, ip, ie, isl, iu, iw, is, nH + 1]) :
      l.prod ≤ nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  obtain ⟨-, hx, hP8⟩ := prods hJ hiδ hip hie hiu hiw his
  have h8 : nJ * iδ * ip * ie * iu * iw * is
      ≤ nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) := by
    have : 1 ≤ isl * (nH + 1) := one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    calc nJ * iδ * ip * ie * iu * iw * is = nJ * iδ * ip * ie * iu * iw * is * 1 := by ring
      _ ≤ nJ * iδ * ip * ie * iu * iw * is * (isl * (nH + 1)) := by gcongr
      _ = _ := by ring
  have hL : log (nJ * iδ * ip * ie * iu * iw * is) ≤ nJ * iδ * ip * ie * iu * iw * is :=
    Real.log_le_self (by positivity)
  have hL0 : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have hΛ : log (nJ * iδ) ≤ nJ * iδ := Real.log_le_self (by positivity)
  have hΛ0 : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have hM4 : 4 ≤ max (ie * is) (max iu iw) := le_max_of_le_left (by nlinarith)
  have hLM1 : 1 ≤ log (max (ie * is) (max iu iw)) := one_le_log_of_three_le (by linarith)
  set P := nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)
  have hJP : nJ ≤ P := by simpa using hsub [nJ] (by simp)
  have hxP : nJ * iδ ≤ P := by simpa [mul_assoc] using hsub [nJ, iδ] (by simp)
  have heP : ie ≤ P := by simpa using hsub [ie] (by simp)
  have hslP : isl ≤ P := by simpa using hsub [isl] (by simp)
  have huP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hwP : iw ≤ P := by simpa using hsub [iw] (by simp)
  have hsP : is ≤ P := by simpa using hsub [is] (by simp)
  have hnP : nH + 1 ≤ P := by simpa using hsub [nH + 1] (by simp)
  have hesP : ie * is ≤ P := by
    simpa using hsub [ie, is] (.cons _ <| .cons _ <| .cons _ <| .cons_cons _ <| .cons _ <|
      .cons _ <| .cons _ <| .cons_cons _ <| .cons _ .slnil)
  have hdpsP : iδ * ip * is ≤ P := by
    simpa [mul_assoc] using hsub [iδ, ip, is] (.cons _ <| .cons_cons _ <| .cons_cons _ <|
      .cons _ <| .cons _ <| .cons _ <| .cons _ <| .cons_cons _ <| .cons _ .slnil)
  have hP : 16 ≤ P := by linarith
  have hP1 : 1 ≤ P := by linarith
  set ℓ := log (max (ie * is) (max iu iw))
  have hℓP : ℓ ≤ P := (Real.log_le_self (by linarith)).trans (max_le hesP (max_le huP hwP))
  have hZ : log (ℓ * iδ * ip * is) ≤ P ^ 2 := by
    refine (Real.log_le_self (by positivity)).trans ?_
    calc ℓ * iδ * ip * is = ℓ * (iδ * ip * is) := by ring
      _ ≤ P * P := by gcongr
      _ = P ^ 2 := by ring
  have hZ0 : 0 ≤ log (ℓ * iδ * ip * is) :=
    Real.log_nonneg (by linarith [Z_ge hLM1 hiδ hip his])
  have h1 : nJ * log (ℓ * iδ * ip * is) * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
      ≤ P * P ^ 2 * (P ^ 2 * P ^ 4 + P ^ 2 * P ^ 4) := by
    gcongr
  have h1b : log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
      ≤ P * (P ^ 2 * P ^ 4 + P ^ 2) := by
    gcongr <;> linarith
  have h1' : isl * log (nJ * iδ) ≤ P * P := by gcongr; linarith
  have h2 : nH * (ie * is ^ 2 + iu) ≤ P * (P * P ^ 2 + P) := by gcongr; linarith
  have h3 : P * P ^ 2 * (P ^ 2 * P ^ 4 + P ^ 2 * P ^ 4) + P * (P ^ 2 * P ^ 4 + P ^ 2) + P * P
      + P * (P * P ^ 2 + P) ≤ 7 * P ^ 13 := by
    have : P ^ 2 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 3 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 4 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 7 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 9 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    ring_nf; linarith
  have h4 : 2 ^ 32 * (7 * P ^ 13) ≤ P ^ 22 := by
    have : (16 : ℝ) ^ 9 ≤ P ^ 9 := pow_le_pow_left₀ (by norm_num) hP 9
    calc 2 ^ 32 * (7 * P ^ 13) ≤ P ^ 9 * P ^ 13 := by
          rw [← mul_assoc]; gcongr; exact le_trans (by norm_num) this
      _ = P ^ 22 := by ring
  have : 0 ≤ P ^ 13 := by positivity
  linarith

private def rsc (ie iu is : ℝ) : ℝ := rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie)

private lemma le_of_one_le_mul {a b c : ℝ} (ha : 0 ≤ a) (hc : 1 ≤ c) (h : a * c = b) : a ≤ b :=
  h ▸ le_mul_of_one_le_right ha hc

private lemma one_le_mul' {a b : ℝ} (ha : 1 ≤ a) (hb : 1 ≤ b) : 1 ≤ a * b :=
  one_le_mul_of_one_le_of_one_le ha hb

private def kW : ClusterVec → ℝ := fun (_, _, _, ie, _, iu, iw, is, _) =>
  kB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4)

private def nsuffW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
  nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ))

private def nprefW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  nprefB (1 / is) (rsc ie iu is) (min (1 / ie) (1 / iu)) (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip)
    (1 / isl) nJ nH (nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ)))

open scoped Classical in
private def atomTolW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is then
    min (min (1 / is * min (1 / ie * (1 / is)) (1 / iu)
          / (2 ^ 19 * max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))))
        (min (1 / isl) (1 / iu)
          / (2 ^ 14 * (nJ + 3) * max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH)))))
      (min (1 / isl) (1 / iu) ^ 2
        / (512 * log (max 2 (64 * nJ * max 1 (kW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))
          * (3 + 2 * logb 2 (max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))))
          / (1 / (2 * iδ))))))
  else 1

open scoped Classical in
private def sufAtomTolW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is then
    1 / ip / (2 ^ 16 * max 1 (nsuffW (nJ, iδ, ip, ie, isl, iu, iw, is, nH)))
  else 1

private lemma tol_pos (v : ClusterVec) : 0 < atomTolW v ∧ 0 < sufAtomTolW v := by
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := v
  simp only [atomTolW, sufAtomTolW]
  split_ifs with h
  · obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, his⟩ := h
    have : 0 < log (max 2 (64 * nJ * max 1 (kW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))
        * (3 + 2 * logb 2 (max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))))
        / (1 / (2 * iδ)))) := Real.log_pos (lt_of_lt_of_le one_lt_two (le_max_left _ _))
    refine ⟨lt_min (lt_min (by positivity) (by positivity)) (by positivity), by positivity⟩
  · exact ⟨one_pos, one_pos⟩

private lemma rsc_facts {ie iu is : ℝ} (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (his : 2 ≤ is) :
    0 < rsc ie iu is ∧ 1 / rsc ie iu is ≤ ie * is ^ 2 + iu :=
  ⟨rs_pos (by linarith) (by linarith) (by linarith), rs_le (by linarith) (by linarith) his⟩

private lemma logM_nonneg {ie iu iw is : ℝ} (hie : 2 ≤ ie) (his : 2 ≤ is) :
    0 ≤ log (max (ie * is) (max iu iw)) :=
  Real.log_nonneg (le_max_of_le_left (by nlinarith))

private lemma kW_isBigO : kW =O[𝓟 vecRegion] (fun (_, _, _, ie, _, iu, iw, is, _) =>
    log (max (ie * is) (max iu iw)) * is ^ 2 : ClusterVec → ℝ) := by
  refine isBigO_on 110 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, -, hiu, hiw, his, -⟩ := region hb
  have := logM_nonneg (iu := iu) (iw := iw) hie his
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := kB_bound hie hiu hiw his hr hr2
  dsimp only [kW]
  exact ⟨h0, by positivity, h⟩

private lemma nsuffW_isBigO : nsuffW =O[𝓟 vecRegion] (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
    log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip : ClusterVec → ℝ) := by
  refine isBigO_on 240 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨-, hiδ, hip, hie, -, hiu, hiw, his, -⟩ := region hb
  have := logM_nonneg (iu := iu) (iw := iw) hie his
  have : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := nsuffB_bound hiδ hip hie hiu hiw his hr hr2
  dsimp only [nsuffW]
  exact ⟨h0, by positivity, h⟩

private lemma nprefW_isBigO : nprefW =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is) * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu) : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 32) fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, hx, hP8⟩ := prods hJ hiδ hip hie hiu hiw his
  have : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have : 0 ≤ log (log (max (ie * is) (max iu iw)) * iδ * ip * is) := by
    have : 1 ≤ log (max (ie * is) (max iu iw)) :=
      one_le_log_of_three_le (le_max_of_le_left (by nlinarith))
    exact Real.log_nonneg (by linarith [Z_ge this hiδ hip his])
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hiδ hip hie hiu hiw his hr hr2
  obtain ⟨h0, h⟩ := nprefB_bound hJ hiδ hip hie hisl hiu hiw his hnH hr hr2 hs0 hs
  dsimp only [nprefW]
  exact ⟨h0, by positivity, h⟩

set_option maxHeartbeats 4000000 in
private lemma atomTolW_isBigO : (fun v => 1 / atomTolW v) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      (nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
              * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
          + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
          + nH * (ie * is ^ 2 + iu)) * (nJ * (iu + isl) + ie * is ^ 2 + iu * is)
        + nJ * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * isl ^ 2
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 53) fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, hx, hP7⟩ := prods hJ hiδ hip hie hiu hiw his
  have he : 1 ≤ ie := by linarith
  have hu : 1 ≤ iu := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hP79 : nJ * iδ * ip * ie * iu * iw * is
      ≤ nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) :=
    le_of_one_le_mul (by positivity) (one_le_mul' (by linarith : 1 ≤ isl)
      (by linarith : 1 ≤ nH + 1)) (by ring)
  have hx7 : nJ * iδ ≤ nJ * iδ * ip * ie * iu * iw * is :=
    le_of_one_le_mul (by positivity) (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' hip he)
      hu) hiw) hs1) (by ring)
  have hL7 : 1 ≤ log (nJ * iδ * ip * ie * iu * iw * is) :=
    one_le_log_of_three_le (by linarith)
  have hL9 : 1 ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) :=
    one_le_log_of_three_le (by linarith)
  have hΛ : log (nJ * iδ) ≤ log (nJ * iδ * ip * ie * iu * iw * is) :=
    Real.log_le_log (by positivity) hx7
  have hΛ0 : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have hΛ9 : log (nJ * iδ) ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) :=
    hΛ.trans (Real.log_le_log (by positivity) hP79)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hiδ hip hie hiu hiw his hr hr2
  obtain ⟨hN0, hN⟩ := nprefB_bound hJ hiδ hip hie hisl hiu hiw his hnH hr hr2 hs0 hs
  have hG := gn_le_pow hJ hiδ hip hie hisl hiu hiw his hnH
  rw [show ∀ a b c d : ℝ, a + b + c + d = a + b + d + c by intros; ring] at hN hG
  have hZ1 : 1 ≤ log (log (max (ie * is) (max iu iw)) * iδ * ip * is) := by
    have : 1 ≤ log (max (ie * is) (max iu iw)) :=
      one_le_log_of_three_le (le_max_of_le_left (by nlinarith))
    exact one_le_log_of_three_le (by linarith [Z_ge this hiδ hip his])
  have hcond : 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is :=
    ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩
  dsimp only
  simp only [atomTolW, nprefW, if_pos hcond]
  set K := max 1 (kW (nJ, iδ, ip, ie, isl, iu, iw, is, nH)) with hKdef
  set N := nprefB (1 / is) (rsc ie iu is) (min (1 / ie) (1 / iu)) (1 / (2 * iδ)) (1 / (2 * iδ))
    (1 / ip) (1 / isl) nJ nH (nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ)))
  set L9 := log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) with hL9e
  set G := nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
    * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
    + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2) + nH * (ie * is ^ 2 + iu)
  set r := rsc ie iu is
  have hG0 : 0 ≤ G := by positivity
  have hGΛ : log (nJ * iδ) * (ie * is ^ 2 + iu * is) ≤ 2 * G := by
    have h1 : ie * is ^ 2 ≤ ie ^ 2 * is ^ 4 :=
      le_of_one_le_mul (by positivity) (one_le_mul' he (one_le_pow₀ (n := 2) hs1)) (by ring)
    have h2 : iu * is ≤ iu ^ 2 + ie ^ 2 * is ^ 4 := by
      have : is ^ 2 ≤ ie ^ 2 * is ^ 4 :=
        le_of_one_le_mul (by positivity) (one_le_mul' (one_le_pow₀ (n := 2) he)
          (one_le_pow₀ (n := 2) hs1)) (by ring)
      nlinarith only [sq_nonneg (iu - is), this, sq_nonneg iu, sq_nonneg is]
    have h3 : log (nJ * iδ) * (ie * is ^ 2 + iu * is)
        ≤ log (nJ * iδ * ip * ie * iu * iw * is) * (2 * (ie ^ 2 * is ^ 4 + iu ^ 2)) := by
      gcongr <;> linarith [sq_nonneg iu]
    have : 0 ≤ nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
      * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4) := by positivity
    have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
    linarith
  have hG1 : 1 ≤ 2 ^ 32 * (G + isl * log (nJ * iδ)) := by
    have h1 : 1 ≤ ie ^ 2 * is ^ 4 :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (one_le_pow₀ (by linarith))
    have h2 : 1 ≤ ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4 := by
      have : 0 ≤ iu ^ 2 * is ^ 4 := by positivity
      linarith
    have : 1 ≤ nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
        * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4) :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hJ hZ1) h2
    have : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2) :=
      mul_nonneg (by linarith) (by positivity)
    have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
    have : 0 ≤ isl * log (nJ * iδ) := by positivity
    linarith
  have hM : max 1 N ≤ 2 ^ 32 * (G + isl * log (nJ * iδ)) := max_le hG1 hN
  have hM1 : 1 ≤ max 1 N := le_max_left _ _
  have hlb0 : 0 ≤ logb 2 (max 1 N) := logb_nonneg one_lt_two hM1
  have hA0 : 0 < 1 / is * min (1 / ie * (1 / is)) (1 / iu) / (2 ^ 19 * max 1 N) := by
    positivity
  have hE0 : 0 < min (1 / isl) (1 / iu) / (2 ^ 14 * (nJ + 3) * max 1 N) := by positivity
  have hK1 : 1 ≤ K := le_max_left _ _
  have hB0 : 0 < min (1 / isl) (1 / iu) ^ 2 / (512 * log (max 2
      (64 * nJ * K * (3 + 2 * logb 2 (max 1 N)) / (1 / (2 * iδ))))) := by
    have := Real.log_pos (lt_of_lt_of_le one_lt_two (le_max_left 2
      (64 * nJ * K * (3 + 2 * logb 2 (max 1 N)) / (1 / (2 * iδ)))))
    positivity
  refine ⟨(one_div_pos.2 (lt_min (lt_min hA0 hE0) hB0)).le, by positivity, ?_⟩
  refine (one_div_min_le (lt_min hA0 hE0) hB0).trans ?_
  refine (add_le_add (one_div_min_le hA0 hE0) le_rfl).trans ?_
  rw [one_div_div, one_div_div, one_div_div,
    show 2 ^ 19 * max 1 N / (1 / is * min (1 / ie * (1 / is)) (1 / iu))
      = 2 ^ 19 * (max 1 N * (1 / min (1 / ie * (1 / is)) (1 / iu)) * is) by
      field_simp,
    show 2 ^ 14 * (nJ + 3) * max 1 N / min (1 / isl) (1 / iu)
      = 2 ^ 14 * ((nJ + 3) * max 1 N * (1 / min (1 / isl) (1 / iu))) by ring]
  set X := 64 * nJ * K * (3 + 2 * logb 2 (max 1 N)) / (1 / (2 * iδ)) with hXdef
  have hX2 : 2 ≤ X := by
    rw [hXdef, div_one_div']
    have h1 : 1 ≤ nJ * K := one_le_mul' hJ hK1
    have h2 : 3 ≤ 3 + 2 * logb 2 (max 1 N) := by linarith
    have h3 := mul_le_mul h1 h2 (by norm_num) (by positivity)
    have h4 : 4 ≤ 2 * iδ := by linarith
    have h5 := mul_le_mul h3 h4 (by norm_num) (by positivity)
    nlinarith
  have hmX : max 2 X = X := max_eq_right hX2
  rw [hmX, show 512 * log X / min (1 / isl) (1 / iu) ^ 2
      = 512 * (log X * (1 / min (1 / isl) (1 / iu)) ^ 2) by
    rw [one_div_pow, mul_one_div, mul_div_assoc]]
  have hma : 1 / min (1 / ie * (1 / is)) (1 / iu) ≤ ie * is + iu := by
    have := one_div_min_le (x := 1 / ie * (1 / is)) (y := 1 / iu) (by positivity) (by positivity)
    rwa [one_div_one_div, show 1 / (1 / ie * (1 / is)) = ie * is by field_simp] at this
  have h1 : max 1 N * (1 / min (1 / ie * (1 / is)) (1 / iu)) * is
      ≤ (2 ^ 32 * (G + isl * log (nJ * iδ))) * (ie * is + iu) * is := by
    gcongr
  have hm : 1 / min (1 / isl) (1 / iu) ≤ isl + iu := by
    have := one_div_min_le (x := 1 / isl) (y := 1 / iu) (by positivity) (by positivity)
    rwa [one_div_one_div, one_div_one_div] at this
  have h1e : (nJ + 3) * max 1 N * (1 / min (1 / isl) (1 / iu))
      ≤ (4 * nJ) * (2 ^ 32 * (G + isl * log (nJ * iδ))) * (isl + iu) := by
    gcongr; linarith
  have hm2 : (1 / min (1 / isl) (1 / iu)) ^ 2 ≤ 2 * (isl ^ 2 + iu ^ 2) :=
    (pow_le_pow_left₀ (by positivity) hm 2).trans add_sq_le
  have hlb : 3 + 2 * logb 2 (max 1 N) ≤ 91 * L9 := by
    have hlog := log_le_mul_log_of_le_pow 22 (by positivity) (hM.trans hG)
    rw [← hL9e] at hlog
    have hl0 : 0 ≤ log (max 1 N) := Real.log_nonneg hM1
    have h2 : 1 / 2 < log 2 := by linarith [log_two_gt_d9]
    have : log (max 1 N) / log 2 ≤ 2 * log (max 1 N) := by
      have := mul_le_mul_of_nonneg_left h2.le hl0
      rw [div_le_iff₀ (by linarith)]; linarith
    rw [Real.logb]
    push_cast at hlog
    linarith only [hlog, this, hL9]
  -- the hash's log is a few copies of `L9`
  set P9 := nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) with hP9
  have hxP9 : nJ * iδ ≤ P9 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' hip he)
      (by linarith : 1 ≤ isl)) hu) hiw) hs1) (by linarith : 1 ≤ nH + 1)) (by ring)
  have hP16 : 16 ≤ P9 := by linarith
  have hLP : L9 ≤ P9 := by rw [hL9e]; exact Real.log_le_self (by positivity)
  -- the family's size is a few powers of `P9`
  have hKle : K ≤ P9 ^ 5 := by
    obtain ⟨-, hk⟩ := kB_bound hie hiu hiw his hr hr2
    have hM1 : ie * is ≤ P9 := le_of_one_le_mul (by positivity)
      (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' hJ
        (by linarith : 1 ≤ iδ)) hip) (by linarith : 1 ≤ isl)) hu) hiw)
        (by linarith : 1 ≤ nH + 1)) (by rw [hP9]; ring)
    have hM2 : iu ≤ P9 := le_of_one_le_mul (by positivity)
      (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul'
        hJ (by linarith : 1 ≤ iδ)) hip) he) (by linarith : 1 ≤ isl)) hiw) hs1)
        (by linarith : 1 ≤ nH + 1)) (by rw [hP9]; ring)
    have hM3 : iw ≤ P9 := le_of_one_le_mul (by positivity)
      (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul'
        hJ (by linarith : 1 ≤ iδ)) hip) he) (by linarith : 1 ≤ isl)) hu) hs1)
        (by linarith : 1 ≤ nH + 1)) (by rw [hP9]; ring)
    have hMP : max (ie * is) (max iu iw) ≤ P9 := max_le hM1 (max_le hM2 hM3)
    have hlM : log (max (ie * is) (max iu iw)) ≤ P9 :=
      (Real.log_le_self (by positivity)).trans hMP
    have his9 : is ≤ P9 := le_of_one_le_mul (by positivity)
      (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul'
        hJ (by linarith : 1 ≤ iδ)) hip) he) (by linarith : 1 ≤ isl)) hu) hiw)
        (by linarith : 1 ≤ nH + 1)) (by rw [hP9]; ring)
    have hP2 : (110 : ℝ) ≤ P9 ^ 2 := by nlinarith
    have hk' : kB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) ≤ P9 ^ 5 := by
      have h1 : log (max (ie * is) (max iu iw)) * is ^ 2 ≤ P9 * P9 ^ 2 :=
        mul_le_mul hlM (pow_le_pow_left₀ (by linarith) his9 2) (by positivity) (by positivity)
      calc kB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4)
          ≤ 110 * (log (max (ie * is) (max iu iw)) * is ^ 2) := hk
        _ ≤ P9 ^ 2 * (P9 * P9 ^ 2) := by
          have hlM0 : 0 ≤ log (max (ie * is) (max iu iw)) :=
            Real.log_nonneg (le_max_of_le_right (le_max_of_le_left (by linarith)))
          exact mul_le_mul hP2 h1 (mul_nonneg hlM0 (sq_nonneg is)) (by positivity)
        _ = P9 ^ 5 := by ring
    refine max_le (one_le_pow₀ (by linarith)) ?_
    simpa only [kW] using hk'
  have hXle : X ≤ P9 ^ 11 := by
    rw [hXdef, div_one_div']
    have h4 : (16 : ℝ) ^ 4 ≤ P9 ^ 4 := pow_le_pow_left₀ (by norm_num) hP16 4
    calc 64 * nJ * K * (3 + 2 * logb 2 (max 1 N)) * (2 * iδ)
        = 128 * (nJ * iδ) * K * (3 + 2 * logb 2 (max 1 N)) := by ring
      _ ≤ 128 * P9 * P9 ^ 5 * (91 * P9) := by gcongr; linarith
      _ = 11648 * P9 ^ 7 := by ring
      _ ≤ P9 ^ 4 * P9 ^ 7 := by gcongr; linarith
      _ = P9 ^ 11 := by ring
  have hLXL : log X ≤ 11 * L9 := by
    have := log_le_mul_log_of_le_pow 11 (by linarith) hXle
    rw [hL9e]; push_cast at this; exact this
  have h2 : log X * (1 / min (1 / isl) (1 / iu)) ^ 2 ≤ 11 * L9 * (2 * (isl ^ 2 + iu ^ 2)) := by
    gcongr
  -- `L9` against `u²` is under the first summand
  have hsplit : L9 = log (nJ * iδ * ip * ie * iu * iw * is) + log isl + log (nH + 1) := by
    rw [hL9e, hP9, show nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)
      = nJ * iδ * ip * ie * iu * iw * is * isl * (nH + 1) by ring,
      Real.log_mul (by positivity) (by linarith), Real.log_mul (by positivity) (by linarith)]
  have hli : log isl ≤ isl := Real.log_le_self (by linarith)
  have hln : log (nH + 1) ≤ nH := by
    have := Real.log_le_sub_one_of_pos (by linarith : (0 : ℝ) < nH + 1); linarith
  have hiu2 : 0 ≤ iu ^ 2 := sq_nonneg iu
  have hGa : 0 ≤ nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
    * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4) := by positivity
  have hGc : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
  have hGb : log (nJ * iδ * ip * ie * iu * iw * is) * iu ^ 2
      ≤ log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2) :=
    mul_le_mul_of_nonneg_left (by have : 0 ≤ ie ^ 2 * is ^ 4 := (by positivity); linarith)
      (by linarith)
  have hGu : log (nJ * iδ * ip * ie * iu * iw * is) * iu ^ 2 ≤ G := by linarith
  have hGi : log isl * iu ^ 2 ≤ isl * G := by
    have h := mul_le_mul hli (le_mul_of_one_le_left hiu2 hL7) hiu2 (by linarith)
    have h2 : isl * (log (nJ * iδ * ip * ie * iu * iw * is) * iu ^ 2) ≤ isl * G :=
      mul_le_mul_of_nonneg_left hGu (by linarith)
    linarith
  have hGn : log (nH + 1) * iu ^ 2 ≤ G * iu := by
    have h0 : log (nH + 1) * iu ^ 2 ≤ nH * iu ^ 2 := mul_le_mul_of_nonneg_right hln hiu2
    have e : nH * (ie * is ^ 2 + iu) * iu = nH * ie * is ^ 2 * iu + nH * iu ^ 2 := by ring
    have : 0 ≤ nH * ie * is ^ 2 * iu := by positivity
    have hGb0 : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2) :=
      mul_nonneg (by linarith) (by positivity)
    have h4 : nH * (ie * is ^ 2 + iu) * iu ≤ G * iu :=
      mul_le_mul_of_nonneg_right (by linarith) (by linarith)
    linarith
  have hLu : L9 * iu ^ 2 ≤ G + isl * G + G * iu := by
    rw [hsplit, add_mul, add_mul]
    linarith
  have hΛu : log (nJ * iδ) * iu ≤ 2 * G := by
    have hiu : iu ≤ ie * is ^ 2 + iu * is := by
      have : iu ≤ iu * is := le_mul_of_one_le_right (by linarith) hs1
      have : 0 ≤ ie * is ^ 2 := by positivity
      linarith
    have := mul_le_mul_of_nonneg_left hiu hΛ0
    linarith
  have hJ1 : ∀ y : ℝ, 0 ≤ y → y ≤ nJ * y := fun y hy => le_mul_of_one_le_left hy hJ
  have cG : G ≤ nJ * G * iu := by
    have := hJ1 (G * iu) (by positivity)
    have := le_mul_of_one_le_right hG0 hu
    linarith [show nJ * (G * iu) = nJ * G * iu by ring]
  have cGi : isl * G ≤ nJ * G * isl := by
    have := hJ1 (isl * G) (by positivity)
    linarith [show nJ * (isl * G) = nJ * G * isl by ring]
  have cGu : G * iu ≤ nJ * G * iu := by
    have := hJ1 (G * iu) (by positivity)
    linarith [show nJ * (G * iu) = nJ * G * iu by ring]
  have hLi : L9 * isl ^ 2 ≤ nJ * L9 * isl ^ 2 := by
    have := hJ1 (L9 * isl ^ 2) (by positivity)
    linarith [show nJ * (L9 * isl ^ 2) = nJ * L9 * isl ^ 2 by ring]
  have : 0 ≤ G * (ie * is ^ 2 + iu * is) := by positivity
  have : 0 ≤ nJ * G * isl := by positivity
  have : 0 ≤ nJ * G * iu := by positivity
  have : 0 ≤ nJ * L9 * isl ^ 2 := by positivity
  have s1 : 2 ^ 19 * (max 1 N * (1 / min (1 / ie * (1 / is)) (1 / iu)) * is)
      ≤ 2 ^ 51 * (G * (ie * is ^ 2 + iu * is)) + 2 ^ 52 * (nJ * G * isl) := by
    have e : 2 ^ 32 * (G + isl * log (nJ * iδ)) * (ie * is + iu) * is
        = 2 ^ 32 * (G * (ie * is ^ 2 + iu * is))
          + 2 ^ 32 * (isl * (log (nJ * iδ) * (ie * is ^ 2 + iu * is))) := by ring
    have c1 : isl * (log (nJ * iδ) * (ie * is ^ 2 + iu * is)) ≤ isl * (2 * G) :=
      mul_le_mul_of_nonneg_left hGΛ (by linarith)
    linarith
  have s2 : 2 ^ 14 * ((nJ + 3) * max 1 N * (1 / min (1 / isl) (1 / iu)))
      ≤ 2 ^ 50 * (nJ * G * isl) + 2 ^ 48 * (nJ * G * iu) + 2 ^ 48 * (nJ * L9 * isl ^ 2) := by
    have e : 4 * nJ * (2 ^ 32 * (G + isl * log (nJ * iδ))) * (isl + iu)
        = 2 ^ 34 * (nJ * G * isl) + 2 ^ 34 * (nJ * G * iu)
          + 2 ^ 34 * (nJ * isl ^ 2 * log (nJ * iδ))
          + 2 ^ 34 * (nJ * isl * (log (nJ * iδ) * iu)) := by ring
    have c2 : nJ * isl ^ 2 * log (nJ * iδ) ≤ nJ * isl ^ 2 * L9 :=
      mul_le_mul_of_nonneg_left hΛ9 (by positivity)
    have c3 : nJ * isl * (log (nJ * iδ) * iu) ≤ nJ * isl * (2 * G) :=
      mul_le_mul_of_nonneg_left hΛu (by positivity)
    have e2 : nJ * isl ^ 2 * L9 = nJ * L9 * isl ^ 2 := by ring
    have e3 : nJ * isl * (2 * G) = 2 * (nJ * G * isl) := by ring
    linarith
  have s3 : 512 * (log X * (1 / min (1 / isl) (1 / iu)) ^ 2)
      ≤ 11264 * (nJ * L9 * isl ^ 2) + 11264 * (G + isl * G + G * iu) := by
    have e : 11 * L9 * (2 * (isl ^ 2 + iu ^ 2)) = 22 * (L9 * isl ^ 2) + 22 * (L9 * iu ^ 2) := by
      ring
    linarith
  have e : G * (nJ * (iu + isl) + ie * is ^ 2 + iu * is)
      = nJ * G * iu + nJ * G * isl + G * (ie * is ^ 2 + iu * is) := by ring
  linarith

private lemma sufAtomTolW_isBigO : (fun v => 1 / sufAtomTolW v) =O[𝓟 vecRegion]
    (fun (_, iδ, ip, ie, _, iu, iw, is, _) =>
      ip * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip)
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 40) fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, -⟩ := region hb
  obtain ⟨-, hx, -⟩ := prods hJ hiδ hip hie hiu hiw his
  have hL4 : 1 ≤ log (max (ie * is) (max iu iw)) :=
    one_le_log_of_three_le (le_max_of_le_left (by nlinarith))
  have hLJ : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hiδ hip hie hiu hiw his hr hr2
  have hcond : 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is :=
    ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩
  dsimp only
  simp only [sufAtomTolW, nsuffW, if_pos hcond]
  set N := nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ))
  set G := log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip
  have hG1 : 1 ≤ 240 * G := by
    have : 1 ≤ log (max (ie * is) (max iu iw)) * ip * is ^ 2 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hL4 hip)
        (one_le_pow₀ (by linarith))
    have : 0 ≤ log iδ * ip := by positivity
    linarith
  have hM : max 1 N ≤ 240 * G := max_le hG1 hs
  have hA0 : 0 < 1 / ip / (2 ^ 16 * max 1 N) := by positivity
  refine ⟨(one_div_pos.2 hA0).le, by positivity, ?_⟩
  rw [one_div_div, show 2 ^ 16 * max 1 N / (1 / ip) = 2 ^ 16 * (max 1 N * ip) by
    field_simp]
  have : max 1 N * ip ≤ (240 * G) * ip := by gcongr
  have : 0 ≤ ip * G := by positivity
  linarith

/-- No strings harvested. -/
private def toVec (x : TrainFamilyInputs) : ClusterVec :=
  (x.nJ, 1 / x.δ, 1 / x.pAP, 1 / x.εc, 1 / x.εsc, 1 / x.ε_u, 1 / x.κ, 1 / x.s₀, 0)

private lemma toVec_mem {x : TrainFamilyInputs} (hx : x ∈ trainFamilyRegion) :
    toVec x ∈ vecRegion := by
  obtain ⟨hJ, hδ, hδ1, hp, hp1, he₀, he₁, hsl₀, hsl₁, hu, hu1, hw, hw1, hs, hs1, hb⟩ := hx
  have he : 0 < x.εc := lt_min he₀ (by linarith)
  have he1 : x.εc ≤ 1 / 2 := (min_le_left _ _).trans he₁
  have hsl : 0 < x.εsc := lt_min hsl₀ he
  have hsl1 : x.εsc ≤ 1 / 2 := (min_le_left _ _).trans hsl₁
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hJ, (le_div_iff₀ hδ).2 (by linarith), (le_div_iff₀ hp).2 (by linarith),
    (le_div_iff₀ he).2 (by linarith), (le_div_iff₀ hsl).2 (by linarith),
    (le_div_iff₀ hu).2 (by linarith), (le_div_iff₀ hw).2 (by linarith),
    (le_div_iff₀ hs).2 (by linarith), le_rfl⟩

private lemma transfer {f g : ClusterVec → ℝ} {g' : TrainFamilyInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 trainFamilyRegion] g' :=
  (isBigO_comp_on h fun _ hx => toVec_mem hx).congr_right hg

/-- The clustering runs at a quarter of the signal. -/
private def quarter (x : TrainFamilyInputs) : TrainFamilyInputs := { x with s₀ := x.s₀ / 4 }

/-- `toVec` at a quarter of the signal. -/
private def quad : ClusterVec → ClusterVec := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  (nJ, iδ, ip, ie, isl, iu, iw, 4 * is, nH)

private lemma toVec_quarter (x : TrainFamilyInputs) : toVec (quarter x) = quad (toVec x) := by
  simp only [toVec, quarter, quad, TrainFamilyInputs.εc, TrainFamilyInputs.εsc, one_div_div,
    mul_one_div]

private lemma quad_maps : Set.MapsTo quad vecRegion vecRegion := by
  intro b hb
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  simp only [quad, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, by linarith, hnH⟩

private lemma logM_quad {ie iu iw is : ℝ} (hie : 2 ≤ ie) (his : 2 ≤ is) (hiu : 0 ≤ iu)
    (hiw : 0 ≤ iw) :
    1 ≤ log (max (ie * is) (max iu iw))
      ∧ log (max (ie * is) (max iu iw)) ≤ log (max (ie * (4 * is)) (max iu iw))
      ∧ log (max (ie * (4 * is)) (max iu iw)) ≤ 2 * log (max (ie * is) (max iu iw)) := by
  set M := max (ie * is) (max iu iw) with hMd
  have hM : 4 ≤ M := le_max_of_le_left (by nlinarith)
  have hM' : M ≤ max (ie * (4 * is)) (max iu iw) := max_le_max (by nlinarith) le_rfl
  have hM4 : max (ie * (4 * is)) (max iu iw) ≤ 4 * M := by
    refine max_le ?_ ?_
    · rw [show ie * (4 * is) = 4 * (ie * is) by ring]
      exact mul_le_mul_of_nonneg_left (le_max_left _ _) (by norm_num)
    · have := le_max_right (ie * is) (max iu iw)
      linarith
  refine ⟨one_le_log_of_three_le (by linarith), Real.log_le_log (by linarith) hM', ?_⟩
  calc log (max (ie * (4 * is)) (max iu iw)) ≤ log (4 * M) := Real.log_le_log (by linarith) hM4
    _ = log 4 + log M := Real.log_mul (by norm_num) (by linarith)
    _ ≤ 2 * log M := by linarith [Real.log_le_log (by norm_num : (0 : ℝ) < 4) hM]

/-- The logs the shapes take, at a quarter of the signal: `log M` at most doubles, the nested
log at most quadruples, and the products' logs at most double. -/
private lemma quad_logs {nJ iδ ip ie iu iw is : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip)
    (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) :
    0 ≤ log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
      ∧ 0 ≤ log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
      ∧ log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
        ≤ 4 * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
      ∧ 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is)
      ∧ 0 ≤ log (nJ * iδ * ip * ie * iu * iw * (4 * is))
      ∧ log (nJ * iδ * ip * ie * iu * iw * (4 * is))
        ≤ 2 * log (nJ * iδ * ip * ie * iu * iw * is) := by
  obtain ⟨h1, h2, h3⟩ := logM_quad (iu := iu) (iw := iw) hie his (by linarith) (by linarith)
  obtain ⟨-, -, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  set L := log (max (ie * is) (max iu iw))
  set L' := log (max (ie * (4 * is)) (max iu iw))
  have hZ : 4 ≤ L * iδ * ip * is := by
    have a : 2 ≤ L * iδ := by nlinarith
    have b : 2 ≤ L * iδ * ip := by nlinarith
    nlinarith
  have hZ' : L * iδ * ip * is ≤ L' * iδ * ip * (4 * is) := by
    have : L * iδ * ip ≤ L' * iδ * ip := by
      have := mul_le_mul_of_nonneg_right h2 (by positivity : (0 : ℝ) ≤ iδ * ip)
      linarith [show L * iδ * ip = L * (iδ * ip) by ring,
        show L' * iδ * ip = L' * (iδ * ip) by ring]
    have h4 : 0 ≤ L' * iδ * ip := by nlinarith
    nlinarith
  have hZ8 : L' * iδ * ip * (4 * is) ≤ 8 * (L * iδ * ip * is) := by
    have := mul_le_mul_of_nonneg_right h3 (by positivity : (0 : ℝ) ≤ iδ * ip * (4 * is))
    linarith [show L' * iδ * ip * (4 * is) = L' * (iδ * ip * (4 * is)) by ring,
      show 2 * L * (iδ * ip * (4 * is)) = 8 * (L * iδ * ip * is) by ring]
  have e : nJ * iδ * ip * ie * iu * iw * (4 * is) = 4 * (nJ * iδ * ip * ie * iu * iw * is) := by
    ring
  refine ⟨Real.log_nonneg (by linarith), Real.log_nonneg (by linarith),
    (Real.log_le_log (by linarith) hZ8).trans (log_eight_mul_le (by linarith)),
    Real.log_nonneg (by linarith), Real.log_nonneg (by rw [e]; linarith), ?_⟩
  rw [e]
  exact log_four_mul_le (by linarith)

private lemma kW_quad : (fun v => kW (quad v)) =O[𝓟 vecRegion]
    (fun (_, _, _, ie, _, iu, iw, is, _) =>
      log (max (ie * is) (max iu iw)) * is ^ 2 : ClusterVec → ℝ) := by
  refine (isBigO_comp_on kW_isBigO quad_maps).trans (isBigO_on 32 fun b hb => ?_)
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨-, -, -, hie, -, hiu, hiw, his, -⟩ := region hb
  obtain ⟨h1, h2, h3⟩ := logM_quad (iu := iu) (iw := iw) hie his (by linarith) (by linarith)
  dsimp only [Function.comp_apply, quad]
  refine ⟨mul_nonneg (by linarith) (sq_nonneg _), mul_nonneg (by linarith) (sq_nonneg _), ?_⟩
  nlinarith [mul_le_mul_of_nonneg_right h3 (sq_nonneg is)]

private lemma nsuffW_quad : (fun v => nsuffW (quad v)) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
      log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip : ClusterVec → ℝ) := by
  refine (isBigO_comp_on nsuffW_isBigO quad_maps).trans (isBigO_on 32 fun b hb => ?_)
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨-, hiδ, hip, hie, -, hiu, hiw, his, -⟩ := region hb
  obtain ⟨h1, h2, h3⟩ := logM_quad (iu := iu) (iw := iw) hie his (by linarith) (by linarith)
  have hl : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  dsimp only [Function.comp_apply, quad]
  have hp : 0 ≤ ip * is ^ 2 := by positivity
  refine ⟨by nlinarith, by nlinarith, ?_⟩
  nlinarith [mul_le_mul_of_nonneg_right h3 hp]

private lemma sufAtomTolW_quad : (fun v => 1 / sufAtomTolW (quad v)) =O[𝓟 vecRegion]
    (fun (_, iδ, ip, ie, _, iu, iw, is, _) =>
      ip * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip)
      : ClusterVec → ℝ) := by
  refine (isBigO_comp_on sufAtomTolW_isBigO quad_maps).trans (isBigO_on 32 fun b hb => ?_)
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨-, hiδ, hip, hie, -, hiu, hiw, his, -⟩ := region hb
  obtain ⟨h1, h2, h3⟩ := logM_quad (iu := iu) (iw := iw) hie his (by linarith) (by linarith)
  have hl : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  dsimp only [Function.comp_apply, quad]
  have hp : 0 ≤ ip * (ip * is ^ 2) := by positivity
  have hq : 0 ≤ ip * (log iδ * ip) := by positivity
  refine ⟨by nlinarith, by nlinarith, ?_⟩
  nlinarith [mul_le_mul_of_nonneg_right h3 hp]

/-- `nprefW`'s shape without its `isl` term, at a quarter of the signal. -/
private lemma core_quad {nJ iδ ip ie iu iw is nH : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip)
    (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) (hnH : 0 ≤ nH) :
    0 ≤ nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
          * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + nH * (ie * is ^ 2 + iu)
      ∧ 0 ≤ nJ * log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
          * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 * (4 * is) ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * (4 * is)) * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2)
        + nH * (ie * (4 * is) ^ 2 + iu)
      ∧ nJ * log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
          * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 * (4 * is) ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * (4 * is)) * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2)
        + nH * (ie * (4 * is) ^ 2 + iu)
        ≤ 1024 * (nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
          * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + nH * (ie * is ^ 2 + iu)) := by
  obtain ⟨a0, a0', ha, b0, b0', hb⟩ := quad_logs hJ hiδ hip hie hiu hiw his
  set L1 := log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
  set L1' := log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
  set L2 := log (nJ * iδ * ip * ie * iu * iw * is)
  set L2' := log (nJ * iδ * ip * ie * iu * iw * (4 * is))
  set A := ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4 with hA
  set Bq := ie ^ 2 * is ^ 4 + iu ^ 2 with hB
  set C := ie * is ^ 2 + iu with hC
  have eA : ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 * (4 * is) ^ 4 = 256 * A := by rw [hA]; ring
  have eB : ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 ≤ 256 * Bq := by
    rw [hB]; nlinarith [sq_nonneg iu]
  have eC : ie * (4 * is) ^ 2 + iu ≤ 16 * C := by rw [hC]; nlinarith [sq_nonneg is]
  have hA0 : 0 ≤ A := by positivity
  have hB0 : 0 ≤ Bq := by positivity
  have hC0 : 0 ≤ C := by positivity
  have hB'0 : 0 ≤ ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 := by positivity
  have hC'0 : 0 ≤ ie * (4 * is) ^ 2 + iu := by positivity
  rw [eA]
  have t1 : nJ * L1' * (256 * A) ≤ 1024 * (nJ * L1 * A) := by
    have := mul_le_mul_of_nonneg_left ha (by positivity : (0 : ℝ) ≤ nJ * (256 * A))
    nlinarith
  have t2 : L2' * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2) ≤ 512 * (L2 * Bq) := by
    have := mul_le_mul hb eB hB'0 (by positivity)
    nlinarith
  have t3 : nH * (ie * (4 * is) ^ 2 + iu) ≤ 16 * (nH * C) := by
    have := mul_le_mul_of_nonneg_left eC hnH
    nlinarith
  have p1 : 0 ≤ nJ * L1 * A := by positivity
  have p2 : 0 ≤ L2 * Bq := by positivity
  have p3 : 0 ≤ nH * C := by positivity
  refine ⟨by positivity, by positivity, ?_⟩
  nlinarith

private lemma nprefW_quad : (fun v => nprefW (quad v)) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
          * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu) : ClusterVec → ℝ) := by
  refine (isBigO_comp_on nprefW_isBigO quad_maps).trans (isBigO_on 1024 fun b hb => ?_)
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨c0, c0', hc⟩ := core_quad hJ hiδ hip hie hiu hiw his hnH
  have hl : 0 ≤ isl * log (nJ * iδ) := mul_nonneg (by linarith) (Real.log_nonneg (by nlinarith))
  dsimp only [Function.comp_apply, quad]
  refine ⟨by linarith, by linarith, by linarith⟩

set_option maxHeartbeats 1000000 in
private lemma atomTolW_quad : (fun v => 1 / atomTolW (quad v)) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      (nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
              * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
          + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
          + nH * (ie * is ^ 2 + iu)) * (nJ * (iu + isl) + ie * is ^ 2 + iu * is)
        + nJ * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * isl ^ 2
      : ClusterVec → ℝ) := by
  refine (isBigO_comp_on atomTolW_isBigO quad_maps).trans (isBigO_on (2 ^ 14) fun b hb => ?_)
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨c0, c0', hc⟩ := core_quad hJ hiδ hip hie hiu hiw his hnH
  obtain ⟨-, -, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  dsimp only [Function.comp_apply, quad]
  set Y := nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) with hY
  have hY16 : 16 ≤ Y := by
    have h1 : 1 ≤ isl * (nH + 1) := by nlinarith
    have : nJ * iδ * ip * ie * iu * iw * is ≤ Y := by
      rw [hY]
      have e : nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)
          = (nJ * iδ * ip * ie * iu * iw * is) * (isl * (nH + 1)) := by ring
      rw [e]
      exact le_mul_of_one_le_right (by linarith) h1
    linarith
  have eY : nJ * iδ * ip * ie * isl * iu * iw * (4 * is) * (nH + 1) = 4 * Y := by rw [hY]; ring
  have hlY := log_four_mul_le (by linarith : 4 ≤ Y)
  have hlY0 : 0 ≤ log Y := Real.log_nonneg (by linarith)
  set Bv := nJ * (iu + isl) + ie * is ^ 2 + iu * is with hBv
  have hBv0 : 0 ≤ Bv := by positivity
  have hBq : nJ * (iu + isl) + ie * (4 * is) ^ 2 + iu * (4 * is) ≤ 16 * Bv := by
    rw [hBv]; nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ nJ) (by linarith : (0 : ℝ) ≤ iu + isl)]
  have hBq0 : 0 ≤ nJ * (iu + isl) + ie * (4 * is) ^ 2 + iu * (4 * is) := by positivity
  rw [eY]
  set A := nJ * log (log (max (ie * is) (max iu iw)) * iδ * ip * is)
          * (ie ^ 2 * is ^ 4 + iu ^ 2 * is ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 4 + iu ^ 2)
        + nH * (ie * is ^ 2 + iu)
  set A' := nJ * log (log (max (ie * (4 * is)) (max iu iw)) * iδ * ip * (4 * is))
          * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2 * (4 * is) ^ 4)
        + log (nJ * iδ * ip * ie * iu * iw * (4 * is)) * (ie ^ 2 * (4 * is) ^ 4 + iu ^ 2)
        + nH * (ie * (4 * is) ^ 2 + iu)
  have h1 := mul_le_mul hc hBq hBq0 (by positivity)
  have h2 : nJ * log (4 * Y) * isl ^ 2 ≤ 2 * (nJ * log Y * isl ^ 2) := by
    have := mul_le_mul_of_nonneg_left hlY (by positivity : (0 : ℝ) ≤ nJ * isl ^ 2)
    nlinarith
  have h3 : 0 ≤ nJ * log Y * isl ^ 2 := by positivity
  have h4 : 0 ≤ A * Bv := mul_nonneg c0 hBv0
  have h5 : 0 ≤ nJ * log (4 * Y) * isl ^ 2 := by
    have := Real.log_nonneg (by linarith : (1 : ℝ) ≤ 4 * Y)
    positivity
  refine ⟨by linarith [mul_nonneg c0' hBq0], by linarith, by linarith⟩

private lemma half_eq (a : ℝ) : 1 / (2 * (1 / a)) = a / 2 := by
  rw [one_div, one_div, mul_inv, inv_inv, mul_comm, div_eq_mul_inv]

theorem trainFamilyClaim_holds : TrainFamilyClaim := by
  have hq : ∀ x, toVec (quarter x) = quad (toVec x) := toVec_quarter
  refine ⟨fun x => kW (toVec (quarter x)), fun x => nsuffW (toVec (quarter x)),
    fun x => nprefW (toVec (quarter x)), fun x => atomTolW (toVec (quarter x)),
    fun x => sufAtomTolW (toVec (quarter x)),
    (transfer kW_quad fun x => by
      simp only [toVec, TrainFamilyInputs.ℓ_conf]; ring_nf).congr_left
      fun x => by rw [hq],
    (transfer nsuffW_quad fun x => by
      simp only [toVec, TrainFamilyInputs.ℓ_conf]; ring_nf).congr_left
      fun x => by rw [hq],
    (transfer nprefW_quad fun x => by
      simp only [toVec, TrainFamilyInputs.ℓ_conf]; ring_nf).congr_left
      fun x => by rw [hq],
    (transfer atomTolW_quad fun x => by
      simp only [toVec, TrainFamilyInputs.ℓ_conf]; ring_nf).congr_left
      fun x => by dsimp only; rw [hq],
    (transfer sufAtomTolW_quad fun x => by
      simp only [toVec, TrainFamilyInputs.ℓ_conf]; ring_nf).congr_left
      fun x => by dsimp only; rw [hq],
    fun x => tol_pos _, ?_⟩
  intro Ω _ μ _ S _ J _ x O populations uni Pre Suf hx hcard hgap huni hflat
  obtain ⟨nJ, δ, pAP, εcov₀, ε_s₀, ε_u, κ, s₀, β⟩ := x
  simp only [trainFamilyRegion, Set.mem_ofPred_eq] at hx hcard hgap
  obtain ⟨hJ, hδ, hδ1, hpAP, hpAP1, he₀, he₁, hsl₀, hsl₁, hu, hu1, hw, hw1, hs, hs1, hb⟩ := hx
  obtain ⟨η₀, rfl⟩ : ∃ η₀, s₀ = 4 * (1 / 2 - η₀) := ⟨1 / 2 - s₀ / 4, by ring⟩
  subst hcard
  have hη₀ : η₀ < 1 / 2 := by linarith
  have hq4 : 4 * (1 / 2 - η₀) / 4 = 1 / 2 - η₀ := by ring
  simp only [TrainFamilyOutcome, quarter, toVec, TrainFamilyInputs.εc, TrainFamilyInputs.εsc, hq4,
    sub_sub_cancel]
  -- the schedule is sized for coverage `min(εcov, β/2)` and redraw share `min(ε_s, that)`
  set εcov := min εcov₀ (β / 2) with hεcov
  set ε_s := min ε_s₀ εcov with hεs
  have he : 0 < εcov := lt_min he₀ (by linarith)
  have he1 : εcov ≤ 1 / 2 := (min_le_left _ _).trans he₁
  have hsl : 0 < ε_s := lt_min hsl₀ he
  have hsl1 : ε_s ≤ εcov := min_le_right _ _
  have hεβ : 2 * εcov ≤ β := by have := min_le_right εcov₀ (β / 2); linarith
  have hεc0 : εcov ≤ εcov₀ := min_le_left _ _
  have hεs0 : ε_s ≤ ε_s₀ := min_le_left _ _
  set v : ClusterVec := ((populations.card : ℝ), 1 / δ, 1 / pAP, 1 / εcov, 1 / ε_s, 1 / ε_u,
    1 / κ, 1 / (1 / 2 - η₀), 0) with hv
  have hH0 : (allHarvest (fun _ : J => (∅ : Finset S))).card = 0 :=
    Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun s hs => by
      simp [allHarvest] at hs)
  obtain ⟨states, hnp, -, hkn, hband, hκ, hmain⟩ := clustering_guarantee_of_correct O
    populations populations uni (fun _ => ∅) Pre Suf η₀ ε_u εcov ε_s (δ / 2) (δ / 2) pAP (κ ^ 4) εcov₀ hgap
    hη₀ huni (Finset.Subset.refl _) hflat (fun _ p hp => absurd hp (Finset.notMem_empty p)) hpAP hu hu1
    (by positivity) (by linarith) he he1 hsl hsl1 hεc0 he₁ (by positivity) (by linarith)
    (by positivity)
    (pow_le_one₀ hw.le hw1)
  simp only [hH0, CharP.cast_eq_zero] at hnp
  have hs0 : 0 < 1 / 2 - η₀ := by linarith
  have hcard : (0 : ℝ) < populations.card := by
    exact_mod_cast Finset.card_pos.2 ⟨uni, huni⟩
  have hr : 0 < rateScale η₀ ε_u εcov := by
    have := rs_pos (ie := 1 / εcov) (iu := 1 / ε_u) (is := 1 / (1 / 2 - η₀)) (by positivity)
      (by positivity) (by positivity)
    simpa only [one_div_one_div, sub_sub_cancel] using this
  have hcond : 0 < (populations.card : ℝ) ∧ 0 < 1 / δ ∧ 0 < 1 / pAP ∧ 0 < 1 / εcov
      ∧ 0 < 1 / ε_s ∧ 0 < 1 / ε_u ∧ 0 < 1 / (1 / 2 - η₀) :=
    ⟨hcard, by positivity, by positivity, by positivity, by positivity, by positivity,
      by positivity⟩
  have ek : kW v = kB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) (κ ^ 4) := by
    simp only [v, kW, rsc, one_div_one_div, sub_sub_cancel]
  have es : nsuffW v
      = nsuffB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) (κ ^ 4) pAP (δ / 2) := by
    simp only [v, nsuffW, rsc, one_div_one_div, sub_sub_cancel, half_eq]
  have ep : nprefW v = nprefB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) (min εcov ε_u) (δ / 2) (δ / 2)
      pAP ε_s populations.card 0 (nsuffW v) := by
    simp only [v, nprefW, nsuffW, rsc, one_div_one_div, sub_sub_cancel, half_eq]
  have ea : atomTolW v = min (min ((1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) ε_u
        / (2 ^ 19 * max 1 (nprefW v)))
        (min ε_s ε_u / (2 ^ 14 * (populations.card + 3) * max 1 (nprefW v))))
      (min ε_s ε_u ^ 2
        / (512 * log (max 2 (64 * populations.card * max 1 (kW v)
          * (3 + 2 * logb 2 (max 1 (nprefW v))) / (δ / 2))))) := by
    simp only [v, atomTolW, if_pos hcond, one_div_one_div, half_eq]
  have ec : sufAtomTolW v = pAP / (2 ^ 16 * max 1 (nsuffW v)) := by
    simp only [v, sufAtomTolW, if_pos hcond, one_div_one_div, half_eq]
  have hks : ∀ B ∈ states, (B.k : ℝ) ≤ kW v ∧ (B.nsuff : ℝ) ≤ nsuffW v := fun B hB => by
    rw [ek, es]; exact hkn B hB
  have hnpW : ∀ B ∈ states, (B.npref : ℝ) ≤ nprefW v := fun B hB => by
    refine (hnp B hB).trans ?_
    rw [ep]
    unfold nprefB
    have hd : (0 : ℝ) < δ / 2 * (δ / 2) * pAP * rateScale η₀ ε_u εcov := by positivity
    have hX0 : 0 ≤ ((populations.card : ℝ) + 2) * B.nsuff
        / (δ / 2 * (δ / 2) * pAP * rateScale η₀ ε_u εcov) := by positivity
    have hX : ((populations.card : ℝ) + 2) * B.nsuff
          / (δ / 2 * (δ / 2) * pAP * rateScale η₀ ε_u εcov)
        ≤ ((populations.card : ℝ) + 2) * nsuffW v
          / (δ / 2 * (δ / 2) * pAP * rateScale η₀ ε_u εcov) := by
      gcongr; exact (hks B hB).2
    have hY0 : 0 ≤ 64 * ((B.nsuff : ℝ) + 1) / (δ / 2) := by positivity
    have hY : 64 * ((B.nsuff : ℝ) + 1) / (δ / 2) ≤ 64 * (nsuffW v + 1) / (δ / 2) := by
      gcongr; exact (hks B hB).2
    have hc : (0 : ℝ) ≤ 2 ^ 22 * (populations.card : ℝ) := by positivity
    have hd' : (0 : ℝ) ≤ (1 / 2 - η₀) ^ 4 * min εcov ε_u ^ 2 := by positivity
    exact add_le_add (add_le_add (add_le_add (div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_left (log_le_log_max hY0 hY) hc) hd')
      (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (log_le_log_max hX0 hX)
        (by norm_num)) (by positivity))) le_rfl) le_rfl
  refine ⟨states, fun B hB => ⟨hnpW B hB, hks B hB⟩, hband,
    fun B hB N hN p r => hκ B hB N hN κ p r hw.le le_rfl, ?_⟩
  intro D Dsf hD hDsf hpre hbalL hbalN hsuf hpAP' hatom hcoll
  rw [ea] at hatom
  rw [ec] at hcoll
  have hM : 0 < max 1 (nprefW v) := lt_max_of_lt_left one_pos
  have hlb0 : 0 ≤ logb 2 (max 1 (nprefW v)) := logb_nonneg one_lt_two (le_max_left _ _)
  refine (le_of_eq (by ring)).trans ((hmain D Dsf hD hDsf hpre (hεβ.trans hbalL)
    (hεβ.trans hbalN) hsuf hpAP'
    (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB t => ?_)).trans
    (measureReal_mono (fun r hr => ⟨hr.1, fun B hB =>
      ⟨le_trans (by linarith) (hr.2 B hB).1, fun j hj =>
        ⟨le_trans (by linarith) ((hr.2 B hB).2 j hj).1, ((hr.2 B hB).2 j hj).2⟩⟩⟩)
      (measure_ne_top _ _)))
  · have h : (D j).real {t} ≤ (1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) ε_u
        / (2 ^ 19 * max 1 (nprefW v)) :=
      (le_min_iff.1 (le_min_iff.1 (hatom j hj t)).1).1
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    calc 2 ^ 19 * (B.npref : ℝ) * (D j).real {t}
        ≤ 2 ^ 19 * max 1 (nprefW v) * (D j).real {t} := by
          gcongr; exact (hnpW B hB).trans (le_max_right _ _)
      _ ≤ _ := by linarith
  · have h : (D j).real {t}
        ≤ min ε_s ε_u / (2 ^ 14 * (populations.card + 3) * max 1 (nprefW v)) :=
      (le_min_iff.1 (le_min_iff.1 (hatom j hj t)).1).2
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    calc 2 ^ 14 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
        ≤ 2 ^ 14 * ((populations.card : ℝ) + 3) * max 1 (nprefW v) * (D j).real {t} := by
          gcongr; exact (hnpW B hB).trans (le_max_right _ _)
      _ ≤ _ := by linarith
  · have h : (D j).real {t} ≤ min ε_s ε_u ^ 2
        / (512 * log (max 2 (64 * populations.card * max 1 (kW v)
          * (3 + 2 * logb 2 (max 1 (nprefW v))) / (δ / 2)))) :=
      (le_min_iff.1 (hatom j hj t)).2
    have hX2 : (0 : ℝ) < log (max 2 (64 * populations.card * max 1 (kW v)
        * (3 + 2 * logb 2 (max 1 (nprefW v))) / (δ / 2))) :=
      Real.log_pos (lt_of_lt_of_le one_lt_two (le_max_left _ _))
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    have hlog : (Nat.log 2 B.npref : ℝ) ≤ logb 2 (max 1 (nprefW v)) := by
      rcases Nat.eq_zero_or_pos B.npref with h0 | h0
      · simpa [h0] using hlb0
      · have := Real.natLog_le_logb B.npref 2
        push_cast at this
        exact this.trans (logb_le_logb_of_le one_lt_two (by exact_mod_cast h0)
          ((hnpW B hB).trans (le_max_right _ _)))
    rcases Nat.eq_zero_or_pos B.k with hk0 | hk0
    · rw [hk0]
      simp only [Nat.cast_zero, mul_zero, zero_mul, zero_div, Real.log_zero]
      positivity
    have hBk0 : (0 : ℝ) < B.k := by exact_mod_cast hk0
    have hl : log (64 * (populations.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ))
          / (δ / 2))
        ≤ log (max 2 (64 * populations.card * max 1 (kW v)
          * (3 + 2 * logb 2 (max 1 (nprefW v))) / (δ / 2))) := by
      refine Real.log_le_log (by positivity) (le_trans ?_ (le_max_right _ _))
      gcongr
      exact (hks B hB).1.trans (le_max_right _ _)
    calc 512 * log (64 * (populations.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ))
          / (δ / 2)) * (D j).real {t}
        ≤ 512 * log (max 2 (64 * populations.card * max 1 (kW v)
          * (3 + 2 * logb 2 (max 1 (nprefW v))) / (δ / 2))) * (D j).real {t} := by gcongr
      _ ≤ _ := by linarith
  · have h := hcoll t
    have : 0 < max 1 (nsuffW v) := lt_max_of_lt_left one_pos
    rw [le_div_iff₀ (by positivity)] at h
    have hc0 : 0 ≤ Dsf.real {t} := measureReal_nonneg
    calc 2 ^ 16 * (B.nsuff : ℝ) * Dsf.real {t}
        ≤ 2 ^ 16 * max 1 (nsuffW v) * Dsf.real {t} := by
          gcongr; exact (hks B hB).2.trans (le_max_right _ _)
      _ ≤ _ := by linarith

end OrthoDFA
