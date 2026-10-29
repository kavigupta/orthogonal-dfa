import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Claims.Clustering
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.BigO

/-!
# The clustering's costs, asymptotically

The witnesses are `ClusteringGuarantee`'s bounds at its `δ = α = δ/2`, the prefix count's log
taken at the pool's ceiling; the tolerances are its per-state atom and collision conditions at the
ceilings, and `1` off the parameters' range. They are stated and bounded on `toVec x`, the counts
and the rates' reciprocals, and carried back to `ClusterInputs`. `ClusteringClaimSimple` follows
from `ClusteringClaim`, each tight shape bounded by its product.
-/

noncomputable section

namespace OrthoDFA

open Real Asymptotics Filter MeasureTheory

private def kB (s r c : ℝ) : ℝ :=
  25 * max (4 * log (20 / r)) (log (1 / c)) / (8 * s ^ 2) + 5 / s + 7

private def nsuffB (s r c pAP δ : ℝ) : ℝ :=
  2 * kB s r c / pAP + 16 * log (3 / δ) / pAP ^ 2

private def nprefB (s r δ α pAP εs nJ nH n : ℝ) : ℝ :=
  161000 * nJ ^ 2 * log (max 1 ((nJ + 2) * n / (δ * α * pAP * r))) / (s ^ 4 * r ^ 2)
    + 6 * log (256 * nJ / δ) / εs + 2 ^ 11 * nH / r

private lemma rs_eq {ie iu is : ℝ} (his : 0 < is) :
    rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie) = min (16 / (ie * (is + 2) ^ 2)) (1 / iu) := by
  unfold rateScale
  have : (1 - 2 * (1 / 2 - 1 / is)) / (1 - (1 / 2 - 1 / is)) = 4 / (is + 2) := by
    rw [show 1 - (1 / 2 - 1 / is) = (is + 2) / (2 * is) by field_simp; ring,
      show 1 - 2 * (1 / 2 - 1 / is) = 2 / is by field_simp; ring, div_div_div_eq,
      div_eq_div_iff (by positivity) (by positivity)]
    ring
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
  have h1 : 0 < 16 / (ie * (is + 2) ^ 2) := by positivity
  refine (one_div_min_le h1 (by positivity)).trans ?_
  rw [one_div_div, one_div_one_div]
  have : (is + 2) ^ 2 ≤ 16 * is ^ 2 := by nlinarith
  have := mul_le_mul_of_nonneg_left this hie.le
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
      ∧ kB (1 / is) r ((1 / iw) ^ 4) ≤ 70 * (log (max (ie * is) (max iu iw)) * is ^ 2) := by
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
  rw [show 25 * m / (8 * (1 / is) ^ 2) + 5 / (1 / is) + 7 = 25 * m * is ^ 2 / 8 + 5 * is + 7 by
    field_simp]
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
      ≤ 140 * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2) := by
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
  rw [show 2 * k / (1 / ip) + 16 * l / (1 / ip) ^ 2 = 2 * k * ip + 16 * l * ip ^ 2 by field_simp]
  have := mul_le_mul_of_nonneg_right hk (show 0 ≤ ip by linarith)
  have := mul_le_mul_of_nonneg_right hJl (sq_nonneg ip)
  have : 0 ≤ log iδ * ip ^ 2 := by positivity
  constructor
  · positivity
  · nlinarith

private lemma nprefB_bound {nJ iδ ip ie isl iu iw is nH r n : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hisl : 2 ≤ isl) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) (hn0 : 0 ≤ n)
    (hn : n ≤ 140 * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2)) :
    0 ≤ nprefB (1 / is) r (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl) nJ nH n
    ∧ nprefB (1 / is) r (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl) nJ nH n ≤ 2 ^ 23 *
      (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
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
  have hLM0 : 0 ≤ log (max (ie * is) (max iu iw)) := Real.log_nonneg (by linarith)
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
  have hn' : n ≤ 280 * P ^ 4 := by
    have : log (max (ie * is) (max iu iw)) * ip * is ^ 2 ≤ P * P * P ^ 2 := by gcongr
    have : log iδ * ip ^ 2 ≤ P * P ^ 2 := by
      gcongr; exact (Real.log_le_self (by linarith)).trans hδP
    have : P * P ^ 2 ≤ P * P * P ^ 2 := by
      gcongr; exact le_mul_of_one_le_left (by positivity) (by linarith)
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
        ≤ (3 * P) * (280 * P ^ 4) * (2 * P) * (2 * P) * P * (2 * P ^ 3) := by
          gcongr; linarith
      _ = 6720 * P ^ 11 := by ring
      _ ≤ P ^ 4 * P ^ 11 := by gcongr; linarith
      _ = P ^ 15 := by ring
  have hl := log_le_mul_log_of_le_pow 15 (by positivity) hX
  have hl0 : 0 ≤ log (max 1 ((nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r))) :=
    Real.log_nonneg (le_max_left _ _)
  have hs : log (256 * nJ / (1 / (2 * iδ))) ≤ 10 * log (nJ * iδ) := by
    refine (log_le_mul_log_of_le_pow (P := nJ * iδ) 10 (by positivity) ?_).trans (by norm_num)
    have : (2 : ℝ) ^ 9 ≤ (nJ * iδ) ^ 9 := pow_le_pow_left₀ (by norm_num) hx 9
    rw [div_one_div']
    nlinarith
  have hs0 : 0 ≤ log (256 * nJ / (1 / (2 * iδ))) :=
    Real.log_nonneg (by rw [div_one_div']; nlinarith)
  unfold nprefB
  generalize log (max 1 ((nJ + 2) * n / (1 / (2 * iδ) * (1 / (2 * iδ)) * (1 / ip) * r))) = l at *
  generalize log (256 * nJ / (1 / (2 * iδ))) = ls at *
  rw [show 161000 * nJ ^ 2 * l / ((1 / is) ^ 4 * r ^ 2) + 6 * ls / (1 / isl) + 2 ^ 11 * nH / r
    = 161000 * (nJ ^ 2 * l * is ^ 4 * (1 / r) ^ 2) + 6 * (ls * isl) + 2 ^ 11 * (nH * (1 / r)) by
    field_simp]
  have h3 := mul_le_mul_of_nonneg_right hs (by linarith : (0 : ℝ) ≤ isl)
  have : 0 ≤ isl * log (nJ * iδ) := by positivity
  have hR0 : 0 ≤ 1 / r := by positivity
  have hR2 : (1 / r) ^ 2 ≤ 2 * (ie ^ 2 * is ^ 4 + iu ^ 2) := by
    have := pow_le_pow_left₀ hR0 hr2 2
    have := add_sq_le (a := ie * is ^ 2) (b := iu)
    linarith
  have h1 : nJ ^ 2 * l * is ^ 4 * (1 / r) ^ 2
      ≤ nJ ^ 2 * (15 * log P) * is ^ 4 * (2 * (ie ^ 2 * is ^ 4 + iu ^ 2)) := by
    gcongr; exact_mod_cast hl
  have h2 : nH * (1 / r) ≤ nH * (ie * is ^ 2 + iu) := by gcongr
  have : 0 ≤ nJ ^ 2 * log P * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) :=
    mul_nonneg (mul_nonneg (by positivity) (by linarith)) (by positivity)
  have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
  constructor
  · positivity
  · linarith

private lemma gn_le_pow {nJ iδ ip ie isl iu iw is nH : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hisl : 2 ≤ isl) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) :
    2 ^ 23 * (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu))
      ≤ (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) ^ 20 := by
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
  set P := nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)
  have hJP : nJ ≤ P := by simpa using hsub [nJ] (by simp)
  have hxP : nJ * iδ ≤ P := by simpa [mul_assoc] using hsub [nJ, iδ] (by simp)
  have heP : ie ≤ P := by simpa using hsub [ie] (by simp)
  have hslP : isl ≤ P := by simpa using hsub [isl] (by simp)
  have huP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hsP : is ≤ P := by simpa using hsub [is] (by simp)
  have hnP : nH + 1 ≤ P := by simpa using hsub [nH + 1] (by simp)
  have hP : 16 ≤ P := by linarith
  have hP1 : 1 ≤ P := by linarith
  have h1 : nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
      ≤ P ^ 2 * P * (P ^ 2 * P ^ 8 + P ^ 2 * P ^ 4) := by gcongr; linarith
  have h1' : isl * log (nJ * iδ) ≤ P * P := by gcongr; linarith
  have h2 : nH * (ie * is ^ 2 + iu) ≤ P * (P * P ^ 2 + P) := by gcongr; linarith
  have h3 : P ^ 2 * P * (P ^ 2 * P ^ 8 + P ^ 2 * P ^ 4) + P * P + P * (P * P ^ 2 + P)
      ≤ 5 * P ^ 13 := by
    have : P ^ 2 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 4 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 9 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    ring_nf; linarith
  have h4 : 2 ^ 26 * P ^ 13 ≤ P ^ 20 := by
    have : (16 : ℝ) ^ 7 ≤ P ^ 7 := pow_le_pow_left₀ (by norm_num) hP 7
    calc 2 ^ 26 * P ^ 13 ≤ P ^ 7 * P ^ 13 := by
          gcongr; exact le_trans (by norm_num) this
      _ = P ^ 20 := by ring
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
  nprefB (1 / is) (rsc ie iu is) (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl) nJ nH
    (nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ)))

open scoped Classical in
private def atomTolW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is then
    min (min (rsc ie iu is * (1 / is)
          / (2 ^ 18 * (nJ + 3) * max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))))
        (1 / isl / (2 ^ 8 * (nJ + 3) * max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH)))))
      (1 / (2 * iδ) * min (1 / isl) (1 / iu) ^ 2
        / (2 ^ 15 * nJ
          * (3 + 2 * logb 2 (max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))))))
  else 1

open scoped Classical in
private def collTolW : ClusterVec → ℝ := fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is then
    1 / (2 * iδ) * (1 / ip)
      / (2 ^ 12 * (nJ + 1) * max 1 (nsuffW (nJ, iδ, ip, ie, isl, iu, iw, is, nH)))
  else 1

private lemma tol_pos (v : ClusterVec) : 0 < atomTolW v ∧ 0 < collTolW v := by
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := v
  simp only [atomTolW, collTolW]
  split_ifs with h
  · obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, his⟩ := h
    have := rs_pos hie hiu his
    have : 0 ≤ logb 2 (max 1 (nprefW (nJ, iδ, ip, ie, isl, iu, iw, is, nH))) :=
      logb_nonneg one_lt_two (le_max_left _ _)
    unfold rsc at *
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
  refine isBigO_on 70 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, -, hiu, hiw, his, -⟩ := region hb
  have := logM_nonneg (iu := iu) (iw := iw) hie his
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := kB_bound hie hiu hiw his hr hr2
  dsimp only [kW]
  exact ⟨h0, by positivity, h⟩

private lemma nsuffW_isBigO : nsuffW =O[𝓟 vecRegion] (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
    log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2 : ClusterVec → ℝ) := by
  refine isBigO_on 140 fun b hb => ?_
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
      nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu) : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 23) fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, hx, hP8⟩ := prods hJ hiδ hip hie hiu hiw his
  have : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hiδ hip hie hiu hiw his hr hr2
  obtain ⟨h0, h⟩ := nprefB_bound hJ hiδ hip hie hisl hiu hiw his hnH hr hr2 hs0 hs
  dsimp only [nprefW]
  exact ⟨h0, by positivity, h⟩

private lemma atomTolW_isBigO : (fun v => 1 / atomTolW v) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      nJ * (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
              * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
          + nH * (ie * is ^ 2 + iu)) * (ie * is ^ 3 + iu * is + isl)
        + nJ * iδ * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * (isl ^ 2 + iu ^ 2)
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 46) fun b hb => ?_
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
  rw [show ∀ a c b : ℝ, a + c + b = a + b + c by intros; ring] at hN hG
  have hcond : 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < isl ∧ 0 < iu ∧ 0 < is :=
    ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩
  dsimp only
  simp only [atomTolW, nprefW, if_pos hcond]
  set N := nprefB (1 / is) (rsc ie iu is) (1 / (2 * iδ)) (1 / (2 * iδ)) (1 / ip) (1 / isl) nJ nH
    (nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ)))
  set L9 := log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) with hL9e
  set G := nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
    * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) + nH * (ie * is ^ 2 + iu)
  set r := rsc ie iu is
  have hG0 : 0 ≤ G := by positivity
  have hGΛ : log (nJ * iδ) * (ie * is ^ 3 + iu * is) ≤ G := by
    have h1 : ie * is ^ 3 ≤ ie ^ 2 * is ^ 8 :=
      le_of_one_le_mul (by positivity) (one_le_mul' he (one_le_pow₀ (n := 5) hs1)) (by ring)
    have h2 : iu * is ≤ iu ^ 2 * is ^ 4 :=
      le_of_one_le_mul (by positivity) (one_le_mul' hu (one_le_pow₀ (n := 3) hs1)) (by ring)
    have h3 : log (nJ * iδ) * (ie * is ^ 3 + iu * is)
        ≤ nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) :=
      calc log (nJ * iδ) * (ie * is ^ 3 + iu * is)
          ≤ log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) := by
            gcongr
        _ ≤ _ := le_of_one_le_mul (by positivity) (one_le_pow₀ (n := 2) hJ) (by ring)
    have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
    linarith
  have hG1 : 1 ≤ 2 ^ 23 * (G + isl * log (nJ * iδ)) := by
    have h1 : 1 ≤ ie ^ 2 * is ^ 8 :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (one_le_pow₀ (by linarith))
    have h2 : 1 ≤ ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4 := by
      have : 0 ≤ iu ^ 2 * is ^ 4 := by positivity
      linarith
    have : 1 ≤ nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hJ) hL7) h2
    have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
    have : 0 ≤ isl * log (nJ * iδ) := by positivity
    linarith
  have hM : max 1 N ≤ 2 ^ 23 * (G + isl * log (nJ * iδ)) := max_le hG1 hN
  have hM1 : 1 ≤ max 1 N := le_max_left _ _
  have hlb0 : 0 ≤ logb 2 (max 1 N) := logb_nonneg one_lt_two hM1
  have hA0 : 0 < r * (1 / is) / (2 ^ 18 * (nJ + 3) * max 1 N) := by positivity
  have hE0 : 0 < 1 / isl / (2 ^ 8 * (nJ + 3) * max 1 N) := by positivity
  have hB0 : 0 < 1 / (2 * iδ) * min (1 / isl) (1 / iu) ^ 2
      / (2 ^ 15 * nJ * (3 + 2 * logb 2 (max 1 N))) := by positivity
  refine ⟨(one_div_pos.2 (lt_min (lt_min hA0 hE0) hB0)).le, by positivity, ?_⟩
  refine (one_div_min_le (lt_min hA0 hE0) hB0).trans ?_
  refine (add_le_add (one_div_min_le hA0 hE0) le_rfl).trans ?_
  rw [one_div_div, one_div_div, one_div_div,
    show 2 ^ 18 * (nJ + 3) * max 1 N / (r * (1 / is))
      = 2 ^ 18 * ((nJ + 3) * max 1 N * (1 / r) * is) by field_simp,
    show 2 ^ 8 * (nJ + 3) * max 1 N / (1 / isl) = 2 ^ 8 * ((nJ + 3) * max 1 N * isl) by
      field_simp,
    show 2 ^ 15 * nJ * (3 + 2 * logb 2 (max 1 N)) / (1 / (2 * iδ) * min (1 / isl) (1 / iu) ^ 2)
      = 2 ^ 16 * (nJ * (3 + 2 * logb 2 (max 1 N)) * iδ * (1 / min (1 / isl) (1 / iu)) ^ 2) by
      field_simp]
  have h1 : (nJ + 3) * max 1 N * (1 / r) * is
      ≤ (4 * nJ) * (2 ^ 23 * (G + isl * log (nJ * iδ))) * (ie * is ^ 2 + iu) * is := by
    gcongr; linarith
  have h1e : (nJ + 3) * max 1 N * isl ≤ (4 * nJ) * (2 ^ 23 * (G + isl * log (nJ * iδ))) * isl := by
    gcongr; linarith
  have hm : 1 / min (1 / isl) (1 / iu) ≤ isl + iu := by
    have := one_div_min_le (x := 1 / isl) (y := 1 / iu) (by positivity) (by positivity)
    rwa [one_div_one_div, one_div_one_div] at this
  have hm2 : (1 / min (1 / isl) (1 / iu)) ^ 2 ≤ 2 * (isl ^ 2 + iu ^ 2) :=
    (pow_le_pow_left₀ (by positivity) hm 2).trans add_sq_le
  have hlb : 3 + 2 * logb 2 (max 1 N) ≤ 83 * L9 := by
    have hlog := log_le_mul_log_of_le_pow 20 (by positivity) (hM.trans hG)
    rw [← hL9e] at hlog
    have hl0 : 0 ≤ log (max 1 N) := Real.log_nonneg hM1
    have h2 : 1 / 2 < log 2 := by linarith [log_two_gt_d9]
    have : log (max 1 N) / log 2 ≤ 2 * log (max 1 N) := by
      have := mul_le_mul_of_nonneg_left h2.le hl0
      rw [div_le_iff₀ (by linarith)]; linarith
    rw [Real.logb]
    push_cast at hlog
    linarith only [hlog, this, hL9]
  have h2 : nJ * (3 + 2 * logb 2 (max 1 N)) * iδ * (1 / min (1 / isl) (1 / iu)) ^ 2
      ≤ nJ * (83 * L9) * iδ * (2 * (isl ^ 2 + iu ^ 2)) := by gcongr
  have c1 : nJ * isl * (log (nJ * iδ) * (ie * is ^ 3 + iu * is)) ≤ nJ * isl * G :=
    mul_le_mul_of_nonneg_left hGΛ (by positivity)
  have c2 : nJ * isl ^ 2 * log (nJ * iδ) ≤ nJ * isl ^ 2 * (iδ * L9) :=
    mul_le_mul_of_nonneg_left (hΛ9.trans (le_of_one_le_mul (by linarith) (by linarith : 1 ≤ iδ)
      (by ring))) (by positivity)
  have : 0 ≤ nJ * G * (ie * is ^ 3 + iu * is) := by positivity
  have : 0 ≤ nJ * G * isl := by positivity
  have : 0 ≤ nJ * iδ * L9 * isl ^ 2 := by positivity
  have : 0 ≤ nJ * iδ * L9 * iu ^ 2 := by positivity
  linarith

private lemma collTolW_isBigO : (fun v => 1 / collTolW v) =O[𝓟 vecRegion]
    (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
      nJ * iδ * ip * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2)
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 22) fun b hb => ?_
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
  simp only [collTolW, nsuffW, if_pos hcond]
  set N := nsuffB (1 / is) (rsc ie iu is) ((1 / iw) ^ 4) (1 / ip) (1 / (2 * iδ))
  set G := log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2
  have hG1 : 1 ≤ 140 * G := by
    have : 1 ≤ log (max (ie * is) (max iu iw)) * ip * is ^ 2 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hL4 hip)
        (one_le_pow₀ (by linarith))
    have : 0 ≤ log iδ * ip ^ 2 := by positivity
    linarith
  have hM : max 1 N ≤ 140 * G := max_le hG1 hs
  have hA0 : 0 < 1 / (2 * iδ) * (1 / ip) / (2 ^ 12 * (nJ + 1) * max 1 N) := by positivity
  refine ⟨(one_div_pos.2 hA0).le, by positivity, ?_⟩
  rw [one_div_div, show 2 ^ 12 * (nJ + 1) * max 1 N / (1 / (2 * iδ) * (1 / ip))
      = 2 ^ 13 * ((nJ + 1) * max 1 N * iδ * ip) by field_simp]
  have : (nJ + 1) * max 1 N * iδ * ip ≤ (2 * nJ) * (140 * G) * iδ * ip := by gcongr; linarith
  have : 0 ≤ nJ * iδ * ip * G := by positivity
  linarith

private lemma nsuff_le {iδ ip ie iu iw is : ℝ} (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip)
    (hie : 2 ≤ ie) (his : 2 ≤ is) :
    log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2
      ≤ (log (max (ie * is) (max iu iw)) + log iδ) * ip ^ 2 * is ^ 2 := by
  have hL := logM_nonneg (iu := iu) (iw := iw) hie his
  have hΛ : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  have hip2 : ip ≤ ip ^ 2 := by nlinarith
  have his2 : 1 ≤ is ^ 2 := one_le_pow₀ (by linarith)
  have e1 : log (max (ie * is) (max iu iw)) * ip * is ^ 2
      ≤ log (max (ie * is) (max iu iw)) * ip ^ 2 * is ^ 2 := by gcongr
  have e2 : log iδ * ip ^ 2 ≤ log iδ * ip ^ 2 * is ^ 2 :=
    le_mul_of_one_le_right (by positivity) his2
  linarith

private lemma npref_le {nJ iδ ip ie iu iw is nH : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hip : 1 ≤ ip)
    (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) (hnH : 0 ≤ nH) :
    nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + nH * (ie * is ^ 2 + iu)
      ≤ 4 * (nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * iu * iw * is)
        * ie ^ 2 * iu ^ 2 * is ^ 8) := by
  obtain ⟨-, -, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  have hL : 1 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := one_le_log_of_three_le (by linarith)
  have hK : 1 ≤ nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) :=
    one_le_mul' (one_le_pow₀ hJ) hL
  have he : 1 ≤ ie := by linarith
  have hu : 1 ≤ iu := by linarith
  have hs : 1 ≤ is := by linarith
  have m1 : ie ^ 2 * is ^ 8 ≤ ie ^ 2 * iu ^ 2 * is ^ 8 :=
    le_of_one_le_mul (by positivity) (one_le_pow₀ (n := 2) hu) (by ring)
  have m2 : iu ^ 2 * is ^ 4 ≤ ie ^ 2 * iu ^ 2 * is ^ 8 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_pow₀ (n := 2) he) (one_le_pow₀ (n := 4) hs)) (by ring)
  have m3 : ie * is ^ 2 ≤ ie ^ 2 * iu ^ 2 * is ^ 8 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_mul' he (one_le_pow₀ (n := 2) hu)) (one_le_pow₀ (n := 6) hs)) (by ring)
  have m4 : iu ≤ ie ^ 2 * iu ^ 2 * is ^ 8 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_mul' (one_le_pow₀ (n := 2) he) hu) (one_le_pow₀ (n := 8) hs)) (by ring)
  have e : 4 * (nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * iu * iw * is) * ie ^ 2 * iu ^ 2
      * is ^ 8) = 4 * (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * iu ^ 2 * is ^ 8))
        + 4 * (nH * (ie ^ 2 * iu ^ 2 * is ^ 8)
          * (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is))) := by ring
  rw [e]
  set K := nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
  set M := ie ^ 2 * iu ^ 2 * is ^ 8
  have e1 : K * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) ≤ K * (2 * M) :=
    mul_le_mul_of_nonneg_left (by linarith) (by linarith)
  have e2 : nH * (ie * is ^ 2 + iu) ≤ nH * (2 * M) := mul_le_mul_of_nonneg_left (by linarith) hnH
  have e3 : nH * M ≤ nH * M * K := le_mul_of_one_le_right (by positivity) hK
  have : 0 ≤ K * M := by positivity
  have : 0 ≤ nH * M * K := by positivity
  linarith

private lemma nsuff_dom : (fun (_, iδ, ip, ie, _, iu, iw, is, _) =>
    log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2 : ClusterVec → ℝ)
    =O[𝓟 vecRegion] (fun (_, iδ, ip, ie, _, iu, iw, is, _) =>
      (log (max (ie * is) (max iu iw)) + log iδ) * ip ^ 2 * is ^ 2 : ClusterVec → ℝ) := by
  refine isBigO_on 1 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨-, hiδ, hip, hie, -, -, -, his, -⟩ := region hb
  have := logM_nonneg (iu := iu) (iw := iw) hie his
  have : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  dsimp only
  exact ⟨by positivity, by positivity, by rw [one_mul]; exact nsuff_le hiδ hip hie his⟩

private lemma npref_dom : (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
    nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
      + isl * log (nJ * iδ) + nH * (ie * is ^ 2 + iu) : ClusterVec → ℝ)
    =O[𝓟 vecRegion] (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is) * ie ^ 2 * isl * iu ^ 2
        * is ^ 8 : ClusterVec → ℝ) := by
  refine isBigO_on 5 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, hx, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  have he : 1 ≤ ie := by linarith
  have hu : 1 ≤ iu := by linarith
  have hs : 1 ≤ is := by linarith
  have hl : 1 ≤ isl := by linarith
  have hP78 : nJ * iδ * ip * ie * iu * iw * is ≤ nJ * iδ * ip * ie * isl * iu * iw * is :=
    le_of_one_le_mul (by positivity) hl (by ring)
  have hx7 : nJ * iδ ≤ nJ * iδ * ip * ie * iu * iw * is :=
    le_of_one_le_mul (by positivity) (one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' hip he)
      hu) hiw) hs) (by ring)
  have hL7 : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have hL78 : log (nJ * iδ * ip * ie * iu * iw * is)
      ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is) := Real.log_le_log (by positivity) hP78
  have hL8 : 0 ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is) := by linarith
  have hΛ0 : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have hΛ : log (nJ * iδ) ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is) :=
    (Real.log_le_log (by positivity) hx7).trans hL78
  have hK : 1 ≤ nJ ^ 2 * (nH + 1) * ie ^ 2 * iu ^ 2 * is ^ 8 :=
    one_le_mul' (one_le_mul' (one_le_mul' (one_le_mul' (one_le_pow₀ hJ) (by linarith))
      (one_le_pow₀ he)) (one_le_pow₀ hu)) (one_le_pow₀ hs)
  have h1 := npref_le hJ hiδ hip hie hiu hiw his hnH
  have h2 : nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * iu * iw * is) * ie ^ 2 * iu ^ 2 * is ^ 8
      ≤ nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is) * ie ^ 2 * isl
        * iu ^ 2 * is ^ 8 :=
    calc _ ≤ nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is) * ie ^ 2 * iu ^ 2
          * is ^ 8 := by gcongr
      _ ≤ _ := le_of_one_le_mul (by positivity) hl (by ring)
  have h3 : isl * log (nJ * iδ) ≤ nJ ^ 2 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is)
      * ie ^ 2 * isl * iu ^ 2 * is ^ 8 :=
    calc isl * log (nJ * iδ) ≤ isl * log (nJ * iδ * ip * ie * isl * iu * iw * is) := by gcongr
      _ ≤ _ := le_of_one_le_mul (by positivity) hK (by ring)
  dsimp only
  exact ⟨by positivity, by positivity, by linarith⟩

private lemma atom_dom : (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
    nJ * (nJ ^ 2 * log (nJ * iδ * ip * ie * iu * iw * is)
            * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + nH * (ie * is ^ 2 + iu)) * (ie * is ^ 3 + iu * is + isl)
      + nJ * iδ * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * (isl ^ 2 + iu ^ 2)
    : ClusterVec → ℝ)
    =O[𝓟 vecRegion] (fun (nJ, iδ, ip, ie, isl, iu, iw, is, nH) =>
      nJ ^ 3 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1))
        * iδ * ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11 : ClusterVec → ℝ) := by
  refine isBigO_on 16 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, hisl, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, -, hP⟩ := prods hJ hiδ hip hie hiu hiw his
  have he : 1 ≤ ie := by linarith
  have hu : 1 ≤ iu := by linarith
  have hs : 1 ≤ is := by linarith
  have hl : 1 ≤ isl := by linarith
  have hP79 : nJ * iδ * ip * ie * iu * iw * is
      ≤ nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1) :=
    le_of_one_le_mul (by positivity) (one_le_mul' hl (by linarith : 1 ≤ nH + 1)) (by ring)
  have hL7 : 0 ≤ log (nJ * iδ * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have hL9 : 0 ≤ log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) :=
    Real.log_nonneg (by linarith)
  have hF := (npref_le hJ hiδ hip hie hiu hiw his hnH).trans (by gcongr : 4 * (nJ ^ 2 * (nH + 1)
    * log (nJ * iδ * ip * ie * iu * iw * is) * ie ^ 2 * iu ^ 2 * is ^ 8) ≤ 4 * (nJ ^ 2 * (nH + 1)
    * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * ie ^ 2 * iu ^ 2 * is ^ 8))
  have t1 : ie * is ^ 3 ≤ ie * isl * iu * is ^ 3 :=
    le_of_one_le_mul (by positivity) (one_le_mul' hl hu) (by ring)
  have t2 : iu * is ≤ ie * isl * iu * is ^ 3 :=
    le_of_one_le_mul (by positivity) (one_le_mul' (one_le_mul' he hl) (one_le_pow₀ (n := 2) hs))
      (by ring)
  have t3 : isl ≤ ie * isl * iu * is ^ 3 :=
    le_of_one_le_mul (by positivity) (one_le_mul' (one_le_mul' he hu) (one_le_pow₀ (n := 3) hs))
      (by ring)
  have hA := mul_le_mul (mul_le_mul_of_nonneg_left hF (by positivity : (0 : ℝ) ≤ nJ))
    (show ie * is ^ 3 + iu * is + isl ≤ 3 * (ie * isl * iu * is ^ 3) by linarith) (by positivity)
    (by positivity)
  have hZ := le_mul_of_one_le_right (show 0 ≤ nJ ^ 3 * (nH + 1)
    * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) * ie ^ 3 * isl * iu ^ 3 * is ^ 11
    by positivity) (one_le_mul' (by linarith : 1 ≤ iδ) hl)
  have t4 : isl ^ 2 ≤ ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_mul' (one_le_pow₀ (n := 3) he) (one_le_pow₀ (n := 3) hu))
      (one_le_pow₀ (n := 11) hs)) (by ring)
  have t5 : iu ^ 2 ≤ ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11 := le_of_one_le_mul (by positivity)
    (one_le_mul' (one_le_mul' (one_le_mul' (one_le_pow₀ (n := 3) he) (one_le_pow₀ (n := 2) hl))
      hu) (one_le_pow₀ (n := 11) hs)) (by ring)
  have hB := mul_le_mul_of_nonneg_left
    (show isl ^ 2 + iu ^ 2 ≤ 2 * (ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11) by linarith)
    (show 0 ≤ nJ * iδ * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1)) by positivity)
  have hB2 := le_mul_of_one_le_right (show 0 ≤ nJ * iδ
    * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1))
    * (ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11) by positivity)
    (one_le_mul' (one_le_pow₀ (n := 2) hJ) (by linarith : 1 ≤ nH + 1))
  have : 0 ≤ nJ ^ 3 * (nH + 1) * log (nJ * iδ * ip * ie * isl * iu * iw * is * (nH + 1))
      * iδ * ie ^ 3 * isl ^ 2 * iu ^ 3 * is ^ 11 := by positivity
  dsimp only
  exact ⟨by positivity, by positivity, by linarith⟩

private lemma coll_dom : (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
    nJ * iδ * ip * (log (max (ie * is) (max iu iw)) * ip * is ^ 2 + log iδ * ip ^ 2)
    : ClusterVec → ℝ)
    =O[𝓟 vecRegion] (fun (nJ, iδ, ip, ie, _, iu, iw, is, _) =>
      nJ * (log (max (ie * is) (max iu iw)) + log iδ) * iδ * ip ^ 3 * is ^ 2
      : ClusterVec → ℝ) := by
  refine isBigO_on 1 fun b hb => ?_
  obtain ⟨nJ, iδ, ip, ie, isl, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hip, hie, -, -, -, his, -⟩ := region hb
  have := logM_nonneg (iu := iu) (iw := iw) hie his
  have : 0 ≤ log iδ := Real.log_nonneg (by linarith)
  have := mul_le_mul_of_nonneg_left (nsuff_le (iu := iu) (iw := iw) hiδ hip hie his)
    (show 0 ≤ nJ * iδ * ip by positivity)
  dsimp only
  exact ⟨by positivity, by positivity, by linarith⟩

/-- No strings harvested. -/
private def toVec (x : ClusterInputs) : ClusterVec :=
  (x.nJ, 1 / x.δ, 1 / x.pAP, 1 / x.εcov, 1 / x.ε_s, 1 / x.ε_u, 1 / x.κ, 1 / x.s₀, 0)

private lemma toVec_mem {x : ClusterInputs} (hx : x ∈ clusterRegion) : toVec x ∈ vecRegion := by
  obtain ⟨hJ, hδ, hδ1, hp, hp1, he, he1, hsl, hsl1, hu, hu1, hw, hw1, hs, hs1⟩ := hx
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hJ, (le_div_iff₀ hδ).2 (by linarith), (le_div_iff₀ hp).2 (by linarith),
    (le_div_iff₀ he).2 (by linarith), (le_div_iff₀ hsl).2 (by linarith),
    (le_div_iff₀ hu).2 (by linarith), (le_div_iff₀ hw).2 (by linarith),
    (le_div_iff₀ hs).2 (by linarith), le_rfl⟩

private lemma transfer {f g : ClusterVec → ℝ} {g' : ClusterInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 clusterRegion] g' :=
  (isBigO_comp_on h fun _ hx => toVec_mem hx).congr_right hg

private lemma dominate {f g : ClusterVec → ℝ} {f' g' : ClusterInputs → ℝ}
    (h : f =O[𝓟 vecRegion] g) (hf : ∀ x, f (toVec x) = f' x) (hg : ∀ x, g (toVec x) = g' x) :
    f' =O[𝓟 clusterRegion] g' :=
  (transfer h hg).congr_left hf

private lemma half_eq (a : ℝ) : 1 / (2 * (1 / a)) = a / 2 := by
  rw [one_div, one_div, mul_inv, inv_inv, mul_comm, div_eq_mul_inv]

theorem clusteringClaim_holds : ClusteringClaim := by
  refine ⟨fun x => kW (toVec x), fun x => nsuffW (toVec x), fun x => nprefW (toVec x),
    fun x => atomTolW (toVec x), fun x => collTolW (toVec x),
    transfer kW_isBigO fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf,
    transfer nsuffW_isBigO fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf,
    transfer nprefW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer atomTolW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer collTolW_isBigO fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf,
    fun x => tol_pos _, ?_⟩
  intro Ω _ μ _ S _ J _ x O populations tame uni Pre Suf hx hcard hηle huni htame hflat
  obtain ⟨nJ, δ, pAP, εcov, ε_s, ε_u, κ, s₀⟩ := x
  simp only [clusterRegion, Set.mem_ofPred_eq] at hx hcard hηle
  obtain ⟨hJ, hδ, hδ1, hpAP, hpAP1, he, he1, hsl, hsl1, hu, hu1, hw, hw1, hs, hs1⟩ := hx
  obtain ⟨η₀, rfl⟩ : ∃ η₀, s₀ = 1 / 2 - η₀ := ⟨1 / 2 - s₀, by ring⟩
  subst hcard
  rw [sub_sub_cancel] at hηle
  have hη₀ : η₀ < 1 / 2 := by linarith
  simp only [ClusteringOutcome, toVec, sub_sub_cancel]
  set v : ClusterVec := ((populations.card : ℝ), 1 / δ, 1 / pAP, 1 / εcov, 1 / ε_s, 1 / ε_u,
    1 / κ, 1 / (1 / 2 - η₀), 0) with hv
  have hH0 : (allHarvest (fun _ : J => (∅ : Finset S))).card = 0 :=
    Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun s hs => by
      simp [allHarvest] at hs)
  obtain ⟨states, hnp, -, hkn, hband, hκ, hmain⟩ := clustering_guarantee_of_correct O
    populations tame uni (fun _ => ∅) Pre Suf η₀ ε_u εcov ε_s (δ / 2) (δ / 2) pAP (κ ^ 4) hηle
    hη₀ huni htame hflat (fun _ p hp => absurd hp (Finset.notMem_empty p)) hpAP hu hu1
    (by positivity) (by linarith) he he1 hsl (by positivity) (by linarith) (by positivity)
    (pow_le_one₀ hw.le hw1)
  simp only [hH0, CharP.cast_eq_zero] at hnp
  have hs0 : 0 < 1 / 2 - η₀ := by linarith
  have hcard : (0 : ℝ) < populations.card := by
    exact_mod_cast Finset.card_pos.2 ⟨uni, htame huni⟩
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
  have ep : nprefW v = nprefB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) (δ / 2) (δ / 2) pAP ε_s
      populations.card 0 (nsuffW v) := by
    simp only [v, nprefW, nsuffW, rsc, one_div_one_div, sub_sub_cancel, half_eq]
  have ea : atomTolW v = min (min (rateScale η₀ ε_u εcov * (1 / 2 - η₀)
        / (2 ^ 18 * (populations.card + 3) * max 1 (nprefW v)))
        (ε_s / (2 ^ 8 * (populations.card + 3) * max 1 (nprefW v))))
      (δ / 2 * min ε_s ε_u ^ 2
        / (2 ^ 15 * populations.card * (3 + 2 * logb 2 (max 1 (nprefW v))))) := by
    simp only [v, atomTolW, if_pos hcond, rsc, one_div_one_div, sub_sub_cancel, half_eq]
  have ec : collTolW v = δ / 2 * pAP / (2 ^ 12 * (populations.card + 1) * max 1 (nsuffW v)) := by
    simp only [v, collTolW, if_pos hcond, one_div_one_div, half_eq]
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
    have hc : (0 : ℝ) ≤ 161000 * (populations.card : ℝ) ^ 2 := by positivity
    have hd' : (0 : ℝ) ≤ (1 / 2 - η₀) ^ 4 * rateScale η₀ ε_u εcov ^ 2 := by positivity
    exact add_le_add (add_le_add (div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_left (log_le_log_max hX0 hX) hc) hd') le_rfl) le_rfl
  refine ⟨states, fun B hB => ⟨hnpW B hB, hks B hB⟩, hband,
    fun B hB N hN p r => hκ B hB N hN κ p r hw.le le_rfl, ?_⟩
  intro D Dsf hD hDsf hpre hsuf hpAP' hatom hcoll
  rw [ea] at hatom
  rw [ec] at hcoll
  have hM : 0 < max 1 (nprefW v) := lt_max_of_lt_left one_pos
  have hlb0 : 0 ≤ logb 2 (max 1 (nprefW v)) := logb_nonneg one_lt_two (le_max_left _ _)
  refine (le_of_eq (by ring)).trans (hmain D Dsf hD hDsf hpre hsuf hpAP'
    (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB => ?_))
  · have h : (D j).real {t} ≤ rateScale η₀ ε_u εcov * (1 / 2 - η₀)
        / (2 ^ 18 * (populations.card + 3) * max 1 (nprefW v)) :=
      (le_min_iff.1 (le_min_iff.1 (hatom j hj t)).1).1
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    calc 2 ^ 18 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
        ≤ 2 ^ 18 * ((populations.card : ℝ) + 3) * max 1 (nprefW v) * (D j).real {t} := by
          gcongr; exact (hnpW B hB).trans (le_max_right _ _)
      _ ≤ _ := by linarith
  · have h : (D j).real {t} ≤ ε_s / (2 ^ 8 * (populations.card + 3) * max 1 (nprefW v)) :=
      (le_min_iff.1 (le_min_iff.1 (hatom j hj t)).1).2
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    calc 2 ^ 8 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
        ≤ 2 ^ 8 * ((populations.card : ℝ) + 3) * max 1 (nprefW v) * (D j).real {t} := by
          gcongr; exact (hnpW B hB).trans (le_max_right _ _)
      _ ≤ _ := by linarith
  · have h : (D j).real {t} ≤ δ / 2 * min ε_s ε_u ^ 2
        / (2 ^ 15 * populations.card * (3 + 2 * logb 2 (max 1 (nprefW v)))) :=
      (le_min_iff.1 (hatom j hj t)).2
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    have hlog : (Nat.log 2 B.npref : ℝ) ≤ logb 2 (max 1 (nprefW v)) := by
      rcases Nat.eq_zero_or_pos B.npref with h0 | h0
      · simpa [h0] using hlb0
      · have := Real.natLog_le_logb B.npref 2
        push_cast at this
        exact this.trans (logb_le_logb_of_le one_lt_two (by exact_mod_cast h0)
          ((hnpW B hB).trans (le_max_right _ _)))
    calc 2 ^ 15 * (populations.card : ℝ) * (3 + 2 * (Nat.log 2 B.npref : ℝ)) * (D j).real {t}
        ≤ 2 ^ 15 * (populations.card : ℝ) * (3 + 2 * logb 2 (max 1 (nprefW v)))
          * (D j).real {t} := by gcongr
      _ ≤ _ := by linarith
  · have h := hcoll
    have : 0 < max 1 (nsuffW v) := lt_max_of_lt_left one_pos
    rw [le_div_iff₀ (by positivity)] at h
    have hc0 : 0 ≤ collisionMass Dsf := tsum_nonneg fun t => sq_nonneg _
    calc 2 ^ 12 * ((populations.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Dsf
        ≤ 2 ^ 12 * ((populations.card : ℝ) + 1) * max 1 (nsuffW v) * collisionMass Dsf := by
          gcongr; exact (hks B hB).2.trans (le_max_right _ _)
      _ ≤ _ := by linarith

theorem clusteringClaimSimple_holds : ClusteringClaimSimple := by
  obtain ⟨k, nsuff, npref, atomTol, collTol, hk, hs, hp, ha, hc, hw⟩ :=
    clusteringClaim_holds
  exact ⟨k, nsuff, npref, atomTol, collTol, hk,
    hs.trans (dominate nsuff_dom (fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf)
      fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf),
    hp.trans (dominate npref_dom (fun x => by simp only [toVec]; ring_nf)
      fun x => by simp only [toVec]; ring_nf),
    ha.trans (dominate atom_dom (fun x => by simp only [toVec]; ring_nf)
      fun x => by simp only [toVec]; ring_nf),
    hc.trans (dominate coll_dom (fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf)
      fun x => by simp only [toVec, ClusterInputs.ℓ_conf]; ring_nf), hw⟩

end OrthoDFA
