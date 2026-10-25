import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Claims.Clustering
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.BigO

/-!
# The clustering's costs, asymptotically

The witnesses are `ClusteringGuarantee`'s bounds, the prefix count's log taken at the pool's
ceiling; the tolerances are its per-state atom and collision conditions at the ceilings, and `1`
off the parameters' range. They are stated and bounded on `toVec x`, the counts and the rates'
reciprocals, and carried back to `ClusterInputs`.
-/

noncomputable section

namespace OrthoDFA

open Real Asymptotics Filter MeasureTheory

private def kB (s r εw : ℝ) : ℝ :=
  25 * max (4 * log (20 / r)) (log (1 / εw)) / (8 * s ^ 2) + 5 / s + 7

private def nsuffB (s r εw pAP nJ δ : ℝ) : ℝ :=
  2 * kB s r εw / pAP + 16 * log ((nJ + 2) / δ) / pAP ^ 2

private def nprefB (s r δ α pAP nJ nH n : ℝ) : ℝ :=
  161000 * nJ ^ 2 * log (max 1 ((nJ + 2) * n / (δ * α * pAP * r))) / (s ^ 4 * r ^ 2)
    + 2 ^ 11 * nH / r

private def npref₀B (s r εu δ nJ : ℝ) : ℝ :=
  max (5 * log (128 * nJ / δ) / εu ^ 2) (2 ^ 13 * log (32 * nJ / δ) / (r * s))

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

/-- `(nJ, 1/δ, 1/α, 1/pAP, 1/εcov, 1/ε_u, 1/ε_w, 1/s₀, nH)`. -/
private abbrev ClusterVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set ClusterVec := Set.Ici (1, 2, 2, 1, 2, 2, 1, 2, 0)

private lemma region {nJ iδ iα ip ie iu iw is nH : ℝ}
    (hb : (nJ, iδ, iα, ip, ie, iu, iw, is, nH) ∈ vecRegion) :
    1 ≤ nJ ∧ 2 ≤ iδ ∧ 2 ≤ iα ∧ 1 ≤ ip ∧ 2 ≤ ie ∧ 2 ≤ iu ∧ 1 ≤ iw ∧ 2 ≤ is ∧ 0 ≤ nH := by
  simpa only [vecRegion, Set.mem_Ici, Prod.mk_le_mk] using hb

private lemma prods {nJ iδ iα ip ie iu iw is : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hiα : 2 ≤ iα)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) :
    8 ≤ ie * iu * iw * is ∧ 2 ≤ nJ * iδ ∧ 32 ≤ nJ * iδ * iα * ip * ie * iu * iw * is := by
  have h1 : 4 ≤ ie * iu := by nlinarith
  have h2 : 4 ≤ ie * iu * iw := by nlinarith
  have h3 : 2 ≤ nJ * iδ := by nlinarith
  have h4 : 4 ≤ nJ * iδ * iα := by nlinarith
  have h5 : 4 ≤ nJ * iδ * iα * ip := by nlinarith
  have h6 : 8 ≤ nJ * iδ * iα * ip * ie := by nlinarith
  have h7 : 16 ≤ nJ * iδ * iα * ip * ie * iu := by nlinarith
  have h8 : 16 ≤ nJ * iδ * iα * ip * ie * iu * iw := by nlinarith
  exact ⟨by nlinarith, h3, by nlinarith⟩

private lemma kB_bound {ie iu iw is r : ℝ} (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw)
    (his : 2 ≤ is) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) :
    0 ≤ kB (1 / is) r (1 / iw)
      ∧ kB (1 / is) r (1 / iw) ≤ 70 * (log (ie * iu * iw * is) * is ^ 2) := by
  have hl : ∀ x ∈ [ie, iu, iw, is], 1 ≤ x := fun x hx => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl <;> linarith
  have hsub (l : List ℝ) (hs : l.Sublist [ie, iu, iw, is]) : l.prod ≤ ie * iu * iw * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have hP : 8 ≤ ie * iu * iw * is := by
    have : 4 ≤ ie * iu := by nlinarith
    have : 4 ≤ ie * iu * iw := by nlinarith
    nlinarith
  set P := ie * iu * iw * is
  have hiwP : iw ≤ P := by simpa using hsub [iw] (by simp)
  have hiuP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hisP : is ≤ P := by simpa using hsub [is] (by simp)
  have hieis : ie * is ≤ P := by simpa using hsub [ie, is] (by simp)
  have hL : 1 ≤ log P := one_le_log_of_three_le (by linarith)
  have h20 : log (20 / r) ≤ 4 * log P := by
    refine (log_le_mul_log_of_le_pow (P := P) 4 (by positivity) ?_).trans (by norm_num)
    have : ie * is ^ 2 ≤ P * P := by
      calc ie * is ^ 2 = (ie * is) * is := by ring
        _ ≤ P * P := by gcongr
    have h1 : 1 / r ≤ 2 * P ^ 2 := by nlinarith
    have h2 : 64 * P ^ 2 ≤ P ^ 4 := by
      have : 64 ≤ P ^ 2 := by nlinarith
      nlinarith [mul_nonneg (sub_nonneg.2 this) (sq_nonneg P)]
    rw [div_eq_mul_one_div]
    nlinarith
  have hw : log iw ≤ log P := Real.log_le_log (by linarith) hiwP
  have hw0 : 0 ≤ log iw := Real.log_nonneg hiw
  have hm : max (4 * log (20 / r)) (log iw) ≤ 16 * log P := max_le (by linarith) (by linarith)
  have hm0 : 0 ≤ max (4 * log (20 / r)) (log iw) := le_max_of_le_right hw0
  unfold kB
  rw [one_div_one_div]
  generalize max (4 * log (20 / r)) (log iw) = m at *
  rw [show 25 * m / (8 * (1 / is) ^ 2) + 5 / (1 / is) + 7 = 25 * m * is ^ 2 / 8 + 5 * is + 7 by
    field_simp]
  have h1 : is ≤ is ^ 2 * log P := by nlinarith
  have h2 : 1 ≤ is ^ 2 * log P := by nlinarith
  have h3 := mul_le_mul_of_nonneg_right hm (sq_nonneg is)
  constructor
  · positivity
  · nlinarith

private lemma nsuffB_bound {nJ iδ ip ie iu iw is r : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is) (hr : 0 < r)
    (hr2 : 1 / r ≤ ie * is ^ 2 + iu) :
    0 ≤ nsuffB (1 / is) r (1 / iw) (1 / ip) nJ (1 / iδ)
      ∧ nsuffB (1 / is) r (1 / iw) (1 / ip) nJ (1 / iδ)
      ≤ 140 * (log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2) := by
  obtain ⟨hk0, hk⟩ := kB_bound hie hiu hiw his hr hr2
  have hx : 2 ≤ nJ * iδ := by nlinarith
  have hJl : log ((nJ + 2) / (1 / iδ)) ≤ 3 * log (nJ * iδ) := by
    refine (log_le_mul_log_of_le_pow (P := nJ * iδ) 3 (by positivity) ?_).trans (by norm_num)
    rw [div_one_div']
    refine (show (nJ + 2) * iδ ≤ 3 * (nJ * iδ) by nlinarith).trans ?_
    generalize nJ * iδ = x at *
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ x) (by nlinarith : (0 : ℝ) ≤ x ^ 2 - 4)]
  have hJ0 : 0 ≤ log ((nJ + 2) / (1 / iδ)) := Real.log_nonneg (by
    rw [div_one_div']; nlinarith)
  have hLJ : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  unfold nsuffB
  generalize kB (1 / is) r (1 / iw) = k at *
  generalize log ((nJ + 2) / (1 / iδ)) = l at *
  rw [show 2 * k / (1 / ip) + 16 * l / (1 / ip) ^ 2 = 2 * k * ip + 16 * l * ip ^ 2 by field_simp]
  have := mul_le_mul_of_nonneg_right hk (show 0 ≤ ip by linarith)
  have := mul_le_mul_of_nonneg_right hJl (sq_nonneg ip)
  have : 0 ≤ log (nJ * iδ) * ip ^ 2 := by positivity
  constructor
  · positivity
  · nlinarith

private lemma log_le_eight {nJ iδ : ℝ} (c : ℝ) (hc0 : 1 ≤ c) (hc : c ≤ 128) (hJ : 1 ≤ nJ)
    (hiδ : 2 ≤ iδ) :
    0 ≤ log (c * nJ / (1 / iδ)) ∧ log (c * nJ / (1 / iδ)) ≤ 8 * log (nJ * iδ) := by
  have hx : 2 ≤ nJ * iδ := by nlinarith
  rw [div_one_div', show c * nJ * iδ = c * (nJ * iδ) by ring]
  generalize nJ * iδ = x at *
  have h7 : (2 : ℝ) ^ 7 ≤ x ^ 7 := pow_le_pow_left₀ (by norm_num) hx 7
  refine ⟨Real.log_nonneg (by nlinarith),
    (log_le_mul_log_of_le_pow (P := x) 8 (by nlinarith) ?_).trans (by norm_num)⟩
  nlinarith [mul_nonneg (sub_nonneg.2 hc) (by linarith : (0 : ℝ) ≤ x),
    mul_nonneg (sub_nonneg.2 h7) (by linarith : (0 : ℝ) ≤ x)]

private lemma npref₀B_bound {nJ iδ ie iu is r : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ) (hie : 2 ≤ ie)
    (hiu : 2 ≤ iu) (his : 2 ≤ is) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) :
    0 ≤ npref₀B (1 / is) r (1 / iu) (1 / iδ) nJ ∧ npref₀B (1 / is) r (1 / iu) (1 / iδ) nJ
      ≤ 2 ^ 16 * (log (nJ * iδ) * (iu ^ 2 + iu * is + ie * is ^ 3)) := by
  obtain ⟨h1₀, h1⟩ := log_le_eight 128 (by norm_num) le_rfl hJ hiδ
  obtain ⟨h2₀, h2⟩ := log_le_eight 32 (by norm_num) (by norm_num) hJ hiδ
  have hLJ : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by nlinarith)
  unfold npref₀B
  generalize log (128 * nJ / (1 / iδ)) = l₁ at *
  generalize log (32 * nJ / (1 / iδ)) = l₂ at *
  rw [show 5 * l₁ / (1 / iu) ^ 2 = 5 * l₁ * iu ^ 2 by field_simp,
    show 2 ^ 13 * l₂ / (r * (1 / is)) = 2 ^ 13 * (l₂ * (1 / r)) * is by field_simp]
  have hR := mul_le_mul h2 hr2 (by positivity) (by positivity)
  have hR' := mul_le_mul_of_nonneg_right hR (show 0 ≤ is by linarith)
  have h1' := mul_le_mul_of_nonneg_right h1 (sq_nonneg iu)
  have : 0 ≤ log (nJ * iδ) * (iu * is) := by positivity
  have : 0 ≤ log (nJ * iδ) * (ie * is ^ 3) := by positivity
  have : 0 ≤ log (nJ * iδ) * iu ^ 2 := by positivity
  refine ⟨le_max_of_le_left (by positivity), max_le ?_ ?_⟩ <;> nlinarith

private lemma nprefB_bound {nJ iδ iα ip ie iu iw is nH r n : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hiα : 2 ≤ iα) (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) (hr : 0 < r) (hr2 : 1 / r ≤ ie * is ^ 2 + iu) (hn0 : 0 ≤ n)
    (hn : n ≤ 140 * (log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2)) :
    0 ≤ nprefB (1 / is) r (1 / iδ) (1 / iα) (1 / ip) nJ nH n
    ∧ nprefB (1 / is) r (1 / iδ) (1 / iα) (1 / ip) nJ nH n ≤ 2 ^ 23 *
      (nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + nH * (ie * is ^ 2 + iu)) := by
  have hl : ∀ x ∈ [nJ, iδ, iα, ip, ie, iu, iw, is], 1 ≤ x := fun x hx => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> linarith
  have hsub (l : List ℝ) (hs : l.Sublist [nJ, iδ, iα, ip, ie, iu, iw, is]) :
      l.prod ≤ nJ * iδ * iα * ip * ie * iu * iw * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  obtain ⟨hP4, hx, hP⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have hL4 : log (ie * iu * iw * is) ≤ ie * iu * iw * is := Real.log_le_self (by positivity)
  have hLJ : log (nJ * iδ) ≤ nJ * iδ := Real.log_le_self (by positivity)
  have hL40 : 0 ≤ log (ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have hLJ0 : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  have h4P : ie * iu * iw * is ≤ nJ * iδ * iα * ip * ie * iu * iw * is := by
    simpa [mul_assoc] using hsub _ (List.drop_sublist 4 _)
  have hPJ : nJ * iδ ≤ nJ * iδ * iα * ip * ie * iu * iw * is := by
    simpa [mul_assoc] using hsub [nJ, iδ] (by simp)
  set P := nJ * iδ * iα * ip * ie * iu * iw * is
  have hJP : nJ ≤ P := by simpa using hsub [nJ] (by simp)
  have hδP : iδ ≤ P := by simpa using hsub [iδ] (by simp)
  have hαP : iα ≤ P := by simpa using hsub [iα] (by simp)
  have hpP : ip ≤ P := by simpa using hsub [ip] (by simp)
  have heP : ie ≤ P := by simpa using hsub [ie] (by simp)
  have huP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hsP : is ≤ P := by simpa using hsub [is] (by simp)
  have hL8 : 1 ≤ log P := one_le_log_of_three_le (by linarith)
  have hn' : n ≤ 280 * P ^ 4 := by
    have : log (ie * iu * iw * is) * ip * is ^ 2 ≤ P * P * P ^ 2 := by gcongr; linarith
    have : log (nJ * iδ) * ip ^ 2 ≤ P * P ^ 2 := by gcongr; linarith
    have : P * P ^ 2 ≤ P * P * P ^ 2 := by
      gcongr; exact le_mul_of_one_le_left (by positivity) (by linarith)
    linarith
  have hR : 1 / r ≤ 2 * P ^ 3 := by
    have : ie * is ^ 2 ≤ P * P ^ 2 := by gcongr
    have : P ≤ P * P ^ 2 := le_mul_of_one_le_right (by positivity) (one_le_pow₀ (by linarith))
    linarith
  have hX : max 1 ((nJ + 2) * n / (1 / iδ * (1 / iα) * (1 / ip) * r)) ≤ P ^ 14 := by
    have h1 : (1 : ℝ) ≤ P ^ 14 := one_le_pow₀ (by linarith)
    refine max_le h1 ?_
    rw [show (nJ + 2) * n / (1 / iδ * (1 / iα) * (1 / ip) * r)
      = (nJ + 2) * n * iδ * iα * ip * (1 / r) by field_simp]
    have h3 : (32 : ℝ) ^ 3 ≤ P ^ 3 := pow_le_pow_left₀ (by norm_num) hP 3
    calc (nJ + 2) * n * iδ * iα * ip * (1 / r)
        ≤ (3 * P) * (280 * P ^ 4) * P * P * P * (2 * P ^ 3) := by
          gcongr; linarith
      _ = 1680 * P ^ 11 := by ring
      _ ≤ P ^ 3 * P ^ 11 := by gcongr; linarith
      _ = P ^ 14 := by ring
  have hl := log_le_mul_log_of_le_pow 14 (by positivity) hX
  have hl0 : 0 ≤ log (max 1 ((nJ + 2) * n / (1 / iδ * (1 / iα) * (1 / ip) * r))) :=
    Real.log_nonneg (le_max_left _ _)
  unfold nprefB
  generalize log (max 1 ((nJ + 2) * n / (1 / iδ * (1 / iα) * (1 / ip) * r))) = l at *
  rw [show 161000 * nJ ^ 2 * l / ((1 / is) ^ 4 * r ^ 2) + 2 ^ 11 * nH / r
    = 161000 * (nJ ^ 2 * l * is ^ 4 * (1 / r) ^ 2) + 2 ^ 11 * (nH * (1 / r)) by field_simp]
  have hR0 : 0 ≤ 1 / r := by positivity
  have hR2 : (1 / r) ^ 2 ≤ 2 * (ie ^ 2 * is ^ 4 + iu ^ 2) := by
    have := pow_le_pow_left₀ hR0 hr2 2
    have := add_sq_le (a := ie * is ^ 2) (b := iu)
    linarith
  have h1 : nJ ^ 2 * l * is ^ 4 * (1 / r) ^ 2
      ≤ nJ ^ 2 * (14 * log P) * is ^ 4 * (2 * (ie ^ 2 * is ^ 4 + iu ^ 2)) := by
    gcongr; exact_mod_cast hl
  have h2 : nH * (1 / r) ≤ nH * (ie * is ^ 2 + iu) := by gcongr
  have : 0 ≤ nJ ^ 2 * log P * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) :=
    mul_nonneg (mul_nonneg (by positivity) (by linarith)) (by positivity)
  have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
  constructor
  · positivity
  · linarith

private lemma gn_le_pow {nJ iδ iα ip ie iu iw is nH : ℝ} (hJ : 1 ≤ nJ) (hiδ : 2 ≤ iδ)
    (hiα : 2 ≤ iα) (hip : 1 ≤ ip) (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (hiw : 1 ≤ iw) (his : 2 ≤ is)
    (hnH : 0 ≤ nH) :
    2 ^ 23 * (nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) + nH * (ie * is ^ 2 + iu))
      ≤ (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)) ^ 18 := by
  have hl : ∀ x ∈ [nJ, iδ, iα, ip, ie, iu, iw, is, nH + 1], 1 ≤ x := fun x hx => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> linarith
  have hsub (l : List ℝ) (hs : l.Sublist [nJ, iδ, iα, ip, ie, iu, iw, is, nH + 1]) :
      l.prod ≤ nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1) := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  obtain ⟨-, -, hP8⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have h8 : nJ * iδ * iα * ip * ie * iu * iw * is
      ≤ nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1) :=
    le_mul_of_one_le_right (by positivity) (by linarith)
  have hL : log (nJ * iδ * iα * ip * ie * iu * iw * is) ≤ nJ * iδ * iα * ip * ie * iu * iw * is :=
    Real.log_le_self (by positivity)
  have hL0 : 0 ≤ log (nJ * iδ * iα * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  set P := nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)
  have hJP : nJ ≤ P := by simpa using hsub [nJ] (by simp)
  have heP : ie ≤ P := by simpa using hsub [ie] (by simp)
  have huP : iu ≤ P := by simpa using hsub [iu] (by simp)
  have hsP : is ≤ P := by simpa using hsub [is] (by simp)
  have hnP : nH + 1 ≤ P := by simpa using hsub [nH + 1] (by simp)
  have hP : 32 ≤ P := by linarith
  have hP1 : 1 ≤ P := by linarith
  have h1 : nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
      ≤ P ^ 2 * P * (P ^ 2 * P ^ 8 + P ^ 2 * P ^ 4) := by gcongr; linarith
  have h2 : nH * (ie * is ^ 2 + iu) ≤ P * (P * P ^ 2 + P) := by gcongr; linarith
  have h3 : P ^ 2 * P * (P ^ 2 * P ^ 8 + P ^ 2 * P ^ 4) + P * (P * P ^ 2 + P) ≤ 4 * P ^ 13 := by
    have : P ^ 2 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 4 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    have : P ^ 9 ≤ P ^ 13 := pow_le_pow_right₀ hP1 (by norm_num)
    ring_nf; linarith
  have h4 : 2 ^ 25 * P ^ 13 ≤ P ^ 18 := by
    have : (32 : ℝ) ^ 5 ≤ P ^ 5 := pow_le_pow_left₀ (by norm_num) hP 5
    calc 2 ^ 25 * P ^ 13 ≤ P ^ 5 * P ^ 13 := by
          gcongr; exact (by norm_num : (2 : ℝ) ^ 25 = 32 ^ 5).trans_le this
      _ = P ^ 18 := by ring
  linarith

private def rsc (ie iu is : ℝ) : ℝ := rateScale (1 / 2 - 1 / is) (1 / iu) (1 / ie)

private def kW : ClusterVec → ℝ := fun (_, _, _, _, ie, iu, iw, is, _) =>
  kB (1 / is) (rsc ie iu is) (1 / iw)

private def nsuffW : ClusterVec → ℝ := fun (nJ, iδ, _, ip, ie, iu, iw, is, _) =>
  nsuffB (1 / is) (rsc ie iu is) (1 / iw) (1 / ip) nJ (1 / iδ)

private def nprefW : ClusterVec → ℝ := fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
  nprefB (1 / is) (rsc ie iu is) (1 / iδ) (1 / iα) (1 / ip) nJ nH
    (nsuffB (1 / is) (rsc ie iu is) (1 / iw) (1 / ip) nJ (1 / iδ))

private def npref₀W : ClusterVec → ℝ := fun (nJ, iδ, _, _, ie, iu, _, is, _) =>
  npref₀B (1 / is) (rsc ie iu is) (1 / iu) (1 / iδ) nJ

open scoped Classical in
private def atomTolW : ClusterVec → ℝ := fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < iu ∧ 0 < is then
    min (rsc ie iu is * (1 / is)
        / (2 ^ 18 * (nJ + 3) * max 1 (nprefW (nJ, iδ, iα, ip, ie, iu, iw, is, nH))))
      (1 / iδ * min (1 / ie) (1 / iu) ^ 2
        / (2 ^ 33 * nJ * (3 + 2 * logb 2 (max 1 (nprefW (nJ, iδ, iα, ip, ie, iu, iw, is, nH))))))
  else 1

open scoped Classical in
private def collTolW : ClusterVec → ℝ := fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
  if 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < iu ∧ 0 < is then
    1 / iδ * (1 / ip) / (2 ^ 12 * (nJ + 1) * max 1 (nsuffW (nJ, iδ, iα, ip, ie, iu, iw, is, nH)))
  else 1

private lemma tol_pos (v : ClusterVec) : 0 < atomTolW v ∧ 0 < collTolW v := by
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := v
  simp only [atomTolW, collTolW]
  split_ifs with h
  · obtain ⟨hJ, hiδ, hip, hie, hiu, his⟩ := h
    have := rs_pos hie hiu his
    have : 0 ≤ logb 2 (max 1 (nprefW (nJ, iδ, iα, ip, ie, iu, iw, is, nH))) :=
      logb_nonneg one_lt_two (le_max_left _ _)
    unfold rsc at *
    refine ⟨lt_min (by positivity) (by positivity), by positivity⟩
  · exact ⟨one_pos, one_pos⟩

private lemma rsc_facts {ie iu is : ℝ} (hie : 2 ≤ ie) (hiu : 2 ≤ iu) (his : 2 ≤ is) :
    0 < rsc ie iu is ∧ 1 / rsc ie iu is ≤ ie * is ^ 2 + iu :=
  ⟨rs_pos (by linarith) (by linarith) (by linarith), rs_le (by linarith) (by linarith) his⟩

private lemma kW_isBigO : kW =O[𝓟 vecRegion] (fun (_, _, _, _, ie, iu, iw, is, _) =>
    log (ie * iu * iw * is) * is ^ 2 : ClusterVec → ℝ) := by
  refine isBigO_on 70 fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, -⟩ := region hb
  obtain ⟨hP4, -, -⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have : 0 ≤ log (ie * iu * iw * is) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := kB_bound hie hiu hiw his hr hr2
  dsimp only [kW]
  exact ⟨h0, by positivity, h⟩

private lemma nsuffW_isBigO : nsuffW =O[𝓟 vecRegion] (fun (nJ, iδ, _, ip, ie, iu, iw, is, _) =>
    log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2 : ClusterVec → ℝ) := by
  refine isBigO_on 140 fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, -⟩ := region hb
  obtain ⟨hP4, hx, -⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have : 0 ≤ log (ie * iu * iw * is) := Real.log_nonneg (by linarith)
  have : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := nsuffB_bound hJ hiδ hip hie hiu hiw his hr hr2
  dsimp only [nsuffW]
  exact ⟨h0, by positivity, h⟩

private lemma npref₀W_isBigO : npref₀W =O[𝓟 vecRegion] (fun (nJ, iδ, _, _, ie, iu, _, is, _) =>
    log (nJ * iδ) * (iu ^ 2 + iu * is + ie * is ^ 3) : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 16) fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, -⟩ := region hb
  obtain ⟨-, hx, -⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨h0, h⟩ := npref₀B_bound hJ hiδ hie hiu his hr hr2
  dsimp only [npref₀W]
  exact ⟨h0, by positivity, h⟩

private lemma nprefW_isBigO : nprefW =O[𝓟 vecRegion]
    (fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
      nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + nH * (ie * is ^ 2 + iu) : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 23) fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, -, hP8⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have : 0 ≤ log (nJ * iδ * iα * ip * ie * iu * iw * is) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hJ hiδ hip hie hiu hiw his hr hr2
  obtain ⟨h0, h⟩ := nprefB_bound hJ hiδ hiα hip hie hiu hiw his hnH hr hr2 hs0 hs
  dsimp only [nprefW]
  exact ⟨h0, by positivity, h⟩

private lemma atomTolW_isBigO : (fun v => 1 / atomTolW v) =O[𝓟 vecRegion]
    (fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
      nJ * (nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
              * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
          + nH * (ie * is ^ 2 + iu)) * (ie * is ^ 3 + iu * is)
        + nJ * iδ * log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)) * (ie ^ 2 + iu ^ 2)
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 44) fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, hnH⟩ := region hb
  obtain ⟨-, -, hP8⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have hL8 : 1 ≤ log (nJ * iδ * iα * ip * ie * iu * iw * is) :=
    one_le_log_of_three_le (by linarith)
  have hL9 : 1 ≤ log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)) :=
    one_le_log_of_three_le (by nlinarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hJ hiδ hip hie hiu hiw his hr hr2
  obtain ⟨hN0, hN⟩ := nprefB_bound hJ hiδ hiα hip hie hiu hiw his hnH hr hr2 hs0 hs
  have hG := gn_le_pow hJ hiδ hiα hip hie hiu hiw his hnH
  have hcond : 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < iu ∧ 0 < is :=
    ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩
  dsimp only
  simp only [atomTolW, nprefW, if_pos hcond]
  set N := nprefB (1 / is) (rsc ie iu is) (1 / iδ) (1 / iα) (1 / ip) nJ nH
    (nsuffB (1 / is) (rsc ie iu is) (1 / iw) (1 / ip) nJ (1 / iδ))
  set G := nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
    * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) + nH * (ie * is ^ 2 + iu)
  set r := rsc ie iu is
  have hG0 : 0 ≤ G := by positivity
  have hG1 : 1 ≤ 2 ^ 23 * G := by
    have h1 : 1 ≤ ie ^ 2 * is ^ 8 :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (one_le_pow₀ (by linarith))
    have h2 : 1 ≤ ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4 := by
      have : 0 ≤ iu ^ 2 * is ^ 4 := by positivity
      linarith
    have : 1 ≤ nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
        * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4) :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hJ) hL8) h2
    have : 0 ≤ nH * (ie * is ^ 2 + iu) := by positivity
    linarith
  have hM : max 1 N ≤ 2 ^ 23 * G := max_le hG1 hN
  have hM1 : 1 ≤ max 1 N := le_max_left _ _
  have hlb0 : 0 ≤ logb 2 (max 1 N) := logb_nonneg one_lt_two hM1
  have hA0 : 0 < r * (1 / is) / (2 ^ 18 * (nJ + 3) * max 1 N) := by positivity
  have hB0 : 0 < 1 / iδ * min (1 / ie) (1 / iu) ^ 2
      / (2 ^ 33 * nJ * (3 + 2 * logb 2 (max 1 N))) := by positivity
  refine ⟨(one_div_pos.2 (lt_min hA0 hB0)).le, by positivity, ?_⟩
  refine (one_div_min_le hA0 hB0).trans ?_
  rw [one_div_div, one_div_div,
    show 2 ^ 18 * (nJ + 3) * max 1 N / (r * (1 / is))
      = 2 ^ 18 * ((nJ + 3) * max 1 N * (1 / r) * is) by field_simp,
    show 2 ^ 33 * nJ * (3 + 2 * logb 2 (max 1 N)) / (1 / iδ * min (1 / ie) (1 / iu) ^ 2)
      = 2 ^ 33 * (nJ * (3 + 2 * logb 2 (max 1 N)) * iδ * (1 / min (1 / ie) (1 / iu)) ^ 2) by
      field_simp]
  have h1 : (nJ + 3) * max 1 N * (1 / r) * is
      ≤ (4 * nJ) * (2 ^ 23 * G) * (ie * is ^ 2 + iu) * is := by
    gcongr; linarith
  have hm : 1 / min (1 / ie) (1 / iu) ≤ ie + iu := by
    have := one_div_min_le (x := 1 / ie) (y := 1 / iu) (by positivity) (by positivity)
    rwa [one_div_one_div, one_div_one_div] at this
  have hm2 : (1 / min (1 / ie) (1 / iu)) ^ 2 ≤ 2 * (ie ^ 2 + iu ^ 2) :=
    (pow_le_pow_left₀ (by positivity) hm 2).trans add_sq_le
  have hlb : 3 + 2 * logb 2 (max 1 N)
      ≤ 75 * log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)) := by
    have hlog := log_le_mul_log_of_le_pow 18 (by positivity) (hM.trans hG)
    have hl0 : 0 ≤ log (max 1 N) := Real.log_nonneg hM1
    have h2 : 1 / 2 < log 2 := by linarith [log_two_gt_d9]
    have : log (max 1 N) / log 2 ≤ 2 * log (max 1 N) := by
      rw [div_le_iff₀ (by linarith)]; nlinarith
    rw [Real.logb]
    push_cast at hlog
    linarith
  have h2 : nJ * (3 + 2 * logb 2 (max 1 N)) * iδ * (1 / min (1 / ie) (1 / iu)) ^ 2
      ≤ nJ * (75 * log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1))) * iδ
        * (2 * (ie ^ 2 + iu ^ 2)) := by gcongr
  have : 0 ≤ nJ * G * (ie * is ^ 3 + iu * is) := by positivity
  have : 0 ≤ nJ * iδ * log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1))
      * (ie ^ 2 + iu ^ 2) := by positivity
  linarith

private lemma collTolW_isBigO : (fun v => 1 / collTolW v) =O[𝓟 vecRegion]
    (fun (nJ, iδ, _, ip, ie, iu, iw, is, _) =>
      nJ * iδ * ip * (log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2)
      : ClusterVec → ℝ) := by
  refine isBigO_on (2 ^ 21) fun b hb => ?_
  obtain ⟨nJ, iδ, iα, ip, ie, iu, iw, is, nH⟩ := b
  obtain ⟨hJ, hiδ, hiα, hip, hie, hiu, hiw, his, -⟩ := region hb
  obtain ⟨hP4, hx, -⟩ := prods hJ hiδ hiα hip hie hiu hiw his
  have hL4 : 1 ≤ log (ie * iu * iw * is) := one_le_log_of_three_le (by linarith)
  have hLJ : 0 ≤ log (nJ * iδ) := Real.log_nonneg (by linarith)
  obtain ⟨hr, hr2⟩ := rsc_facts hie hiu his
  obtain ⟨hs0, hs⟩ := nsuffB_bound hJ hiδ hip hie hiu hiw his hr hr2
  have hcond : 0 < nJ ∧ 0 < iδ ∧ 0 < ip ∧ 0 < ie ∧ 0 < iu ∧ 0 < is :=
    ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩
  dsimp only
  simp only [collTolW, nsuffW, if_pos hcond]
  set N := nsuffB (1 / is) (rsc ie iu is) (1 / iw) (1 / ip) nJ (1 / iδ)
  set G := log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2
  have hG1 : 1 ≤ 140 * G := by
    have : 1 ≤ log (ie * iu * iw * is) * ip * is ^ 2 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hL4 hip)
        (one_le_pow₀ (by linarith))
    have : 0 ≤ log (nJ * iδ) * ip ^ 2 := by positivity
    linarith
  have hM : max 1 N ≤ 140 * G := max_le hG1 hs
  have hA0 : 0 < 1 / iδ * (1 / ip) / (2 ^ 12 * (nJ + 1) * max 1 N) := by positivity
  refine ⟨(one_div_pos.2 hA0).le, by positivity, ?_⟩
  rw [one_div_div, show 2 ^ 12 * (nJ + 1) * max 1 N / (1 / iδ * (1 / ip))
      = 2 ^ 12 * ((nJ + 1) * max 1 N * iδ * ip) by field_simp]
  have : (nJ + 1) * max 1 N * iδ * ip ≤ (2 * nJ) * (140 * G) * iδ * ip := by gcongr; linarith
  have : 0 ≤ nJ * iδ * ip * G := by positivity
  linarith

private def toVec (x : ClusterInputs) : ClusterVec :=
  (x.nJ, 1 / x.δ, 1 / x.α, 1 / x.pAP, 1 / x.εcov, 1 / x.ε_u, 1 / x.ε_w, 1 / x.s₀, x.nH)

private lemma transfer {f g : ClusterVec → ℝ} {g' : ClusterInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 clusterRegion] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hJ, hδ, hδ1, hα, hα1, hp, hp1, he, he1, hu, hu1, hw, hw1, hs, hs1⟩ := hx
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hJ, (le_div_iff₀ hδ).2 (by linarith), (le_div_iff₀ hα).2 (by linarith),
    (le_div_iff₀ hp).2 (by linarith), (le_div_iff₀ he).2 (by linarith),
    (le_div_iff₀ hu).2 (by linarith), (le_div_iff₀ hw).2 (by linarith),
    (le_div_iff₀ hs).2 (by linarith), Nat.cast_nonneg _⟩

theorem clusteringClaim_holds : ClusteringClaim := by
  refine ⟨fun x => kW (toVec x), fun x => nsuffW (toVec x), fun x => nprefW (toVec x),
    fun x => npref₀W (toVec x), fun x => atomTolW (toVec x), fun x => collTolW (toVec x),
    transfer kW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer nsuffW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer npref₀W_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer nprefW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer atomTolW_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer collTolW_isBigO fun x => by simp only [toVec]; ring_nf,
    fun x => tol_pos _, ?_⟩
  intro Ω _ μ _ S _ J _ x O populations tame uni H Pre Suf hx hcard hnH hηle huni htame hflat hH
  obtain ⟨nJ, nH, δ, α, pAP, εcov, ε_u, ε_w, s₀⟩ := x
  simp only [clusterRegion, Set.mem_ofPred_eq] at hx hcard hnH hηle
  obtain ⟨hJ, hδ, hδ1, hα, hα1, hpAP, hpAP1, he, he1, hu, hu1, hw, hw1, hs, hs1⟩ := hx
  obtain ⟨η₀, rfl⟩ : ∃ η₀, s₀ = 1 / 2 - η₀ := ⟨1 / 2 - s₀, by ring⟩
  subst hcard hnH
  rw [sub_sub_cancel] at hηle
  have hη₀ : η₀ < 1 / 2 := by linarith
  simp only [ClusteringOutcome, toVec, sub_sub_cancel]
  set v : ClusterVec := ((populations.card : ℝ), 1 / δ, 1 / α, 1 / pAP, 1 / εcov, 1 / ε_u, 1 / ε_w,
    1 / (1 / 2 - η₀), ((allHarvest H).card : ℝ)) with hv
  obtain ⟨states, hnp, hnp₀, hkn, hband, hmain⟩ := clustering_guarantee_of_correct O
    populations tame uni H Pre Suf η₀ ε_u εcov α δ pAP ε_w hηle hη₀ huni htame hflat hH hpAP hu
    hu1 hα hα1 he he1 hδ (by linarith) hw hw1
  have hs0 : 0 < 1 / 2 - η₀ := by linarith
  have hcard : (0 : ℝ) < populations.card := by
    exact_mod_cast Finset.card_pos.2 ⟨uni, htame huni⟩
  have hr : 0 < rateScale η₀ ε_u εcov := by
    have := rs_pos (ie := 1 / εcov) (iu := 1 / ε_u) (is := 1 / (1 / 2 - η₀)) (by positivity)
      (by positivity) (by positivity)
    simpa only [one_div_one_div, sub_sub_cancel] using this
  have hcond : 0 < (populations.card : ℝ) ∧ 0 < 1 / δ ∧ 0 < 1 / pAP ∧ 0 < 1 / εcov
      ∧ 0 < 1 / ε_u ∧ 0 < 1 / (1 / 2 - η₀) :=
    ⟨hcard, by positivity, by positivity, by positivity, by positivity, by positivity⟩
  have ek : kW v = kB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) ε_w := by
    simp only [v, kW, rsc, one_div_one_div, sub_sub_cancel]
  have es : nsuffW v = nsuffB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) ε_w pAP populations.card δ := by
    simp only [v, nsuffW, rsc, one_div_one_div, sub_sub_cancel]
  have ep : nprefW v = nprefB (1 / 2 - η₀) (rateScale η₀ ε_u εcov) δ α pAP populations.card
      (allHarvest H).card (nsuffW v) := by
    simp only [v, nprefW, nsuffW, rsc, one_div_one_div, sub_sub_cancel]
  have ep₀ : npref₀W v = npref₀B (1 / 2 - η₀) (rateScale η₀ ε_u εcov) ε_u δ populations.card := by
    simp only [v, npref₀W, rsc, one_div_one_div, sub_sub_cancel]
  have ea : atomTolW v = min (rateScale η₀ ε_u εcov * (1 / 2 - η₀)
        / (2 ^ 18 * (populations.card + 3) * max 1 (nprefW v)))
      (δ * min εcov ε_u ^ 2
        / (2 ^ 33 * populations.card * (3 + 2 * logb 2 (max 1 (nprefW v))))) := by
    simp only [v, atomTolW, if_pos hcond, rsc, one_div_one_div, sub_sub_cancel]
  have ec : collTolW v = δ * pAP / (2 ^ 12 * (populations.card + 1) * max 1 (nsuffW v)) := by
    simp only [v, collTolW, if_pos hcond, one_div_one_div]
  have hks : ∀ B ∈ states, (B.k : ℝ) ≤ kW v ∧ (B.nsuff : ℝ) ≤ nsuffW v := fun B hB => by
    rw [ek, es]; exact hkn B hB
  have hnpW : ∀ B ∈ states, (B.npref : ℝ) ≤ nprefW v := fun B hB => by
    refine (hnp B hB).trans ?_
    rw [ep]
    unfold nprefB
    have hd : (0 : ℝ) < δ * α * pAP * rateScale η₀ ε_u εcov := by positivity
    have hX0 : 0 ≤ ((populations.card : ℝ) + 2) * B.nsuff
        / (δ * α * pAP * rateScale η₀ ε_u εcov) := by positivity
    have hX : ((populations.card : ℝ) + 2) * B.nsuff / (δ * α * pAP * rateScale η₀ ε_u εcov)
        ≤ ((populations.card : ℝ) + 2) * nsuffW v / (δ * α * pAP * rateScale η₀ ε_u εcov) := by
      gcongr; exact (hks B hB).2
    have hc : (0 : ℝ) ≤ 161000 * (populations.card : ℝ) ^ 2 := by positivity
    have hd' : (0 : ℝ) ≤ (1 / 2 - η₀) ^ 4 * rateScale η₀ ε_u εcov ^ 2 := by positivity
    exact add_le_add (div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_left (log_le_log_max hX0 hX) hc) hd') le_rfl
  refine ⟨states, fun B hB => ⟨hnpW B hB, hks B hB⟩, ?_, hband, ?_⟩
  · obtain ⟨B, hB, h⟩ := hnp₀
    exact ⟨B, hB, by rw [ep₀]; exact h⟩
  intro D Dsf hD hDsf hpre hsuf hpAP' hatom hcoll
  rw [ea] at hatom
  rw [ec] at hcoll
  have hM : 0 < max 1 (nprefW v) := lt_max_of_lt_left one_pos
  have hlb0 : 0 ≤ logb 2 (max 1 (nprefW v)) := logb_nonneg one_lt_two (le_max_left _ _)
  refine hmain D Dsf hD hDsf hpre hsuf hpAP' (fun B hB j hj t => ?_) (fun B hB j hj t => ?_)
    (fun B hB => ?_)
  · have h : (D j).real {t} ≤ rateScale η₀ ε_u εcov * (1 / 2 - η₀)
        / (2 ^ 18 * (populations.card + 3) * max 1 (nprefW v)) := (le_min_iff.1 (hatom j hj t)).1
    rw [le_div_iff₀ (by positivity)] at h
    have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
    calc 2 ^ 18 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
        ≤ 2 ^ 18 * ((populations.card : ℝ) + 3) * max 1 (nprefW v) * (D j).real {t} := by
          gcongr; exact (hnpW B hB).trans (le_max_right _ _)
      _ ≤ _ := by linarith
  · have h : (D j).real {t} ≤ δ * min εcov ε_u ^ 2
        / (2 ^ 33 * populations.card * (3 + 2 * logb 2 (max 1 (nprefW v)))) :=
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
    calc 2 ^ 33 * (populations.card : ℝ) * (3 + 2 * (Nat.log 2 B.npref : ℝ)) * (D j).real {t}
        ≤ 2 ^ 33 * (populations.card : ℝ) * (3 + 2 * logb 2 (max 1 (nprefW v)))
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

end OrthoDFA
