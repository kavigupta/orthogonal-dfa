import RLStar.Proofs.TrieRoundInv
import RLStar.Proofs.QueryAdaptive
import RLStar.Proofs.RandomRound
import RLStar.Proofs.RoundEnd
import RLStar.Proofs.TrieCap
import RLStar.LowLevelClaims.TrieRound

/-!
# The trie harvest's law on the states rarely read undecided

A probe walked from `k` harvests `x`'s length-`p` prefix followed by a midfix, `p ≥ k`, read
undecided.
- Where `p ≥ k_w` the string is in `pot k_w x T`, whatever the expanded prefixes. These probes'
  mass concentrates at scale `p_w`, over the trees of `classSet |Q|` (`light_noise_le`).
- Where `p < k_w` it is in `heavySet k_w T` (`trieHarvestLaw_low_le`), which `TrieEndsWell` reads
  after the round as one program.
-/

noncomputable section

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory Query
open scoped ENNReal

variable {α : Type*} [Fintype α]

open scoped Classical in
/-- The strings `v m` with `|v| < k` and `m` a midfix of `T`. -/
def heavySet (k : ℕ) (T : DTree α) : Finset (FreeMonoid α) :=
  ((Finset.range k).biUnion lenK).biUnion fun v => (mids T).image (v * ·)

theorem card_heavySet (k : ℕ) (T : DTree α) :
    (heavySet k T).card ≤ (∑ j ∈ Finset.range k, Fintype.card α ^ j) * (mids T).card := by
  classical
  unfold heavySet
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _v ∈ (Finset.range k).biUnion lenK, (mids T).card :=
        Finset.sum_le_sum fun v _ => Finset.card_image_le
    _ = ((Finset.range k).biUnion (lenK (α := α))).card * (mids T).card := by
        rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := by
        refine Nat.mul_le_mul_right _ (Finset.card_biUnion_le.trans
          (Finset.sum_le_sum fun j _ => ?_))
        unfold lenK
        refine Finset.card_image_le.trans (le_of_eq ?_)
        simp

variable (read : FreeMonoid α → ARU)

omit [Fintype α] in
theorem startString_sub (o : Outcome α) : ∀ z ∈ startString o, z ∈ endsString o := by
  cases o <;> simp [startString, endsString]

omit [Fintype α] in
theorem trieSiteHarvest_spec (X : Finset (FreeMonoid α)) (H : HypothesisDFA α) (site : TrieSite)
    (x : FreeMonoid α) :
    (∀ z ∈ trieSiteHarvest read X H site x,
      z ∈ pot (walkStartIn X x) x H.tree ∧ read z = .undecided)
      ∧ (trieSiteHarvest read X H site x).length ≤ 1 := by
  unfold trieSiteHarvest trieSiteHarvestQ
  rw [eval_map]
  have he : (probeOrAgreeQ H.tree H.edges (walkStartIn X x) x).eval read
      = probeOrAgree read H.tree H.edges (walkStartIn X x) x := rfl
  rw [he]
  cases site with
  | starts =>
    refine ⟨fun z hz => probeOrAgree_ends read H.tree x H.edges _ z (startString_sub _ z hz), ?_⟩
    simp only
    cases probeOrAgree read H.tree H.edges (walkStartIn X x) x <;> simp [startString]
  | ends => exact ⟨probeOrAgree_ends read H.tree x H.edges _, endsString_length _⟩
  | middles => exact ⟨probeOrAgree_pt read H.tree x H.edges _, middleString_length _⟩

variable (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (X : Finset (FreeMonoid α))
  (H : HypothesisDFA α) (site : TrieSite)

theorem trieHarvestLaw_le (A : Set (FreeMonoid α)) :
    trieHarvestLaw read D X H site A ≤ D {x | ∃ z ∈ trieSiteHarvest read X H site x, z ∈ A} := by
  classical
  set B := {x | ∃ z ∈ trieSiteHarvest read X H site x, z ∈ A}
  unfold trieHarvestLaw
  rw [bind_diracSum_apply, ← lintegral_indicator_one (Set.to_countable B).measurableSet]
  refine lintegral_mono fun x => ?_
  by_cases hx : x ∈ B
  · rw [Set.indicator_of_mem hx, Pi.one_apply]
    have := (List.length_filter_le (· ∈ A) _).trans (trieSiteHarvest_spec read X H site x).2
    calc _ ≤ ((1 : ℕ) : ℝ≥0∞) := Nat.cast_le.2 this
      _ = 1 := Nat.cast_one
  · rw [Set.indicator_of_notMem hx]
    have : (trieSiteHarvest read X H site x).filter (· ∈ A) = [] :=
      List.filter_eq_nil_iff.2 fun z hz hzA => hx ⟨z, hz, by simpa using hzA⟩
    simp [this]

instance : IsFiniteMeasure (trieHarvestLaw read D X H site) :=
  ⟨(trieHarvestLaw_le read D X H site Set.univ).trans_lt (measure_lt_top D _)⟩

theorem trieHarvestLaw_real_le (A : Set (FreeMonoid α)) :
    (trieHarvestLaw read D X H site).real A
      ≤ D.real {x | ∃ z ∈ trieSiteHarvest read X H site x, z ∈ A} :=
  ENNReal.toReal_mono (measure_ne_top D _) (trieHarvestLaw_le read D X H site A)

theorem trieHarvestLaw_univ_le :
    (trieHarvestLaw read D X H site).real Set.univ ≤ 1 :=
  (trieHarvestLaw_real_le read D X H site _).trans measureReal_le_one

theorem trieHarvestLaw_spread : Spread D 0 (trieHarvestLaw read D X H site) := by
  refine spread_of_fresh D 0 _ (fun x => ?_) fun x z _ => by simp
  have hl := (trieSiteHarvest_spec read X H site x).2
  rcases h : trieSiteHarvest read X H site x with _ | ⟨a, _ | ⟨b, l⟩⟩ <;> simp_all

/-- The law's mass at the states read undecided at most `t` of the time is at most the probes
reading undecided there a string `x`'s first `k` letters or more followed by a midfix, once no
heavy string is so read. -/
theorem trieHarvestLaw_low_le {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (k : ℕ) (t : ℝ)
    (hheavy : ∀ z ∈ heavySet k H.tree, ¬ (read z = .undecided ∧ U (M.eval z.toList) ≤ t)) :
    (trieHarvestLaw read D X H site).real {z | U (M.eval z.toList) ≤ t}
      ≤ D.real {x | ∃ z ∈ pot k x H.tree, read z = .undecided ∧ U (M.eval z.toList) ≤ t} := by
  classical
  refine (trieHarvestLaw_real_le read D X H site _).trans
    (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _))
  obtain ⟨z, hz, hzt⟩ := hx
  obtain ⟨hpot, hund⟩ := (trieSiteHarvest_spec read X H site x).1 z hz
  obtain ⟨p, hkp, hpx, m, hm, rfl⟩ := mem_pot_iff hpot
  by_cases hp : k ≤ p
  · exact ⟨_, mem_pot hp hpx hm, hund, hzt⟩
  · refine absurd ⟨hund, hzt⟩ (hheavy _ ?_)
    unfold heavySet
    refine Finset.mem_biUnion.2 ⟨prefixOf x p, Finset.mem_biUnion.2 ⟨p, ?_, ?_⟩,
      Finset.mem_image.2 ⟨m, hm, rfl⟩⟩
    · exact Finset.mem_range.2 (by omega)
    · exact mem_lenK.2 (length_prefixOf hpx)

/-! ## The light strings -/

section Light

variable {σ : Type*} (M : DFA α σ) (U : σ → ℝ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (rd : FreeMonoid α → Ω → ARU)

/-- At one tree and rate `t`, the probes reading undecided, at a state read undecided at most `t`
of the time, a string of `pot k x T` exceed twice `θ₀ = (L + 1)(n + 1)t` by `η` with chance at
most `exp(−η / p₀)`, where `θ₀ ≤ 1/8`. -/
theorem light_tail_one (hmeas : ∀ z, Measurable (rd z)) (hind : iIndepFun rd μ)
    (hU : ∀ z, μ.real {ω | rd z ω = .undecided} = U (M.eval z.toList)) {L n k : ℕ}
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {p₀ η t : ℝ} (hp₀ : 0 < p₀)
    (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (T : DTree α)
    (hT : (mids T).card ≤ n + 1) (ht : 0 ≤ t) (ht8 : 8 * ((L + 1) * (n + 1) * t) ≤ 1) :
    μ {ω | 2 * ((L + 1) * (n + 1) * t) + η
        < D.real {x | ∃ z ∈ pot k x T, rd z ω = .undecided ∧ U (M.eval z.toList) ≤ t}}
      ≤ ENNReal.ofReal (Real.exp (-(η / p₀))) := by
  classical
  set θ₀ : ℝ := (L + 1) * (n + 1) * t
  have hθ₀ : 0 ≤ θ₀ := by positivity
  set P : (FreeMonoid α → ARU) → FreeMonoid α → Prop := fun r x =>
    ∃ z ∈ pot k x T, r z = .undecided ∧ U (M.eval z.toList) ≤ t
  have hPc : ∀ x r r', (∀ z ∈ pot k x T, r z = r' z) → (P r x ↔ P r' x) := by
    intro x r r' h
    simp only [P]
    exact ⟨fun ⟨z, hz, h1, h2⟩ => ⟨z, hz, (h z hz) ▸ h1, h2⟩,
      fun ⟨z, hz, h1, h2⟩ => ⟨z, hz, (h z hz).symm ▸ h1, h2⟩⟩
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun r x => if P r x then 1 else 0
  have hTB := block_tail hmeas hind D hp₀ hpre hθ₀ ht8 (fun x => pot k x T)
    (fun x z hz => (pot_pre hz).2) uf (fun _ _ => 1)
    (fun x r r' h => by simp only [uf, hPc x r r' h])
    (fun _ _ _ _ => rfl) (fun r x _ => by simp only [uf]; split_ifs <;> norm_num)
    (fun r x _ => by norm_num) (η := η) (L := L)
    fun x hx => by
      simp only [uf]
      rw [integral_ite_strings rd hmeas (pot k x T) (fun r => P r x) fun r r' h => hPc x r r' h]
      simp only [integral_const, probReal_univ, smul_eq_mul, mul_one]
      have hxL : x.toList.length ≤ L := mem_upTo.1 hx
      have hsub : {ω | P (rd · ω) x} ⊆ ⋃ z ∈ (pot k x T).filter fun z => U (M.eval z.toList) ≤ t,
          {ω | rd z ω = .undecided} := by
        intro ω ⟨z, hz, h1, h2⟩
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hz, h2⟩) h1
      calc _ ≤ _ := measureReal_mono hsub (measure_ne_top μ _)
        _ ≤ ∑ z ∈ (pot k x T).filter fun z => U (M.eval z.toList) ≤ t,
              μ.real {ω | rd z ω = .undecided} := measureReal_biUnion_finset_le _ _
        _ ≤ ∑ _z ∈ (pot k x T).filter fun z => U (M.eval z.toList) ≤ t, t :=
            Finset.sum_le_sum fun z hz => by rw [hU z]; exact (Finset.mem_filter.1 hz).2
        _ ≤ (pot k x T).card * t := by
            rw [Finset.sum_const, nsmul_eq_mul]
            exact mul_le_mul_of_nonneg_right (by exact_mod_cast Finset.card_filter_le _ _) ht
        _ ≤ θ₀ := by
            have hc : ((pot k x T).card : ℝ) ≤ (L + 1) * (n + 1) := by
              have := (card_pot k x T).trans (Nat.mul_le_mul (Nat.succ_le_succ hxL) hT)
              exact_mod_cast this
            exact mul_le_mul_of_nonneg_right hc ht
  refine le_trans (measure_mono fun ω hω => ?_) hTB
  rw [Set.mem_ofPred_eq] at hω ⊢
  rw [measureReal_eq_sum_upTo D hL] at hω
  have hs : ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 1 := by
    simp only [mul_one]
    rw [sum_measureReal_singleton]
    exact measureReal_le_one
  have : 2 * θ₀ * ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 2 * θ₀ := by nlinarith
  have heq : ∑ x ∈ upTo L, D.real {x} * uf (rd · ω) x
      = ∑ x ∈ upTo L, D.real {x} * if x ∈ {x | P (rd · ω) x} then 1 else 0 :=
    Finset.sum_congr rfl fun x _ => by
      simp only [uf]
      split_ifs with h1 h2 h2 <;> first | rfl | exact absurd h1 h2 | exact absurd h2 h1
  linarith

/-- At every state's rate `t = U q` with `8(L + 1)(n + 1)t ≤ 1` and every tree of `classSet n`,
the light probes are at most `2(L + 1)(n + 1)t + η`, but for `n_Q K exp(−η / p₀)`. -/
theorem light_noise_le [Fintype σ] (hmeas : ∀ z, Measurable (rd z)) (hind : iIndepFun rd μ)
    (hU : ∀ z, μ.real {ω | rd z ω = .undecided} = U (M.eval z.toList)) {L n k : ℕ}
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {p₀ η : ℝ} (hp₀ : 0 < p₀)
    (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) :
    μ {ω | ∃ q, 0 ≤ U q ∧ 8 * ((L + 1) * (n + 1) * U q) ≤ 1 ∧
        ∃ T ∈ (classSet n : Finset (DTree α)), 2 * ((L + 1) * (n + 1) * U q) + η
          < D.real {x | ∃ z ∈ pot k x T, rd z ω = .undecided ∧ U (M.eval z.toList) ≤ U q}}
      ≤ ENNReal.ofReal (Fintype.card σ * (1 + (n + 1) ^ 3 * Fintype.card α) ^ n
        * Real.exp (-(η / p₀))) := by
  classical
  set e := ENNReal.ofReal (Real.exp (-(η / p₀)))
  have hsub : {ω | ∃ q, 0 ≤ U q ∧ 8 * ((L + 1) * (n + 1) * U q) ≤ 1 ∧
        ∃ T ∈ (classSet n : Finset (DTree α)), 2 * ((L + 1) * (n + 1) * U q) + η
          < D.real {x | ∃ z ∈ pot k x T, rd z ω = .undecided ∧ U (M.eval z.toList) ≤ U q}}
      ⊆ ⋃ q ∈ (Finset.univ.filter fun q => 0 ≤ U q ∧ 8 * ((L + 1) * (n + 1) * U q) ≤ 1),
        ⋃ T ∈ (classSet n : Finset (DTree α)),
        {ω | 2 * ((L + 1) * (n + 1) * U q) + η
          < D.real {x | ∃ z ∈ pot k x T, rd z ω = .undecided ∧ U (M.eval z.toList) ≤ U q}} := by
    intro ω ⟨q, h0, h8, T, hT, h⟩
    exact Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_univ _, h0, h8⟩)
      (Set.mem_biUnion hT h)
  refine (measure_mono hsub).trans ((measure_biUnion_finset_le _ _).trans ?_)
  calc _ ≤ ∑ _q ∈ (Finset.univ.filter fun q => 0 ≤ U q ∧ 8 * ((L + 1) * (n + 1) * U q) ≤ 1),
        ∑ _T ∈ (classSet n : Finset (DTree α)), e := by
        refine Finset.sum_le_sum fun q hq => (measure_biUnion_finset_le _ _).trans
          (Finset.sum_le_sum fun T hT => ?_)
        obtain ⟨-, h0, h8⟩ := Finset.mem_filter.1 hq
        exact light_tail_one D M U rd hmeas hind hU hL hp₀ hpre T (mids_le_of_class hT) h0 h8
    _ ≤ ∑ _q : σ, ((1 + (n + 1) ^ 3 * Fintype.card α) ^ n : ℕ) * e := by
        refine (Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)).trans
          (Finset.sum_le_sum fun q _ => ?_)
        rw [Finset.sum_const, nsmul_eq_mul]
        gcongr
        exact_mod_cast classSet_card n
    _ = _ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← mul_assoc]
        simp only [e]
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity)]
        push_cast
        ring_nf

end Light

end Random

end OrthoDFA

end
