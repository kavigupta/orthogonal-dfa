import RLStar.LowLevelClaims.Level5Claim
import RLStar.Proofs.SpurAccount
import RLStar.Proofs.SpurCount

/-!
# Level 5: the proof

Level 3 within an allowance of `f` leaves (`Level3AllowanceCorrect`) fails but for `7δ/8`, plus
the chance that a step grows the tree past `|Q| + f + 2` leaves. Where no two probes share their
first `walkStart` letters, such a step comes after at least `f + 2` fresh-bad rounds
(`grows_badCount`, from `account`), which `RaceTailBound` bounds with the rounds' reads of
`count6`. Two probes share their first letters with chance at most `p₀` a pair
(`pi_notDistinct`).
-/

namespace OrthoDFA

namespace Random

namespace Spur

open MeasureTheory Eps Kappa RaceTail AdaptiveReads

variable {α : Type*} [Fintype α] [DecidableEq α]

section Grows

variable {r : FreeMonoid α → ARU} {C : Level3Settings} {n : ℕ}

theorem growsRun_state : ∀ (xs : List (FreeMonoid α)) (s₀ : Level3State α),
    GrowsRun r C n s₀ xs → ∃ j ≤ xs.length, ∃ s, stateAt C r s₀ xs j = some s
      ∧ n + 2 < s.tree.leaves.length
  | [], _, h => h.elim
  | x :: xs, s₀, h => by
    rcases h with ⟨s', hs', hlt⟩ | ⟨s', hs', h⟩
    · exact ⟨1, by simp, s', by simp [stateAt, hs'], hlt⟩
    · obtain ⟨j, hj, s, hs, hlt⟩ := growsRun_state xs s' h
      exact ⟨j + 1, by simp; omega, s, by simp [stateAt, hs', hs], hlt⟩

theorem leaves_le_of_not_growsRun : ∀ (xs : List (FreeMonoid α)) (s : Level3State α),
    s.tree.leaves.length ≤ n + 2 → ¬ GrowsRun r C n s xs →
      (runProbes r C s xs).1.tree.leaves.length ≤ n + 2
  | [], _, h, _ => h
  | x :: xs, s, h, hg => by
    simp only [runProbes_nil, runProbes_cons]
    rcases ha : applyProbe r C s x with s' | e
    · refine leaves_le_of_not_growsRun xs s' ?_ fun h' => hg (Or.inr ⟨s', ha, h'⟩)
      by_contra hc
      exact hg (Or.inl ⟨s', ha, by omega⟩)
    · exact h

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) {xs : List (FreeMonoid α)}
  {R : ℕ}

open scoped Classical in
/-- A step growing the tree past `n + 2` leaves, where no string read wrong starts like two probes,
follows at least `n + 2 - |Q|` fresh-bad rounds. -/
theorem grows_badCount (hc : ∀ z ∈ before (rounds6 C xs n R) (xs.length * (R + 1)) r,
      Wrong M side r z → Unshared xs C.walkStart z) (hR : (n + 2) * Fintype.card α ≤ R)
    (hgrow : GrowsRun r C n initialState xs) :
    n + 2 ≤ Fintype.card σ
      + badCount (rounds6 C xs n R) (wrongSet M side) (xs.length * (R + 1)) r := by
  have hP : ∃ j, j ≤ xs.length ∧ ∃ s, stateAt C r initialState xs j = some s
      ∧ n + 2 < s.tree.leaves.length := by
    obtain ⟨j, hj, s, hs, hlt⟩ := growsRun_state xs initialState hgrow
    exact ⟨j, hj, s, hs, hlt⟩
  obtain ⟨hj, s, hs, hlt⟩ := Nat.find_spec hP
  have hmin : ∀ i < Nat.find hP, ∀ s', stateAt C r initialState xs i = some s' →
      s'.tree.leaves.length ≤ n + 2 := fun i hi s' hs' => by
    by_contra hc
    exact Nat.find_min hP hi ⟨by omega, s', hs', by omega⟩
  obtain ⟨-, -, -, hacc⟩ := account (M := M) (side := side) hc hR (Nat.find hP) s hs hmin
  have himg := img_card_le M side s.tree
  have hmono := badCount_mono (rounds6 C xs n R) (wrongSet M side) r
    (Nat.mul_le_mul_right (R + 1) hj)
  omega

end Grows

open scoped Classical in
/-- A round reading wrong makes a fresh-bad round. -/
theorem one_le_badCount {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B : Z → Set R) {N : ℕ}
    {r : Z → R} {z : Z} (hz : z ∈ before Q N r) (hzB : r z ∈ B z) : 1 ≤ badCount Q B N r := by
  have hP : ∃ i, i < N ∧ ∃ y ∈ Q i r, r y ∈ B y := by
    obtain ⟨i, hi, hzi⟩ := (mem_before Q).1 hz
    exact ⟨i, hi, z, hzi, hzB⟩
  obtain ⟨hi₀, y, hy, hyB⟩ := Nat.find_spec hP
  have hfresh : FreshBad Q B (Nat.find hP) r := by
    refine ⟨y, Finset.mem_filter.2 ⟨hy, fun i hi hyi => ?_⟩, hyB⟩
    exact Nat.find_min hP hi ⟨by omega, y, hyi, hyB⟩
  unfold badCount
  exact Finset.card_pos.2 ⟨_, Finset.mem_filter.2 ⟨Finset.mem_range.2 hi₀, hfresh⟩⟩

/-! ## A round's reads -/

section Card

variable (r : FreeMonoid α → ARU) (C : Level3Settings) (xs : List (FreeMonoid α)) (n R L : ℕ)

open scoped Classical in
theorem rounds6_card (hL : ∀ x ∈ xs, x.toList.length ≤ L) (ρ : ℕ) :
    (rounds6 C xs n R ρ r).card ≤ max (probeSifts L * (n + 1)) (n + 2)
      ∧ ((rounds6 C xs n R ρ r).filter fun z => r z = .undecided).card ≤ 1 := by
  unfold rounds6
  split_ifs with hS
  swap
  · simp
  rcases hs : stateAt C r initialState xs (ρ / (R + 1)) with _ | s
  · simp
  rcases hx : xs[ρ / (R + 1)]? with _ | x
  · simp
  dsimp only
  have hinv := (inv_keys_stateAt r C xs _ s hs).1
  have hsm := hS _ le_rfl s hs
  unfold stepRound
  split_ifs with h0
  · have hxL : x.toList.length ≤ L := hL x (List.mem_of_getElem? hx)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    refine ⟨((probeReads_card_mids r hxL).trans (Nat.mul_le_mul_left _ hmid)).trans
      (le_max_left _ _), ?_⟩
    exact (probeReads_und_card r s.tree s.edges C.walkStart x).trans
      (ends_mid_le_one _)
  · rcases (edgeList s)[ρ % (R + 1) - 1]? with _ | e
    · simp
    · obtain ⟨h1, h2⟩ := resiftOf_card r C hinv (x := x) e
      exact ⟨h1.trans (hsm.trans (le_max_right _ _)), h2⟩

end Card

/-! ## Two probes sharing their first letters -/

section Collide

open scoped ENNReal

variable (P : PrefixDistribution α)

/-- Probes at least `k` long that start with `u`. -/
def startsWith (k : ℕ) (u : List α) : Set (FreeMonoid α) :=
  {x | k ≤ x.toList.length ∧ x.toList.take k = u}

omit [DecidableEq α] in
theorem pi_pair_le {N : ℕ} {i j : Fin N} (hij : i ≠ j) (A : Set (FreeMonoid α)) :
    (Measure.pi fun _ : Fin N => P.D) {d | d i ∈ A ∧ d j ∈ A} ≤ P.D A * P.D A := by
  classical
  have : {d : Fin N → FreeMonoid α | d i ∈ A ∧ d j ∈ A}
      = Set.univ.pi fun l => if l = i ∨ l = j then A else Set.univ := by
    ext d
    simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, true_implies]
    constructor
    · rintro ⟨h1, h2⟩ l
      split_ifs with h
      · rcases h with rfl | rfl
        · exact h1
        · exact h2
      · trivial
    · intro h
      exact ⟨by simpa using h i, by simpa using h j⟩
  rw [this, Measure.pi_pi]
  rw [← Finset.prod_erase_mul _ _ (Finset.mem_univ j),
    ← Finset.prod_erase_mul _ _ (Finset.mem_erase.2 ⟨hij, Finset.mem_univ i⟩)]
  simp only [true_or, or_true, if_true]
  calc _ ≤ 1 * P.D A * P.D A := by
        gcongr
        refine Finset.prod_le_one' fun l _ => ?_
        split_ifs
        · exact prob_le_one
        · simp
    _ = _ := by ring

omit [DecidableEq α] in
theorem collide_le (k : ℕ) (p₀ : ℝ)
    (hpre : ∀ u, P.D.real {x | prefixOf x k = u} ≤ p₀) {N : ℕ} {i j : Fin N} (hij : i ≠ j) :
    (Measure.pi fun _ : Fin N => P.D) {d | k ≤ (d i).toList.length ∧ k ≤ (d j).toList.length
      ∧ (d i).toList.take k = (d j).toList.take k} ≤ ENNReal.ofReal p₀ := by
  classical
  have : Countable (List α) := inferInstance
  have hsub : {d : Fin N → FreeMonoid α | k ≤ (d i).toList.length ∧ k ≤ (d j).toList.length
      ∧ (d i).toList.take k = (d j).toList.take k}
      ⊆ ⋃ u : List α, {d | d i ∈ startsWith k u ∧ d j ∈ startsWith k u} := by
    intro d ⟨h1, h2, h3⟩
    exact Set.mem_iUnion.2 ⟨(d i).toList.take k, ⟨h1, rfl⟩, ⟨h2, h3.symm⟩⟩
  have hA : ∀ u, P.D (startsWith k u) ≤ ENNReal.ofReal p₀ := by
    intro u
    calc P.D (startsWith k u) ≤ P.D {x | prefixOf x k = FreeMonoid.ofList u} :=
          measure_mono fun x hx => by
            show prefixOf x k = FreeMonoid.ofList u
            simp only [prefixOf, hx.2]
      _ = ENNReal.ofReal (P.D.real {x | prefixOf x k = FreeMonoid.ofList u}) :=
          (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
      _ ≤ _ := ENNReal.ofReal_le_ofReal (hpre _)
  have hdisj : Pairwise (Function.onFun Disjoint fun u : List α => startsWith k u) := by
    intro u v huv
    refine Set.disjoint_left.2 fun x hu hv => huv ?_
    exact hu.2.symm.trans hv.2
  calc _ ≤ ∑' u, (Measure.pi fun _ : Fin N => P.D)
          {d | d i ∈ startsWith k u ∧ d j ∈ startsWith k u} :=
        (measure_mono hsub).trans (measure_iUnion_le _)
    _ ≤ ∑' u, ENNReal.ofReal p₀ * P.D (startsWith k u) := ENNReal.tsum_le_tsum fun u =>
        (pi_pair_le P hij _).trans (by gcongr; exact hA u)
    _ = ENNReal.ofReal p₀ * P.D (⋃ u, startsWith k u) := by
        rw [ENNReal.tsum_mul_left, measure_iUnion hdisj fun u =>
          (Set.to_countable _).measurableSet]
    _ ≤ ENNReal.ofReal p₀ * 1 := by gcongr; exact prob_le_one
    _ = _ := mul_one _

theorem card_lt_pairs (N : ℕ) :
    (((Finset.univ : Finset (Fin N × Fin N)).filter fun q => q.1 < q.2).card : ℝ)
      = N * (N - 1 : ℝ) / 2 := by
  classical
  set A := (Finset.univ : Finset (Fin N × Fin N)).filter fun q => q.1 < q.2
  set B := (Finset.univ : Finset (Fin N × Fin N)).filter fun q => q.2 < q.1
  set D := (Finset.univ : Finset (Fin N × Fin N)).filter fun q => q.1 = q.2
  have hAB : B.card = A.card :=
    Finset.card_equiv (Equiv.prodComm _ _) fun q => by simp [A, B]
  have hD : D.card = N := by
    have : D = (Finset.univ : Finset (Fin N)).image fun i => (i, i) := by
      ext q; simp only [D, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
      constructor
      · intro h; exact ⟨q.1, by ext <;> simp [h]⟩
      · rintro ⟨i, rfl⟩; rfl
    rw [this, Finset.card_image_of_injective _ (fun i j h => by simpa using congrArg Prod.fst h),
      Finset.card_univ, Fintype.card_fin]
  have htot : A.card + B.card + D.card = N * N := by
    have h1 : A ∪ B ∪ D = Finset.univ := by
      ext q; simp only [A, B, D, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and,
        iff_true]
      rcases lt_trichotomy q.1 q.2 with h | h | h
      · exact Or.inl (Or.inl h)
      · exact Or.inr h
      · exact Or.inl (Or.inr h)
    have hd1 : Disjoint A B :=
      Finset.disjoint_filter.2 fun q _ h1 h2 => absurd (h1.trans h2) (lt_irrefl _)
    have hd2 : Disjoint (A ∪ B) D := by
      rw [Finset.disjoint_union_left]
      exact ⟨Finset.disjoint_filter.2 fun q _ h1 h2 => by rw [h2] at h1; exact lt_irrefl _ h1,
        Finset.disjoint_filter.2 fun q _ h1 h2 => by rw [h2] at h1; exact lt_irrefl _ h1⟩
    have := congrArg Finset.card h1
    rw [Finset.card_union_of_disjoint hd2, Finset.card_union_of_disjoint hd1, Finset.card_univ,
      Fintype.card_prod, Fintype.card_fin] at this
    exact this
  rw [hAB, hD] at htot
  have : (A.card : ℝ) * 2 + N = N * N := by exact_mod_cast (by omega : A.card * 2 + N = N * N)
  linarith

omit [DecidableEq α] in
theorem pi_notDistinct (k : ℕ) (p₀ : ℝ)
    (hpre : ∀ u, P.D.real {x | prefixOf x k = u} ≤ p₀) (N : ℕ) :
    (Measure.pi fun _ : Fin N => P.D) {d | ¬ Distinct (List.ofFn d) k}
      ≤ ENNReal.ofReal (N * (N - 1 : ℝ) / 2 * p₀) := by
  classical
  set pairs := (Finset.univ : Finset (Fin N × Fin N)).filter fun q => q.1 < q.2
  have hsub : {d : Fin N → FreeMonoid α | ¬ Distinct (List.ofFn d) k}
      ⊆ ⋃ q ∈ pairs, {d | k ≤ (d q.1).toList.length ∧ k ≤ (d q.2).toList.length
        ∧ (d q.1).toList.take k = (d q.2).toList.take k} := by
    intro d hd
    simp only [Distinct, not_forall, Set.mem_ofPred_eq] at hd
    obtain ⟨i, j, x, y, hij, hx, hy, hlx, hly, hxy⟩ := hd
    have hxy := not_not.1 hxy
    rw [List.getElem?_ofFn] at hx hy
    have hi : i < N := by
      by_contra h; simp [h] at hx
    have hj : j < N := by
      by_contra h; simp [h] at hy
    simp only [hi, hj, Option.some.injEq, dite_true] at hx hy
    subst hx hy
    rcases lt_or_gt_of_ne hij with h | h
    · exact Set.mem_biUnion (x := (⟨i, hi⟩, ⟨j, hj⟩)) (by simp [pairs, h])
        ⟨hlx, hly, by simpa using hxy⟩
    · exact Set.mem_biUnion (x := (⟨j, hj⟩, ⟨i, hi⟩)) (by simp [pairs, h])
        ⟨hly, hlx, by simpa using hxy.symm⟩
  have hcard : (pairs.card : ℝ) = N * (N - 1 : ℝ) / 2 := card_lt_pairs N
  calc _ ≤ ∑ q ∈ pairs, (Measure.pi fun _ : Fin N => P.D) {d | k ≤ (d q.1).toList.length
          ∧ k ≤ (d q.2).toList.length ∧ (d q.1).toList.take k = (d q.2).toList.take k} :=
        (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
    _ ≤ ∑ _q ∈ pairs, ENNReal.ofReal p₀ := Finset.sum_le_sum fun q hq =>
        collide_le P k p₀ hpre (ne_of_lt (Finset.mem_filter.1 hq).2)
    _ = ENNReal.ofReal (pairs.card * p₀) := by
        rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]
    _ = _ := by rw [hcard]

end Collide

end Spur

end Random

end OrthoDFA

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw + κ U q`. -/
structure KappaEpsReadOracle {α : Type*} {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    {σ : Type*} (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw κ : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept}
    ≤ εw + κ * U (M.eval z.toList)

end Kappa

end Random

open MeasureTheory Random Random.Kappa Random.Spur RaceTail AdaptiveReads

lemma pairs_nonneg (B : ℕ) {p₀ : ℝ} (hp : 0 ≤ p₀) : 0 ≤ (B : ℝ) * (B - 1 : ℝ) / 2 * p₀ := by
  rcases Nat.eq_zero_or_pos B with h0 | h0
  · simp [h0]
  · have : (1 : ℝ) ≤ B := by exact_mod_cast h0
    have : (0 : ℝ) ≤ B - 1 := by linarith
    positivity

lemma pairs_min_le (B : ℕ) {p₀ : ℝ} (hp : 0 ≤ p₀) (x : ℝ) :
    (B : ℝ) * (B - 1 : ℝ) / 2 * p₀ * min 1 x ≤ (B : ℝ) * (B - 1 : ℝ) / 2 * p₀ :=
  mul_le_of_le_one_right (pairs_nonneg B hp) (min_le_left _ _)

/-- `Level5RoundCorrect` with an added floor `εw` on each read's wrong chance. -/
def Level5EpsRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (εw κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaEpsReadOracle μ M side U εw κ) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ) (f : ℕ) (t t' : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ + f
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (2 + Nat.clog 2 P.L))
    let η := p₀ * Real.log (16 * (nQ + 1) * K / δ)
    let w : ℕ := min (P.L + 1) (3 + Nat.clog 2 P.L) * (nQ + 1)
    let c : ℝ := εw * max w (nQ + 2) + κ
    let b : ℝ := C.probeBudget * w + (nQ + 2) * nα * (nQ + 2) * (nQ + 1)
    let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + (nQ + 2) * nα * (nQ + 1)
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → δ < 1 → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ εw → 0 ≤ κ → 0 ≤ t → 0 ≤ t' → t * c ≤ 1 - Real.exp (-(t' * c)) →
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
        * min 1 (Real.exp (t' * (εw * b + κ * m)) / (1 + t))
      + Real.exp (t' * (εw * b + κ * m)) / (1 + t) ^ (f + 2) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }
          ∨ nQ + 2 < result.hypothesis.tree.leaves.length} ∂μ
      ≤ ENNReal.ofReal δ

theorem level5EpsRoundCorrect_holds : Level5EpsRoundCorrect := by
  intro α _ _ σ _ M side U εw κ Ω _ μ _ R P C p₀ f t t'
  dsimp only
  intro hpre hδ hδ1 hy0 hyρ hρ1 hP hεw hκ ht ht' htc hfin
  classical
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set nQ := Fintype.card σ + f
  set nα := Fintype.card α
  set Rr := (nQ + 2) * nα
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
  set w : ℕ := min (P.L + 1) (3 + Nat.clog 2 P.L) * (nQ + 1)
  set c : ℝ := εw * ((max w (nQ + 2) : ℕ) : ℝ) + κ
  set b : ℝ := C.probeBudget * w + (nQ + 2) * nα * (nQ + 2) * (nQ + 1)
  set m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + (nQ + 2) * nα * (nQ + 1)
  have hp₀ : 0 < p₀ := (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  have hall := level3AllowanceCorrect_holds M U μ R.read R.measurable R.indep R.undecided P C p₀ f
  dsimp only at hall
  refine (lintegral_mono fun ω => measure_mono fun d hd => ?_).trans
    ((hall hpre hδ hδ1 hy0 hyρ hρ1 hP).trans ?_)
  · rcases hd with hd | hd
    · exact Or.inl hd
    · by_contra hc
      simp only [Set.mem_ofPred_eq, not_or] at hc
      have h0 : (initialState : Level3State α).tree.leaves.length ≤ nQ + 2 := by
        simp [initialState, newHypothesis, DTree.leaves]
      rw [indecisiveLStarLevel3_tree] at hd
      exact absurd (leaves_le_of_not_growsRun _ _ h0 hc.2) (not_le.2 hd)
  set π := Measure.pi fun _ : Fin C.probeBudget => P.D
  set X : ℝ := εw * b + κ * m
  set tail := Real.exp (t' * X) / (1 + t) ^ (f + 2)
  set tail₁ : ℝ := min 1 (Real.exp (t' * X) / (1 + t))
  have htail0 : 0 ≤ tail := by positivity
  have htail₁0 : 0 ≤ tail₁ := le_min zero_le_one (by positivity)
  -- no string read wrong starts like two probes
  set Cl : (Fin C.probeBudget → FreeMonoid α) → Ω → Prop := fun d ω =>
    ∀ z ∈ before (rounds6 C (List.ofFn d) nQ Rr) ((List.ofFn d).length * (Rr + 1)) (R.read · ω),
      Wrong M side (R.read · ω) z → Unshared (List.ofFn d) C.walkStart z
  -- `g` fresh-bad rounds, at a drawn point
  have hrace : ∀ d : Fin C.probeBudget → FreeMonoid α, π {d} ≠ 0 → ∀ g : ℕ,
      μ {ω | g ≤ badCount (rounds6 C (List.ofFn d) nQ Rr) (wrongSet M side)
          ((List.ofFn d).length * (Rr + 1)) (R.read · ω)}
        ≤ ENNReal.ofReal (Real.exp (t' * X) / (1 + t) ^ g) := by
    intro d hd g
    have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ P.L := by
      intro x hx
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      by_contra hlt
      apply hd
      have hpi : π {d} = ∏ j, P.D {d j} := by
        rw [← Measure.pi_pi]
        congr 1
        ext d'
        simp [funext_iff]
      rw [hpi]
      refine Finset.prod_eq_zero (Finset.mem_univ i) (measure_mono_null ?_
        (ae_iff.1 P.length_le))
      intro x hx
      rw [Set.mem_singleton_iff] at hx
      subst hx
      exact hlt
    set xs := List.ofFn d
    have hlen : xs.length = C.probeBudget := List.length_ofFn
    have hc0 : (0 : ℝ) ≤ c := by positivity
    refine raceTailBound_holds μ R.read (wrongSet M side) (fun _ => {ARU.undecided}) εw κ
      (rounds6 C xs nQ Rr) g (xs.length * (Rr + 1)) b m c t t'
      R.measurable R.indep hεw hκ (fun z => ?_) (fun j r r' _ h => rounds6_local le_rfl j r r' h)
      (fun r => ?_) (fun r => ?_) (fun r j _ => ?_) hc0 ht ht' htc
    · have h1 := R.wrong z
      have h2 := R.undecided z
      simp only [wrongSet, Set.mem_ofPred_eq, Set.mem_singleton_iff]
      rw [h2]; exact h1
    · obtain ⟨-, hrd⟩ := count6 (r := r) (C := C) (xs := xs) (n := nQ) (R := Rr) hL hy0 hyρ
        hρ1 hδ hδ1.le
      calc _ ≤ ∑ j ∈ Finset.range (xs.length * (Rr + 1)),
            ((rounds6 C xs nQ Rr j r).card : ℝ) := by
            refine (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
              fun _ _ _ => by positivity).trans (Finset.sum_le_sum fun j _ => ?_)
            exact_mod_cast Finset.card_le_card (Finset.filter_subset _ _)
        _ ≤ Mb6 nQ P.L nα xs.length := hrd
        _ = b := by
            simp only [Mb6, edgesMax, probeSifts, b, w, hlen]
            push_cast
            ring
    · obtain ⟨hu, -⟩ := count6 (r := r) (C := C) (xs := xs) (n := nQ) (R := Rr) (L := P.L) hL
        hy0 hyρ hρ1 hδ hδ1.le
      have hcost := sum_costAt_le C S hδ hδ1.le
      calc _ ≤ ∑ j ∈ Finset.range (xs.length * (Rr + 1)),
            (((rounds6 C xs nQ Rr j r).filter fun z => r z = .undecided).card : ℝ) := by
            refine (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
              fun _ _ _ => by positivity).trans (Finset.sum_le_sum fun j _ => ?_)
            refine Nat.cast_le.2 (Finset.card_le_card fun z hz => ?_)
            simp only [newQ, Finset.mem_filter, Set.mem_singleton_iff] at hz
            exact Finset.mem_filter.2 ⟨hz.1.1, hz.2⟩
        _ ≤ Mu6 C nQ nα := hu
        _ ≤ m := by
            simp only [Mu6, edgesMax, m]
            have : Level2.stretches nQ nα = S := rfl
            rw [this]
            push_cast
            linarith
    · obtain ⟨h1, h2⟩ := rounds6_card r C xs nQ Rr P.L hL j
      simp only [c]
      rw [show εw * ((max w (nQ + 2) : ℕ) : ℝ) + κ = εw * ((max w (nQ + 2) : ℕ) : ℝ) + κ * 1
        by ring]
      gcongr
      · exact_mod_cast (Finset.card_le_card (Finset.filter_subset _ _)).trans h1
      · norm_cast
        refine (Finset.card_le_card fun z hz => ?_).trans h2
        simp only [newQ, Finset.mem_filter, Set.mem_singleton_iff] at hz
        exact Finset.mem_filter.2 ⟨hz.1.1, hz.2⟩
  -- a sequence growing the tree past the allowance has a wrong read starting like two probes, or
  -- `f + 2` fresh-bad rounds
  have hpt : ∀ d : Fin C.probeBudget → FreeMonoid α, π {d} ≠ 0 →
      μ {ω | d ∈ {d | GrowsRun (R.read · ω) C nQ initialState (List.ofFn d)}}
        ≤ {d | ¬ Distinct (List.ofFn d) C.walkStart}.indicator 1 d * ENNReal.ofReal tail₁
          + ENNReal.ofReal tail := by
    intro d hd
    have hsplit : {ω | d ∈ {d | GrowsRun (R.read · ω) C nQ initialState (List.ofFn d)}}
        ⊆ {ω | ¬ Cl d ω} ∪ {ω | Cl d ω ∧ GrowsRun (R.read · ω) C nQ initialState (List.ofFn d)} :=
      fun ω hω => by
        by_cases h : Cl d ω
        · exact Or.inr ⟨h, hω⟩
        · exact Or.inl h
    refine (measure_mono hsplit).trans ((measure_union_le _ _).trans (add_le_add ?_ ?_))
    · by_cases hdist : Distinct (List.ofFn d) C.walkStart
      · have : {ω | ¬ Cl d ω} = ∅ := Set.eq_empty_of_forall_notMem fun ω h =>
          h fun z _ _ => unshared_of_distinct hdist z
        rw [this, measure_empty]
        exact bot_le
      rw [Set.indicator_of_mem (show d ∈ {d | ¬ Distinct (List.ofFn d) C.walkStart} from hdist),
        Pi.one_apply, one_mul, ENNReal.ofReal_min, ENNReal.ofReal_one]
      refine le_min prob_le_one ?_
      refine (measure_mono fun ω hω => ?_).trans ((hrace d hd 1).trans (by rw [pow_one]))
      simp only [Cl, Set.mem_setOf_eq, not_forall] at hω ⊢
      obtain ⟨z, hz, hzw, -⟩ := hω
      exact one_le_badCount _ _ hz hzw
    · refine (measure_mono fun ω hω => ?_).trans (hrace d hd (f + 2))
      have := grows_badCount (R := Rr) M side hω.1 le_rfl hω.2
      have hnQ : nQ = Fintype.card σ + f := rfl
      simp only [Set.mem_setOf_eq]
      omega
  have hG : ∫⁻ ω, π {d | GrowsRun (R.read · ω) C nQ initialState (List.ofFn d)} ∂μ
      ≤ ENNReal.ofReal (C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀) * ENNReal.ofReal tail₁
        + ENNReal.ofReal tail := by
    refine (Eps.lintegral_le_tsum_of_points μ π _ _ hpt).trans ?_
    have e : ∑' d, π {d} * ({d | ¬ Distinct (List.ofFn d) C.walkStart}.indicator 1 d
          * ENNReal.ofReal tail₁ + ENNReal.ofReal tail)
        = π {d | ¬ Distinct (List.ofFn d) C.walkStart} * ENNReal.ofReal tail₁
          + ENNReal.ofReal tail := by
      simp_rw [mul_add, ← mul_assoc]
      rw [ENNReal.tsum_add, ENNReal.tsum_mul_right, ENNReal.tsum_mul_right,
        ← Eps.measure_eq_tsum π]
      congr 1
      have h1 : ∑' d, π {d} = 1 := by
        rw [← measure_univ (μ := π), Eps.measure_eq_tsum π Set.univ]
        simp
      rw [h1, one_mul]
    rw [e]
    gcongr
    exact pi_notDistinct P C.walkStart p₀ hpre _
  calc _ ≤ ENNReal.ofReal (7 * C.δ / 8) + (ENNReal.ofReal
        (C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀) * ENNReal.ofReal tail₁
          + ENNReal.ofReal tail) := by
        gcongr
    _ ≤ ENNReal.ofReal (7 * C.δ / 8) + ENNReal.ofReal (C.δ / 8) := by
        gcongr
        have hpair := pairs_nonneg C.probeBudget hp₀.le
        rw [← ENNReal.ofReal_mul hpair, ← ENNReal.ofReal_add (by positivity) htail0]
        exact ENNReal.ofReal_le_ofReal hfin
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1; ring

theorem level5RoundCorrect_holds : Level5RoundCorrect := by
  intro α _ _ σ _ M side U κ Ω _ μ _ R P C p₀ f t t'
  dsimp only
  intro hpre hδ hδ1 hy0 hyρ hρ1 hP hκ ht ht' htc hfin
  exact level5EpsRoundCorrect_holds M side U 0 κ μ
    ⟨R.read, R.measurable, R.indep, R.undecided, fun z => by simpa using R.wrong z⟩ P C p₀ f t t'
    hpre hδ hδ1 hy0 hyρ hρ1 hP le_rfl hκ ht ht' (by simpa using htc) (by simpa using hfin)

end OrthoDFA
