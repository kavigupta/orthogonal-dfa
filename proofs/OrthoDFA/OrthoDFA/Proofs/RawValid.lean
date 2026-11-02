import OrthoDFA.Proofs.RawSides

/-!
# Whatever the loop returns is valid, against each tame population's own law

At each assignment of the hash the stream-level validity holds against the reserve side's law.
Where the hash splits a tame population evenly, and splits the family's sets evenly too, that
law's verdict is the population's own up to `εs/2`; the hash fails to with chance linear in
`ρ`, which the collision cap pays for.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]
variable {η₀ indecisionLimit crossLimit εs : ℝ} {hsz : ℕ}

lemma log_ladder_le {N : ℕ} (hN : 2 ^ 19 ≤ N) : 2048 * (3 + 2 * Nat.log 2 N) ≤ N := by
  have hk : 19 ≤ Nat.log 2 N := Nat.le_log_of_pow_le (by norm_num) hN
  have hpow : 2 ^ Nat.log 2 N ≤ N := Nat.pow_log_le_self 2 (by omega)
  have key : ∀ k, 19 ≤ k → 2048 * (3 + 2 * k) ≤ 2 ^ k := by
    intro k hk
    induction k, hk using Nat.le_induction with
    | base => norm_num
    | succ k hk ih =>
      have : 4096 ≤ 2 ^ k := le_trans (by norm_num) (Nat.pow_le_pow_right (by norm_num) hk)
      rw [pow_succ]
      omega
  exact le_trans (key _ hk) hpow

open scoped Classical in
lemma card_stoppable_le (populations : Finset J) (εcov δ α pAP ρ : ℝ) :
    (stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ).card
      ≤ Nat.log 2 (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
        + 1 := by
  classical
  refine le_trans (Finset.card_filter_le _ _) ?_
  rw [schedule]
  refine le_trans Finset.card_image_le ?_
  rw [Finset.card_range, ladderLen]

open scoped Classical in
lemma npref_le_of_mem_stoppable {populations : Finset J} {εcov δ α pAP ρ : ℝ} {B : State}
    (hB : B ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ) :
    B.npref ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
  classical
  have hs := (Finset.mem_filter.1 hB).1
  rw [schedule] at hs
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hs
  exact Nat.div_le_self _ _

set_option maxHeartbeats 1000000 in
/-- What the solved count buys the conversion: the hash's tails, over every rung and tame
population, and its windows' mass, are small against `δ` and `e = min εs il / 8`. -/
lemma solved_conversion (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit) (hind1 : indecisionLimit ≤ 1 / 2) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) (hρ0 : 0 < ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    2 ^ 19 ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
    ∧ (populations.card : ℝ) * Real.exp (-((prefCount η₀ populations indecisionLimit εcov εs δ α
        pAP crossLimit hsz : ℝ) / 2)) ≤ δ / 32
    ∧ (populations.card : ℝ) * (3 + 2 * (Nat.log 2 (prefCount η₀ populations indecisionLimit
        εcov εs δ α pAP crossLimit hsz) : ℝ))
        * (2 * Real.exp (-(min εs indecisionLimit / 8 / 2) ^ 2 / (2 * ρ))) ≤ δ / 32
    ∧ (populations.card : ℝ) * (4 * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP
        crossLimit hsz : ℝ)) * ρ ≤ min εs indecisionLimit / 8 / 4
    ∧ (hsz : ℝ) * ρ ≤ min εs indecisionLimit / 8 / 4 := by
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set e : ℝ := min εs indecisionLimit / 8 with hedef
  set P : ℝ := (populations.card : ℝ) with hP
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔdef
  set L : ℝ := Real.log (128 * P * ((M : ℝ) + 1) / δ) with hLdef
  have hcb0 : 0 < cb := cutBudget_pos η₀ hη₀ hε hind
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hη₀ hε hind
  have hcb16 : cb ≤ 1 / 16 := by
    have h := (cutBudget_le η₀ indecisionLimit εcov).2.1
    rw [sig] at h
    linarith
  have hΔle : Δ ≤ 1 / 2048 := by
    have hs := sig_pos η₀ hη₀
    have hs2 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    have hβ : shiftScale indecisionLimit εcov ≤ 1 / 2 :=
      (min_le_left _ _).trans (min_le_right _ _)
    have hβ0 := shiftScale_pos hε hind
    rw [hΔdef, shiftBudget]
    have : sig η₀ ^ 2 ≤ 1 / 4 := by nlinarith
    nlinarith
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hNR0 : (0 : ℝ) ≤ (N : ℝ) := Nat.cast_nonneg _
  have hL4 : 4 ≤ L := by
    rw [hLdef, Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    have h4 : Real.exp 4 = Real.exp 1 ^ 4 := by
      rw [← Real.exp_nat_mul]; norm_num
    rw [h4, le_div_iff₀ hδ]
    have he0 : 0 < Real.exp 1 := Real.exp_pos 1
    have h81 : Real.exp 1 ^ 4 ≤ 81 := by
      have h3 : Real.exp 1 ≤ 3 := by linarith
      calc Real.exp 1 ^ 4 ≤ 3 ^ 4 := pow_le_pow_left₀ he0.le h3 4
        _ = 81 := by norm_num
    nlinarith
  have c1 : 64 * L / Δ ≤ (N : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * L / Δ⌉₊ ≤ N := by
      rw [hNdef, prefCount, ← hMdef, ← hΔdef, ← hP, ← hLdef]; omega
    exact_mod_cast hle
  have c3 : 128 * (hsz : ℝ) / cb ≤ (N : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈128 * (hsz : ℝ) / cb⌉₊ ≤ N := by
      rw [hNdef, prefCount, ← hcb]; omega
    exact_mod_cast hle
  rw [div_le_iff₀ (by positivity)] at c1 c3
  have hNL : 131072 * L ≤ (N : ℝ) := by nlinarith
  have hN20 : (2 : ℝ) ^ 19 ≤ N := by nlinarith
  have hlogR : 2048 * (3 + 2 * (Nat.log 2 N : ℝ)) ≤ N := by
    have hN20' : 2 ^ 19 ≤ N := by exact_mod_cast hN20
    exact_mod_cast log_ladder_le hN20'
  have hhszN : (hsz : ℝ) ≤ N := by nlinarith
  have hNpos : (0 : ℝ) < N := by linarith
  -- the collision cap against `e`
  have he0 : 0 < e := div_pos (lt_min hεs hind) (by norm_num)
  have hwin : ρ * (128 * P * N) ≤ min εs indecisionLimit := by
    have h := le_trans hρcap (le_trans (min_le_left _ _) (min_le_right _ _))
    rw [← hP, ← hNdef, le_div_iff₀ (by positivity)] at h
    exact h
  set Lg : ℝ := Real.log (64 * P * (3 + 2 * (Nat.log 2 N : ℝ)) / δ) with hLg
  have hLg0 : 0 < Lg := by
    rw [hLg]
    refine Real.log_pos ?_
    rw [lt_div_iff₀ hδ]
    have : (0 : ℝ) ≤ (Nat.log 2 N : ℝ) := Nat.cast_nonneg _
    nlinarith
  have hhash : ρ * (512 * Lg) ≤ min εs indecisionLimit ^ 2 := by
    have h := le_trans hρcap (min_le_right _ _)
    rw [← hP, ← hNdef, ← hLg, le_div_iff₀ (by positivity)] at h
    exact h
  have hmin : min εs indecisionLimit = 8 * e := by rw [hedef]; ring
  rw [hmin] at hwin hhash
  clear hmin
  refine ⟨by exact_mod_cast hN20, ?_, ?_, ?_, ?_⟩
  · -- the fill's tail
    have hL : L ≤ (N : ℝ) / 2 := by linarith
    have hexp : Real.exp (-((N : ℝ) / 2)) ≤ Real.exp (-L) := Real.exp_le_exp.2 (by linarith)
    have hexpL : Real.exp (-L) = δ / (128 * P * ((M : ℝ) + 1)) := by
      rw [hLdef, Real.exp_neg, Real.exp_log (by positivity)]
      field_simp
    rw [hexpL] at hexp
    have : P * (δ / (128 * P * ((M : ℝ) + 1))) ≤ δ / 32 := by
      rw [show P * (δ / (128 * P * ((M : ℝ) + 1))) = δ / (128 * ((M : ℝ) + 1)) by
        field_simp]
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith
    calc P * Real.exp (-((N : ℝ) / 2)) ≤ P * (δ / (128 * P * ((M : ℝ) + 1))) :=
          mul_le_mul_of_nonneg_left hexp (by linarith)
      _ ≤ δ / 32 := this
  · -- the hash's tails over every rung
    have hlog0 : (0 : ℝ) < 3 + 2 * (Nat.log 2 N : ℝ) := by positivity
    have hLge : Lg ≤ (e / 2) ^ 2 / (2 * ρ) := by
      rw [le_div_iff₀ (by positivity)]
      nlinarith
    have hexp : Real.exp (-(e / 2) ^ 2 / (2 * ρ)) ≤ Real.exp (-Lg) :=
      Real.exp_le_exp.2 (by rw [neg_div]; linarith)
    have hexpL : Real.exp (-Lg) = δ / (64 * P * (3 + 2 * (Nat.log 2 N : ℝ))) := by
      rw [hLg, Real.exp_neg, Real.exp_log (by positivity)]
      field_simp
    rw [hexpL] at hexp
    calc P * (3 + 2 * (Nat.log 2 N : ℝ)) * (2 * Real.exp (-(e / 2) ^ 2 / (2 * ρ)))
        ≤ P * (3 + 2 * (Nat.log 2 N : ℝ)) * (2 * (δ / (64 * P * (3 + 2 * (Nat.log 2 N : ℝ))))) :=
          by gcongr
      _ = δ / 32 := by field_simp; ring
  · -- the windows' mass
    have e1 : ρ * (128 * P * N) = 128 * (P * (4 * (N : ℝ)) * ρ) / 4 := by ring
    rw [e1] at hwin
    linarith
  · -- the harvested strings' mass
    have h1 : ρ * N ≤ 8 * e / (128 * P) := by
      rw [le_div_iff₀ (by positivity)]; linarith
    have h2 : 8 * e / (128 * P) ≤ e / 4 := by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]; nlinarith
    calc (hsz : ℝ) * ρ ≤ N * ρ := mul_le_mul_of_nonneg_right hhszN hρ0.le
      _ ≤ e / 4 := by linarith

open scoped Classical in
/-- `D` on `A`, or `D` itself when `A` is null. -/
noncomputable def condSide (D : Measure S) (A : Set S) : Measure S :=
  if D A = 0 then D else D[|A]

instance (D : Measure S) [IsProbabilityMeasure D] (A : Set S) :
    IsProbabilityMeasure (condSide D A) := by
  classical
  unfold condSide
  split_ifs with h
  · infer_instance
  · exact cond_isProbabilityMeasure h

open scoped Classical in
/-- Population `j`'s table-side law under the hash `g`. -/
noncomputable def tableLaw (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool)
    (j : J) : Measure S :=
  if j = uni then D j else condSide (D j) (onSide (allHarvest H) true g)

open scoped Classical in
/-- Population `j`'s certification law under the hash `g`. -/
noncomputable def certLaw (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool)
    (j : J) : Measure S :=
  if j = uni then D j else condSide (D j) (onSide (allHarvest H) false g)

instance (H : J → Finset S) (uni : J) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (g : S → Bool) (j : J) : IsProbabilityMeasure (tableLaw H uni D g j) := by
  classical
  unfold tableLaw; split_ifs <;> infer_instance

instance (H : J → Finset S) (uni : J) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (g : S → Bool) (j : J) : IsProbabilityMeasure (certLaw H uni D g j) := by
  classical
  unfold certLaw; split_ifs <;> infer_instance

lemma measurable_cutMass (O : Oracle μ S) (D : Measure S) [IsProbabilityMeasure D] (lo hi : ℕ)
    (A : Finset S) : Measurable (fun ω => D.real {p | cutCorrect O lo hi A p ω}) := by
  have h : (fun ω => D.real {p | cutCorrect O lo hi A p ω})
      = fun ω => 1 - D.real {p | ¬ cutCorrect O lo hi A p ω} := by
    funext ω
    rw [show {p : S | cutCorrect O lo hi A p ω} = {p | ¬ cutCorrect O lo hi A p ω}ᶜ by
      ext; simp, measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
  rw [h]
  exact measurable_const.sub (measurable_badMassReal O D lo hi A)

open scoped Classical in
/-- What the loop is held to at a tame population: its cut wrong on at most its gate's limit,
twice the redraw fraction and, read through one side of the hash, `εs/2`, and at most
`2·indecisionLimit` of it undecided. -/
def goodD (O : Oracle μ S) (populations : Finset J) (uni : J) (D : J → Measure S)
    (εcov εs c indecisionLimit : ℝ) (y : Run Ω S J) (B : State) (j : J) : Prop :=
  1 - gateLimit uni εcov j - 2 * c - (if j = uni then 0 else εs / 2)
      ≤ (D j).real {p | cutCorrect O B.lo (B.hi + 1) (familyBy rule O.mq populations y B) p
          (oracleNoise y)}
    ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations y B) p
        (oracleNoise y)} ≤ 2 * indecisionLimit

open scoped Classical in
lemma measurableSet_good (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)] (εcov εs c indecisionLimit : ℝ)
    (B : State) :
    MeasurableSet {y : Run Ω S J | ∀ j ∈ tame,
      goodD (rule := rule) O populations uni D εcov εs c indecisionLimit y B j} := by
  classical
  refine measurableSet_family_noise O populations B
    (fun A ω => ∀ j ∈ tame, (1 - gateLimit uni εcov j - 2 * c
        - (if j = uni then 0 else εs / 2)
        ≤ (D j).real {p | cutCorrect O B.lo (B.hi + 1) (insert 1 A) p ω})
      ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A) p ω}
        ≤ 2 * indecisionLimit) (fun A => ?_)
  have : {ω | ∀ j ∈ tame, (1 - gateLimit uni εcov j - 2 * c
        - (if j = uni then 0 else εs / 2)
        ≤ (D j).real {p | cutCorrect O B.lo (B.hi + 1) (insert 1 A) p ω})
      ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A) p ω}
        ≤ 2 * indecisionLimit}
      = ⋂ j ∈ tame, ({ω | 1 - gateLimit uni εcov j - 2 * c
          - (if j = uni then 0 else εs / 2)
          ≤ (D j).real {p | cutCorrect O B.lo (B.hi + 1) (insert 1 A) p ω}}
        ∩ {ω | (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A) p ω}
          ≤ 2 * indecisionLimit}) := by
    ext ω; simp
  rw [this]
  exact Finset.measurableSet_biInter _ (fun j _ =>
    (measurableSet_le measurable_const (measurable_cutMass O (D j) _ _ _)).inter
      (measurableSet_le (measurable_undecidedMassReal O (D j) _ _ _) measurable_const))

open scoped Classical in
set_option maxHeartbeats 8000000 in
/-- Whatever the loop returns, at whatever rung, is good on every tame population, except with
chance `δ/2 + α`. -/
theorem raw_valid {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations tame : Finset J) (uni : J) (huni : uni ∈ tame) (htame : tame ⊆ populations)
    (H : J → Finset S) (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (εcov α δ ρ pAP : ℝ) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hind : 0 < indecisionLimit) (hind1 : indecisionLimit ≤ 1 / 2) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (hεcov : 0 < εcov) (hεs : 0 < εs) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP0 : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
      (allHarvest H).card)
    (hρsf : collisionMass Dsf
      ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (runMeasure μ D Dsf).real (streams H uni ⁻¹' ⋃ B : {B : State // B ∈ stoppable η₀
        populations indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card ρ},
      (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
        ∩ {y | ¬ ∀ j ∈ tame, goodD (rule := rule) O populations uni D εcov εs
          (rejFrac η₀ indecisionLimit εcov εs) indecisionLimit y B.val j}))
      ≤ δ / 2 + α := by
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  have hcard : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hρpos : 0 < ρ := pos_of_atoms (D uni) (hρ uni huni)
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  set hz : ℕ := (allHarvest H).card with hhz
  obtain ⟨hN20, hfillT, hhashT, hwinM, hhszM⟩ := solved_conversion (crossLimit := crossLimit)
    (hsz := hz) (α := α) (pAP := pAP) η₀ populations hη0 hη₀ hεcov hεs hind hind1 hδ hδ1 hcard
    hρpos hρcap
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz with hNdef
  set e : ℝ := min εs indecisionLimit / 8 with hedef
  set K : ℕ := 4 * N with hK
  set Hs : Finset S := allHarvest H with hHs
  set st := stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz ρ with hst
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hc
  have hc0 : 0 < c := rejFrac_pos η₀ hη₀ hεcov hεs hind
  have hcle : c ≤ indecisionLimit / 256 := by
    have h1 := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀ (εcov := εcov)).2.1
    have hs2 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    have := mul_le_mul_of_nonneg_right hs2 hind.le
    linarith
  have he0 : 0 < e := div_pos (lt_min hεs hind) (by norm_num)
  have heε : e ≤ εs / 8 := by rw [hedef]; linarith [min_le_left εs indecisionLimit]
  have heil : e ≤ indecisionLimit / 8 := by
    rw [hedef]; linarith [min_le_right εs indecisionLimit]
  have he8 : e ≤ 1 / 8 := by linarith
  have hNpos : 0 < N := lt_of_lt_of_le (pow_pos (by norm_num) 19) hN20
  clear hN20
  set Y := streams (Ω := Ω) H uni with hY
  -- the harvested strings' mass
  have hHsM : ∀ j ∈ tame, (D j).real ↑Hs ≤ e / 4 := fun j hj =>
    le_trans (measureReal_window_le (D j) (hρ j hj) Hs) hhszM
  -- the events
  set T : (S → Bool) → Set (Run Ω S J) := fun g => ⋃ B : {B : State // B ∈ st},
    (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
      ∩ FailAt rule O populations tame uni (certLaw H uni D g) εcov c (27 / 16 * indecisionLimit)
        B.val) with hT
  set BalBad : Set (S → Bool) := ⋃ j ∈ tame, sideBad (D j) Hs e with hBalBad
  set Bad : Set (Draws Ω S J) := {x | ¬ fillOK H uni populations K N x}
    ∪ ⋃ B ∈ st, ⋃ j ∈ tame, ⋃ b : Bool,
      imbSet (rule := rule) O populations H uni (D j) K N e b B with hBad
  set E : Set (Draws Ω S J) := Y ⁻¹' ⋃ B : {B : State // B ∈ st},
      (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
        ∩ {y | ¬ ∀ j ∈ tame, goodD (rule := rule) O populations uni D εcov εs c indecisionLimit y
          B.val j}) with hE
  -- at each even hash, the stream-level validity against the reserve side's law
  have hslice : ∀ g ∉ BalBad,
      (streamMeasure μ (sideLawT H uni D g) (sideLawC H uni D g) Dsf).real (T g)
        ≤ 7 * δ / 16 + α := by
    intro g hg
    have hbal : ∀ j ∈ tame,
        |(D j).real (onSide Hs true g) - (D j).real (onSide Hs false g)| ≤ e := by
      intro j hj
      by_contra hc'
      exact hg (Set.mem_biUnion hj (not_le.1 hc'))
    have hthird : ∀ j ∈ tame, 1 / 3 ≤ (D j).real (onSide Hs true g)
        ∧ 1 / 3 ≤ (D j).real (onSide Hs false g) := fun j hj =>
      side_third (D j) Hs g he8 (by linarith [hHsM j hj]) (hbal j hj)
    have hA0 : ∀ j ∈ tame, D j (onSide Hs true g) ≠ 0 := fun j hj h => by
      have := (hthird j hj).1
      rw [measureReal_def, h] at this; norm_num at this
    have hR0 : ∀ j ∈ tame, D j (onSide Hs false g) ≠ 0 := fun j hj h => by
      have := (hthird j hj).2
      rw [measureReal_def, h] at this; norm_num at this
    have hcoll := poolDraws_le_of_cap (indecisionLimit := indecisionLimit)
        (crossLimit := crossLimit) η₀ populations hpAP0 hpAP1 hδ
        (tsum_nonneg (fun a => sq_nonneg _)) hρsf
    have h := validity_of_returned (rule := rule) (crossLimit := crossLimit) (hsz := hz) hflat O
      populations tame uni huni htame (sideLawT H uni D g) (sideLawC H uni D g) Dsf
      (fun j => if j = uni then [] else (H j).toList) (tableLaw H uni D g) (certLaw H uni D g)
      (fun j _ => by split_ifs; exact List.nodup_nil; exact Finset.nodup_toList _)
      (fun j hj => by
        by_cases hju : j = uni
        · subst hju
          simp only [if_true, tableLaw, sideLawT_uni, prependMeasure_nil]
        · rw [tableLaw, if_neg hju, if_neg hju, condSide,
            if_neg (show ¬ (D j (onSide (allHarvest H) true g) = 0) from hA0 j hj)]
          exact sideLawT_named H uni D g hju (hA0 j hj))
      (fun j hj => by
        by_cases hju : j = uni
        · subst hju
          simp only [certLaw, if_true, sideLawC_uni]
        · rw [certLaw, if_neg hju, condSide,
            if_neg (show ¬ (D j (onSide (allHarvest H) false g) = 0) from hR0 j hj)]
          exact sideLawC_named H uni D g hju (hR0 j hj))
      (ae_sideLawT_mem H uni D g populations hsupp hH)
      (ae_sideLawC_mem H uni D g populations hsupp) hsuppSf indecisionLimit εcov α
      (27 / 16 * indecisionLimit) hind (by linarith) hηle hη₀ hα0 hα1 ρ (collisionMass Dsf)
      (fun j hj t => by
        by_cases hju : j = uni
        · subst hju
          simp only [certLaw, tableLaw, if_true]
          exact ⟨by linarith [hρ j hj t], by linarith [hρ j hj t]⟩
        · rw [certLaw, tableLaw, if_neg hju, if_neg hju, condSide, condSide,
            if_neg (show ¬ (D j (onSide (allHarvest H) true g) = 0) from hA0 j hj),
            if_neg (show ¬ (D j (onSide (allHarvest H) false g) = 0) from hR0 j hj)]
          exact ⟨cond_atom_le (D j) (hρ j hj) _ (hthird j hj).2 t,
            cond_atom_le (D j) (hρ j hj) _ (hthird j hj).1 t⟩)
      le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hεcov hεs δ hδ hδ1 pAP hpAP0 hpAP1 hpAPBound
      hcoll
    exact h
  -- off the bad events, a family bad for a population's law is bad for its reserve side's
  have hconv : ∀ x : Draws Ω S J, x.2 ∉ BalBad → x ∉ Bad → x ∈ E → Y x ∈ T x.2 := by
    intro x hg hx hxE
    simp only [hBad, Set.mem_union, Set.mem_iUnion, Set.mem_ofPred_eq, not_or, not_exists,
      not_not] at hx
    obtain ⟨hfill, himb⟩ := hx
    obtain ⟨B, hret, hbad⟩ := Set.mem_iUnion.1 hxE
    refine Set.mem_iUnion.2 ⟨B, hret, ?_⟩
    by_contra hnf
    apply hbad
    simp only [FailAt, Set.mem_ofPred_eq, not_not] at hnf
    intro j hj
    obtain ⟨hcutc, hundc⟩ := hnf j hj
    have hB : B.val.npref ≤ N := npref_le_of_mem_stoppable B.2
    set F := clusterBy rule O.mq populations (Y x) B.val with hF
    have hF1 : (1 : S) ∉ F := Finset.notMem_erase _ _
    have hmono : (D j).real {p | cutCorrect O B.val.lo B.val.hi F p (oracleNoise (Y x))}
        ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
          (familyBy rule O.mq populations (Y x) B.val) p (oracleNoise (Y x))} :=
      measureReal_mono (fun p hp => cutCorrect_insert_one O _ _ _ hF1 p _ hp)
        (measure_ne_top _ _)
    by_cases hju : j = uni
    · subst hju
      simp only [certLaw, if_true] at hcutc hundc
      refine ⟨?_, by linarith⟩
      simp only [if_true, sub_zero]
      linarith
    · -- the conversion
      have hbalj : |(D j).real (onSide Hs true x.2) - (D j).real (onSide Hs false x.2)| ≤ e := by
        by_contra hc'
        exact hg (Set.mem_biUnion hj (not_le.1 hc'))
      have hthird := side_third (D j) Hs x.2 he8 (by linarith [hHsM j hj]) hbalj
      have hR0 : D j (onSide Hs false x.2) ≠ 0 := fun h => by
        have := hthird.2
        rw [measureReal_def, h] at this; norm_num at this
      have hDc : certLaw H uni D x.2 j = (D j)[|onSide Hs false x.2] := by
        rw [certLaw, if_neg hju, condSide,
          if_neg (show ¬ (D j (onSide (allHarvest H) false x.2) = 0) from hR0)]
      rw [hDc] at hcutc hundc
      have : IsProbabilityMeasure ((D j)[|onSide Hs false x.2]) := cond_isProbabilityMeasure hR0
      have hfam : ∀ b, famSetF (rule := rule) O populations H uni K N b B.val x
          = famSet (rule := rule) O populations b (Y x) B.val := fun b => by
        unfold famSetF
        rw [if_pos (by exact hfill)]
      have himbj : ∀ b, |(D j).real (famSet (rule := rule) O populations b (Y x) B.val
            ∩ onSide Hs true x.2)
          - (D j).real (famSet (rule := rule) O populations b (Y x) B.val
            ∩ onSide Hs false x.2)| ≤ e := by
        intro b
        have := himb B.val B.2 j hj b
        simp only [imbSet, Set.mem_ofPred_eq, not_lt, hfam] at this
        exact this
      have hHj : (D j).real ↑Hs ≤ 1 / 4 := by linarith [hHsM j hj]
      have hcutW := le_cond_add (D j) Hs x.2 (famSet (rule := rule) O populations true (Y x)
        B.val) he0.le he8 hHj (himbj true) hbalj
      have hundW := le_cond_add (D j) Hs x.2 (famSet (rule := rule) O populations false (Y x)
        B.val) he0.le he8 hHj (himbj false) hbalj
      simp only [famSet, if_true, Bool.false_eq_true, if_false] at hcutW hundW
      have hcompl : ∀ (ν : Measure S) [IsProbabilityMeasure ν],
          ν.real {p | ¬ cutCorrect O B.val.lo B.val.hi F p (oracleNoise (Y x))}
            = 1 - ν.real {p | cutCorrect O B.val.lo B.val.hi F p (oracleNoise (Y x))} := by
        intro ν _
        rw [show {p : S | ¬ cutCorrect O B.val.lo B.val.hi F p (oracleNoise (Y x))}
            = {p | cutCorrect O B.val.lo B.val.hi F p (oracleNoise (Y x))}ᶜ by ext; simp,
          measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
      rw [hcompl, hcompl] at hcutW
      have hlim : gateLimit uni εcov j = 1 / 2 := by simp [gateLimit, hju]
      rw [hlim] at hcutc
      refine ⟨?_, ?_⟩
      · simp only [hju, if_false, hlim]
        linarith [hHsM j hj]
      · linarith [hHsM j hj]
  -- measurability
  have hEm : MeasurableSet E := by
    refine measurable_streams H uni (MeasurableSet.iUnion (fun B => ?_))
    exact (measurableSet_retBy O populations uni η₀ indecisionLimit εcov α B.val).inter
      (measurableSet_good (rule := rule) O populations tame uni D εcov εs c indecisionLimit
        B.val).compl
  have hBadm : MeasurableSet Bad :=
    (measurableSet_fillOK populations H uni K N).compl.union
      (Finset.measurableSet_biUnion _ (fun B _ => Finset.measurableSet_biUnion _ (fun j _ =>
        MeasurableSet.iUnion (fun b => measurableSet_imbSet O populations H uni (D j) K N e b B))))
  have hBalm : MeasurableSet BalBad :=
    Finset.measurableSet_biUnion _ (fun j _ => measurableSet_sideBad (D j) Hs e)
  -- at each hash, the bad draws
  have key : ∀ g, baseRun μ D Dsf {r | (r, g) ∈ E}
      ≤ BalBad.indicator 1 g + ENNReal.ofReal (7 * δ / 16 + α)
        + baseRun μ D Dsf {r | (r, g) ∈ Bad} := by
    intro g
    by_cases hg : g ∈ BalBad
    · rw [Set.indicator_of_mem hg, Pi.one_apply, add_assoc]
      exact le_trans prob_le_one le_self_add
    · rw [Set.indicator_of_notMem hg, zero_add]
      have hsub : {r | (r, g) ∈ E} ⊆ (fun r => Y (r, g)) ⁻¹' T g ∪ {r | (r, g) ∈ Bad} := by
        intro r hr
        by_cases hb : (r, g) ∈ Bad
        · exact Or.inr hb
        · exact Or.inl (hconv (r, g) hg hb hr)
      refine le_trans (measure_mono hsub)
        (le_trans (measure_union_le _ _) (add_le_add ?_ le_rfl))
      have hm : Measurable (fun r : Run Ω S J => Y (r, g)) :=
        (measurable_streams H uni).comp (measurable_id.prodMk measurable_const)
      refine le_trans (Measure.le_map_apply hm.aemeasurable _) ?_
      rw [map_streams H uni D Dsf g, ← ofReal_measureReal]
      exact ENNReal.ofReal_le_ofReal (hslice g hg)
  have hint : runMeasure μ D Dsf E
      ≤ coins S BalBad + ENNReal.ofReal (7 * δ / 16 + α) + runMeasure μ D Dsf Bad := by
    rw [runMeasure_eq_prod, Measure.prod_apply_symm hEm, Measure.prod_apply_symm hBadm]
    refine le_trans (lintegral_mono key) (le_of_eq ?_)
    have hf : Measurable (fun g : S → Bool => BalBad.indicator 1 g
        + ENNReal.ofReal (7 * δ / 16 + α)) :=
      (measurable_one.indicator hBalm).add measurable_const
    have e1 : ∫⁻ g, (BalBad.indicator 1 g + ENNReal.ofReal (7 * δ / 16 + α)
          + baseRun μ D Dsf {r | (r, g) ∈ Bad}) ∂(coins S)
        = ∫⁻ g, (BalBad.indicator 1 g + ENNReal.ofReal (7 * δ / 16 + α)) ∂(coins S)
          + ∫⁻ g, baseRun μ D Dsf {r | (r, g) ∈ Bad} ∂(coins S) :=
      lintegral_add_left hf _
    have e2 : ∫⁻ g, (BalBad.indicator 1 g + ENNReal.ofReal (7 * δ / 16 + α)) ∂(coins S)
        = ∫⁻ g, BalBad.indicator 1 g ∂(coins S)
          + ∫⁻ _g, ENNReal.ofReal (7 * δ / 16 + α) ∂(coins S) :=
      lintegral_add_left (measurable_one.indicator hBalm) _
    rw [e1, e2, lintegral_indicator_one hBalm, lintegral_const, measure_univ, mul_one]
    rfl
  have hreal : (runMeasure μ D Dsf).real E
      ≤ (coins S).real BalBad + (7 * δ / 16 + α) + (runMeasure μ D Dsf).real Bad := by
    have h1 : coins S BalBad + ENNReal.ofReal (7 * δ / 16 + α) + runMeasure μ D Dsf Bad
        = ENNReal.ofReal ((coins S).real BalBad + (7 * δ / 16 + α)
          + (runMeasure μ D Dsf).real Bad) := by
      rw [ENNReal.ofReal_add (by positivity) measureReal_nonneg,
        ENNReal.ofReal_add measureReal_nonneg (by positivity),
        ofReal_measureReal (measure_ne_top _ _), ofReal_measureReal (measure_ne_top _ _)]
    have h2 : 0 ≤ (coins S).real BalBad + (7 * δ / 16 + α) + (runMeasure μ D Dsf).real Bad := by
      positivity
    exact ENNReal.toReal_le_of_le_ofReal h2 (h1 ▸ hint)
  -- the hash's tails
  set X : ℝ := 2 * Real.exp (-(e / 2) ^ 2 / (2 * ρ)) with hXdef
  have hX0 : 0 ≤ X := by positivity
  have hBal : (coins S).real BalBad ≤ tame.card * X := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    calc ∑ j ∈ tame, (coins S).real (sideBad (D j) Hs e) ≤ ∑ _j ∈ tame, X :=
          Finset.sum_le_sum (fun j hj => coins_sideBad_le (D j) (hρ j hj) Hs he0)
      _ = tame.card * X := by rw [Finset.sum_const, nsmul_eq_mul]
  have hfillB : (runMeasure μ D Dsf).real {x | ¬ fillOK H uni populations K N x}
      ≤ populations.card * Real.exp (-((N : ℝ) / 2)) :=
    runMeasure_coins_le D Dsf _ (measurableSet_fillOK populations H uni K N).compl
      (by positivity) (fun r => coins_notFill_le populations H uni N hNpos r)
  have himbB : ∀ B ∈ st, ∀ j ∈ tame, ∀ b,
      (runMeasure μ D Dsf).real (imbSet (rule := rule) O populations H uni (D j) K N e b B)
        ≤ X := by
    intro B hB j hj b
    refine runMeasure_coins_le D Dsf _ (measurableSet_imbSet O populations H uni (D j) K N e b B)
      hX0 (fun r => coins_imbSet_le O populations H uni (D j) (hρ j hj) K N he0 b B
        (npref_le_of_mem_stoppable hB) r ?_)
    refine le_trans (measureReal_window_le (D j) (hρ j hj) _) (le_trans ?_ hwinM)
    have hw := card_winAll_le H populations K r
    have : ((winAll H populations K r).card : ℝ) ≤ populations.card * (4 * N) := by
      rw [hK] at hw; exact_mod_cast hw
    exact mul_le_mul_of_nonneg_right this hρpos.le
  have hBadB : (runMeasure μ D Dsf).real Bad
      ≤ populations.card * Real.exp (-((N : ℝ) / 2)) + st.card * (tame.card * (2 * X)) := by
    refine le_trans (measureReal_union_le _ _) (add_le_add hfillB ?_)
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    calc ∑ B ∈ st, (runMeasure μ D Dsf).real (⋃ j ∈ tame, ⋃ b : Bool,
          imbSet (rule := rule) O populations H uni (D j) K N e b B)
        ≤ ∑ _B ∈ st, tame.card * (2 * X) := by
          refine Finset.sum_le_sum (fun B hB => ?_)
          refine le_trans (measureReal_biUnion_finset_le _ _) ?_
          calc ∑ j ∈ tame, (runMeasure μ D Dsf).real (⋃ b : Bool,
                imbSet (rule := rule) O populations H uni (D j) K N e b B)
              ≤ ∑ _j ∈ tame, 2 * X := by
                refine Finset.sum_le_sum (fun j hj => ?_)
                refine le_trans (measureReal_iUnion_fintype_le _) ?_
                rw [Fintype.sum_bool]
                linarith [himbB B hB j hj true, himbB B hB j hj false]
            _ = tame.card * (2 * X) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ = st.card * (tame.card * (2 * X)) := by rw [Finset.sum_const, nsmul_eq_mul]
  -- everything against `δ/16`
  have htc : (tame.card : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_le_card htame
  have hstc : (st.card : ℝ) ≤ (Nat.log 2 N : ℝ) + 1 := by
    exact_mod_cast card_stoppable_le populations εcov δ α pAP ρ
  have hcount : (tame.card : ℝ) * X + st.card * (tame.card * (2 * X))
      ≤ populations.card * (3 + 2 * (Nat.log 2 N : ℝ)) * X := by
    have h1 : (tame.card : ℝ) * X + st.card * (tame.card * (2 * X))
        = tame.card * (1 + 2 * st.card) * X := by ring
    rw [h1]
    have h2 : (tame.card : ℝ) * (1 + 2 * st.card)
        ≤ populations.card * (3 + 2 * (Nat.log 2 N : ℝ)) :=
      mul_le_mul htc (by linarith) (by positivity) (by positivity)
    exact mul_le_mul_of_nonneg_right h2 hX0
  show (runMeasure μ D Dsf).real E ≤ δ / 2 + α
  linarith

end OrthoDFA
