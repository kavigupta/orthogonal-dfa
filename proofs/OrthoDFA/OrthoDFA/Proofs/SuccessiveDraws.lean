import OrthoDFA.Utils.SuccessiveDraws
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.BitSum

/-!
# Sampling without replacement against binomials

`hitCount A l` counts the draws of `l` landing in the set `A` fixes from the draws before them.
Peeling the first draw, the count reaches `c + 1` with chance the first set's mass times the rest's
chance of reaching `c`, plus the remainder times its chance of reaching `c + 1`; with every set's
mass at most `q` that is at most `Bin(n + 1, q)`'s. Rejections and kept draws in `P` are such
counts.
-/

namespace OrthoDFA

open MeasureTheory

variable {S : Type*}

open Classical in
/-- How many of `l`'s draws land in the set `A` fixes from the draws before them. -/
noncomputable def hitCount (A : List S → Set S) : List S → ℕ
  | [] => 0
  | a :: l => (if a ∈ A [] then 1 else 0) + hitCount (fun l' => A (a :: l')) l

lemma hitCount_compl_add : ∀ (l : List S) (A : List S → Set S),
    hitCount (fun l' => (A l')ᶜ) l + hitCount A l = l.length
  | [], _ => rfl
  | a :: l, A => by
    have ih := hitCount_compl_add l (fun l' => A (a :: l'))
    simp only [hitCount, List.length_cons, Set.mem_compl_iff]
    by_cases ha : a ∈ A [] <;> simp [ha] <;> omega

section Combinatorics

variable [DecidableEq S]

/-- Where a draw is rejected: in `T`, or a repeat. -/
def rejectSet (T : Finset S) (l : List S) : Set S := ↑T ∪ {z | z ∈ l}

/-- Where a draw is kept in `P`. -/
def keptIn (P : Set S) (T : Finset S) (l : List S) : Set S := P \ rejectSet T l

lemma rejectSet_cons (T : Finset S) (a : S) (l : List S) :
    rejectSet T (a :: l) = rejectSet (insert a T) l := by
  ext z
  simp only [rejectSet, Set.mem_union, Finset.mem_coe, Set.mem_ofPred_eq, List.mem_cons,
    Finset.mem_insert]
  tauto

lemma hitCount_reject : ∀ (l : List S) (T : Finset S),
    hitCount (rejectSet T) l + (l.toFinset \ T).card = l.length
  | [], _ => by simp [hitCount]
  | a :: l, T => by
    have ih := hitCount_reject l (insert a T)
    have hfun : (fun l' => rejectSet T (a :: l')) = rejectSet (insert a T) :=
      funext fun l' => rejectSet_cons T a l'
    simp only [hitCount, hfun, List.length_cons, List.toFinset_cons]
    by_cases ha : a ∈ T
    · have h1 : a ∈ rejectSet T [] := by simp [rejectSet, ha]
      rw [if_pos h1, Finset.insert_sdiff_of_mem _ ha, Finset.insert_eq_of_mem ha] at *
      omega
    · have h1 : a ∉ rejectSet T [] := by simp [rejectSet, ha]
      have hsd : insert a l.toFinset \ T = insert a (l.toFinset \ insert a T) := by
        ext z
        simp only [Finset.mem_sdiff, Finset.mem_insert]
        by_cases hz : z = a
        · subst hz; simp [ha]
        · simp [hz]
      rw [if_neg h1, hsd, Finset.card_insert_of_notMem (by simp)]
      omega

open Classical in
lemma hitCount_kept (P : Set S) : ∀ (l : List S) (T : Finset S),
    hitCount (keptIn P T) l = ((l.toFinset \ T).filter (· ∈ P)).card
  | [], _ => by simp [hitCount]
  | a :: l, T => by
    have ih := hitCount_kept P l (insert a T)
    have hfun : (fun l' => keptIn P T (a :: l')) = keptIn P (insert a T) :=
      funext fun l' => by rw [keptIn, keptIn, rejectSet_cons]
    simp only [hitCount, hfun, List.toFinset_cons]
    by_cases ha : a ∈ T
    · have h1 : a ∉ keptIn P T [] := by simp [keptIn, rejectSet, ha]
      rw [if_neg h1, zero_add, ih, Finset.insert_sdiff_of_mem _ ha, Finset.insert_eq_of_mem ha]
    · have hsd : insert a l.toFinset \ T = insert a (l.toFinset \ insert a T) := by
        ext z
        simp only [Finset.mem_sdiff, Finset.mem_insert]
        by_cases hz : z = a
        · subst hz; simp [ha]
        · simp [hz]
      have hnm : a ∉ l.toFinset \ insert a T := by simp
      rw [hsd, Finset.filter_insert, ih]
      by_cases hP : a ∈ P
      · have h1 : a ∈ keptIn P T [] := by simp [keptIn, rejectSet, ha, hP]
        rw [if_pos h1, if_pos hP, Finset.card_insert_of_notMem (fun h => hnm
          (Finset.mem_filter.1 h).1)]
        omega
      · have h1 : a ∉ keptIn P T [] := by simp [keptIn, rejectSet, hP]
        rw [if_neg h1, if_neg hP, zero_add]

lemma toFinset_ofFn {n : ℕ} (y : Fin n → S) :
    (List.ofFn y).toFinset = Finset.univ.image y := by
  ext z
  simp [List.mem_ofFn]

end Combinatorics

section Measure

variable [MeasurableSpace S] [Countable S] [MeasurableSingletonClass S]

lemma measurableSet_pi_fin (n : ℕ) (E : Set (Fin n → S)) : MeasurableSet E :=
  (Set.to_countable E).measurableSet

lemma measurableSet_prod_pi_fin (n : ℕ) (E : Set (S × (Fin n → S))) : MeasurableSet E :=
  (Set.to_countable E).measurableSet

/-- The law of `n + 1` draws is the first draw's times the rest's. -/
lemma pi_succ_real (D : Measure S) [IsProbabilityMeasure D] (n : ℕ)
    (E : Set (S × (Fin n → S))) :
    (Measure.pi fun _ : Fin (n + 1) => D).real
        {y | (y 0, fun j : Fin n => y j.succ) ∈ E}
      = (D.prod (Measure.pi fun _ : Fin n => D)).real E := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => D) 0
  have hpre : {y : Fin (n + 1) → S | (y 0, fun j : Fin n => y j.succ) ∈ E}
      = (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => S) 0) ⁻¹' E := by
    ext y
    simp [MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv]
    rfl
  rw [hpre, measureReal_def, measureReal_def,
    hmp.measure_preimage (measurableSet_prod_pi_fin n E).nullMeasurableSet]

/-- Every set of mass at most `q`: the count reaches `c` no more often than `Bin(n, q)` does. -/
theorem hitCount_tail (D : Measure S) [IsProbabilityMeasure D] {q : ℝ} (hq0 : 0 ≤ q)
    (hq1 : q ≤ 1) : ∀ (n : ℕ) (A : List S → Set S),
    (∀ l : List S, l.length < n → D.real (A l) ≤ q) → ∀ c : ℕ,
    (Measure.pi fun _ : Fin n => D).real {y | c ≤ hitCount A (List.ofFn y)}
      ≤ binomSfGe n q c
  | n, _, _, 0 => by rw [binomSfGe_zero_right]; exact measureReal_le_one
  | 0, A, _, c + 1 => by
    have : {y : Fin 0 → S | c + 1 ≤ hitCount A (List.ofFn y)} = ∅ := by
      ext y; simp [hitCount]
    rw [this, measureReal_empty]
    exact binomSfGe_nonneg hq0 hq1 _
  | n + 1, A, hA, c + 1 => by
    set ν := Measure.pi fun _ : Fin n => D
    set A0 := A []
    set E : Set (S × (Fin n → S)) := {p | c + 1 ≤ hitCount A (p.1 :: List.ofFn p.2)} with hE
    have hset : {y : Fin (n + 1) → S | c + 1 ≤ hitCount A (List.ofFn y)}
        = {y | (y 0, fun j : Fin n => y j.succ) ∈ E} := by
      ext y
      simp [hE, List.ofFn_succ]
    rw [hset, pi_succ_real D n E]
    set Fc := binomSfGe n q c
    set Fc1 := binomSfGe n q (c + 1)
    have hFc0 : 0 ≤ Fc := binomSfGe_nonneg hq0 hq1 _
    have hFc10 : 0 ≤ Fc1 := binomSfGe_nonneg hq0 hq1 _
    have hanti : Fc1 ≤ Fc := binomSfGe_antitone hq0 hq1 _
    have hsec : ∀ a, ν (Prod.mk a ⁻¹' E)
        ≤ A0.indicator (fun _ => ENNReal.ofReal Fc) a
          + A0ᶜ.indicator (fun _ => ENNReal.ofReal Fc1) a := by
      intro a
      have hA' : ∀ l : List S, l.length < n → D.real ((fun l' => A (a :: l')) l) ≤ q :=
        fun l hl => hA (a :: l) (by simp; omega)
      by_cases ha : a ∈ A0
      · have hpre : Prod.mk a ⁻¹' E
            = {z : Fin n → S | c ≤ hitCount (fun l' => A (a :: l')) (List.ofFn z)} := by
          ext z
          simp only [hE, Set.mem_preimage, Set.mem_ofPred_eq, hitCount]
          rw [if_pos ha]
          omega
        rw [Set.indicator_of_mem ha, Set.indicator_of_notMem (by simpa using ha), add_zero, hpre,
          ← ENNReal.ofReal_toReal (measure_ne_top ν _)]
        exact ENNReal.ofReal_le_ofReal (hitCount_tail D hq0 hq1 n _ hA' c)
      · have hpre : Prod.mk a ⁻¹' E
            = {z : Fin n → S | c + 1 ≤ hitCount (fun l' => A (a :: l')) (List.ofFn z)} := by
          ext z
          simp only [hE, Set.mem_preimage, Set.mem_ofPred_eq, hitCount]
          rw [if_neg ha, zero_add]
        rw [Set.indicator_of_notMem ha, Set.indicator_of_mem (by simpa using ha), zero_add,
          hpre, ← ENNReal.ofReal_toReal (measure_ne_top ν _)]
        exact ENNReal.ofReal_le_ofReal (hitCount_tail D hq0 hq1 n _ hA' (c + 1))
    have hA0m : MeasurableSet A0 := (Set.to_countable A0).measurableSet
    have hprod : (D.prod ν) E
        ≤ ENNReal.ofReal Fc * D A0 + ENNReal.ofReal Fc1 * D A0ᶜ := by
      rw [Measure.prod_apply (measurableSet_prod_pi_fin n E)]
      calc ∫⁻ a, ν (Prod.mk a ⁻¹' E) ∂D
          ≤ ∫⁻ a, (A0.indicator (fun _ => ENNReal.ofReal Fc) a
              + A0ᶜ.indicator (fun _ => ENNReal.ofReal Fc1) a) ∂D := lintegral_mono hsec
        _ = ENNReal.ofReal Fc * D A0 + ENNReal.ofReal Fc1 * D A0ᶜ := by
          rw [lintegral_add_left ((measurable_const.indicator hA0m)),
            lintegral_indicator_const hA0m, lintegral_indicator_const hA0m.compl]
    have hx0 : 0 ≤ D.real A0 := measureReal_nonneg
    have hxq : D.real A0 ≤ q := hA [] (by simp)
    have hcompl : D.real A0ᶜ = 1 - D.real A0 := by
      rw [measureReal_compl hA0m, probReal_univ]
    calc (D.prod ν).real E
        ≤ (ENNReal.ofReal Fc * D A0 + ENNReal.ofReal Fc1 * D A0ᶜ).toReal :=
          ENNReal.toReal_mono (by finiteness) hprod
      _ = Fc * D.real A0 + Fc1 * D.real A0ᶜ := by
          rw [ENNReal.toReal_add (by finiteness) (by finiteness), ENNReal.toReal_mul,
            ENNReal.toReal_mul, ENNReal.toReal_ofReal hFc0, ENNReal.toReal_ofReal hFc10]
          rfl
      _ ≤ q * Fc + (1 - q) * Fc1 := by
          rw [hcompl]; nlinarith
      _ = binomSfGe (n + 1) q (c + 1) := (binomSfGe_succ n q c).symm

/-- Every set of mass at least `q`: the count stays at or below `c` no more often than
`Bin(n, q)` does. -/
theorem hitCount_lower (D : Measure S) [IsProbabilityMeasure D] {q : ℝ} (hq0 : 0 ≤ q)
    (hq1 : q ≤ 1) (n : ℕ) (A : List S → Set S)
    (hA : ∀ l : List S, l.length < n → q ≤ D.real (A l)) (c : ℕ) :
    (Measure.pi fun _ : Fin n => D).real {y | hitCount A (List.ofFn y) ≤ c}
      ≤ 1 - binomSfGe n q (c + 1) := by
  rcases Nat.lt_or_ge c n with hcn | hcn
  · have hset : {y : Fin n → S | hitCount A (List.ofFn y) ≤ c}
        = {y | n - c ≤ hitCount (fun l' => (A l')ᶜ) (List.ofFn y)} := by
      ext y
      have := hitCount_compl_add (List.ofFn y) A
      rw [List.length_ofFn] at this
      simp only [Set.mem_ofPred_eq]
      omega
    have hAc : ∀ l : List S, l.length < n → D.real (A l)ᶜ ≤ 1 - q := fun l hl => by
      rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
      linarith [hA l hl]
    rw [hset]
    refine le_trans (hitCount_tail D (by linarith) (by linarith) n _ hAc (n - c)) (le_of_eq ?_)
    rw [binomSfGe_symm (1 - q) n (n - c) (by omega), sub_sub_cancel,
      show n + 1 - (n - c) = c + 1 by omega]
  · rw [binomSfGe_eq_zero _ _ (by omega), sub_zero]
    exact measureReal_le_one

end Measure

theorem successiveDrawsTails_holds : SuccessiveDrawsTails := by
  intro S _ _ _ _ D _ n T t P ht1 hT
  classical
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · refine ⟨fun c => ?_, fun c => ?_, fun c _ => ?_⟩
    · rcases c with _ | c
      · rw [binomSfGe_zero_right]; exact measureReal_le_one
      · have : {y : Fin 0 → S | c + 1 + (Finset.univ.image y \ T).card ≤ 0} = ∅ := by
          ext y; simp
        rw [this, measureReal_empty, binomSfGe_zero_left]
    · rcases c with _ | c
      · rw [binomSfGe_zero_right]; exact measureReal_le_one
      · have : {y : Fin 0 → S | c + 1 ≤ ((Finset.univ.image y \ T).filter (· ∈ P)).card} = ∅ := by
          ext y; simp
        rw [this, measureReal_empty, binomSfGe_zero_left]
    · rw [binomSfGe_zero_left, sub_zero]; exact measureReal_le_one
  have hrej : ∀ l : List S, l.length < n → D.real (rejectSet T l) ≤ t := fun l hl => by
    have h := hT l.toFinset (lt_of_le_of_lt (List.toFinset_card_le l) hl)
    have hs : rejectSet T l = ↑(T ∪ l.toFinset) := by
      ext z; simp [rejectSet]
    rwa [hs]
  have ht0 : 0 ≤ t := le_trans measureReal_nonneg (hrej [] (by simpa using hn))
  have hP1 : D.real P ≤ 1 := measureReal_le_one
  refine ⟨fun c => ?_, fun c => ?_, fun c htP => ?_⟩
  · have hset : {y : Fin n → S | c + (Finset.univ.image y \ T).card ≤ n}
        = {y | c ≤ hitCount (rejectSet T) (List.ofFn y)} := by
      ext y
      have := hitCount_reject (List.ofFn y) T
      rw [List.length_ofFn, toFinset_ofFn] at this
      simp only [Set.mem_ofPred_eq]
      omega
    rw [hset]
    exact hitCount_tail D ht0 ht1 n _ hrej c
  · have hset : {y : Fin n → S | c ≤ ((Finset.univ.image y \ T).filter (· ∈ P)).card}
        = {y | c ≤ hitCount (keptIn P T) (List.ofFn y)} := by
      ext y
      rw [Set.mem_ofPred_eq, Set.mem_ofPred_eq, hitCount_kept, toFinset_ofFn]
    rw [hset]
    exact hitCount_tail D measureReal_nonneg hP1 n _
      (fun l _ => measureReal_mono Set.sdiff_subset) c
  · have hset : {y : Fin n → S | ((Finset.univ.image y \ T).filter (· ∈ P)).card ≤ c}
        = {y | hitCount (keptIn P T) (List.ofFn y) ≤ c} := by
      ext y
      rw [Set.mem_ofPred_eq, Set.mem_ofPred_eq, hitCount_kept, toFinset_ofFn]
    rw [hset]
    refine hitCount_lower D (by linarith) (by linarith) n _ (fun l hl => ?_) c
    have hsub : P ⊆ keptIn P T l ∪ rejectSet T l := by
      intro z hz
      by_cases h : z ∈ rejectSet T l
      · exact Or.inr h
      · exact Or.inl ⟨hz, h⟩
    have := le_trans (measureReal_mono hsub (measure_ne_top D _)) (measureReal_union_le _ _)
    linarith [hrej l hl]

end OrthoDFA
