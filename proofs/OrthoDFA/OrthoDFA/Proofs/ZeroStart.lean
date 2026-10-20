import OrthoDFA.Proofs.DFABridge

/-!
# A start equivalent to a covered state

Re-rooting at a covered `q` accurate to `1 − η` at every length up to `|Q|`, with
`η < |Σ|^(−|Q|)`, classifies every word of at most `|Q|` letters as the target does: one word
weighs `|Σ|^(−ℓ)`. Two states of an `|Q|`-state DFA agreeing on every word of at most `|Q|` letters
agree on every word (`agree_of_agreeUpTo`), so the start is equivalent to `q`.

Then every reachable state reads as a state a covered state reaches in fewer than `|Q|` letters,
which the window's probes visit `c |Σ|^(−|Q|)/N` of the time, so the clustering's bound on the
undecided reads over the window bounds every state's (`undecided_le_of_start`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Words

variable {σ : Type*} (M : _root_.DFA α σ)

/-- `p` and `q` classify every word of at most `n` letters alike. -/
def AgreeUpTo (n : ℕ) (p q : σ) : Prop :=
  ∀ w : List α, w.length ≤ n → (M.evalFrom p w ∈ M.accept ↔ M.evalFrom q w ∈ M.accept)

theorem agreeUpTo_succ {n : ℕ} {p q : σ} :
    AgreeUpTo M (n + 1) p q ↔
      (p ∈ M.accept ↔ q ∈ M.accept) ∧ ∀ a, AgreeUpTo M n (M.step p a) (M.step q a) := by
  constructor
  · intro h
    refine ⟨h [] (by simp), fun a w hw => h (a :: w) (by simp; omega)⟩
  · rintro ⟨h0, h1⟩ w hw
    rcases w with _ | ⟨a, w⟩
    · exact h0
    · exact h1 a w (by simp at hw; omega)

theorem agreeUpTo_mono {n m : ℕ} (h : n ≤ m) {p q : σ} (hpq : AgreeUpTo M m p q) :
    AgreeUpTo M n p q := fun w hw => hpq w (hw.trans h)

/-- Agreement up to `n` that already gives agreement up to `n + 1` gives it up to every length. -/
theorem agreeUpTo_of_stable {n : ℕ} (hs : ∀ p q, AgreeUpTo M n p q → AgreeUpTo M (n + 1) p q) :
    ∀ m p q, AgreeUpTo M n p q → AgreeUpTo M m p q := by
  intro m
  induction m with
  | zero => intro p q h; exact agreeUpTo_mono M (Nat.zero_le n) h
  | succ m ih =>
    intro p q h
    have h1 := (agreeUpTo_succ M).1 (hs p q h)
    exact (agreeUpTo_succ M).2 ⟨h1.1, fun a => ih _ _ (agreeUpTo_mono M (Nat.le_succ n)
      (hs _ _ (h1.2 a)))⟩

/-- `AgreeUpTo` as a setoid. -/
def agreeSetoid (n : ℕ) : Setoid σ where
  r := AgreeUpTo M n
  iseqv := ⟨fun _ _ _ => Iff.rfl, fun h w hw => (h w hw).symm,
    fun h h' w hw => (h w hw).trans (h' w hw)⟩

variable [Fintype σ]

open scoped Classical in
theorem card_agree_lt {n : ℕ} (hn : ¬ ∀ p q, AgreeUpTo M n p q → AgreeUpTo M (n + 1) p q) :
    Fintype.card (Quotient (agreeSetoid M n)) < Fintype.card (Quotient (agreeSetoid M (n + 1))) := by
  push Not at hn
  obtain ⟨p, q, hpq, hnot⟩ := hn
  set f : Quotient (agreeSetoid M (n + 1)) → Quotient (agreeSetoid M n) :=
    Quotient.map id fun a b h => agreeUpTo_mono M (Nat.le_succ n) h
  refine Fintype.card_lt_of_surjective_not_injective f ?_ ?_
  · intro x
    obtain ⟨a, rfl⟩ := Quotient.exists_rep x
    exact ⟨Quotient.mk _ a, rfl⟩
  · intro hinj
    apply hnot
    have : f (Quotient.mk _ p) = f (Quotient.mk _ q) := Quotient.sound hpq
    exact Quotient.exact (hinj this)

open scoped Classical in
theorem card_agree_ge : ∀ n, n ≤ Fintype.card σ →
    (∀ i < n, ¬ ∀ p q, AgreeUpTo M i p q → AgreeUpTo M (i + 1) p q) →
      Nonempty σ → n + 1 ≤ Fintype.card (Quotient (agreeSetoid M n))
  | 0, _, _, hne => by
    obtain ⟨p⟩ := hne
    exact Fintype.card_pos_iff.2 ⟨Quotient.mk _ p⟩
  | n + 1, hn, hs, hne => by
    have h1 := card_agree_ge n (by omega) (fun i hi => hs i (by omega)) hne
    have h2 := card_agree_lt M (hs n (by omega))
    omega

/-- Two states agreeing on every word of at most `|Q|` letters agree on every word. -/
theorem agree_of_agreeUpTo {p q : σ} (h : AgreeUpTo M (Fintype.card σ) p q) (w : List α) :
    M.evalFrom p w ∈ M.accept ↔ M.evalFrom q w ∈ M.accept := by
  classical
  have hne : Nonempty σ := ⟨p⟩
  have hst : ∃ i < Fintype.card σ, ∀ p q, AgreeUpTo M i p q → AgreeUpTo M (i + 1) p q := by
    by_contra hno
    push Not at hno
    have h1 := card_agree_ge M (Fintype.card σ) le_rfl (fun i hi => by
      push Not
      exact hno i hi) hne
    have h2 : Fintype.card (Quotient (agreeSetoid M (Fintype.card σ))) ≤ Fintype.card σ :=
      Fintype.card_le_of_surjective _ (Quotient.mk_surjective (s := agreeSetoid M _))
    omega
  obtain ⟨i, hi, hs⟩ := hst
  exact agreeUpTo_of_stable M hs w.length p q (agreeUpTo_mono M hi.le h) w le_rfl

/-- A state reached from `p` is reached in fewer than `|Q|` letters. -/
theorem exists_short_word : ∀ (n : ℕ) (p : σ) (w : List α), w.length ≤ n →
    ∃ w' : List α, w'.length < Fintype.card σ ∧ M.evalFrom p w' = M.evalFrom p w := by
  intro n
  induction n with
  | zero =>
    intro p w hn
    by_cases h : w.length < Fintype.card σ
    · exact ⟨w, h, rfl⟩
    · haveI : Nonempty σ := ⟨p⟩
      have := Fintype.card_pos (α := σ)
      omega
  | succ n ih =>
    intro p w hn
    by_cases h : w.length < Fintype.card σ
    · exact ⟨w, h, rfl⟩
    · obtain ⟨q, a, b, c, hw, -, hb, ha, hbq, hc⟩ := M.evalFrom_split (s := p) (not_lt.1 h) rfl
      have hlen : (a ++ c).length ≤ n := by
        have : b.length ≠ 0 := fun h0 => hb (List.length_eq_zero_iff.1 h0)
        have e := congrArg List.length hw
        simp only [List.length_append] at e ⊢
        omega
      obtain ⟨w', hw', he⟩ := ih p (a ++ c) hlen
      refine ⟨w', hw', he.trans ?_⟩
      rw [hw]
      simp only [DFA.evalFrom_of_append, ha, hbq]

end Words

section Accuracy

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ)

lemma uniformStrings_singleton_ge (t : FreeMonoid α) :
    1 / (Fintype.card α : ℝ) ^ t.toList.length ≤ (uniformStrings t.toList.length).real {t} := by
  rw [uniformStrings_real]
  have : 1 ≤ Nat.card {v : List.Vector α t.toList.length //
      FreeMonoid.ofList v.toList ∈ ({t} : Set _)} := by
    rw [Nat.one_le_iff_ne_zero, Nat.card_ne_zero]
    exact ⟨⟨⟨⟨t.toList, rfl⟩, by simp⟩⟩, inferInstance⟩
  exact div_le_div_of_nonneg_right (by exact_mod_cast this) (by positivity)

/-- Re-rooting accurate to `1 − η` at every length up to `|Q|`, with `η |Σ|^|Q| < 1`, makes the
start equivalent to the covered start. -/
theorem start_agree_of_accuracy {S : Set σ} {q : σ} {η : ℝ}
    (hη : η * (Fintype.card α : ℝ) ^ Fintype.card σ < 1)
    (hacc : ∀ ℓ ≤ Fintype.card σ, 1 - η ≤ rerootAccuracy M ℓ S q) (w : List α) :
    M.evalFrom q w ∈ M.accept ↔ M.evalFrom M.start w ∈ M.accept := by
  refine agree_of_agreeUpTo M (fun u hu => ?_) w
  by_contra hno
  set t : FreeMonoid α := FreeMonoid.ofList u
  have htl : t.toList = u := rfl
  set G := {x : FreeMonoid α | (∀ i ≤ t.toList.length, M.evalFrom q (prefixOf x i).toList ∈ S)
    ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  have htG : t ∉ G := fun h => hno (by rw [← htl]; exact h.2)
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have h1 := hacc t.toList.length (by rw [htl]; exact hu)
  have h2 := uniformStrings_singleton_ge (α := α) t
  have hdis : Disjoint G {t} := Set.disjoint_singleton_right.2 htG
  have h3 : (uniformStrings t.toList.length).real G + (uniformStrings t.toList.length).real {t}
      ≤ 1 := by
    rw [← measureReal_union hdis (measurableSet_singleton t)]
    exact measureReal_le_one
  have h4 : η < 1 / (Fintype.card α : ℝ) ^ t.toList.length := by
    rcases le_or_gt η 0 with h0 | h0
    · exact lt_of_le_of_lt h0 (by positivity)
    rw [lt_div_iff₀ (by positivity)]
    calc η * (Fintype.card α : ℝ) ^ t.toList.length
        ≤ η * (Fintype.card α : ℝ) ^ Fintype.card σ :=
          mul_le_mul_of_nonneg_left (pow_le_pow_right₀ hA (by rw [htl]; exact hu)) h0.le
      _ < 1 := hη
  have : rerootAccuracy M t.toList.length S q = (uniformStrings t.toList.length).real G := rfl
  linarith

end Accuracy

section Reads

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)

include hL in
/-- Strings at states classifying every word alike read alike. -/
theorem readProb_eq_of_agree (F : Finset (FreeMonoid α)) (kl kh : ℕ) {w w' : FreeMonoid α}
    (h : ∀ v : List α, M.evalFrom (M.eval w.toList) v ∈ M.accept
      ↔ M.evalFrom (M.eval w'.toList) v ∈ M.accept) :
    readProb O F kl kh w = readProb O F kl kh w' := by
  classical
  funext rd
  rw [readProb_eq_readLaw, readProb_eq_readLaw]
  congr 1
  unfold acceptCount
  congr 1
  refine Finset.filter_congr fun v _ => ?_
  rw [hL, hL]
  simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
  exact h v.toList

/-- The strings of length `i + t` reaching `q` by their first `i` letters and then `w`, `|w| = t`,
are at least those of length `i` reaching `q`, scaled by `|Σ|^(−t)`. -/
theorem uniformStrings_reach_ge (i : ℕ) (q : σ) (w : List α) :
    (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q}
        / (Fintype.card α : ℝ) ^ w.length
      ≤ (uniformStrings (i + w.length)).real
        {y : FreeMonoid α | M.eval y.toList = M.evalFrom q w} := by
  classical
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, uavg_split]
  unfold uavg
  have hA : (0 : ℝ) < (Fintype.card α : ℝ) ^ w.length := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  rw [div_div, pow_add]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  set wv : List.Vector α w.length := ⟨w, rfl⟩
  refine Finset.sum_le_sum fun u _ => ?_
  refine le_trans ?_ (Finset.single_le_sum (f := fun z : List.Vector α w.length =>
    {y : FreeMonoid α | M.eval y.toList = M.evalFrom q w}.indicator 1
      (FreeMonoid.ofList (u.toList ++ z.toList)))
    (fun z _ => Set.indicator_nonneg (fun _ _ => zero_le_one) _) (Finset.mem_univ wv))
  by_cases hu : M.eval (FreeMonoid.ofList u.toList).toList = q
  · rw [Set.indicator_of_mem (show FreeMonoid.ofList u.toList ∈
      {y : FreeMonoid α | M.eval y.toList = q} from hu),
      Set.indicator_of_mem (show FreeMonoid.ofList (u.toList ++ wv.toList) ∈
        {y : FreeMonoid α | M.eval y.toList = M.evalFrom q w} by
        simp only [Set.mem_setOf_eq, FreeMonoid.toList_ofList, DFA.eval,
          DFA.evalFrom_of_append] at hu ⊢
        rw [hu]; rfl)]
    simp only [Pi.one_apply, le_refl]
  · rw [Set.indicator_of_notMem (show FreeMonoid.ofList u.toList ∉
      {y : FreeMonoid α | M.eval y.toList = q} from hu)]
    exact Set.indicator_nonneg (fun _ _ => zero_le_one) _

include hL in
/-- Where the start is equivalent to a covered `q` and the family, with its seed, leaves the window
undecided at most `u` on average, every state reads undecided at most `u N |Σ|^|Q| / c` of the
time. -/
theorem undecided_le_of_start (L : ℕ) {c u : ℝ} (hc : 0 < c) {q : σ}
    (hq : q ∈ coveredStates M L c)
    (hstart : ∀ w : List α, M.evalFrom q w ∈ M.accept ↔ M.evalFrom M.start w ∈ M.accept)
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) (lo hi : ℕ)
    (hund : ∫ p, undecidedProb O lo (hi + 1) (insert 1 F) p
      ∂(windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) ≤ u) (s : σ) :
    familyReadDist O M F lo (hi + 1) s .undecided
      ≤ u * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
        * (Fintype.card α : ℝ) ^ Fintype.card σ / c := by
  classical
  set k := (L + 1) / 2
  set L₁ := L + Fintype.card σ + 3
  have hkL₁ : k ≤ L₁ := by omega
  set D : Measure (FreeMonoid α) := windowStrings k L₁
  haveI : IsProbabilityMeasure D := windowStrings_isProbability hkL₁
  set N : ℝ := (L₁ : ℝ) + 1 - k
  have hN : 0 < N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    show 0 < (L₁ : ℝ) + 1 - k
    linarith
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hu0 : 0 ≤ u := le_trans (integral_nonneg fun _ => measureReal_nonneg) hund
  by_cases hreach : ∃ w : FreeMonoid α, M.eval w.toList = s
  swap
  · have : familyReadDist O M F lo (hi + 1) s .undecided = 0 := by
      simp only [familyReadDist, dif_neg hreach]
    rw [this]
    positivity
  obtain ⟨w₀, hw₀⟩ := hreach
  obtain ⟨w, hw, hws⟩ := exists_short_word M w₀.toList.length M.start w₀.toList le_rfl
  -- `s` reads as the state `q` reaches by `w`
  set s' := M.evalFrom q w
  obtain ⟨y', hy'⟩ : ∃ y' : FreeMonoid α, M.eval y'.toList = s' := by
    obtain ⟨x₀, hx₀⟩ : ∃ x₀ : FreeMonoid α, M.eval x₀.toList = q := by
      by_contra hno
      push Not at hno
      have h0 : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
          ∧ M.eval (prefixOf x j).toList = q} = ∅ := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
        exact fun j _ _ => hno _
      have h := hq
      simp only [coveredStates, Set.mem_setOf_eq, h0, measureReal_empty] at h
      linarith
    refine ⟨x₀ * FreeMonoid.ofList w, ?_⟩
    simp only [FreeMonoid.toList_mul, FreeMonoid.toList_ofList, DFA.eval,
      DFA.evalFrom_of_append] at hx₀ ⊢
    rw [hx₀]
  have hsame : familyReadDist O M F lo (hi + 1) s
      = familyReadDist O M F lo (hi + 1) s' := by
    rw [← hw₀, ← readProb_eq_familyReadDist O M hL, ← hy',
      ← readProb_eq_familyReadDist O M hL]
    refine readProb_eq_of_agree M O hL F lo (hi + 1) fun v => ?_
    rw [hy']
    show M.evalFrom (M.evalFrom M.start w₀.toList) v ∈ M.accept
      ↔ M.evalFrom (M.evalFrom q w) v ∈ M.accept
    rw [← hws, ← DFA.evalFrom_of_append, ← DFA.evalFrom_of_append]
    exact (hstart (w ++ v)).symm
  rw [hsame]
  -- the window visits `s'`
  set Sq := {y : FreeMonoid α | M.eval y.toList = s'}
  have hocc := occupancy_of_covered M hq
  have hvis : c / (Fintype.card α : ℝ) ^ Fintype.card σ ≤ N * D.real Sq := by
    rw [windowStrings_real hkL₁, mul_div_cancel₀ _ hN.ne']
    have hw1 : (Fintype.card α : ℝ) ^ w.length ≤ (Fintype.card α : ℝ) ^ Fintype.card σ :=
      pow_le_pow_right₀ hA hw.le
    have hpow : (0 : ℝ) < (Fintype.card α : ℝ) ^ w.length := by positivity
    calc c / (Fintype.card α : ℝ) ^ Fintype.card σ
        ≤ c / (Fintype.card α : ℝ) ^ w.length :=
          div_le_div_of_nonneg_left hc.le hpow hw1
      _ ≤ (∑ i ∈ Finset.Icc k L, (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q})
            / (Fintype.card α : ℝ) ^ w.length := div_le_div_of_nonneg_right hocc hpow.le
      _ = ∑ i ∈ Finset.Icc k L, (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q}
            / (Fintype.card α : ℝ) ^ w.length := Finset.sum_div _ _ _
      _ ≤ ∑ i ∈ Finset.Icc k L, (uniformStrings (i + w.length)).real Sq :=
          Finset.sum_le_sum fun i _ => uniformStrings_reach_ge M i q w
      _ = ∑ j ∈ (Finset.Icc k L).image (· + w.length), (uniformStrings j).real Sq := by
          rw [Finset.sum_image fun a _ b _ h => by simpa using h]
      _ ≤ ∑ j ∈ Finset.Icc k L₁, (uniformStrings j).real Sq :=
          Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => by
            obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hj
            simp only [Finset.mem_Icc] at hi ⊢
            omega) fun _ _ _ => measureReal_nonneg
  -- the family's bound over the window
  set U := familyReadDist O M F lo (hi + 1) s' .undecided
  have hU0 : 0 ≤ U := by
    simp only [U, familyReadDist]
    split_ifs
    · exact measureReal_nonneg
    · exact le_rfl
  have hlow : U * D.real Sq ≤ u := by
    set g := fun p => undecidedProb O lo (hi + 1) (insert 1 F) p
    have hg0 : ∀ p, 0 ≤ g p := fun p => measureReal_nonneg
    have hle : ∀ p, Sq.indicator (fun _ => U) p ≤ g p := by
      intro p
      by_cases hp : p ∈ Sq
      · rw [Set.indicator_of_mem hp]
        have := readProb_undecided_le O hF lo hi p
        rw [readProb_eq_familyReadDist O M hL, Set.mem_setOf_eq.mp hp] at this
        exact this
      · rw [Set.indicator_of_notMem hp]
        exact hg0 p
    calc U * D.real Sq = ∫ p, Sq.indicator (fun _ => U) p ∂D := by
          rw [integral_indicator_const _ (measurableSet_of_countable _), smul_eq_mul, mul_comm]
      _ ≤ ∫ p, g p ∂D := integral_mono
          ((integrable_const U).indicator (measurableSet_of_countable _))
          (integrable_of_unit D g hg0 fun _ => measureReal_le_one) hle
      _ ≤ u := hund
  have hApos : (0 : ℝ) < (Fintype.card α : ℝ) ^ Fintype.card σ := by positivity
  rw [le_div_iff₀ hc]
  have h1 : U * (c / (Fintype.card α : ℝ) ^ Fintype.card σ) ≤ U * (N * D.real Sq) :=
    mul_le_mul_of_nonneg_left hvis hU0
  have h2 : U * c = U * (c / (Fintype.card α : ℝ) ^ Fintype.card σ)
      * (Fintype.card α : ℝ) ^ Fintype.card σ := by field_simp
  have h3 : U * (N * D.real Sq) * (Fintype.card α : ℝ) ^ Fintype.card σ
      ≤ u * N * (Fintype.card α : ℝ) ^ Fintype.card σ := by
    have : U * (N * D.real Sq) = N * (U * D.real Sq) := by ring
    rw [this]
    have := mul_le_mul_of_nonneg_left hlow hN.le
    nlinarith
  show U * c ≤ u * ((L₁ : ℕ) + 1 - (k : ℕ) : ℝ) * (Fintype.card α : ℝ) ^ Fintype.card σ
  calc U * c ≤ U * (N * D.real Sq) * (Fintype.card α : ℝ) ^ Fintype.card σ := by
        rw [h2]; exact mul_le_mul_of_nonneg_right h1 hApos.le
    _ ≤ _ := h3

end Reads

end OrthoDFA

end
