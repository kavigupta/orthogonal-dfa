import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.UniformStrings

/-!
# A start the window visits covers every state it reaches

A word leads from `e` to `q` only if one shorter than `|Q|` does (`exists_short_word`), so a run
that can still reach a set `G` meets it within any `|Q|` letters with chance at least
`ρ = |Σ|^{-(|Q|-1)}`, and has not met it after `n` letters with chance at most `(1 − ρ)^⌊n/|Q|⌋`
(`avoid_le`).

Let `k = ⌈L/2⌉` and `δT = (1 − ρ)^⌊k/|Q|⌋`. A set of states entered only from itself, and left from
each of its states by some word, holds at most `δT` of the strings at `k` (`stay_le`). A string that
visits `q₀` from `k` on is at `k` in a state that reaches `q₀`. The states that reach `q₀` but are
not reached by it are such a set, and so are all those that reach `q₀`, if `q₀` reaches a state
that does not reach it back. So a start visited from `k` on by `c₀ > δT` of the strings is reached
back from every state it reaches, at least `c₀ − δT` of the strings are at `k` in a state it
reaches, and from there each state it reaches is met by `L` with chance at least
`1 − (1 − ρ)^⌊(L − k)/|Q|⌋` (`visit_of_reach`). A run from `q₀` stays in those states, so its
accuracy over any set is at most that over the covered states (`ceil_of_visited`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α]

section Split

/-- The strings of length `j + t`, split after `j` letters. -/
def vecAppendEquiv (j t : ℕ) : List.Vector α (j + t) ≃ List.Vector α j × List.Vector α t where
  toFun v := (⟨v.toList.take j, by simp⟩, ⟨v.toList.drop j, by simp⟩)
  invFun w := ⟨w.1.toList ++ w.2.toList, by simp⟩
  left_inv v := by apply List.Vector.toList_injective; simp
  right_inv w := by
    obtain ⟨⟨w₁, h₁⟩, ⟨w₂, h₂⟩⟩ := w
    refine Prod.ext (List.Vector.toList_injective ?_) (List.Vector.toList_injective ?_)
    · simp [List.Vector.toList, ← h₁]
    · simp [List.Vector.toList, ← h₁]

open scoped Classical in
lemma uniformStrings_real_sum (n : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings n).real S
      = (∑ v : List.Vector α n, if FreeMonoid.ofList v.toList ∈ S then (1 : ℝ) else 0)
        / (Fintype.card α : ℝ) ^ n := by
  rw [uniformStrings_real, Nat.card_eq_fintype_card, Fintype.card_subtype, Finset.card_filter]
  push_cast
  rfl

variable [Nonempty α]

/-- A uniform string of length `j + t` is a uniform first `j` letters followed by a uniform rest. -/
lemma uniformStrings_real_append (j t : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings (j + t)).real S = ∑ w : List.Vector α j,
      (uniformStrings t).real {z | FreeMonoid.ofList w.toList * z ∈ S}
        / (Fintype.card α : ℝ) ^ j := by
  classical
  have hA : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [uniformStrings_real_sum]
  simp_rw [uniformStrings_real_sum]
  rw [← (vecAppendEquiv j t).symm.sum_comp, Fintype.sum_prod_type, pow_add, Finset.sum_div]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [div_div, mul_comm ((Fintype.card α : ℝ) ^ t)]
  congr 2

lemma uniformStrings_real_prefix (j t : ℕ) (P : Set (FreeMonoid α)) :
    (uniformStrings (j + t)).real {x | prefixOf x j ∈ P} = (uniformStrings j).real P := by
  classical
  rw [uniformStrings_real_append, uniformStrings_real_sum, Finset.sum_div]
  refine Finset.sum_congr rfl fun w _ => ?_
  have e : ∀ z : FreeMonoid α, prefixOf (FreeMonoid.ofList w.toList * z) j
      = FreeMonoid.ofList w.toList := fun z => by
    simp [prefixOf, List.take_left' w.toList_length]
  simp only [Set.mem_ofPred_eq, e]
  split_ifs with h
  · simp [h]
  · simp [h]

omit [Nonempty α] in
lemma uniformStrings_real_singleton (n : ℕ) (v : List.Vector α n) :
    (uniformStrings n).real {FreeMonoid.ofList v.toList} = ((Fintype.card α : ℝ) ^ n)⁻¹ := by
  classical
  rw [uniformStrings_real_sum]
  have e : ∀ u : List.Vector α n, (FreeMonoid.ofList u.toList ∈ ({FreeMonoid.ofList v.toList} :
      Set (FreeMonoid α))) ↔ u = v := fun u => by
    simp only [Set.mem_singleton_iff]
    exact ⟨fun h => List.Vector.toList_injective (FreeMonoid.ofList.injective h), fun h => h ▸ rfl⟩
  simp only [e, Finset.sum_ite_eq', Finset.mem_univ, if_true, one_div]

end Split


variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ)

/-- Some word leads from `e` to `q`. -/
def Reach (e q : σ) : Prop := ∃ w : List α, M.evalFrom e w = q

omit [Fintype α] [Nonempty α] in
theorem exists_short_word {e q : σ} {w : List α} (h : M.evalFrom e w = q) :
    ∃ w' : List α, w'.length < Fintype.card σ ∧ M.evalFrom e w' = q := by
  induction hn : w.length using Nat.strong_induction_on generalizing w with
  | _ n ih =>
    by_cases hlen : Fintype.card σ ≤ w.length
    · obtain ⟨p, a, b, c, rfl, -, hb, ha, -, hc⟩ := M.evalFrom_split hlen h
      have hb' := List.length_pos_of_ne_nil hb
      simp only [List.length_append] at hn
      exact ih (a ++ c).length (by simp; omega) (by rw [DFA.evalFrom_of_append, ha, hc]) rfl
    · exact ⟨w, by omega, h⟩

omit [Fintype α] [Nonempty α] [Fintype σ] in
theorem evalFrom_mem_of_closed {P : Set σ} (hP : ∀ e ∈ P, ∀ a, M.step e a ∈ P) :
    ∀ (w : List α) {e : σ}, e ∈ P → M.evalFrom e w ∈ P
  | [], _, he => he
  | a :: w, _, he => evalFrom_mem_of_closed hP w (hP _ he a)

omit [Fintype α] [Nonempty α] [Fintype σ] in
theorem mem_of_evalFrom_mem {B : Set σ} (hin : ∀ e a, M.step e a ∈ B → e ∈ B) :
    ∀ (w : List α) {e : σ}, M.evalFrom e w ∈ B → e ∈ B
  | [], _, h => h
  | _ :: w, _, h => hin _ _ (mem_of_evalFrom_mem hin w h)

/-- The runs from `e` that meet no state of `G` within their first `n` letters. -/
def avoidSet (G : Set σ) (e : σ) (n : ℕ) : Set (FreeMonoid α) :=
  {z | ∀ i ≤ n, M.evalFrom e (z.toList.take i) ∉ G}

/-- `|Σ|^{-(|Q|-1)}`, at most the chance of reading a given word shorter than `|Q|`. -/
abbrev wordRate (α σ : Type*) [Fintype α] [Fintype σ] : ℝ :=
  ((Fintype.card α : ℝ) ^ (Fintype.card σ - 1))⁻¹

lemma wordRate_le_pow {l : ℕ} (hl : l < Fintype.card σ) :
    wordRate α σ ≤ ((Fintype.card α : ℝ) ^ l)⁻¹ := by
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  exact inv_anti₀ (by positivity) (pow_le_pow_right₀ hA (by omega))

lemma wordRate_le_one : wordRate α σ ≤ 1 := by
  have : (1 : ℝ) ≤ (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) :=
    one_le_pow₀ (by exact_mod_cast Fintype.card_pos)
  exact inv_le_one_of_one_le₀ this

omit [Nonempty α] in
lemma wordRate_nonneg : 0 ≤ wordRate α σ := by unfold wordRate; positivity

omit [Fintype σ] in
/-- Within `B` letters, a run that can meet `G` along a word of chance at least `r`, at most `B`
long, meets it with chance at least `r`. -/
theorem avoid_block (G : Set σ) {e : σ} {B : ℕ} {r : ℝ} (hr : r ≤ 1)
    (hG : e ∉ G → ∃ w : List α, w.length ≤ B ∧ r ≤ ((Fintype.card α : ℝ) ^ w.length)⁻¹
      ∧ M.evalFrom e w ∈ G) :
    (uniformStrings B).real (avoidSet M G e B) ≤ 1 - r := by
  by_cases he : e ∈ G
  · have : avoidSet M G e B = ∅ :=
      Set.eq_empty_of_forall_notMem fun z hz => hz 0 (Nat.zero_le _) (by simpa using he)
    rw [this, measureReal_empty]
    linarith
  obtain ⟨w, hw, hrw, hwG⟩ := hG he
  have hsplit : B = w.length + (B - w.length) := by omega
  set Pre : Set (FreeMonoid α) := {x | prefixOf x w.length ∈ ({FreeMonoid.ofList w} : Set _)}
  have hsub : avoidSet M G e B ⊆ Preᶜ := by
    intro z hz hpre
    have : z.toList.take w.length = w := by
      have := congrArg FreeMonoid.toList (Set.mem_singleton_iff.1 hpre)
      simpa [prefixOf] using this
    exact hz w.length hw (by rw [this]; exact hwG)
  have hpre : (uniformStrings B).real Pre = ((Fintype.card α : ℝ) ^ w.length)⁻¹ := by
    have h1 := uniformStrings_real_prefix (α := α) w.length (B - w.length)
      ({FreeMonoid.ofList w} : Set _)
    rw [← hsplit] at h1
    rw [h1]
    exact uniformStrings_real_singleton w.length ⟨w, rfl⟩
  have hc : (uniformStrings B).real Preᶜ = 1 - (uniformStrings B).real Pre := by
    rw [measureReal_compl (MeasurableSpace.measurableSet_top), probReal_univ]
  have := measureReal_mono (μ := uniformStrings (α := α) B) hsub
  linarith

omit [Fintype σ] in
/-- A run that, from each state of the closed `P` outside `G`, can meet `G` along a word of chance
at least `r`, at most `B` long, has not met `G` after `m B` letters with chance at most
`(1 − r)^m`. -/
theorem avoid_le (G P : Set σ) {B : ℕ} {r : ℝ} (hr : r ≤ 1) (hP : ∀ e ∈ P, ∀ a, M.step e a ∈ P)
    (hG : ∀ e ∈ P, e ∉ G → ∃ w : List α, w.length ≤ B ∧ r ≤ ((Fintype.card α : ℝ) ^ w.length)⁻¹
      ∧ M.evalFrom e w ∈ G) :
    ∀ (m : ℕ) (e : σ), e ∈ P →
      (uniformStrings (m * B)).real (avoidSet M G e (m * B)) ≤ (1 - r) ^ m := by
  classical
  intro m
  induction m with
  | zero => intro e _; rw [pow_zero]; exact measureReal_le_one
  | succ m ih =>
    intro e he
    have hA : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    -- each first block either meets `G`, or leaves a run from a state of `P`
    have hterm : ∀ w : List.Vector α B,
        (uniformStrings (m * B)).real
          {z | FreeMonoid.ofList w.toList * z ∈ avoidSet M G e (B + m * B)}
        ≤ (1 - r) ^ m * (if FreeMonoid.ofList w.toList ∈ avoidSet M G e B then 1 else 0) := by
      intro w
      split_ifs with hw
      · rw [mul_one]
        refine (measureReal_mono fun z hz => ?_).trans
          (ih _ (evalFrom_mem_of_closed M hP w.toList he))
        intro i hi
        have h := hz (B + i) (by omega)
        simp only [FreeMonoid.toList_mul, FreeMonoid.toList_ofList] at h
        rwa [show B + i = w.toList.length + i by simp, List.take_length_add_append,
          DFA.evalFrom_of_append] at h
      · rw [mul_zero]
        have : {z | FreeMonoid.ofList w.toList * z ∈ avoidSet M G e (B + m * B)} = ∅ := by
          refine Set.eq_empty_of_forall_notMem fun z hz => hw fun i hi => ?_
          have h := hz i (by omega)
          simp only [FreeMonoid.toList_mul, FreeMonoid.toList_ofList] at h
          rw [List.take_append_of_le_length (by simp; omega)] at h
          simpa using h
        rw [this, measureReal_empty]
    have hsum := Finset.sum_le_sum fun (w : List.Vector α B) (_ : w ∈ Finset.univ) =>
      div_le_div_of_nonneg_right (hterm w) (pow_pos hA B).le
    have hblock := avoid_block M G hr (hG e he)
    rw [show (m + 1) * B = B + m * B by ring, uniformStrings_real_append]
    refine hsum.trans ?_
    rw [← Finset.sum_div, ← Finset.mul_sum, mul_div_assoc, ← uniformStrings_real_sum, pow_succ]
    exact mul_le_mul_of_nonneg_left hblock (pow_nonneg (by linarith) _)

omit [Fintype σ] in
/-- `avoid_le` after `n` letters: chance at most `(1 − r)^⌊n/B⌋`. -/
theorem avoid_le' (G P : Set σ) {B : ℕ} {r : ℝ} (hr : r ≤ 1) (hP : ∀ e ∈ P, ∀ a, M.step e a ∈ P)
    (hG : ∀ e ∈ P, e ∉ G → ∃ w : List α, w.length ≤ B ∧ r ≤ ((Fintype.card α : ℝ) ^ w.length)⁻¹
      ∧ M.evalFrom e w ∈ G) (n : ℕ) {e : σ} (he : e ∈ P) :
    (uniformStrings n).real (avoidSet M G e n) ≤ (1 - r) ^ (n / B) := by
  have hm : n / B * B ≤ n := Nat.div_mul_le_self n B
  obtain ⟨t, ht⟩ := Nat.exists_eq_add_of_le hm
  have hsub : avoidSet M G e n ⊆ {x | prefixOf x (n / B * B) ∈ avoidSet M G e (n / B * B)} := by
    intro z hz i hi
    simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take]
    rw [min_eq_left hi]
    exact hz i (hi.trans hm)
  refine (measureReal_mono hsub).trans ?_
  have h := uniformStrings_real_prefix (n / B * B) t (avoidSet M G e (n / B * B))
  rw [← ht] at h
  rw [h]
  exact avoid_le M G P hr hP hG (n / B) e he

/-- A state set entered only from itself, and left from each of its states by some word, holds at
most `(1 − ρ)^⌊n/|Q|⌋` of the strings of length `n`. -/
theorem stay_le (B : Set σ) (hin : ∀ e a, M.step e a ∈ B → e ∈ B)
    (hout : ∀ e ∈ B, ∃ w : List α, M.evalFrom e w ∉ B) (n : ℕ) :
    (uniformStrings n).real {y | M.eval y.toList ∈ B}
      ≤ (1 - wordRate α σ) ^ (n / Fintype.card σ) := by
  have hsub : {y : FreeMonoid α | M.eval y.toList ∈ B} ⊆ avoidSet M Bᶜ M.start n := by
    intro y hy i _ hi
    apply hi
    have e : M.eval y.toList
        = M.evalFrom (M.evalFrom M.start (y.toList.take i)) (y.toList.drop i) := by
      rw [← DFA.evalFrom_of_append, List.take_append_drop]
      rfl
    exact mem_of_evalFrom_mem M hin (y.toList.drop i) (by rw [← e]; exact hy)
  refine (measureReal_mono hsub).trans (avoid_le' M Bᶜ Set.univ wordRate_le_one
    (fun _ _ _ => trivial) (fun e _ he => ?_) n (Set.mem_univ _))
  obtain ⟨w, hw⟩ := hout e (not_not.1 he)
  obtain ⟨w', hw'l, hw'⟩ := exists_short_word M (rfl : M.evalFrom e w = M.evalFrom e w)
  exact ⟨w', hw'l.le, wordRate_le_pow hw'l, by rw [hw']; exact hw⟩

/-- The strings that visit `q` at some position from `k` to `L`. -/
def visitSet (k L : ℕ) (q : σ) : Set (FreeMonoid α) :=
  {x | ∃ j, k ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}

section Visited

variable {M}

omit [Fintype σ] in
/-- A string visiting `q₀` from `k` on is, at `k`, in a state that reaches `q₀`. -/
lemma visit_le_reach {k L : ℕ} (hk : k ≤ L) (q₀ : σ) :
    (uniformStrings L).real (visitSet M k L q₀)
      ≤ (uniformStrings k).real {y | M.eval y.toList ∈ {e | Reach M e q₀}} := by
  have hsub : visitSet M k L q₀
      ⊆ {x | prefixOf x k ∈ {y | M.eval y.toList ∈ {e | Reach M e q₀}}} := by
    rintro x ⟨j, hkj, -, hj⟩
    refine ⟨(x.toList.take j).drop k, ?_⟩
    simp only [prefixOf, FreeMonoid.toList_ofList] at hj ⊢
    rw [DFA.eval, ← DFA.evalFrom_of_append, ← hj, DFA.eval]
    congr 1
    conv_rhs => rw [← List.take_append_drop k (x.toList.take j)]
    rw [List.take_take, min_eq_left hkj]
  refine (measureReal_mono hsub).trans (le_of_eq ?_)
  have h := uniformStrings_real_prefix k (L - k) {y | M.eval y.toList ∈ {e | Reach M e q₀}}
  rwa [Nat.add_sub_cancel' hk] at h

/-- Where `(1 − ρ)^⌊k/|Q|⌋ < c₀`, a start visited from `k` on by at least `c₀` of the strings is
reached back from every state it reaches, and each of those is visited from `k` on by at least
`(c₀ − (1 − ρ)^⌊k/|Q|⌋)(1 − (1 − ρ)^⌊(L − k)/|Q|⌋)` of them. -/
theorem visit_of_reach {k L : ℕ} (hk : k ≤ L) {c₀ : ℝ} {q₀ : σ}
    (hq₀ : c₀ ≤ (uniformStrings L).real (visitSet M k L q₀))
    (hT : (1 - wordRate α σ) ^ (k / Fintype.card σ) < c₀) :
    (∀ q, Reach M q₀ q → Reach M q q₀) ∧
    ∀ q, Reach M q₀ q → (c₀ - (1 - wordRate α σ) ^ (k / Fintype.card σ))
        * (1 - (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ))
      ≤ (uniformStrings L).real (visitSet M k L q) := by
  classical
  have hρ1 := wordRate_le_one (α := α) (σ := σ)
  have hρ0 := wordRate_nonneg (α := α) (σ := σ)
  set U : Set σ := {e | Reach M e q₀} with hU
  set R : Set σ := {e | Reach M q₀ e} with hR
  have hUin : ∀ e a, M.step e a ∈ U → e ∈ U := fun e a ⟨w, hw⟩ => ⟨a :: w, hw⟩
  have hW : (uniformStrings L).real (visitSet M k L q₀)
      ≤ (uniformStrings k).real {y | M.eval y.toList ∈ U} := visit_le_reach hk q₀
  -- `q₀` is reached back from every state it reaches
  have hrec : ∀ q, Reach M q₀ q → Reach M q q₀ := by
    intro q₁ ⟨w₁, hw₁⟩
    by_contra hno
    have hout : ∀ e ∈ U, ∃ w : List α, M.evalFrom e w ∉ U := fun e ⟨w, hw⟩ =>
      ⟨w ++ w₁, by rw [DFA.evalFrom_of_append, hw, hw₁]; exact hno⟩
    have := stay_le M U hUin hout k
    linarith
  refine ⟨hrec, fun q ⟨wq, hwq⟩ => ?_⟩
  -- the states that reach `q₀` but are not reached by it hold at most `(1 − ρ)^⌊k/|Q|⌋` at `k`
  have hB : (uniformStrings k).real {y | M.eval y.toList ∈ U \ R}
      ≤ (1 - wordRate α σ) ^ (k / Fintype.card σ) := by
    refine stay_le M (U \ R)
      (fun e a ⟨hUe, hRe⟩ => ⟨hUin e a hUe, fun ⟨w, hw⟩ => hRe ⟨w ++ [a], ?_⟩⟩)
      (fun e ⟨⟨w, hw⟩, _⟩ => ⟨w, fun h => h.2 ⟨[], by rw [hw]; rfl⟩⟩) k
    rw [DFA.evalFrom_append_singleton, hw]
  have hPk : c₀ - (1 - wordRate α σ) ^ (k / Fintype.card σ)
      ≤ (uniformStrings k).real {y | M.eval y.toList ∈ R} := by
    have hsub : {y : FreeMonoid α | M.eval y.toList ∈ U}
        ⊆ {y | M.eval y.toList ∈ R} ∪ {y | M.eval y.toList ∈ U \ R} := fun y hy => by
      by_cases h : M.eval y.toList ∈ R
      · exact Or.inl h
      · exact Or.inr ⟨hy, h⟩
    have := (measureReal_mono (μ := uniformStrings (α := α) k) hsub).trans
      (measureReal_union_le _ _)
    linarith
  -- from a state `q₀` reaches, `q` is met within the window
  have hhit : ∀ e ∈ R, 1 - (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ)
      ≤ (uniformStrings (L - k)).real {z | ∃ i ≤ L - k, M.evalFrom e (z.toList.take i) = q} := by
    intro e he
    have h := avoid_le' M {q} R wordRate_le_one (fun e ⟨w, hw⟩ a => ⟨w ++ [a], by
        rw [DFA.evalFrom_append_singleton, hw]⟩)
      (fun e he _ => by
        obtain ⟨w₀, hw₀⟩ := hrec e he
        obtain ⟨w', hw'l, hw'⟩ := exists_short_word M
          (show M.evalFrom e (w₀ ++ wq) = q by rw [DFA.evalFrom_of_append, hw₀, hwq])
        exact ⟨w', hw'l.le, wordRate_le_pow hw'l, hw'⟩) (L - k) he
    have hc : {z : FreeMonoid α | ∃ i ≤ L - k, M.evalFrom e (z.toList.take i) = q}
        = (avoidSet M {q} e (L - k))ᶜ := by
      ext z; simp [avoidSet]
    rw [hc, measureReal_compl (MeasurableSpace.measurableSet_top), probReal_univ]
    linarith
  have hA : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hterm : ∀ w : List.Vector α k,
      (if FreeMonoid.ofList w.toList ∈ {y : FreeMonoid α | M.eval y.toList ∈ R} then (1 : ℝ) else 0)
          * (1 - (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ))
        ≤ (uniformStrings (L - k)).real
          {z | FreeMonoid.ofList w.toList * z ∈ visitSet M k L q} := by
    intro w
    split_ifs with hw
    · rw [one_mul]
      refine (hhit _ hw).trans (measureReal_mono fun z ⟨i, hi, hz⟩ =>
        ⟨k + i, by omega, by omega, ?_⟩)
      simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList] at hz ⊢
      rw [show k + i = w.toList.length + i by simp, List.take_length_add_append, DFA.eval,
        DFA.evalFrom_of_append]
      exact hz
    · rw [zero_mul]; exact measureReal_nonneg
  have hsum : (uniformStrings k).real {y | M.eval y.toList ∈ R}
        * (1 - (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ))
      ≤ (uniformStrings L).real (visitSet M k L q) := by
    have e := uniformStrings_real_append k (L - k) (visitSet M k L q)
    rw [Nat.add_sub_cancel' hk] at e
    rw [e, uniformStrings_real_sum, div_mul_eq_mul_div, Finset.sum_mul, Finset.sum_div]
    exact Finset.sum_le_sum fun w _ => div_le_div_of_nonneg_right (hterm w) (pow_pos hA k).le
  have hδW : 0 ≤ 1 - (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ) := by
    have : (1 - wordRate α σ) ^ ((L - k) / Fintype.card σ) ≤ 1 :=
      pow_le_one₀ (by linarith) (by linarith)
    linarith
  exact (mul_le_mul_of_nonneg_right hPk hδW).trans hsum

end Visited

omit [Fintype σ] in
/-- A run from `q₀` stays in the states `q₀` reaches. -/
theorem rerootAccuracy_le_of_reach (L : ℕ) (S T : Set σ) (q₀ : σ)
    (hT : ∀ q, Reach M q₀ q → q ∈ T) :
    rerootAccuracy M L S q₀ ≤ rerootAccuracy M L T q₀ :=
  measureReal_mono fun _ hx => ⟨fun _ _ => hT _ ⟨_, rfl⟩, hx.2⟩

/-- A start visited from `k = ⌈L/2⌉` on by at least `c₀` of the strings, with accuracy `1 − γ` over
any set, is covered at `c` with accuracy `1 − γ` over the covered states, once `0 < c` and
`c ≤ (c₀ − (1 − ρ)^⌊k/|Q|⌋)(1 − (1 − ρ)^⌊(L − k)/|Q|⌋)`. -/
theorem ceil_of_visited {L : ℕ} {c₀ c γ : ℝ} (S : Set σ) {q₀ : σ} (hc : 0 < c)
    (hcc : c ≤ (c₀ - (1 - wordRate α σ) ^ ((L + 1) / 2 / Fintype.card σ))
      * (1 - (1 - wordRate α σ) ^ ((L - (L + 1) / 2) / Fintype.card σ)))
    (hq₀ : c₀ ≤ windowVisitRate M L q₀) (hacc : 1 - γ ≤ rerootAccuracy M L S q₀) :
    ∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q := by
  have hρ1 := wordRate_le_one (α := α) (σ := σ)
  have hρ0 := wordRate_nonneg (α := α) (σ := σ)
  have hW1 : (1 - wordRate α σ) ^ ((L - (L + 1) / 2) / Fintype.card σ) ≤ 1 :=
    pow_le_one₀ (by linarith) (by linarith)
  have hT : (1 - wordRate α σ) ^ ((L + 1) / 2 / Fintype.card σ) < c₀ := by
    by_contra h
    push Not at h
    have : (c₀ - (1 - wordRate α σ) ^ ((L + 1) / 2 / Fintype.card σ))
        * (1 - (1 - wordRate α σ) ^ ((L - (L + 1) / 2) / Fintype.card σ)) ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
    linarith
  obtain ⟨-, hcov⟩ := visit_of_reach (M := M) (by omega : (L + 1) / 2 ≤ L) hq₀ hT
  have hT' : ∀ q, Reach M q₀ q → q ∈ coveredStates M L c := fun q hq => hcc.trans (hcov q hq)
  exact ⟨q₀, hT' q₀ ⟨[], rfl⟩, hacc.trans (rerootAccuracy_le_of_reach M L S _ q₀ hT')⟩

end OrthoDFA

end
