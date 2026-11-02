import OrthoDFA.Proofs.GateLoop

/-!
# The centre check passes near the true centre

`centreFits` reads, on the uniform pool's sample, how often the prefixes the family votes
strongly against read accept, and how often those it votes strongly for read reject.  Near the
true centre

    c* = (1 − ηIn + ηOut)/2

with the gap `1 − ηIn − ηOut` at least `4·sig`, the strongly rejected prefixes are the
complement's, but for those the family flips more than a twentieth of the time and those its
vote misfires on, and the language's only where at least half the family flips them or the vote
misfires.  A language prefix there reads accept `gap` more often than the complement's, so a tenth
of them moves the estimate by a tenth of the gap, and the threshold `centre − sig` sits a fifth
of the gap above `ηOut`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- The true centre: halfway between the complement's rate of reading accept and the
language's. -/
noncomputable def Oracle.cmid (O : Oracle μ S) : ℝ := (1 - max O.ηIn 0 + max O.ηOut 0) / 2

/-- A prefix of class `b` that more than a share `f` of `F` flips. -/
def flipHeavy (O : Oracle μ S) (b : Bool) (f : ℝ) (F : Finset S) (p : S) : Prop :=
  O.label p = sideClass b ∧ (F.card : ℝ) * f < flipCount O F p

/-- Markov on the shift the family carries on class `b`. -/
theorem flipHeavy_mass_le (O : Oracle μ S) {ctr η₀ : ℝ} (hη : O.Straddles ctr η₀)
    (hη₀ : η₀ < 1 / 2) (Dj : Measure S) [IsProbabilityMeasure Dj] (F : Finset S) (b : Bool)
    {f : ℝ} (hf : 0 < f) (Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hW : ∑ v ∈ F, wMass O Dj (clsW O ctr b) v ≤ F.card * Δ) :
    Dj.real {p | flipHeavy O b f F p} ≤ Δ / (f * clsShift O ctr b) := by
  classical
  have hw := clsShift_pos O hη hη₀ b
  rcases Nat.eq_zero_or_pos F.card with hF0 | hF
  · have hz : {p | flipHeavy O b f F p} = (∅ : Set S) := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, flipHeavy, not_and,
        not_lt]
      intro _
      simp [flipCount, Finset.card_eq_zero.1 hF0]
    rw [hz, measureReal_empty]
    positivity
  have hFR : (0 : ℝ) < F.card := by exact_mod_cast hF
  have hsub : {p | flipHeavy O b f F p}
      ⊆ {p | F.card * (f * clsShift O ctr b) ≤ ∑ v ∈ F, clsW O ctr b p * O.flip v p} := by
    rintro p ⟨hp, hbad⟩
    simp only [Set.mem_setOf_eq]
    have hsum : ∑ v ∈ F, clsW O ctr b p * O.flip v p = clsShift O ctr b * flipCount O F p := by
      rw [← Finset.mul_sum, ← flipCount_eq_sum]
      unfold clsW; rw [if_pos hp]
    rw [hsum]
    nlinarith
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (wMass_markov O Dj (clsW O ctr b) (clsW_nonneg O hη hη₀ b)
    (clsW_le O hη hη₀ b) F (by positivity)) ?_
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  have hfw : 0 ≤ f * clsShift O ctr b := by positivity
  have := mul_le_mul_of_nonneg_right hW hfw
  linarith [show (F.card : ℝ) * Δ * (f * clsShift O ctr b)
    = Δ * (F.card * (f * clsShift O ctr b)) by ring]

/-- Near the true centre the strong thresholds sit a fifth of the gap inside each class's rate. -/
lemma mid_facts (O : Oracle μ S) {c η₀ : ℝ} (hg : 4 * (1 / 2 - η₀) ≤ O.gap)
    (hc : |c - O.cmid| ≤ (1 / 2 - η₀) / 5) :
    max O.ηOut 0 + O.gap / 5 ≤ c - (1 / 2 - η₀)
      ∧ max O.ηIn 0 + O.gap / 5 ≤ 1 - c - (1 / 2 - η₀)
      ∧ O.cmid = max O.ηOut 0 + O.gap / 2
      ∧ 1 - max O.ηIn 0 = O.cmid + O.gap / 2 := by
  have h := abs_le.1 hc
  have hm : O.cmid = max O.ηOut 0 + O.gap / 2 := by unfold Oracle.cmid Oracle.gap; ring
  have hm' : 1 - max O.ηIn 0 = O.cmid + O.gap / 2 := by unfold Oracle.cmid Oracle.gap; ring
  refine ⟨by linarith [h.1, h.2], by linarith [h.1, h.2], hm, hm'⟩

open scoped Classical in
/-- A prefix whose vote over `F` misses the strong reject threshold the wrong way: one of the
complement's that a twentieth of `F` or fewer flip, voted above it, or one of the language's
that half of `F` or fewer flip, voted at or below it. -/
def missRej (O : Oracle μ S) (η₀ : ℝ) (B : State) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (O.label p = 0 ∧ flipCount O F p ≤ F.card * (1 / 20)
      ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀)) < (voteCount O.mq F p ω : ℝ))
    ∨ (O.label p = 1 ∧ flipCount O F p ≤ F.card * (1 / 2)
      ∧ (voteCount O.mq F p ω : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀)))

open scoped Classical in
/-- `missRej` at the strong accept threshold, the classes swapped. -/
def missAcc (O : Oracle μ S) (η₀ : ℝ) (B : State) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (O.label p = 1 ∧ flipCount O F p ≤ F.card * (1 / 20)
      ∧ (voteCount O.mq F p ω : ℝ) < ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)))
    ∨ (O.label p = 0 ∧ flipCount O F p ≤ F.card * (1 / 2)
      ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)) ≤ (voteCount O.mq F p ω : ℝ))

open scoped Classical in
/-- The seed's count against side `accept` on `A` within `hoeffDev #A a` of its mean, from
above. -/
def readLive (O : Oracle μ S) (accept : Bool) (A : Finset S) (a : ℝ) (ω : Ω) : Prop :=
  (againstOn O.mq accept A ω : ℝ) ≤ ∑ p ∈ A, μ[againstVar O (sideClass accept) p] + hoeffDev A.card a

open scoped Classical in
/-- A count on `A` of those of `X` and of `Y` bounds the count of `X ∪ Y`'s members. -/
lemma card_filter_or_le (A : Finset S) (X Y : S → Prop) :
    ((A.filter (fun p => X p ∨ Y p)).card : ℝ)
      ≤ ((A.filter X).card : ℝ) + ((A.filter Y).card : ℝ) := by
  classical
  have := Finset.card_union_le (A.filter X) (A.filter Y)
  rw [← Finset.filter_or] at this
  exact_mod_cast this

open scoped Classical in
/-- The strong reject side passes the check once the sample's counts are in order: its size is
at least `Smin`, the language holds at most a tenth of it, and the deviations fit in a tenth of
the gap. -/
lemma readBound_rej_le (O : Oracle μ S) {η₀ a aT n εb h e Smin : ℝ} {B : State}
    {F A : Finset S} {ω : Ω} (hFk : F.card = B.k - 1) (hη₀ : η₀ ≤ 1 / 2)
    (hg : 4 * (1 / 2 - η₀) ≤ O.gap) (hc : |B.centre - O.cmid| ≤ (1 / 2 - η₀) / 5)
    (hn : (A.card : ℝ) = n)
    (hout : n * εb ≤ ((A.filter (fun p => O.label p = sideClass false)).card : ℝ))
    (hin1 : ∃ p, O.label p = 1)
    (hheavyO : ((A.filter (flipHeavy O false (1 / 20) F)).card : ℝ) ≤ n * h)
    (hheavyI : ((A.filter (flipHeavy O true (1 / 20) F)).card : ℝ) ≤ n * h)
    (hmiss : ((A.filter (fun p => missRej O η₀ B F p ω)).card : ℝ) ≤ n * e)
    (hlive : readLive O false (strongReject O.mq η₀ B F A ω) aT ω)
    (hSmin : 0 < Smin) (hsize : Smin ≤ n * (εb - h - e)) (hcont : n * (h + e) ≤ Smin / 10)
    (hdev : ∀ s : ℕ, Smin ≤ s → hoeffDev s a + hoeffDev s aT ≤ O.gap / 10 * s) :
    readBound O.mq false (strongReject O.mq η₀ B F A ω) a ω ≤ B.centre - 1 / 2 + η₀ := by
  classical
  obtain ⟨hR, -, -, hm'⟩ := mid_facts O hg hc
  set SR := strongReject O.mq η₀ B F A ω with hSR
  have hvoteR : ∀ p, p ∈ SR ↔ p ∈ A
      ∧ (voteCount O.mq F p ω : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀)) := by
    intro p; rw [hSR, strongReject, Finset.mem_filter]
  -- the complement's prefixes are in `SR`, heavy or missed
  have hcov : A.filter (fun p => O.label p = sideClass false)
      ⊆ SR ∪ (A.filter (flipHeavy O false (1 / 20) F)
        ∪ A.filter (fun p => missRej O η₀ B F p ω)) := by
    intro p hp
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    by_cases hS : p ∈ SR
    · exact Finset.mem_union_left _ hS
    · refine Finset.mem_union_right _ ?_
      by_cases hh : (F.card : ℝ) * (1 / 20) < flipCount O F p
      · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpA, hl, hh⟩)
      · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpA, Or.inl ⟨by simpa [sideClass]
          using hl, not_lt.1 hh, ?_⟩⟩)
        by_contra hv
        exact hS ((hvoteR p).2 ⟨hpA, not_lt.1 hv⟩)
  have hSRcard : Smin ≤ (SR.card : ℝ) := by
    have h1 := Finset.card_le_card hcov
    have h2 := Finset.card_union_le SR (A.filter (flipHeavy O false (1 / 20) F)
        ∪ A.filter (fun p => missRej O η₀ B F p ω))
    have h3 := Finset.card_union_le (A.filter (flipHeavy O false (1 / 20) F))
        (A.filter (fun p => missRej O η₀ B F p ω))
    have h4 := le_trans h1 (le_trans h2 (Nat.add_le_add_left h3 _))
    have : ((A.filter (fun p => O.label p = sideClass false)).card : ℝ)
        ≤ SR.card + ((A.filter (flipHeavy O false (1 / 20) F)).card
          + (A.filter (fun p => missRej O η₀ B F p ω)).card) := by exact_mod_cast h4
    nlinarith
  have hSRpos : 0 < SR.card := by
    have : (0 : ℝ) < SR.card := lt_of_lt_of_le hSmin hSRcard
    exact_mod_cast this
  -- the language's prefixes in `SR`, heavy or missed
  have hinS : SR.filter (fun p => O.label p = 1)
      ⊆ A.filter (flipHeavy O true (1 / 20) F) ∪ A.filter (fun p => missRej O η₀ B F p ω) := by
    intro p hp
    obtain ⟨hpS, hl⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpA, hv⟩ := (hvoteR p).1 hpS
    by_cases hh : (F.card : ℝ) * (1 / 2) < flipCount O F p
    · refine Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpA, by simpa [sideClass] using hl, ?_⟩)
      have : (0 : ℝ) ≤ F.card := Nat.cast_nonneg _
      linarith
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpA, Or.inr ⟨hl, not_lt.1 hh, hv⟩⟩)
  have hinCard : ((SR.filter (fun p => O.label p = 1)).card : ℝ) ≤ SR.card / 10 := by
    have h1 := Finset.card_le_card hinS
    have h2 := Finset.card_union_le (A.filter (flipHeavy O true (1 / 20) F))
      (A.filter (fun p => missRej O η₀ B F p ω))
    have : ((SR.filter (fun p => O.label p = 1)).card : ℝ)
        ≤ (A.filter (flipHeavy O true (1 / 20) F)).card
          + (A.filter (fun p => missRej O η₀ B F p ω)).card := by exact_mod_cast h1.trans h2
    nlinarith
  -- the mean the seed reads accept at on `SR`
  obtain ⟨pin, hpin⟩ := hin1
  have hηIn0 : 0 ≤ O.ηIn := by
    have := O.rate_nonneg pin
    rw [rate_of_label O true pin, if_pos (by simp [sideClass, hpin])] at this
    simpa [rightRate] using this
  have hmean : ∑ p ∈ SR, μ[againstVar O (sideClass false) p]
      ≤ SR.card * max O.ηOut 0 + ((SR.filter (fun p => O.label p = 1)).card : ℝ) * O.gap := by
    have hterm : ∀ p ∈ SR, μ[againstVar O (sideClass false) p]
        ≤ max O.ηOut 0 + (if O.label p = 1 then O.gap else 0) := by
      intro p _
      rw [againstVar_mean O (sideClass_bit false) p, rate_of_label O false p]
      unfold rightRate wrongRate sideClass
      rcases O.label_bit p with hl | hl
      · simp only [hl, Bool.false_eq_true, ↓reduceIte]
        norm_num
      · simp only [hl, Bool.false_eq_true, ↓reduceIte]
        norm_num
        rw [max_eq_left hηIn0] at hm'
        linarith [hm', (mid_facts O hg hc).2.2.1]
    refine le_trans (Finset.sum_le_sum hterm) (le_of_eq ?_)
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, Finset.sum_ite,
      Finset.sum_const, Finset.sum_const_zero, nsmul_eq_mul]
    ring
  have hdevS := hdev SR.card hSRcard
  have hd0 : 0 ≤ hoeffDev SR.card a := Real.sqrt_nonneg _
  unfold readBound
  rw [if_neg (by omega), div_le_iff₀ (by exact_mod_cast hSRpos)]
  unfold readLive at hlive
  have hg0 : 0 ≤ O.gap := by linarith
  have hgin := mul_le_mul_of_nonneg_right hinCard hg0
  have hRS := mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg SR.card : (0 : ℝ) ≤ SR.card)
  nlinarith

open scoped Classical in
/-- `readBound_rej_le` at the strong accept side, the classes swapped. -/
lemma readBound_acc_le (O : Oracle μ S) {η₀ a aT n εb h e Smin : ℝ} {B : State}
    {F A : Finset S} {ω : Ω} (hFk : F.card = B.k - 1) (hη₀ : η₀ ≤ 1 / 2)
    (hg : 4 * (1 / 2 - η₀) ≤ O.gap) (hc : |B.centre - O.cmid| ≤ (1 / 2 - η₀) / 5)
    (hn : (A.card : ℝ) = n)
    (hin : n * εb ≤ ((A.filter (fun p => O.label p = sideClass true)).card : ℝ))
    (hout1 : ∃ p, O.label p = 0)
    (hheavyO : ((A.filter (flipHeavy O false (1 / 20) F)).card : ℝ) ≤ n * h)
    (hheavyI : ((A.filter (flipHeavy O true (1 / 20) F)).card : ℝ) ≤ n * h)
    (hmiss : ((A.filter (fun p => missAcc O η₀ B F p ω)).card : ℝ) ≤ n * e)
    (hlive : readLive O true (strongAccept O.mq η₀ B F A ω) aT ω)
    (hSmin : 0 < Smin) (hsize : Smin ≤ n * (εb - h - e)) (hcont : n * (h + e) ≤ Smin / 10)
    (hdev : ∀ s : ℕ, Smin ≤ s → hoeffDev s a + hoeffDev s aT ≤ O.gap / 10 * s) :
    readBound O.mq true (strongAccept O.mq η₀ B F A ω) a ω ≤ 1 / 2 - B.centre + η₀ := by
  classical
  obtain ⟨-, hA, hm, -⟩ := mid_facts O hg hc
  set SA := strongAccept O.mq η₀ B F A ω with hSA
  have hvoteA : ∀ p, p ∈ SA ↔ p ∈ A
      ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)) ≤ (voteCount O.mq F p ω : ℝ) := by
    intro p; rw [hSA, strongAccept, Finset.mem_filter]
  have hcov : A.filter (fun p => O.label p = sideClass true)
      ⊆ SA ∪ (A.filter (flipHeavy O true (1 / 20) F)
        ∪ A.filter (fun p => missAcc O η₀ B F p ω)) := by
    intro p hp
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    by_cases hS : p ∈ SA
    · exact Finset.mem_union_left _ hS
    · refine Finset.mem_union_right _ ?_
      by_cases hh : (F.card : ℝ) * (1 / 20) < flipCount O F p
      · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpA, hl, hh⟩)
      · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpA, Or.inl ⟨by simpa [sideClass]
          using hl, not_lt.1 hh, ?_⟩⟩)
        by_contra hv
        exact hS ((hvoteA p).2 ⟨hpA, not_lt.1 hv⟩)
  have hSAcard : Smin ≤ (SA.card : ℝ) := by
    have h1 := Finset.card_le_card hcov
    have h2 := Finset.card_union_le SA (A.filter (flipHeavy O true (1 / 20) F)
        ∪ A.filter (fun p => missAcc O η₀ B F p ω))
    have h3 := Finset.card_union_le (A.filter (flipHeavy O true (1 / 20) F))
        (A.filter (fun p => missAcc O η₀ B F p ω))
    have h4 := le_trans h1 (le_trans h2 (Nat.add_le_add_left h3 _))
    have : ((A.filter (fun p => O.label p = sideClass true)).card : ℝ)
        ≤ SA.card + ((A.filter (flipHeavy O true (1 / 20) F)).card
          + (A.filter (fun p => missAcc O η₀ B F p ω)).card) := by exact_mod_cast h4
    nlinarith
  have hSApos : 0 < SA.card := by
    have : (0 : ℝ) < SA.card := lt_of_lt_of_le hSmin hSAcard
    exact_mod_cast this
  have houtS : SA.filter (fun p => O.label p = 0)
      ⊆ A.filter (flipHeavy O false (1 / 20) F) ∪ A.filter (fun p => missAcc O η₀ B F p ω) := by
    intro p hp
    obtain ⟨hpS, hl⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpA, hv⟩ := (hvoteA p).1 hpS
    by_cases hh : (F.card : ℝ) * (1 / 2) < flipCount O F p
    · refine Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpA, by simpa [sideClass] using hl, ?_⟩)
      have : (0 : ℝ) ≤ F.card := Nat.cast_nonneg _
      linarith
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpA, Or.inr ⟨hl, not_lt.1 hh, hv⟩⟩)
  have houtCard : ((SA.filter (fun p => O.label p = 0)).card : ℝ) ≤ SA.card / 10 := by
    have h1 := Finset.card_le_card houtS
    have h2 := Finset.card_union_le (A.filter (flipHeavy O false (1 / 20) F))
      (A.filter (fun p => missAcc O η₀ B F p ω))
    have : ((SA.filter (fun p => O.label p = 0)).card : ℝ)
        ≤ (A.filter (flipHeavy O false (1 / 20) F)).card
          + (A.filter (fun p => missAcc O η₀ B F p ω)).card := by exact_mod_cast h1.trans h2
    nlinarith
  obtain ⟨pout, hpout⟩ := hout1
  have hηOut0 : 0 ≤ O.ηOut := by
    have := O.rate_nonneg pout
    rw [rate_of_label O false pout, if_pos (by simp [sideClass, hpout])] at this
    simpa [rightRate] using this
  have hmean : ∑ p ∈ SA, μ[againstVar O (sideClass true) p]
      ≤ SA.card * max O.ηIn 0 + ((SA.filter (fun p => O.label p = 0)).card : ℝ) * O.gap := by
    have hterm : ∀ p ∈ SA, μ[againstVar O (sideClass true) p]
        ≤ max O.ηIn 0 + (if O.label p = 0 then O.gap else 0) := by
      intro p _
      rw [againstVar_mean O (sideClass_bit true) p, rate_of_label O true p]
      unfold rightRate wrongRate sideClass
      rcases O.label_bit p with hl | hl
      · simp only [hl, ↓reduceIte]
        norm_num
        have hg' : O.gap = 1 - max O.ηIn 0 - max O.ηOut 0 := rfl
        rw [max_eq_left hηOut0] at hg'
        linarith
      · simp only [hl, ↓reduceIte]
        norm_num
    refine le_trans (Finset.sum_le_sum hterm) (le_of_eq ?_)
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, Finset.sum_ite,
      Finset.sum_const, Finset.sum_const_zero, nsmul_eq_mul]
    ring
  have hdevS := hdev SA.card hSAcard
  have hd0 : 0 ≤ hoeffDev SA.card a := Real.sqrt_nonneg _
  unfold readBound
  rw [if_neg (by omega), div_le_iff₀ (by exact_mod_cast hSApos)]
  unfold readLive at hlive
  have hg0 : 0 ≤ O.gap := by linarith
  have hgout := mul_le_mul_of_nonneg_right houtCard hg0
  have hAS := mul_le_mul_of_nonneg_left hA (Nat.cast_nonneg SA.card : (0 : ℝ) ≤ SA.card)
  nlinarith

/-- A light prefix's vote misses the strong reject threshold at most `exp(−κ v²/2)` of the time,
`v = voteSlack`, near the true centre and with a family of at least `10/sig`. -/
lemma missRej_prob_le (O : Oracle μ S) {η₀ : ℝ} {B : State} (F : Finset S) (p : S)
    (hFk : F.card = B.k - 1) (hη₀ : η₀ < 1 / 2)
    (hg : 4 * (1 / 2 - η₀) ≤ O.gap) (hc : |B.centre - O.cmid| ≤ (1 / 2 - η₀) / 5)
    (hκ : 10 ≤ (F.card : ℝ) * (1 / 2 - η₀)) :
    μ.real {ω | missRej O η₀ B F p ω}
      ≤ Real.exp (-2 * F.card * (2 * (1 / 2 - η₀) / 5) ^ 2) := by
  classical
  have hσ : 0 < 1 / 2 - η₀ := by linarith
  obtain ⟨hR, -, hm, hm'⟩ := mid_facts O hg hc
  have hg0 : 0 ≤ O.gap := by linarith
  have hgle : O.gap ≤ 1 := by
    unfold Oracle.gap; linarith [le_max_right O.ηIn 0, le_max_right O.ηOut 0]
  have hc0 := abs_le.1 hc
  have hθ : ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀))
      = F.card * (B.centre - (1 / 2 - η₀)) := by rw [hFk]
  have hγ0 : (0 : ℝ) ≤ 2 * (1 / 2 - η₀) / 5 := by positivity
  have hE0 : 0 ≤ Real.exp (-2 * F.card * (2 * (1 / 2 - η₀) / 5) ^ 2) := (Real.exp_pos _).le
  set κ : ℝ := (F.card : ℝ) with hκdef
  have hκ0 : 0 ≤ κ := Nat.cast_nonneg _
  have hθ0 : 0 ≤ κ * (B.centre - (1 / 2 - η₀)) := by
    have : 0 ≤ B.centre - (1 / 2 - η₀) := by linarith [le_max_right O.ηOut 0]
    positivity
  rcases O.label_bit p with hl | hl
  · by_cases hfl : flipCount O F p ≤ F.card * (1 / 20)
    · have hr0 : 0 ≤ O.ηOut := by
        have := O.rate_nonneg p
        rw [rate_of_label O false p, if_pos (by simp [sideClass, hl])] at this
        simpa [rightRate] using this
      have hrate : O.rate p = O.ηOut := by
        rw [rate_of_label O false p, if_pos (by simp [sideClass, hl])]; rfl
      set c' : ℕ := ⌊κ * (B.centre - (1 / 2 - η₀))⌋₊ + 1 with hc'
      have hc'R : κ * (B.centre - (1 / 2 - η₀)) < c' := by
        rw [hc']; push_cast; exact Nat.lt_floor_add_one _
      have hsub : {ω | missRej O η₀ B F p ω} ⊆ {ω | c' ≤ voteCount O.mq F p ω} := by
        intro ω hω
        rcases hω with ⟨-, -, hv⟩ | ⟨hl1, -⟩
        · rw [hθ] at hv
          show c' ≤ voteCount O.mq F p ω
          rw [hc']
          exact (Nat.floor_lt hθ0).2 hv
        · rw [hl] at hl1; norm_num at hl1
      set q : ℝ := max O.ηOut 0 + O.gap / 20 with hq
      have hq0 : 0 ≤ q := by positivity
      have hq1 : q ≤ 1 := by
        have : max O.ηOut 0 ≤ 1 - O.gap := by
          unfold Oracle.gap; linarith [le_max_right O.ηIn 0]
        rw [hq]; linarith
      have hmean : meanVote O F p ≤ F.card * q := by
        have := meanVote_le_flip O F p hl (1 / 20) hg0 hfl
        rw [hrate] at this
        have : O.ηOut ≤ max O.ηOut 0 := le_max_left _ _
        rw [hq]; nlinarith
      have hc'2 : κ * q + 2 ≤ c' := by
        have : κ * q + 2 ≤ κ * (B.centre - (1 / 2 - η₀)) := by
          rw [hq]; nlinarith
        linarith
      have htail := (vote_tails O F p F.card q le_rfl hq0 hq1).2 c' hmean hc'2
      have hexp := binomSfGe_le_exp F.card c' hq0 hq1 hγ0 (by
        have : κ * (q + 2 * (1 / 2 - η₀) / 5) ≤ κ * (B.centre - (1 / 2 - η₀)) := by
          rw [hq]; nlinarith
        linarith)
      exact le_trans (measureReal_mono hsub (measure_ne_top _ _)) (htail.trans hexp)
    · have hz : {ω | missRej O η₀ B F p ω} = (∅ : Set Ω) := by
        ext ω
        constructor
        · rintro (⟨-, h2, -⟩ | ⟨h1, -⟩)
          · exact absurd h2 hfl
          · rw [hl] at h1; norm_num at h1
        · intro h; exact h.elim
      rw [hz, measureReal_empty]; exact hE0
  · by_cases hfl : flipCount O F p ≤ F.card * (1 / 2)
    · have hr0 : 0 ≤ O.ηIn := by
        have := O.rate_nonneg p
        rw [rate_of_label O true p, if_pos (by simp [sideClass, hl])] at this
        simpa [rightRate] using this
      have hrate : O.rate p = O.ηIn := by
        rw [rate_of_label O true p, if_pos (by simp [sideClass, hl])]; rfl
      rw [max_eq_left hr0] at hm'
      set b : ℕ := ⌊κ * (B.centre - (1 / 2 - η₀))⌋₊ with hb
      have hbR : (b : ℝ) ≤ κ * (B.centre - (1 / 2 - η₀)) := Nat.floor_le hθ0
      have hsub : {ω | missRej O η₀ B F p ω} ⊆ {ω | voteCount O.mq F p ω ≤ b} := by
        intro ω hω
        rcases hω with ⟨hl0, -⟩ | ⟨-, -, hv⟩
        · rw [hl] at hl0; norm_num at hl0
        · rw [hθ] at hv
          show voteCount O.mq F p ω ≤ b
          exact Nat.le_floor hv
      have hq0 : 0 ≤ O.cmid := by rw [hm]; positivity
      have hq1 : O.cmid ≤ 1 := by linarith
      have hmean : F.card * O.cmid ≤ meanVote O F p := by
        have := meanVote_ge_flip O F p hl (1 / 2) hg0 hfl
        rw [hrate] at this
        have e : 1 - O.ηIn - O.gap * (1 / 2) = O.cmid := by linarith
        rwa [e] at this
      have hcs : B.centre - (1 / 2 - η₀) ≤ O.cmid - 4 * (1 / 2 - η₀) / 5 := by linarith [hc0.2]
      have hκc := mul_le_mul_of_nonneg_left hcs hκ0
      have hb2 : (b : ℝ) + 2 ≤ κ * O.cmid := by
        have e : κ * (O.cmid - 4 * (1 / 2 - η₀) / 5) = κ * O.cmid - 4 / 5 * (κ * (1 / 2 - η₀)) := by
          ring
        linarith
      have htail := (vote_tails O F p F.card O.cmid le_rfl hq0 hq1).1 b hmean hb2
      have hexp := one_sub_binomSfGe_le_exp F.card b hq0 hq1 hγ0 (by
        have h2 : κ * (O.cmid - 4 * (1 / 2 - η₀) / 5) ≤ κ * (O.cmid - 2 * (1 / 2 - η₀) / 5) :=
          mul_le_mul_of_nonneg_left (by linarith) hκ0
        linarith)
      exact le_trans (measureReal_mono hsub (measure_ne_top _ _)) (htail.trans hexp)
    · have hz : {ω | missRej O η₀ B F p ω} = (∅ : Set Ω) := by
        ext ω
        constructor
        · rintro (⟨h1, -⟩ | ⟨-, h2, -⟩)
          · rw [hl] at h1; norm_num at h1
          · exact absurd h2 hfl
        · intro h; exact h.elim
      rw [hz, measureReal_empty]; exact hE0

/-- `missRej_prob_le` at the strong accept threshold. -/
lemma missAcc_prob_le (O : Oracle μ S) {η₀ : ℝ} {B : State} (F : Finset S) (p : S)
    (hFk : F.card = B.k - 1) (hη₀ : η₀ < 1 / 2)
    (hg : 4 * (1 / 2 - η₀) ≤ O.gap) (hc : |B.centre - O.cmid| ≤ (1 / 2 - η₀) / 5)
    (hκ : 10 ≤ (F.card : ℝ) * (1 / 2 - η₀)) :
    μ.real {ω | missAcc O η₀ B F p ω}
      ≤ Real.exp (-2 * F.card * (2 * (1 / 2 - η₀) / 5) ^ 2) := by
  classical
  have hσ : 0 < 1 / 2 - η₀ := by linarith
  obtain ⟨hR, hA, hm, hm'⟩ := mid_facts O hg hc
  have hg0 : 0 ≤ O.gap := by linarith
  have hc0 := abs_le.1 hc
  have hθ : ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀))
      = F.card * (B.centre + (1 / 2 - η₀)) := by rw [hFk]
  have hγ0 : (0 : ℝ) ≤ 2 * (1 / 2 - η₀) / 5 := by positivity
  have hE0 : 0 ≤ Real.exp (-2 * F.card * (2 * (1 / 2 - η₀) / 5) ^ 2) := (Real.exp_pos _).le
  set κ : ℝ := (F.card : ℝ) with hκdef
  have hκ0 : 0 ≤ κ := Nat.cast_nonneg _
  have hθ0 : 0 ≤ κ * (B.centre + (1 / 2 - η₀)) := by
    have : 0 ≤ B.centre + (1 / 2 - η₀) := by linarith [le_max_right O.ηOut 0]
    positivity
  have hκg : 40 ≤ κ * O.gap := by nlinarith
  rcases O.label_bit p with hl | hl
  · by_cases hfl : flipCount O F p ≤ F.card * (1 / 2)
    · have hr0 : 0 ≤ O.ηOut := by
        have := O.rate_nonneg p
        rw [rate_of_label O false p, if_pos (by simp [sideClass, hl])] at this
        simpa [rightRate] using this
      have hrate : O.rate p = O.ηOut := by
        rw [rate_of_label O false p, if_pos (by simp [sideClass, hl])]; rfl
      rw [max_eq_left hr0] at hm
      set c' : ℕ := ⌈κ * (B.centre + (1 / 2 - η₀))⌉₊ with hc'
      have hc'R : κ * (B.centre + (1 / 2 - η₀)) ≤ c' := Nat.le_ceil _
      have hsub : {ω | missAcc O η₀ B F p ω} ⊆ {ω | c' ≤ voteCount O.mq F p ω} := by
        intro ω hω
        rcases hω with ⟨hl1, -⟩ | ⟨-, -, hv⟩
        · rw [hl] at hl1; norm_num at hl1
        · rw [hθ] at hv
          show c' ≤ voteCount O.mq F p ω
          exact Nat.ceil_le.2 hv
      have hq0 : 0 ≤ O.cmid := by rw [hm]; positivity
      have hq1 : O.cmid ≤ 1 := by linarith [le_max_right O.ηIn 0]
      have hmean : meanVote O F p ≤ F.card * O.cmid := by
        have := meanVote_le_flip O F p hl (1 / 2) hg0 hfl
        rw [hrate] at this
        have e : O.ηOut + O.gap * (1 / 2) = O.cmid := by linarith
        rwa [e] at this
      have hcs : O.cmid + 4 * (1 / 2 - η₀) / 5 ≤ B.centre + (1 / 2 - η₀) := by linarith [hc0.1]
      have hκc := mul_le_mul_of_nonneg_left hcs hκ0
      have hc'2 : κ * O.cmid + 2 ≤ c' := by
        have e : κ * (O.cmid + 4 * (1 / 2 - η₀) / 5) = κ * O.cmid + 4 / 5 * (κ * (1 / 2 - η₀)) := by
          ring
        linarith
      have htail := (vote_tails O F p F.card O.cmid le_rfl hq0 hq1).2 c' hmean hc'2
      have hexp := binomSfGe_le_exp F.card c' hq0 hq1 hγ0 (by
        have h2 : κ * (O.cmid + 2 * (1 / 2 - η₀) / 5) ≤ κ * (O.cmid + 4 * (1 / 2 - η₀) / 5) :=
          mul_le_mul_of_nonneg_left (by linarith) hκ0
        linarith)
      exact le_trans (measureReal_mono hsub (measure_ne_top _ _)) (htail.trans hexp)
    · have hz : {ω | missAcc O η₀ B F p ω} = (∅ : Set Ω) := by
        ext ω
        constructor
        · rintro (⟨h1, -⟩ | ⟨-, h2, -⟩)
          · rw [hl] at h1; norm_num at h1
          · exact absurd h2 hfl
        · intro h; exact h.elim
      rw [hz, measureReal_empty]; exact hE0
  · by_cases hfl : flipCount O F p ≤ F.card * (1 / 20)
    · have hr0 : 0 ≤ O.ηIn := by
        have := O.rate_nonneg p
        rw [rate_of_label O true p, if_pos (by simp [sideClass, hl])] at this
        simpa [rightRate] using this
      have hrate : O.rate p = O.ηIn := by
        rw [rate_of_label O true p, if_pos (by simp [sideClass, hl])]; rfl
      rw [max_eq_left hr0] at hA hm'
      set b : ℕ := ⌊κ * (B.centre + (1 / 2 - η₀))⌋₊ with hb
      have hbR : (b : ℝ) ≤ κ * (B.centre + (1 / 2 - η₀)) := Nat.floor_le hθ0
      have hsub : {ω | missAcc O η₀ B F p ω} ⊆ {ω | voteCount O.mq F p ω ≤ b} := by
        intro ω hω
        rcases hω with ⟨-, -, hv⟩ | ⟨hl0, -⟩
        · rw [hθ] at hv
          show voteCount O.mq F p ω ≤ b
          exact Nat.le_floor hv.le
        · rw [hl] at hl0; norm_num at hl0
      set q : ℝ := 1 - O.ηIn - O.gap / 20 with hq
      have hq1 : q ≤ 1 := by rw [hq]; linarith
      have hq0 : 0 ≤ q := by
        have : O.ηIn + O.gap ≤ 1 := by
          unfold Oracle.gap; rw [max_eq_left hr0]; linarith [le_max_right O.ηOut 0]
        rw [hq]; linarith
      have hmean : F.card * q ≤ meanVote O F p := by
        have := meanVote_ge_flip O F p hl (1 / 20) hg0 hfl
        rw [hrate] at this
        have e : 1 - O.ηIn - O.gap * (1 / 20) = q := by rw [hq]; ring
        rwa [e] at this
      have hcs : B.centre + (1 / 2 - η₀) ≤ q - 3 * O.gap / 20 := by rw [hq]; linarith
      have hκc := mul_le_mul_of_nonneg_left hcs hκ0
      have hb2 : (b : ℝ) + 2 ≤ κ * q := by
        have e : κ * (q - 3 * O.gap / 20) = κ * q - 3 / 20 * (κ * O.gap) := by ring
        linarith
      have htail := (vote_tails O F p F.card q le_rfl hq0 hq1).1 b hmean hb2
      have hexp := one_sub_binomSfGe_le_exp F.card b hq0 hq1 hγ0 (by
        have h2 : κ * (q - 3 * O.gap / 20) ≤ κ * (q - 2 * (1 / 2 - η₀) / 5) :=
          mul_le_mul_of_nonneg_left (by linarith) hκ0
        linarith)
      exact le_trans (measureReal_mono hsub (measure_ne_top _ _)) (htail.trans hexp)
    · have hz : {ω | missAcc O η₀ B F p ω} = (∅ : Set Ω) := by
        ext ω
        constructor
        · rintro (⟨-, h2, -⟩ | ⟨h1, -⟩)
          · exact absurd h2 hfl
          · rw [hl] at h1; norm_num at h1
        · intro h; exact h.elim
      rw [hz, measureReal_empty]; exact hE0

open scoped Classical in
lemma measurableSet_missRej (O : Oracle μ S) (η₀ : ℝ) (B : State) (A₀ : Finset S) (p : S)
    {U : Set S} (hU : ∀ v ∈ A₀, p * v ∈ U) :
    MeasurableSet[noiseAlg O U] {ω | missRej O η₀ B A₀ p ω} := by
  classical
  have e : {ω | missRej O η₀ B A₀ p ω} = {ω | (fun W : Finset S =>
      (O.label p = 0 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 20)
        ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀)) < (W.card : ℝ))
      ∨ (O.label p = 1 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 2)
        ∧ (W.card : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀))))
      (A₀.filter (fun v => O.mq (p * v) ω = 1))} := rfl
  rw [e]
  exact measurableSet_filter_pred_map O (T := U) (A := A₀) (fun v => p * v) hU (fun W : Finset S =>
      (O.label p = 0 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 20)
        ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀)) < (W.card : ℝ))
      ∨ (O.label p = 1 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 2)
        ∧ (W.card : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * (B.centre - (1 / 2 - η₀))))

open scoped Classical in
lemma measurableSet_missAcc (O : Oracle μ S) (η₀ : ℝ) (B : State) (A₀ : Finset S) (p : S)
    {U : Set S} (hU : ∀ v ∈ A₀, p * v ∈ U) :
    MeasurableSet[noiseAlg O U] {ω | missAcc O η₀ B A₀ p ω} := by
  classical
  have e : {ω | missAcc O η₀ B A₀ p ω} = {ω | (fun W : Finset S =>
      (O.label p = 1 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 20)
        ∧ (W.card : ℝ) < ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)))
      ∨ (O.label p = 0 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 2)
        ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)) ≤ (W.card : ℝ)))
      (A₀.filter (fun v => O.mq (p * v) ω = 1))} := rfl
  rw [e]
  exact measurableSet_filter_pred_map O (T := U) (A := A₀) (fun v => p * v) hU (fun W : Finset S =>
      (O.label p = 1 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 20)
        ∧ (W.card : ℝ) < ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)))
      ∨ (O.label p = 0 ∧ flipCount O A₀ p ≤ A₀.card * (1 / 2)
        ∧ ((B.k - 1 : ℕ) : ℝ) * (B.centre + (1 / 2 - η₀)) ≤ (W.card : ℝ)))

variable {J : Type*} [Fintype J] {rule : Clusterer S}

open scoped Classical in
/-- The uniform pool's sample, filled and outside the table, holds more than `e` of its size of
prefixes whose vote misses a strong threshold, on either side. -/
noncomputable def checkMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (uni : J) (η₀ : ℝ) (B : State) (e : ℝ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x)
      (certOf (prefixesAt populations B.npref x) uni B.npref x)
    ∧ (certOf (prefixesAt populations B.npref x) uni B.npref x).card = B.npref
    ∧ 0 < (certOf (prefixesAt populations B.npref x) uni B.npref x).card
    ∧ ¬ ((((certOf (prefixesAt populations B.npref x) uni B.npref x).filter (fun p =>
            (clusterBy rule O.mq populations x B).card = B.k - 1
              ∧ missRej O η₀ B (clusterBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
          ≤ e * ((certOf (prefixesAt populations B.npref x) uni B.npref x).card : ℝ)
        ∧ (((certOf (prefixesAt populations B.npref x) uni B.npref x).filter (fun p =>
            (clusterBy rule O.mq populations x B).card = B.k - 1
              ∧ missAcc O η₀ B (clusterBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
          ≤ e * ((certOf (prefixesAt populations B.npref x) uni B.npref x).card : ℝ))}

open scoped Classical in
lemma measurableSet_checkMiss (O : Oracle μ S) (populations : Finset J) (uni : J) (η₀ : ℝ)
    (B : State) (e : ℝ) : MeasurableSet (checkMiss rule O populations uni η₀ B e) := by
  classical
  set Φ : Finset S → Finset S → Finset S → Ω → Prop := fun P C A ω =>
    ¬ (((A.filter (fun p => (clusterOf rule O B.sc B.scd P C B.k ω).card = B.k - 1
            ∧ missRej O η₀ B (clusterOf rule O B.sc B.scd P C B.k ω) p ω)).card : ℝ)
          ≤ e * (A.card : ℝ)
        ∧ ((A.filter (fun p => (clusterOf rule O B.sc B.scd P C B.k ω).card = B.k - 1
            ∧ missAcc O η₀ B (clusterOf rule O B.sc B.scd P C B.k ω) p ω)).card : ℝ)
          ≤ e * (A.card : ℝ)) with hΦ
  have hR : ∀ (P C : Finset S) (A : Finset S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card) ∧ Φ P C A (oracleNoise x)}
      else ∅) := by
    intro P C A
    split_ifs with hone
    · have hω : MeasurableSet {ω : Ω | Φ P C A ω} := by
        refine measurableSet_of_fam (T := C.powerset)
          (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.sc B.scd P C B.k ω hone))
          (fun A₀ => measurableSet_clusterOf O B.sc B.scd P C B.k hone A₀)
          (fun A₀ => {ω : Ω |
            ¬ (((A.filter (fun p => A₀.card = B.k - 1 ∧ missRej O η₀ B A₀ p ω)).card : ℝ)
                ≤ e * (A.card : ℝ)
              ∧ ((A.filter (fun p => A₀.card = B.k - 1 ∧ missAcc O η₀ B A₀ p ω)).card : ℝ)
                ≤ e * (A.card : ℝ))})
          (fun A₀ => ?_)
        have hd : MeasurableSet[noiseAlg O Set.univ] {ω | ((A.filter (fun p =>
            A₀.card = B.k - 1 ∧ missRej O η₀ B A₀ p ω)).card : ℝ) ≤ e * (A.card : ℝ)} := by
          refine measurableSet_filter_pred' O
            (fun p ω => A₀.card = B.k - 1 ∧ missRej O η₀ B A₀ p ω)
            (fun p _ => ?_) (fun U => ((U.card : ℝ) ≤ e * (A.card : ℝ)))
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using measurableSet_missRej O η₀ B A₀ p (U := Set.univ)
              (fun v _ => Set.mem_univ _)
          · simp [hg]
        have hc : MeasurableSet[noiseAlg O Set.univ] {ω | ((A.filter (fun p =>
            A₀.card = B.k - 1 ∧ missAcc O η₀ B A₀ p ω)).card : ℝ) ≤ e * (A.card : ℝ)} := by
          refine measurableSet_filter_pred' O
            (fun p ω => A₀.card = B.k - 1 ∧ missAcc O η₀ B A₀ p ω)
            (fun p _ => ?_) (fun U => ((U.card : ℝ) ≤ e * (A.card : ℝ)))
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using measurableSet_missAcc O η₀ B A₀ p (U := Set.univ)
              (fun v _ => Set.mem_univ _)
          · simp [hg]
        exact (noiseAlg_le O Set.univ _ (hd.inter hc)).compl
      by_cases hdraw : Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card
      · have hset : {x : Run Ω S J | (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card)
            ∧ Φ P C A (oracleNoise x)} = oracleNoise ⁻¹' {ω : Ω | Φ P C A ω} := by
          ext x
          exact ⟨fun h => h.2, fun h => ⟨hdraw, h⟩⟩
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card)
            ∧ Φ P C A (oracleNoise x)} = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
          exact fun h => absurd h hdraw
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : checkMiss rule O populations uni η₀ B e
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          {x : Run Ω S J | (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card)
            ∧ Φ P C A (oracleNoise x)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) uni B.npref x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), checkMiss, hΦ,
      ← clusterAt_eq_clusterOf O populations B x, and_assoc]
  rw [hrw]
  exact measurableSet_of_run_data_certSet populations uni B _ hR

open scoped Classical in
/-- `frac_selected_le` for both strong thresholds, lifted to the run. -/
theorem measureReal_checkMiss_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre)
    (hsuppSf : Dsf Sufᶜ = 0) (uni : J) (huni : uni ∈ populations) (η₀ : ℝ) (B : State)
    (e Es : ℝ) (hEs0 : 0 ≤ Es) (hEse : Es ≤ e)
    (hEs : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → F.card = B.k - 1 →
      μ.real {ω | missRej O η₀ B F p ω} ≤ Es ∧ μ.real {ω | missAcc O η₀ B F p ω} ≤ Es) :
    (streamMeasure μ Pp Pc Dsf).real (checkMiss rule O populations uni η₀ B e)
      ≤ 2 * Real.exp (-((B.npref : ℝ) * relRate e Es)) := by
  classical
  set R : ℝ := 2 * Real.exp (-((B.npref : ℝ) * relRate e Es)) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : streamMeasure μ Pp Pc Dsf (checkMiss rule O populations uni η₀ B e)
      ≤ ENNReal.ofReal R := by
    refine streamMeasure_slice_le Pp Pc Dsf _
      (measurableSet_checkMiss O populations uni η₀ B e) _ ?_
    filter_upwards [ae_draws_mem_Pre Pp Pc Dsf Pre populations hsuppP,
      ae_cert_mem_Pre Pp Pc Dsf Pre populations hsuppC, ae_sfx_mem_Suf Pp Pc Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
    set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    set Ad : Finset S := freshDraws Pd (fun i => d.2 uni i) B.npref with hAd
    have hPx : ∀ ω : Ω, prefixesAt populations B.npref ((ω, d) : Run Ω S J) = Pd := fun ω => rfl
    have hCx : ∀ ω : Ω, poolAt B.nsuff ((ω, d) : Run Ω S J) = Cd := fun ω => rfl
    have hAx : ∀ ω : Ω, certOf (prefixesAt populations B.npref ((ω, d) : Run Ω S J)) uni B.npref
        ((ω, d) : Run Ω S J) = Ad := fun ω => rfl
    have hP : ∀ q ∈ Pd, q ∈ Pre :=
      table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hdP
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, rfl⟩ := mem_freshDraws hp
      exact hdC uni huni i
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref ∧ 0 < Ad.card
    · obtain ⟨hdisj, hAdm, -⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
      set T : Finset (Finset S) := (Cd.erase 1).powerset with hT
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
          hCx ω ▸ clusterAt_subset O populations B _ hv⟩)
      have hT1 : ∀ t ∈ T, (1 : S) ∉ t := fun t ht h1 =>
        (Finset.mem_erase.1 (Finset.mem_powerset.1 ht h1)).1 rfl
      have hTC : ∀ t ∈ T, t ⊆ Cd := fun t ht =>
        (Finset.mem_powerset.1 ht).trans (Finset.erase_subset _ _)
      have hcongr : ∀ ω ω', (∀ w ∈ readSet Pd Cd, O.noise w ω = O.noise w ω') → fam ω = fam ω' :=
        fun ω ω' h => clusterAt_congr O populations B d (by rw [hPx ω, hCx ω]; exact h)
      have h1 := frac_selected_le hflat O Pd Cd Ad hP hV hA hdisj T ∅
        (Finset.mem_powerset.2 (Finset.empty_subset _)) hTC fam hfamT hcongr
        (fun A₀ p ω => A₀.card = B.k - 1 ∧ missRej O η₀ B A₀ p ω)
        (fun A₀ p U hU => by
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using measurableSet_missRej O η₀ B A₀ p hU
          · simp [hg])
        Es e hEs0 hEse (fun p _ A₀ hA₀ => by
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using (hEs A₀ p (hT1 A₀ hA₀) hg).1
          · simpa [hg] using hEs0)
      have h2 := frac_selected_le hflat O Pd Cd Ad hP hV hA hdisj T ∅
        (Finset.mem_powerset.2 (Finset.empty_subset _)) hTC fam hfamT hcongr
        (fun A₀ p ω => A₀.card = B.k - 1 ∧ missAcc O η₀ B A₀ p ω)
        (fun A₀ p U hU => by
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using measurableSet_missAcc O η₀ B A₀ p hU
          · simp [hg])
        Es e hEs0 hEse (fun p _ A₀ hA₀ => by
          by_cases hg : A₀.card = B.k - 1
          · simpa [hg] using (hEs A₀ p (hT1 A₀ hA₀) hg).2
          · simpa [hg] using hEs0)
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ checkMiss rule O populations uni η₀ B e}
          ⊆ {ω : Ω | e * (Ad.card : ℝ) < ((Ad.filter (fun p =>
                (fam ω).card = B.k - 1 ∧ missRej O η₀ B (fam ω) p ω)).card : ℝ)}
            ∪ {ω : Ω | e * (Ad.card : ℝ) < ((Ad.filter (fun p =>
                (fam ω).card = B.k - 1 ∧ missAcc O η₀ B (fam ω) p ω)).card : ℝ)} := by
        rintro ω ⟨-, -, -, hbad⟩
        rw [hAx ω] at hbad
        simp only [Set.mem_union, Set.mem_setOf_eq]
        by_contra hc
        push_neg at hc
        exact hbad hc
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ((measureReal_union_le _ _).trans ?_)
      rw [hR, ← hcardR, two_mul]
      convert add_le_add h1 h2
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ checkMiss rule O populations uni η₀ B e} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        rw [hAx ω, hPx ω] at h1
        rw [hAx ω] at h2 h3
        exact hdraw ⟨h1, h2, h3⟩
      simp [hsec]
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (checkMiss rule O populations uni η₀ B e)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

end OrthoDFA
