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

end OrthoDFA
