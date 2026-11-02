import OrthoDFA.Proofs.CentreCheck

/-!
# The centre check passes on a clean round

Near the true centre, with each class holding at least `2·εcov` of the uniform pool, the check
passes unless one of the uniform pool's sample counts strays: a class short of `εcov`, a class's
flip-heavy prefixes over their Markov mass, the strong thresholds' misses, or the strong sides'
reads above their mean.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]

lemma straddles_of_mid (O : Oracle μ S) {c η₀ : ℝ} (hη₀ : η₀ ≤ 1 / 2)
    (hg : 4 * (1 / 2 - η₀) ≤ O.gap) (hc : |c - O.cmid| ≤ (1 / 2 - η₀) / 5) :
    O.Straddles c η₀ := by
  obtain ⟨h1, h2, -, -⟩ := mid_facts O hg hc
  have hg0 : 0 ≤ O.gap := by linarith
  exact ⟨by linarith, by linarith⟩

lemma clsShift_ge_of_gap (O : Oracle μ S) {c η₀ : ℝ} (hstr : O.Straddles c η₀)
    (hη₀ : η₀ < 1 / 2) (hg : 4 * sig η₀ ≤ O.gap) (b : Bool) :
    4 * sig η₀ ^ 2 ≤ clsShift O c b := by
  obtain ⟨hm0, -⟩ := margin_mem O hstr b
  have hs := sig_pos η₀ hη₀
  have hsp := span_pos c
  have hsp1 : span c ≤ 1 := span_le_one_of_straddles O hstr hη₀.le
  unfold clsShift
  rw [le_div_iff₀ hsp]
  nlinarith [mul_le_mul hg hm0 hs.le (by linarith)]

open scoped Classical in
lemma card_class_add (O : Oracle μ S) (A : Finset S) (b : Bool) :
    ((A.filter (fun p => O.label p = sideClass b)).card : ℝ)
      + ((A.filter (fun p => O.label p = sideClass (!b))).card : ℝ) = A.card := by
  have h := Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = sideClass b)
  have hneg : A.filter (fun p => ¬ O.label p = sideClass b)
      = A.filter (fun p => O.label p = sideClass (!b)) := by
    refine Finset.filter_congr (fun p _ => ?_)
    rcases O.label_bit p with hl | hl <;> cases b <;> simp [sideClass, hl]
  rw [hneg] at h
  exact_mod_cast h

/-- The Markov mass of class `b`'s prefixes that more than a twentieth of the family flips, at
the family's shift `Δ + c·w_b + g`. -/
noncomputable def flipQ (O : Oracle μ S) (ctr Δ c g : ℝ) (b : Bool) : ℝ :=
  (Δ + c * clsShift O ctr b + g) / (1 / 20 * clsShift O ctr b)

open scoped Classical in
/-- The uniform pool's counts the centre check reads, any of them astray. -/
noncomputable def centreBad (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (uni : J) (Du : Measure S) (η₀ εcov Δ c g e aT : ℝ) (B : State) : Set (Run Ω S J) :=
  (heavyHits rule O populations Du uni B (fun _ p => O.label p = sideClass false) (1 - 2 * εcov)
      εcov
    ∪ heavyHits rule O populations Du uni B (fun _ p => O.label p = sideClass true)
      (1 - 2 * εcov) εcov)
  ∪ (heavyHits rule O populations Du uni B (flipHeavy O true (1 / 20))
      (flipQ O B.centre Δ c g true) (flipQ O B.centre Δ c g true / 32)
    ∪ heavyHits rule O populations Du uni B (flipHeavy O false (1 / 20))
      (flipQ O B.centre Δ c g false) (flipQ O B.centre Δ c g false / 32))
  ∪ checkMiss rule O populations uni η₀ B e
  ∪ livesBad rule O populations uni η₀ B aT

lemma measurableSet_centreBad (O : Oracle μ S) (populations : Finset J) (uni : J)
    (Du : Measure S) (η₀ εcov Δ c g e aT : ℝ) (B : State) :
    MeasurableSet (centreBad rule O populations uni Du η₀ εcov Δ c g e aT B) :=
  ((((measurableSet_heavyHits O populations Du uni B _ _ _).union
      (measurableSet_heavyHits O populations Du uni B _ _ _)).union
    ((measurableSet_heavyHits O populations Du uni B _ _ _).union
      (measurableSet_heavyHits O populations Du uni B _ _ _))).union
    (measurableSet_checkMiss O populations uni η₀ B e)).union
    (measurableSet_livesBad O populations uni η₀ B aT)

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- On a round whose family is full and carries its shift budget, the check passes at a centre
near the true one unless the uniform pool's counts stray. -/
theorem centreFits_of_clean (O : Oracle μ S) (populations : Finset J) (uni : J)
    (Du : Measure S) [IsProbabilityMeasure Du] (B : State) (x : Run Ω S J)
    {η₀ εcov Δ c g e h a aT : ℝ} {r : ℕ}
    (hη₀ : η₀ < 1 / 2) (hg : 4 * sig η₀ ≤ O.gap) (hcm : |B.centre - O.cmid| ≤ sig η₀ / 5)
    (hεcov : 0 < εcov) (hbal : ∀ b, 2 * εcov ≤ Du.real {p | O.label p = sideClass b})
    (hmpos : 0 < B.npref) (hc0 : 0 ≤ c) (hr : (r : ℝ) ≤ c * B.npref) (hΔ : 0 ≤ Δ)
    (hg0 : 0 ≤ g)
    (hgood : certGood populations uni B.npref r x)
    (hfull : (clusterBy rule O.mq populations x B).card = B.k - 1)
    (hdirty : ∀ b, ∑ v ∈ clusterBy rule O.mq populations x B, wMass O Du (clsW O B.centre b) v
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * (Δ + c * clsShift O B.centre b + g))
    (hbad : x ∉ centreBad rule O populations uni Du η₀ εcov Δ c g e aT B)
    (hh : ∀ b, flipQ O B.centre Δ c g b * (33 / 32) + c ≤ h)
    (hsize : εcov / 2 ≤ εcov - c - h - e) (hcont : h + e ≤ 3 * εcov / 40)
    (hdev : ∀ s : ℕ, (B.npref : ℝ) * εcov / 2 ≤ s →
      hoeffDev s a + hoeffDev s aT ≤ O.gap / 20 * s) :
    centreFits O.mq η₀ a B (clusterBy rule O.mq populations x B)
      (certOf (prefixesAt populations B.npref x) uni B.npref x) (oracleNoise x) := by
  classical
  set F := clusterBy rule O.mq populations x B with hFdef
  set A := certOf (prefixesAt populations B.npref x) uni B.npref x with hAdef
  set n : ℝ := (B.npref : ℝ) with hndef
  have hn0 : 0 < n := by rw [hndef]; exact_mod_cast hmpos
  have hs := sig_pos η₀ hη₀
  have hstr := straddles_of_mid O hη₀.le hg hcm
  obtain ⟨hCn, -⟩ := certOf_of_good populations uni B.npref r x hgood
  have hAn : (A.card : ℝ) = n := by rw [hAdef, hCn]
  simp only [centreBad, Set.mem_union, not_or] at hbad
  obtain ⟨⟨⟨⟨hCf, hCt⟩, ⟨hHt, hHf⟩⟩, hM⟩, hL⟩ := hbad
  have hC : ∀ b, x ∉ heavyHits rule O populations Du uni B
      (fun _ p => O.label p = sideClass (!b)) (1 - 2 * εcov) εcov := by
    intro b; cases b
    · exact hCt
    · exact hCf
  have hH : ∀ b, x ∉ heavyHits rule O populations Du uni B (flipHeavy O b (1 / 20))
      (flipQ O B.centre Δ c g b) (flipQ O B.centre Δ c g b / 32) := by
    intro b; cases b
    · exact hHf
    · exact hHt
  -- each class holds at least `εcov − c` of the sample
  have hcls : ∀ b, n * (εcov - c) ≤ ((A.filter (fun p => O.label p = sideClass b)).card : ℝ) := by
    intro b
    have hmass : Du.real {p | O.label p = sideClass (!b)} ≤ 1 - 2 * εcov := by
      have hcompl : {p : S | O.label p = sideClass (!b)} = {p | O.label p = sideClass b}ᶜ := by
        ext p
        simp only [Set.mem_ofPred_eq, Set.mem_compl_iff]
        rcases O.label_bit p with hl | hl <;> cases b <;> simp [sideClass, hl]
      rw [hcompl, measureReal_compl (measurableSet_of_countable _), probReal_univ]
      linarith [hbal b]
    have h1 := heavy_le_of_clean O populations Du uni B x
      (fun _ p => O.label p = sideClass (!b)) hgood hmass (hC b)
    have h2 := card_class_add O A b
    rw [hAn] at h2
    have hrn : (r : ℝ) ≤ c * n := hr
    change ((A.filter (fun p => O.label p = sideClass (!b))).card : ℝ)
      ≤ n * (1 - 2 * εcov + εcov) + r at h1
    nlinarith
  -- flip-heavy prefixes are within their Markov mass
  have hheavy : ∀ b, ((A.filter (flipHeavy O b (1 / 20) F)).card : ℝ) ≤ n * h := by
    intro b
    have hw := clsShift_pos O hstr hη₀ b
    have hmass := flipHeavy_mass_le O hstr hη₀ Du F b (by norm_num : (0 : ℝ) < 1 / 20)
      (Δ + c * clsShift O B.centre b + g) (by positivity) (hdirty b)
    have h1 := heavy_le_of_clean O populations Du uni B x (flipHeavy O b (1 / 20)) hgood
      hmass (hH b)
    have hq0 : 0 ≤ flipQ O B.centre Δ c g b := by unfold flipQ; positivity
    have h2 := mul_le_mul_of_nonneg_left (hh b) hn0.le
    change ((A.filter (flipHeavy O b (1 / 20) F)).card : ℝ)
      ≤ n * (flipQ O B.centre Δ c g b + flipQ O B.centre Δ c g b / 32) + r at h1
    have hrn : (r : ℝ) ≤ c * n := hr
    nlinarith
  -- the misses
  have hmiss : ((A.filter (fun p => missRej O η₀ B F p (oracleNoise x))).card : ℝ) ≤ n * e
      ∧ ((A.filter (fun p => missAcc O η₀ B F p (oracleNoise x))).card : ℝ) ≤ n * e := by
    have hpos : 0 < A.card := by rw [hAdef, hCn]; exact hmpos
    have hdisj : Disjoint (prefixesAt populations B.npref x) A :=
      (disjoint_freshDraws _ _ _).symm
    have hnot : ((A.filter (fun p => F.card = B.k - 1
          ∧ missRej O η₀ B F p (oracleNoise x))).card : ℝ) ≤ e * (A.card : ℝ)
        ∧ ((A.filter (fun p => F.card = B.k - 1
          ∧ missAcc O η₀ B F p (oracleNoise x))).card : ℝ) ≤ e * (A.card : ℝ) := by
      by_contra hcon
      exact hM ⟨hdisj, hCn, hpos, hcon⟩
    have e1 : A.filter (fun p => F.card = B.k - 1 ∧ missRej O η₀ B F p (oracleNoise x))
        = A.filter (fun p => missRej O η₀ B F p (oracleNoise x)) :=
      Finset.filter_congr (fun p _ => by simp [hfull])
    have e2 : A.filter (fun p => F.card = B.k - 1 ∧ missAcc O η₀ B F p (oracleNoise x))
        = A.filter (fun p => missAcc O η₀ B F p (oracleNoise x)) :=
      Finset.filter_congr (fun p _ => by simp [hfull])
    rw [e1, e2, hAn] at hnot
    exact ⟨by linarith [hnot.1], by linarith [hnot.2]⟩
  obtain ⟨hliveR, hliveA⟩ := readsLive_of_not_livesBad O populations uni η₀ B aT hL
  refine centreFits_of_counts O (n := n) (εb := εcov - c) (h := h) (e := e)
    (Smin := n * εcov / 2) hfull hη₀.le hg hcm hAn hcls hheavy hmiss.1 hmiss.2 hliveR hliveA
    (by positivity) ?_ ?_ (fun s hs' => hdev s hs')
  · have := mul_le_mul_of_nonneg_left hsize hn0.le
    linarith
  · have := mul_le_mul_of_nonneg_left hcont hn0.le
    linarith

open scoped Classical in
/-- The check's tails at one assignment of the hash: the classes' and the flip-heavy prefixes'
relative tails, the misses', and the strong sides' reads. -/
theorem measureReal_centreBad_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (uni : J) (huni : uni ∈ populations) (Du : Measure S) [IsProbabilityMeasure Du]
    (hPc : Pc uni = iidStream Du) (B : State) {η₀ εcov Δ c g e aT : ℝ}
    (hη₀ : η₀ < 1 / 2) (hg : 4 * sig η₀ ≤ O.gap) (hcm : |B.centre - O.cmid| ≤ sig η₀ / 5)
    (hκ : 10 ≤ ((B.k - 1 : ℕ) : ℝ) * sig η₀) (hεcov : 0 ≤ εcov)
    (hEe : Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (2 * sig η₀ / 5) ^ 2) ≤ e)
    (haT0 : 0 < aT) (haT1 : aT ≤ 1) (hq : ∀ b, 0 ≤ flipQ O B.centre Δ c g b) :
    (streamMeasure μ Pp Pc Dsf).real (centreBad rule O populations uni Du η₀ εcov Δ c g e aT B)
      ≤ 2 * Real.exp (-((B.npref : ℝ) * relRate (1 - 2 * εcov + εcov) (1 - 2 * εcov)))
        + (Real.exp (-((B.npref : ℝ) * relRate (flipQ O B.centre Δ c g true
              + flipQ O B.centre Δ c g true / 32) (flipQ O B.centre Δ c g true)))
          + Real.exp (-((B.npref : ℝ) * relRate (flipQ O B.centre Δ c g false
              + flipQ O B.centre Δ c g false / 32) (flipQ O B.centre Δ c g false))))
        + 2 * Real.exp (-((B.npref : ℝ) * relRate e
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (2 * sig η₀ / 5) ^ 2))))
        + 2 * aT := by
  classical
  set ν := streamMeasure μ Pp Pc Dsf
  have hC := fun b => measureReal_heavyHits_le (rule := rule) Pp Pc Dsf O populations uni Du hPc
    B (fun _ p => O.label p = sideClass b) (1 - 2 * εcov) εcov hεcov
  have hH := fun b => measureReal_heavyHits_le (rule := rule) Pp Pc Dsf O populations uni Du hPc
    B (flipHeavy O b (1 / 20)) (flipQ O B.centre Δ c g b) (flipQ O B.centre Δ c g b / 32)
    (by have := hq b; positivity)
  have hM := measureReal_checkMiss_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf uni huni η₀ B e _ (Real.exp_pos _).le hEe (fun F p _ hF => by
      have hFR : (F.card : ℝ) = ((B.k - 1 : ℕ) : ℝ) := by rw [hF]
      have hκ' : 10 ≤ (F.card : ℝ) * (1 / 2 - η₀) := by rw [hFR]; exact hκ
      have hcm' : |B.centre - O.cmid| ≤ (1 / 2 - η₀) / 5 := hcm
      have hg' : 4 * (1 / 2 - η₀) ≤ O.gap := hg
      refine ⟨le_trans (missRej_prob_le O F p hF hη₀ hg' hcm' hκ') (le_of_eq ?_),
        le_trans (missAcc_prob_le O F p hF hη₀ hg' hcm' hκ') (le_of_eq ?_)⟩ <;>
      rw [hFR] <;> rfl)
  have hL := measureReal_livesBad_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf uni huni η₀ B haT0 haT1
  unfold centreBad
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ hL)
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ hM)
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · refine le_trans (measureReal_union_le _ _) ?_
    linarith [hC false, hC true]
  · exact le_trans (measureReal_union_le _ _) (add_le_add (hH true) (hH false))

section Passable

variable {η₀ indecisionLimit crossLimit : ℝ} {hsz : ℕ} {εs : ℝ}

/-- The grid's centre count nearest the true centre. -/
noncomputable def midCount (O : Oracle μ S) (κ : ℕ) : ℕ := ⌊(κ : ℝ) * O.cmid + 1 / 2⌋₊

omit [Fintype J] in
/-- With the gap at least `4·sig`, the nearest centre count is one the grid tries, within
`sig/5` of the true centre. -/
lemma midCount_facts (O : Oracle μ S) (populations : Finset J) {εcov δ : ℝ}
    (hη₀ : η₀ < 1 / 2) (hg : 4 * sig η₀ ≤ O.gap) (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    midCount O (famCount η₀ populations indecisionLimit εcov δ crossLimit)
        ∈ centreSet η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit)
      ∧ |(midCount O (famCount η₀ populations indecisionLimit εcov δ crossLimit) : ℝ)
          / famCount η₀ populations indecisionLimit εcov δ crossLimit - O.cmid|
        ≤ sig η₀ / 5 := by
  classical
  have hs := sig_pos η₀ hη₀
  have hg1 := gap_le_one O
  have hη0 : 0 ≤ η₀ := by rw [sig] at hg; linarith
  have hw := famCount_wide (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    (δ := δ) η₀ populations hη0 hη₀ hε hind
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit
  have hκs : 10 ≤ (κ : ℝ) * sig η₀ := by rw [voteSlack] at hw; nlinarith
  have hκ : (0 : ℝ) < κ := by
    rcases eq_or_lt_of_le (Nat.cast_nonneg κ : (0 : ℝ) ≤ κ) with h | h
    · rw [← h] at hκs; linarith
    · exact h
  have hm1 : 2 * sig η₀ ≤ O.cmid := by
    unfold Oracle.cmid; unfold Oracle.gap at hg
    linarith [le_max_right O.ηIn 0, le_max_right O.ηOut 0]
  have hm2 : O.cmid ≤ 1 - 2 * sig η₀ := by
    unfold Oracle.cmid; unfold Oracle.gap at hg
    linarith [le_max_right O.ηIn 0, le_max_right O.ηOut 0]
  set m : ℕ := midCount O κ with hmdef
  have hx0 : 0 ≤ (κ : ℝ) * O.cmid + 1 / 2 := by nlinarith
  have hfl : (m : ℝ) ≤ (κ : ℝ) * O.cmid + 1 / 2 := Nat.floor_le hx0
  have hfl2 : (κ : ℝ) * O.cmid + 1 / 2 < (m : ℝ) + 1 := Nat.lt_floor_add_one _
  have hlo : sig η₀ * κ ≤ m := by nlinarith
  have hhi : (m : ℝ) ≤ (1 - sig η₀) * κ := by nlinarith
  refine ⟨?_, ?_⟩
  · rw [centreSet, Finset.mem_filter, Finset.mem_range]
    refine ⟨?_, hlo, hhi⟩
    have : (m : ℝ) < κ + 1 := by nlinarith
    exact_mod_cast this
  · rw [abs_le]
    have e : (m : ℝ) / κ - O.cmid = ((m : ℝ) - κ * O.cmid) / κ := by field_simp
    rw [e]
    have hk5 : 1 / 2 ≤ (κ : ℝ) * (sig η₀ / 5) := by nlinarith
    constructor
    · rw [le_div_iff₀ hκ]; nlinarith
    · rw [div_le_iff₀ hκ]; nlinarith

/-- The centre of the state at count `m`. -/
lemma centre_solvedStateAt {populations : Finset J} {εcov δ pAP : ℝ} (n m : ℕ)
    (hη₀ : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit) (hη0 : 0 ≤ η₀)
    (hm : m ∈ centres η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit)) :
    (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit n m).centre
      = (m : ℝ) / famCount η₀ populations indecisionLimit εcov δ crossLimit := by
  have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hw := famCount_wide (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    (δ := δ) η₀ populations hη0 hη₀ hε hind
  obtain ⟨hlo, hhi⟩ := loAt_hiAt_of η₀ (bandAt_lt η₀ hm (by linarith) hη₀)
  have hκ : (4 : ℝ) ≤ famCount η₀ populations indecisionLimit εcov δ crossLimit := by
    exact_mod_cast h4
  simp only [State.centre, solvedStateAt]
  rw [hlo, hhi]
  push_cast
  set κ : ℝ := (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
  set b : ℝ := (bandAt η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit) : ℝ)
  rw [show (m : ℝ) - b - 1 + (m + b) + 1 = 2 * m by ring, show κ + 1 - 1 = κ by ring]
  field_simp

set_option maxHeartbeats 4000000 in
/-- The computed ladder holds a state whose round can pass: the top rung at the centre count
nearest the true centre, every clause discharged from the closed forms that define it. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J)
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hgap : 4 * sig η₀ ≤ O.gap) (hpop : populations.Nonempty) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hεs : 0 < εs) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ0 : 0 ≤ ρ) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hρsfcap : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    ∃ B : State, B ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ
      ∧ PassableAt O η₀ populations Dsf indecisionLimit εcov α δ ρ ρsf pAP hsz B
      ∧ |B.centre - O.cmid| ≤ sig η₀ / 5
      ∧ B.npref = prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
      ∧ B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
      ∧ B.k = famCount η₀ populations indecisionLimit εcov δ crossLimit + 1 := by
  classical
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hη₀ hεcov hindLim
  have hv := voteSlack_pos η₀ hη₀
  have hη0' : 0 ≤ η₀ := by have := gap_le_one O; rw [sig] at hgap; linarith
  obtain ⟨hmisκ, h5⟩ := famCount_misfire (crossLimit := crossLimit) η₀ populations (δ := δ) hη0'
    hη₀ hεcov hindLim
  have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  obtain ⟨hmC, hmmid⟩ := midCount_facts (crossLimit := crossLimit) (δ := δ) O populations hη₀
    hgap hεcov hindLim
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set mc : ℕ := midCount O κ with hmcdef
  have hmc : mc ∈ centres η₀ κ := by
    rw [centreSet, Finset.mem_filter] at hmC; exact hmC.2
  have hmcκ : mc ≤ κ := by
    rw [centreSet, Finset.mem_filter, Finset.mem_range] at hmC; omega
  have hb1 := bandAt_lt η₀ hmc (by linarith) hη₀
  obtain ⟨hlo, hhi⟩ := loAt_hiAt_of η₀ hb1
  have hb := bandAt_le η₀ κ hη₀
  have hb0 : (0 : ℝ) ≤ bandAt η₀ κ := Nat.cast_nonneg _
  have hκR : (4 : ℝ) ≤ (κ : ℝ) := by exact_mod_cast h4
  have hκpos' : (0 : ℝ) < κ := by linarith
  have hlohi : loAt η₀ κ mc ≤ hiAt η₀ κ mc := by unfold loAt hiAt; omega
  have hv5 : voteSlack η₀ ≤ 2 / 5 := by rw [voteSlack, sig]; linarith
  have hmcR : sig η₀ * κ ≤ mc ∧ (mc : ℝ) ≤ (1 - sig η₀) * κ := hmc
  -- the mirror centre, which the reject side's misfire is read at
  have hmcR' : ((κ - mc : ℕ) : ℝ) = κ - mc := Nat.cast_sub hmcκ
  have hmc' : κ - mc ∈ centres η₀ κ := by
    simp only [centres, Set.mem_setOf_eq, hmcR']
    exact ⟨by linarith [hmcR.2], by linarith [hmcR.1]⟩
  have hb1' := bandAt_lt η₀ hmc' (by linarith) hη₀
  obtain ⟨hlo', hhi'⟩ := loAt_hiAt_of η₀ hb1'
  set qa : ℝ := (mc : ℝ) / κ + voteSlack η₀ with hqadef
  set qr : ℝ := ((κ - mc : ℕ) : ℝ) / κ + voteSlack η₀ with hqrdef
  obtain ⟨hq0, hq1⟩ := misfire_rate_mem η₀ hmc hη0' hη₀
  obtain ⟨hq0', hq1'⟩ := misfire_rate_mem η₀ hmc' hη0' hη₀
  have hqrv : qr = 1 - (mc : ℝ) / κ + voteSlack η₀ := by
    rw [hqrdef, hmcR']; field_simp
  have hmisle : ∀ c ∈ centres η₀ κ, 1 - binomSfGe κ ((c : ℝ) / κ + voteSlack η₀) (loAt η₀ κ c + 1)
      ≤ misfireTail η₀ κ c := by
    intro c hc
    obtain ⟨h0, h1⟩ := misfire_rate_mem η₀ hc hη0' hη₀
    rw [misfireTail]
    refine sub_le_sub_left (le_trans (binomSfGe_succ_le h0 h1 (hiAt η₀ κ c + 1)) ?_) 1
    exact binomSfGe_antitone' h0 h1 (by unfold loAt hiAt; omega)
  have hκpos : 0 < κ := by omega
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m mc with
      hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  have hBkR : (B.k : ℝ) = κ + 1 := by rw [hBk]; push_cast; ring
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := rfl
  have hBlo : B.lo = loAt η₀ κ mc := rfl
  have hBhi : B.hi = hiAt η₀ κ mc := rfl
  have hBscd : B.scd = 1 := rfl
  have hBsc : B.sc = 1 := rfl
  have hMceil : 2 * ((κ : ℝ) + 1) / pAP ≤ ((B.nsuff : ℕ) : ℝ) := by
    rw [hBM, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊]
  have hcutle := cutBudget_le η₀ indecisionLimit εcov
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast this
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  have hgate := solved_gate (crossLimit := crossLimit) (hsz := hsz) (pAP := pAP) (δ := δ) η₀
    populations hη₀ hη0' hεcov hεs hindLim hαpos (by linarith) hcard1 B hBm hBM hBk
  have hcB : B.centre = (mc : ℝ) / κ :=
    centre_solvedStateAt (pAP := pAP) m mc hη₀ hεcov hindLim hη0' hmc
  have hcm : |B.centre - O.cmid| ≤ sig η₀ / 5 := by rw [hcB]; exact hmmid
  have hstr : O.Straddles B.centre η₀ := straddles_of_mid O hη₀.le hgap hcm
  have hheavyN := solved_heavyNamed (εs := εs) (crossLimit := crossLimit) (hsz := hsz) (α := α)
      (pAP := pAP)
    O B.centre η₀ populations hstr hη₀ hη0' hεcov hindLim hδ hδ1 hcard1
  rw [← hmdef] at hheavyN
  have hcdirty : Real.log (128 * (populations.card : ℝ)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / dirtyRate η₀ indecisionLimit εcov εs
      ≤ (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
        / dirtyRate η₀ indecisionLimit εcov εs⌉₊
        ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
      rw [prefCount]; omega
    exact_mod_cast hle
  have hsignal := solved_signal (crossLimit := crossLimit) (hsz := hsz) (δ := δ) (εs := εs)
    (α := α) (pAP := pAP) η₀ populations hpop hη₀ hεcov hindLim
  obtain ⟨hE1, hE2⟩ := solved_countTails (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations hpop (εcov := εcov) hδ hpAPPositive hη₀
  rw [← hmdef] at hsignal
  rw [← hκdef] at hE1 hE2
  have hround := solved_roundFail (crossLimit := crossLimit) (α := α) (ρ := ρ) (ρsf := ρsf)
      η₀ populations hη₀ hη0' hεcov hεs hδ hδ1 hpAPPositive hpAP1 hindLim
      hcard hρ0 hρsf0 hρcap hρsfcap B hBm hBM (cutBudget η₀ indecisionLimit εcov / 2)
      (by positivity) le_rfl
  have hshare := solved_share η₀ populations hη₀ hη0' hεcov hεs hindLim hδ hδ1
      hpAPPositive hpAP1 hcard hρ0 hρcap B hBm hBk
  have hsched : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz :=
    mem_schedule.2 ⟨0, ladderLen_pos _ _ _ _ _ _, mc, hmC, by rw [pow_zero, Nat.div_one]⟩
  clear_value B κ m
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov εs := rejFrac_pos η₀ hη₀ hεcov hεs hindLim
  obtain ⟨hcm', hcu, -⟩ := rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀
    (εcov := εcov)
  have hΔ0 := shiftBudget_pos η₀ hη₀ hεcov hindLim
  have hs1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have hβ := shiftScale_pos hεcov hindLim
  have hspan : span B.centre ≤ 1 := span_le_one_of_straddles O hstr hη₀.le
  have hsc' : ∀ b, sideCharge O B.centre η₀ (!b) ≤ 5 / sig η₀ ^ 2 * clsSig O B.centre η₀ b := by
    intro b
    refine le_trans (sideCharge_le_clsSig O hstr hη₀ b) (mul_le_mul_of_nonneg_right ?_
      (clsSig_pos O hstr hη₀ b).le)
    exact div_le_div_of_nonneg_right (by linarith) (by positivity)
  have hκv : 8 ≤ (κ : ℝ) * voteSlack η₀ := h5
  refine ⟨B, Finset.mem_filter.2 ⟨hsched, ⟨?_, ?_, ?_, ?_⟩⟩, ⟨9 * pAP / 20, qa, qr,
    cutBudget η₀ indecisionLimit εcov / 2,
    shiftBudget η₀ indecisionLimit εcov, shiftBudget η₀ indecisionLimit εcov,
    cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit,
    heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit,
    apFloor pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    δ / 16, δ / 16, indecisionLimit / 8, min εcov (1 / 2) / 32,
    5 / 8 * cutBudget η₀ indecisionLimit εcov, rejFrac η₀ indecisionLimit εcov εs,
    hoeffDev m (δ / (32 * populations.card
      * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))),
    Real.log (128 * (populations.card : ℝ)
      * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ),
    cutBudget η₀ indecisionLimit εcov / 1024,
    rejCount η₀ indecisionLimit εcov εs m,
    poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    ⌈Real.sqrt ((m : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hcm, hBm, hBM, hBk⟩
  -- Capped
  · rw [hBlo, hBhi]
    exact hlohi
  · rw [hBm]; exact hmpos
  · rw [hBM]
    exact poolCount_found η₀ populations hη₀ hpAPPositive
  · have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by
      rw [mul_comm (8 : ℝ) (m : ℝ), mul_comm δ (m : ℝ),
        mul_div_mul_left _ _ (ne_of_gt hmR)]
    rw [hBm, ← hmdef, hval]
    exact le_add_of_le_of_nonneg hshare (by split_ifs <;> positivity)
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · exact hindLim
  · exact hind1
  · positivity
  · exact hΔ0
  · exact hΔ0
  · exact le_rfl
  · linarith
  · positivity
  · exact hc0.le
  · rw [hBm]
    exact Nat.floor_le (by positivity)
  · exact hpAPBound
  · exact hρsf
  -- each class's heavy prefixes at the uniform pool
  · intro b
    exact (uni_share_le hs hs1 hβ (min_le_right _ _) (min_le_left _ _) rfl hc0.le hcm' hcu
      (clsShift_pos O hstr hη₀ b) (clsShift_le_one O hstr hη₀ b) (clsSig_ge O hstr hη₀ b)
      (clsShift_div_clsSig_le O hstr hη₀ b) (sideCharge_nonneg O B.centre η₀ _) (hsc' b)
      (sideCharge_le_gateA O hstr hη₀ _) (two_sig_le_gateA hη0' hη₀)).1
  · intro b
    exact (uni_share_le hs hs1 hβ (min_le_right _ _) (min_le_left _ _) rfl hc0.le hcm' hcu
      (clsShift_pos O hstr hη₀ b) (clsShift_le_one O hstr hη₀ b) (clsSig_ge O hstr hη₀ b)
      (clsShift_div_clsSig_le O hstr hη₀ b) (sideCharge_nonneg O B.centre η₀ _) (hsc' b)
      (sideCharge_le_gateA O hstr hη₀ _) (two_sig_le_gateA hη0' hη₀)).2
  · exact Real.log_nonneg (by
      rw [le_div_iff₀ hδ]
      have : (0 : ℝ) ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) :=
        Nat.cast_nonneg _
      nlinarith)
  · positivity
  · rw [hBm]
    exact hheavyN.1
  · rw [hBm]
    exact hheavyN.2.1
  · rw [hBm]
    exact hheavyN.2.2.1
  · rw [hBm]
    exact hheavyN.2.2.2
  · linarith
  -- the count rule
  · rw [hBscd, hBsc]
  · exact Real.sqrt_nonneg _
  · exact le_max_left _ _
  · rw [hBm]; exact hsignal
  · rw [hBM, apFloor]
    have hrs : ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
        : ℕ) : ℝ) ≤ pAP * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
          / 20 := Nat.floor_le (by positivity)
    linarith
  · rw [hBkappa, apFloor]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive, hBM] at hceil
    linarith
  · -- a member's shift on a class read off its draws, against the table budget
    have hdr : dirtyRate η₀ indecisionLimit εcov εs
        = shiftBudget η₀ indecisionLimit εcov ^ 2
          / (4 * (shiftBudget η₀ indecisionLimit εcov + rejFrac η₀ indecisionLimit εcov εs)) :=
      rfl
    have hcd := hcdirty
    rw [← hmdef, hdr, div_div_eq_mul_div, div_le_iff₀ (by positivity)] at hcd
    rw [hBm]
    have hL0 : 0 ≤ Real.log (128 * (populations.card : ℝ)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ) := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ]
      have : (1 : ℝ) ≤ (populations.card : ℝ) * ((poolCount η₀ populations indecisionLimit εcov δ
          pAP crossLimit : ℝ) + 1) := by nlinarith [hcard1]
      nlinarith
    nlinarith [mul_nonneg hL0 hc0.le]
  · rw [hBkappa]; exact hE1
  · rw [hBkappa, hBM]; exact hE2
  -- the pool holds a family
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    have hrs : ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
        : ℕ) : ℝ) ≤ pAP * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
          / 20 := Nat.floor_le (by positivity)
    rw [hBM] at hceil
    rw [hBM]
    push_cast
    linarith only [hceil, hrs]
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    have : ((κ + 1 : ℕ) : ℝ) < (B.nsuff : ℝ) := by
      push_cast
      have h1 := mul_le_mul_of_nonneg_left hpAP1 (Nat.cast_nonneg B.nsuff : (0 : ℝ) ≤ B.nsuff)
      have h2 : (0 : ℝ) ≤ κ := Nat.cast_nonneg _
      linarith only [hceil, h1, h2]
    exact_mod_cast this
  -- the thresholds decide, and decide right
  · exact hq0
  · exact hq1
  · exact hq0'
  · exact hq1'
  · intro p
    have h := rate_add_gap_fracAt O hstr hη₀ p
    exact ⟨fracAt_nonneg O hstr hη₀ p, fun hl => by linarith [h.1 hl, hcB],
      fun hl => by rw [hqrv]; linarith [h.2 hl, hcB]⟩
  · rw [hBhi, hBkR, hhi]
    have e : (κ : ℝ) * qa = mc + κ * voteSlack η₀ := by rw [hqadef]; field_simp
    have e2 : ((κ : ℝ) + 1) * qa = κ * qa + qa := by ring
    linarith
  · rw [hBlo, hBkR, hlo]
    have hmk : (mc : ℝ) / κ ≤ 1 := by rw [div_le_one hκpos']; linarith [hmcR.2]
    have e : ((κ : ℝ) + 1) * (1 - qr) = mc - κ * voteSlack η₀ + ((mc : ℝ) / κ - voteSlack η₀) := by
      rw [hqrv]; field_simp; ring
    linarith
  · rw [hBlo, hBkappa, hlo]
    have e : (κ : ℝ) * qa = mc + κ * voteSlack η₀ := by rw [hqadef]; field_simp
    linarith
  · rw [hBhi, hBkappa, hhi]
    have e : (κ : ℝ) * (1 - qr) = mc - κ * voteSlack η₀ := by
      rw [hqrv]; field_simp; ring
    linarith
  · rw [hBhi, hBkappa]
    have hκs : 10 ≤ (κ : ℝ) * sig η₀ := by
      have e : (κ : ℝ) * voteSlack η₀ = 4 / 5 * ((κ : ℝ) * sig η₀) := by rw [voteSlack]; ring
      linarith
    have hbs : (bandAt η₀ κ : ℝ) ≤ 2 / 5 * ((κ : ℝ) * sig η₀) := by
      have e : (κ : ℝ) * voteSlack η₀ / 2 = 2 / 5 * ((κ : ℝ) * sig η₀) := by rw [voteSlack]; ring
      linarith
    have h1 : (mc : ℝ) ≤ κ - (κ : ℝ) * sig η₀ := by
      have e : (1 - sig η₀) * (κ : ℝ) = κ - κ * sig η₀ := by ring
      linarith [hmcR.2]
    have : ((hiAt η₀ κ mc : ℕ) : ℝ) + 1 ≤ κ := by rw [hhi]; linarith
    exact_mod_cast this
  · rw [hBlo, hBhi]; exact hlohi
  -- the gate
  · exact Real.sqrt_nonneg _
  · rw [hBm]
    rw [hBm, hBM] at hgate
    exact hgate
  -- the misfire
  · positivity
  · linarith
  · linarith [hcutle.2.2]
  · rw [hBk, hBhi]; exact hmisκ mc hmc
  · rw [hBk, hBlo, show κ + 1 - loAt η₀ κ mc = hiAt η₀ κ (κ - mc) + 1 + 1 by
      unfold loAt hiAt at *; omega]
    have := hmisκ (κ - mc) hmc'
    rw [misfireTail, hmcR'] at this
    rw [hqrdef, hmcR']
    exact this
  · rw [hBkappa, hBlo]; exact (hmisle mc hmc).trans (hmisκ mc hmc)
  · rw [hBkappa, hBhi, show κ - hiAt η₀ κ mc = loAt η₀ κ (κ - mc) + 1 by
      unfold loAt hiAt at *; omega]
    exact (hmisle (κ - mc) hmc').trans (hmisκ (κ - mc) hmc')
  · rw [hmdef]; linarith [hround]

end Passable

end OrthoDFA
