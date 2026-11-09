import RLStar.Proofs.CentreCheck

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
      ∧ PassableAt O η₀ populations Dsf indecisionLimit εcov α δ ρ ρsf pAP
          (shiftBudget η₀ indecisionLimit εcov) (rejFrac η₀ indecisionLimit εcov εs)
          (shiftBudget η₀ indecisionLimit εcov) hsz B
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
    cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit,
    heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit,
    apFloor pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    δ / 16, δ / 16, indecisionLimit / 8, min εcov (1 / 2) / 32,
    5 / 8 * cutBudget η₀ indecisionLimit εcov,
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

/-- `log (1/sizeLevel c N)` is `log (1/c)` and at most `4 log (2N + 2)` more. -/
lemma log_sizeLevel_one_le {c : ℝ} (hc : 0 < c) (N : ℕ) :
    Real.log (1 / sizeLevel c N) ≤ Real.log (1 / c) + 4 * Real.log (2 * N + 2) := by
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  have e : 1 / sizeLevel c N = 1 / c * (((N : ℝ) + 1) * ((N : ℝ) + 2)) := by
    unfold sizeLevel; field_simp
  rw [e, Real.log_mul (by positivity) (by positivity)]
  have hle : ((N : ℝ) + 1) * ((N : ℝ) + 2) ≤ (2 * N + 2) ^ 4 := by
    have h1 : (1 : ℝ) ≤ 2 * N + 2 := by linarith
    nlinarith [pow_le_pow_right₀ h1 (show 2 ≤ 4 by norm_num)]
  have := Real.log_le_log (by positivity) hle
  rw [Real.log_pow] at this
  push_cast at this
  linarith

/-- `√(sΛ/2)` is within `gap·s/40` once `s` is `800Λ/gap²`. -/
lemma sqrt_le_gap_mul {s Λ γ : ℝ} (hs : 0 ≤ s) (hγ : 0 ≤ γ) (h : 800 * Λ ≤ γ ^ 2 * s) :
    Real.sqrt (s * Λ / 2) ≤ γ / 40 * s := by
  rw [show γ / 40 * s = Real.sqrt ((γ / 40 * s) ^ 2) from (Real.sqrt_sq (by positivity)).symm]
  refine Real.sqrt_le_sqrt ?_
  have := mul_le_mul_of_nonneg_left h hs
  nlinarith

set_option maxHeartbeats 2000000 in
/-- At the top count the centre check's numbers hold: the flip-heavy share `εcov/40` with the
misses' `εcov/20` fits the contamination the check allows, the deviations fit `gap/20`, the
misses' rate sits under half of `εcov/20`, and the tails sum to `δ/16`. -/
lemma solved_centre (O : Oracle μ S) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hη₀ : η₀ < 1 / 2) (hgap : 4 * sig η₀ ≤ O.gap) (hεcov : 0 < εcov) (hε4 : εcov ≤ 1 / 4)
    (hεs : 0 < εs) (hind : 0 < indecisionLimit) (hα : 0 < α) (hα1 : α ≤ 1) (hδ : 0 < δ)
    (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) (B : State)
    (hBm : B.npref = prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hBk : B.k = famCount η₀ populations indecisionLimit εcov δ crossLimit + 1)
    (hstr : O.Straddles B.centre η₀) :
    (∀ b, flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
        (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) b * (33 / 32)
        + rejFrac η₀ indecisionLimit εcov εs ≤ εcov / 40)
    ∧ (∀ b, 0 ≤ flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
        (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) b)
    ∧ (∀ s : ℕ, (B.npref : ℝ) * εcov / 2 ≤ s →
        hoeffDev s (gateLevel α populations B) + hoeffDev s (δ / 128) ≤ O.gap / 20 * s)
    ∧ 10 ≤ ((B.k - 1 : ℕ) : ℝ) * sig η₀
    ∧ Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (2 * sig η₀ / 5) ^ 2) ≤ εcov / 40
    ∧ 2 * Real.exp (-((B.npref : ℝ) * relRate (1 - 2 * εcov + εcov) (1 - 2 * εcov)))
        + (Real.exp (-((B.npref : ℝ) * relRate (flipQ O B.centre (shiftBudget η₀ indecisionLimit
              εcov) (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) true
              + flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
                (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) true / 32)
              (flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
                (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) true)))
          + Real.exp (-((B.npref : ℝ) * relRate (flipQ O B.centre (shiftBudget η₀ indecisionLimit
              εcov) (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) false
              + flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
                (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov) false
                / 32)
              (flipQ O B.centre (shiftBudget η₀ indecisionLimit εcov)
                (rejFrac η₀ indecisionLimit εcov εs) (shiftBudget η₀ indecisionLimit εcov)
                false))))
        + 2 * Real.exp (-((B.npref : ℝ) * relRate (εcov / 20)
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (2 * sig η₀ / 5) ^ 2))))
        + 2 * (δ / 128) ≤ δ / 16 := by
  have hs := sig_pos η₀ hη₀
  have hg1 := gap_le_one O
  have hs4 : sig η₀ ≤ 1 / 4 := by linarith
  have hη0 : 0 ≤ η₀ := by rw [sig] at hs4; linarith
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hcdef
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hη₀ hεcov hind
  have hc0 : 0 < c := rejFrac_pos η₀ hη₀ hεcov hεs hind
  have hNR : (B.npref : ℝ) = N := by rw [hBm]
  have hN1 : (1 : ℝ) ≤ N := by
    have := prefCount_pos (εs := εs) (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) (hsz := hsz) η₀ populations εcov δ α pAP
    exact_mod_cast this
  have hkκ : ((B.k - 1 : ℕ) : ℝ) = κ := by rw [hBk]; simp
  have hκv : 8 ≤ (κ : ℝ) * voteSlack η₀ := famCount_wide (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) (δ := δ) η₀ populations hη0 hη₀ hεcov hind
  have hκs : 10 ≤ (κ : ℝ) * sig η₀ := by
    have e : (κ : ℝ) * voteSlack η₀ = 4 / 5 * ((κ : ℝ) * sig η₀) := by rw [voteSlack]; ring
    linarith
  have hs2 : sig η₀ ^ 2 ≤ 1 / 16 := by nlinarith
  -- the shift budget and the redraws are small against `εcov`
  have hΔle : Δ ≤ sig η₀ ^ 2 * εcov / 512 := by
    have h1 : shiftScale indecisionLimit εcov ≤ εcov := (min_le_left _ _).trans (min_le_left _ _)
    rw [hΔdef, shiftBudget]
    have := mul_le_mul_of_nonneg_left h1 (sq_nonneg (sig η₀))
    linarith
  have hcle : c ≤ εcov / 16384 := by
    have h1 := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀ (εcov := εcov)).1
    have h2 : min εcov (1 / 2) ≤ εcov := min_le_left _ _
    have h3 : 0 ≤ min εcov (1 / 2) := le_min hεcov.le (by norm_num)
    have := mul_le_mul hs2 h2 h3 (by norm_num)
    rw [← hcdef] at h1
    linarith
  -- the flip-heavy mass
  have hw : ∀ b, 4 * sig η₀ ^ 2 ≤ clsShift O B.centre b := clsShift_ge_of_gap O hstr hη₀ hgap
  have hw1 : ∀ b, clsShift O B.centre b ≤ 1 := clsShift_le_one O hstr hη₀
  have hq : ∀ b, flipQ O B.centre Δ c Δ b ≤ 5 * εcov / 256 + 20 * c := by
    intro b
    have hwp : 0 < clsShift O B.centre b := by nlinarith [hw b]
    unfold flipQ
    rw [div_le_iff₀ (by positivity)]
    nlinarith [hw b, hΔle]
  have hqge : ∀ b, 40 * Δ ≤ flipQ O B.centre Δ c Δ b := by
    intro b
    have hwp : 0 < clsShift O B.centre b := by nlinarith [hw b]
    unfold flipQ
    rw [le_div_iff₀ (by positivity)]
    nlinarith [hw1 b, mul_nonneg hc0.le hwp.le]
  have hq0 : ∀ b, 0 ≤ flipQ O B.centre Δ c Δ b := fun b => le_trans (by positivity) (hqge b)
  -- the references' count, which the strong sides' reads and the class tails take
  have hc1 : ⌈50 * (gateDevT η₀ + gateDevU η₀) ^ 2
        * Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 3) / δ)
        / gateRoom η₀ indecisionLimit εcov ^ 2⌉₊ ≤ N := by
    rw [hNdef, prefCount, ← hMdef]; omega
  set W : ℝ := gateDevT η₀ + gateDevU η₀ with hWdef
  set L₁ : ℝ := Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 3) / δ) with hL₁
  have hgR : gateRoom η₀ indecisionLimit εcov = εcov / 5 := by
    rw [gateRoom, min_eq_left (by linarith)]
  have hP1 : 50 * W ^ 2 * L₁ ≤ N * (εcov / 5) ^ 2 := by
    have := le_trans (Nat.le_ceil _) (by exact_mod_cast hc1 :
      ((⌈50 * W ^ 2 * L₁ / gateRoom η₀ indecisionLimit εcov ^ 2⌉₊ : ℕ) : ℝ) ≤ N)
    rw [hgR, div_le_iff₀ (by positivity)] at this
    exact this
  have hgG : gateGap η₀ ≤ 2 * sig η₀ := by
    have := gateKappa_pos hη₀
    unfold gateGap gateA; rw [sig]; linarith
  have hgG0 := gateGap_pos hη0 hη₀
  have hU : 9 / 20 ≤ gateDevU η₀ * sig η₀ ^ 2 := by
    unfold gateDevU
    rw [div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
    nlinarith [mul_le_mul hgG hgG hgG0.le (by linarith)]
  have hD : 9 / 10 ≤ gateDevD η₀ * sig η₀ ^ 2 := by
    unfold gateDevD
    rw [div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
    nlinarith [mul_le_mul hgG hgG hgG0.le (by linarith)]
  obtain ⟨hT0, hD0, hU0⟩ := gateDev_pos hη0 hη₀
  have hW : 81 / 400 ≤ W ^ 2 * sig η₀ ^ 4 := by
    have h1 : 9 / 20 ≤ W * sig η₀ ^ 2 := by
      rw [hWdef]; nlinarith [mul_pos hT0 (pow_pos hs 2)]
    nlinarith [h1]
  set LT : ℝ := Real.log (128 / δ) with hLT
  have hM1 : (1 : ℝ) ≤ M := by
    have h : 0 < ⌈2 * ((κ : ℝ) + 1) / pAP⌉₊ := Nat.ceil_pos.2 (by positivity)
    have : 1 ≤ M := by rw [hMdef, poolCount, ← hκdef]; omega
    exact_mod_cast this
  have hLmaster : 10 * LT ≤ N * εcov ^ 2 * sig η₀ ^ 4 / 25 := by
    rcases le_or_gt LT 0 with h | h
    · have : 0 ≤ N * εcov ^ 2 * sig η₀ ^ 4 / 25 := by positivity
      linarith
    · have hLL : LT ≤ L₁ := Real.log_le_log (by positivity)
        (div_le_div_of_nonneg_right (by nlinarith) hδ.le)
      have h1 := mul_le_mul_of_nonneg_right hP1 (by positivity : (0 : ℝ) ≤ sig η₀ ^ 4)
      have h2 : 50 * (81 / 400) * L₁ ≤ 50 * W ^ 2 * L₁ * sig η₀ ^ 4 := by
        have := mul_le_mul_of_nonneg_left hW (by linarith : (0 : ℝ) ≤ 50 * L₁)
        linarith
      nlinarith
  have hε2 : εcov ^ 2 ≤ εcov / 4 := by nlinarith
  have hs4' : sig η₀ ^ 4 ≤ sig η₀ ^ 2 / 16 := by nlinarith
  have hEsκ : Real.exp (-2 * (κ : ℝ) * (2 * sig η₀ / 5) ^ 2) ≤ εcov / 40 := by
    have hEs := famCount_hoeff (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      η₀ populations εcov δ hη₀
    rw [← hκdef] at hEs
    set cut : ℝ := cutBudget η₀ indecisionLimit εcov with hcutdef
    have hcut0 : 0 < cut := cutBudget_pos η₀ hη₀ hεcov hind
    obtain ⟨hcut1, hcut2, -⟩ := cutBudget_le η₀ indecisionLimit εcov
    rw [← hcutdef] at hcut1 hcut2
    have e1 : -2 * (κ : ℝ) * (2 * sig η₀ / 5) ^ 2 = -(8 * κ * sig η₀ ^ 2 / 25) := by ring
    rw [e1]
    have h1 : Real.exp (-(8 * κ * sig η₀ ^ 2 / 25)) ≤ Real.exp (-Real.log (2 / cut)) :=
      Real.exp_le_exp.2 (by linarith)
    have h2 : Real.exp (-Real.log (2 / cut)) = cut / 2 := by
      rw [Real.exp_neg, Real.exp_log (by positivity)]
      field_simp
    have h3 : cut / 2 ≤ εcov / 40 := by
      have hg := cutBudget_le_gate η₀ indecisionLimit εcov
      rw [← hcutdef] at hg
      have ha : gateA η₀ = 2 * sig η₀ := by rw [gateA, sig]; ring
      have hm : min εcov (1 / 2) ≤ εcov := min_le_left _ _
      have hm0 : 0 ≤ min εcov (1 / 2) := le_min hεcov.le (by norm_num)
      rw [ha] at hg
      have : (2 * sig η₀) ^ 2 * min εcov (1 / 2) ≤ 1 / 4 * εcov := by
        have h4 : (2 * sig η₀) ^ 2 ≤ 1 / 4 := by nlinarith
        exact mul_le_mul h4 hm hm0 (by norm_num)
      linarith
    linarith
  refine ⟨fun b => ?_, hq0, ?_, by rw [hkκ]; exact hκs, ?_, ?_⟩
  · have := hq b
    nlinarith [hcle]
  · -- the deviations
    intro s hs'
    have hgl : gateLevel α populations B
        = sizeLevel (α / (9 * ((populations.card : ℝ) * ((κ : ℝ) + 1)))) N := by
      simp only [gateLevel, hBm, hBk, Nat.cast_add, Nat.cast_one, mul_assoc]
    obtain ⟨hKΛ, hq1⟩ := solved_level_room (εs := εs) (hsz := hsz) (pAP := pAP)
      (crossLimit := crossLimit) (indecisionLimit := indecisionLimit) η₀ populations
      (δ := δ) hη₀ hη0 hεcov hα hα1 hcard
    rw [← hNdef, ← hκdef, hgR] at hKΛ
    rw [← hκdef] at hq1
    set q : ℝ := 9 * ((populations.card : ℝ) * ((κ : ℝ) + 1)) / α with hqdef
    set Λ : ℝ := Real.log q + 4 * Real.log (2 * (N : ℝ) + 2) with hΛdef
    have hΛa : Real.log (1 / gateLevel α populations B) ≤ Λ := by
      rw [hgl]
      have hc' : 0 < α / (9 * ((populations.card : ℝ) * ((κ : ℝ) + 1))) := by positivity
      have := log_sizeLevel_one_le hc' N
      rwa [one_div_div] at this
    have hΛ : 100 * Λ ≤ sig η₀ ^ 2 * εcov * N := by
      rcases le_or_gt Λ 0 with h | h
      · have : 0 ≤ sig η₀ ^ 2 * εcov * N := by positivity
        linarith
      · have hX : 81 / 100 ≤ (gateDevD η₀ + 1) ^ 2 * sig η₀ ^ 4 := by
          have h1 : 9 / 10 ≤ (gateDevD η₀ + 1) * sig η₀ ^ 2 := by nlinarith [hD]
          nlinarith [h1]
        have h2 := mul_le_mul_of_nonneg_right hKΛ (by positivity : (0 : ℝ) ≤ sig η₀ ^ 4)
        have h3 : 200 * (81 / 100) * Λ ≤ 200 * (gateDevD η₀ + 1) ^ 2 * Λ * sig η₀ ^ 4 := by
          have := mul_le_mul_of_nonneg_left hX (by linarith : (0 : ℝ) ≤ 200 * Λ)
          linarith
        have h4 : 9 * (εcov / 5) ^ 2 * N * sig η₀ ^ 4 ≤ sig η₀ ^ 2 * εcov * N := by
          have a1 : (εcov / 5) ^ 2 ≤ εcov / 100 := by nlinarith only [hε4, hεcov]
          have a2 := mul_le_mul a1 hs4' (by positivity) (by positivity)
          have a3 := mul_le_mul_of_nonneg_left a2 (by positivity : (0 : ℝ) ≤ 9 * N)
          have a4 : 0 ≤ sig η₀ ^ 2 * εcov * N := by positivity
          linarith
        linarith
    have hΛT : 100 * LT ≤ sig η₀ ^ 2 * εcov * N := by
      have : N * εcov ^ 2 * sig η₀ ^ 4 / 25 ≤ sig η₀ ^ 2 * εcov * N / 10 := by
        have a2 := mul_le_mul hε2 hs4' (by positivity) (by positivity)
        have a3 := mul_le_mul_of_nonneg_left a2 (by positivity : (0 : ℝ) ≤ N / 25)
        have a4 : 0 ≤ sig η₀ ^ 2 * εcov * N := by positivity
        linarith
      linarith
    have hsR : (0 : ℝ) ≤ s := Nat.cast_nonneg _
    have hg0 : 0 ≤ O.gap := by linarith
    have hsN : (N : ℝ) * εcov / 2 ≤ s := by rw [← hNR]; exact hs'
    have key : ∀ Λ', 100 * Λ' ≤ sig η₀ ^ 2 * εcov * N → 800 * Λ' ≤ O.gap ^ 2 * s := by
      intro Λ' h
      have hg2 : 16 * sig η₀ ^ 2 ≤ O.gap ^ 2 := by nlinarith only [hgap, hs]
      have := mul_le_mul hg2 hsN (by positivity) (by positivity)
      linarith
    have hdT : hoeffDev s (δ / 128) ≤ Real.sqrt (s * LT / 2) :=
      hoeffDev_le_sqrt (by rw [one_div_div])
    calc hoeffDev s (gateLevel α populations B) + hoeffDev s (δ / 128)
        ≤ Real.sqrt (s * Λ / 2) + Real.sqrt (s * LT / 2) :=
          add_le_add (hoeffDev_le_sqrt hΛa) hdT
      _ ≤ O.gap / 40 * s + O.gap / 40 * s :=
          add_le_add (sqrt_le_gap_mul hsR hg0 (key Λ hΛ)) (sqrt_le_gap_mul hsR hg0 (key LT hΛT))
      _ = O.gap / 20 * s := by ring
  · rw [hkκ]; exact hEsκ
  · -- the tails
    rw [hNR, hkκ]
    obtain ⟨-, tth, -⟩ := solved_tails (α := α) (crossLimit := crossLimit) (εs := εs) (hsz := hsz)
      η₀ populations hη0 hη₀ hεcov hind hδ hpAP (by linarith)
    rw [← hNdef, ← hΔdef] at tth
    have hδJ : δ / (128 * (populations.card : ℝ)) ≤ δ / 128 :=
      div_le_div_of_nonneg_left hδ.le (by norm_num) (by linarith)
    have hN0 : (0 : ℝ) ≤ N := by linarith
    have tail : ∀ r : ℝ, LT ≤ N * r → Real.exp (-(N * r)) ≤ δ / 128 := fun r h =>
      exp_neg_le_of_log_le (by positivity) (by rw [one_div_div]; exact h)
    have hr1 : εcov ^ 2 / 2 ≤ relRate (1 - 2 * εcov + εcov) (1 - 2 * εcov) := by
      unfold relRate
      rw [le_div_iff₀ (by linarith)]
      nlinarith only [pow_pos hεcov 3]
    have hLN : LT ≤ N * εcov ^ 2 / 250 := by
      have b1 : sig η₀ ^ 4 ≤ 1 := by nlinarith only [hs4', hs2]
      have b2 := mul_le_mul_of_nonneg_left b1 (by positivity : (0 : ℝ) ≤ N * εcov ^ 2 / 25)
      linarith
    have hLN2 : LT ≤ N * (εcov / 160) := by
      have b3 := mul_le_mul_of_nonneg_left hε2 hN0
      have b4 : 0 ≤ N * εcov := by positivity
      linarith
    have hT1 : Real.exp (-(N * relRate (1 - 2 * εcov + εcov) (1 - 2 * εcov))) ≤ δ / 128 :=
      tail _ (le_trans (by nlinarith only [hLN, hN0, sq_nonneg εcov])
        (mul_le_mul_of_nonneg_left hr1 hN0))
    have hT2 : ∀ b, Real.exp (-(N * relRate (flipQ O B.centre Δ c Δ b
        + flipQ O B.centre Δ c Δ b / 32) (flipQ O B.centre Δ c Δ b))) ≤ δ / 128 := by
      intro b
      have hqp : 0 < flipQ O B.centre Δ c Δ b := lt_of_lt_of_le (by positivity) (hqge b)
      have hr2 : Δ / 1056 ≤ relRate (flipQ O B.centre Δ c Δ b + flipQ O B.centre Δ c Δ b / 32)
          (flipQ O B.centre Δ c Δ b) := by
        unfold relRate
        rw [le_div_iff₀ (by positivity)]
        have := mul_le_mul_of_nonneg_left (hqge b) hqp.le
        nlinarith only [this, mul_pos hΔ0 hqp]
      refine le_trans (Real.exp_le_exp.2 (neg_le_neg (mul_le_mul_of_nonneg_left hr2 hN0)))
        (le_trans tth hδJ)
    have hEs' := hEsκ
    have hr3 : εcov / 160
        ≤ relRate (εcov / 20) (Real.exp (-2 * (κ : ℝ) * (2 * sig η₀ / 5) ^ 2)) := by
      have h := relRate_anti (l := εcov / 20) (l' := εcov / 20) (Real.exp_pos _).le hEs'
        (by linarith) le_rfl
      have e : relRate (εcov / 20) (εcov / 40) = εcov / 160 := by
        unfold relRate; field_simp; ring
      linarith
    have hT3 : Real.exp (-(N * relRate (εcov / 20)
        (Real.exp (-2 * (κ : ℝ) * (2 * sig η₀ / 5) ^ 2)))) ≤ δ / 128 :=
      tail _ (le_trans hLN2 (mul_le_mul_of_nonneg_left hr3 hN0))
    linarith [hT2 true, hT2 false]

end Passable

end OrthoDFA
