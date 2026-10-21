import OrthoDFA.Proofs.MiscutGate
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The adaptive clustering loop, gated: the proof

`ClusteringCorrect`, in `OrthoDFA.Proofs.Schedule`, is what this file proves.  Two parts,
composed by `sound_and_terminating`: `validity_of_returned` — whatever is returned is valid,
whenever it is returned — and `loop_terminates` — the loop returns at some round.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {rule : Clusterer S}
variable {J : Type*} [Fintype J]
variable {η₀ : ℝ}
variable {indecisionLimit : ℝ}
variable {crossLimit : ℝ}
variable {hsz : ℕ}

section Loop

open scoped Classical in
/-- The runs where the clustering stalls on the seed, or overshoots the round's size.  The
gate refuses a stalled family through whichever side the population populates, so this is the liveness
half's obligation, not the round's. -/
noncomputable def stalled (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (B : State)
    (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | ¬ (kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)}

open scoped Classical in
/-- A pool of `k` screened candidates is a cluster of `k`.  The seed is one of them and
the ranking keeps it, so the family has `k − 1` members. -/
lemma stalled_subset (O : Oracle μ S) (populations : Finset J) (B : State) (hkpos : 0 < B.k) :
    stalled rule O populations B (B.k - 1) (B.k - 1)
      ⊆ {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)} := by
  intro x hx
  by_contra hk
  simp only [Set.mem_setOf_eq, Classical.not_not] at hk
  have hcard : (clusterFull rule O populations x B).card = B.k :=
    rule.card_eq _ _ _ B.k (one_mem_screenedAt O populations B x) hk hkpos
  have herase : (clusterBy rule O.mq populations x B).card = B.k - 1 := by
    have := clusterAt_card_add_one (rule := rule) O populations B x
    omega
  exact hx ⟨le_of_eq herase.symm, le_of_eq herase⟩

open scoped Classical in
/-- The liveness half's obligation, discharged: the clustering has a family of the round's
own size except on the events `measureReal_smallScreen_le` prices. -/
theorem measureReal_stalled_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations) (Dj₀ : Measure S)
    [IsProbabilityMeasure Dj₀] (hPp₀ : Pp j₀ = iidStream Dj₀)
    (γ pAP t ρsf ρ : ℝ) (rs rt : ℕ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : ∀ t, Dj₀.real {t} ≤ ρ) :
    (streamMeasure μ Pp Pc Dsf).real (stalled rule O populations B (B.k - 1) (B.k - 1))
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + (rejTail (B.npref + rt) (((B.npref + rt : ℕ) : ℝ) * ρ) rt
          + 2 * ((B.nsuff : ℝ) + 1) * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations Pp Pc Dsf hsuppP hsuppSf B j₀ hj₀ Dj₀ hPp₀
      γ pAP t ρsf ρ rs rt hγ hpAP0 ht hpAPBound hscd hsig hsc hcount hρsf hρsf0 hρj)

lemma exp_tail_anti {m a b : ℝ} (hm : 0 ≤ m) (ha : 0 ≤ a) (hab : a ≤ b) :
    Real.exp (-2 * m * b ^ 2) ≤ Real.exp (-2 * m * a ^ 2) := by
  refine Real.exp_le_exp.2 ?_
  have h := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ ha hab 2) hm
  nlinarith

/-! ### The band -/

/-- The vote's two clean means, pushed by a flipping `flipFracOf` of the family and read
`voteSlack` in, meet the centre `κ/2` exactly. -/
lemma vote_shifts (O : Oracle μ S) {η₀ : ℝ} (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (κ : ℝ) :
    κ * ((O.η + O.gap * flipFracOf O η₀) + voteSlack η₀) ≤ κ / 2
      ∧ κ / 2 ≤ κ * ((1 - O.η - O.gap * flipFracOf O η₀) - voteSlack η₀) := by
  have hg := gap_mul_flipFracOf O η₀ (lt_of_le_of_lt hηle hη₀)
  rw [hg, sig]
  constructor <;> nlinarith

/-- What every rung of the ladder satisfies whatever its prefix count: only the prefix count
and the skip guard change down the ladder, and none of these read either. -/
lemma rung_facts (populations : Finset J) {εcov δ pAP : ℝ} (mi : ℕ)
    (hη₀ : η₀ < 1 / 2) (hεcov : 0 < εcov)
    (hind : 0 < indecisionLimit) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi).k
    ∧ 0 < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi).scd
    ∧ ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi).sc : ℝ)
      ≤ ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi).scd : ℝ)
        * (flipBudget η₀ populations indecisionLimit εcov δ * ((1 - η₀) * (1 - 2 * η₀))
          - 2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2)) := by
  classical
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hind hcard
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set B : State := solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi
    with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBscd : B.scd = ⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  clear_value B κ
  refine ⟨?_, ?_, ?_⟩
  · rw [hBk]; omega
  · rw [hBscd]; omega
  · rw [hBsc, hBscd]
    have hone : (63 : ℝ) ≤ 2 * ((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ +
      1 : ℕ) : ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ := by
      have hcl : (63 : ℝ) / (2 * screenMargin η₀ populations indecisionLimit εcov δ)
          ≤ ((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ) := by
        refine le_trans (Nat.le_ceil _) ?_
        push_cast
        linarith
      rw [div_le_iff₀ (by positivity)] at hcl
      linarith
    have hceilub : (⌈((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
      ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ : ℝ)
        ≤ ((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ)
          * screenMargin η₀ populations indecisionLimit εcov δ + 1 :=
      le_of_lt (Nat.ceil_lt_add_one (by positivity))
    have hFeq : flipBudget η₀ populations indecisionLimit εcov δ * ((1 - η₀) * (1 - 2 * η₀))
        = 128 / 63 * screenMargin η₀ populations indecisionLimit εcov δ := by
      rw [screenMargin, hsval]; ring
    rw [hFeq]
    have hS0 : (0 : ℝ) ≤ ((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 :
      ℕ) : ℝ) :=
      Nat.cast_nonneg _
    nlinarith [hceilub, hone, hγ.le, hS0]

/-- A rung's family is `famCount` without its seed, cut at `loAt` and `hiAt`. -/
lemma band_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) :
    B.k = famCount η₀ populations indecisionLimit εcov δ crossLimit + 1
      ∧ B.lo = loAt η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit)
      ∧ B.hi = hiAt η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit) := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  exact ⟨rfl, rfl, rfl⟩

lemma lohi_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) : B.lo ≤ B.hi := by
  obtain ⟨-, hlo, hhi⟩ := band_of_mem_schedule hB
  rw [hlo, hhi]
  unfold loAt hiAt
  omega

/-- A vote over a family no larger than a rung's own, whose mean lies outside the rung's band,
reaches the band's far side at most `crossLimit` of the time. -/
theorem cross_of_mem_schedule (O : Oracle μ S) {populations : Finset J} {εcov δ α pAP : ℝ}
    {B : State} (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2) (hcross : 0 < crossLimit)
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (F : Finset S) (hF : F.card + 1 ≤ B.k) (p : S) :
    (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
    ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit) := by
  obtain ⟨hBk, hBlo, hBhi⟩ := band_of_mem_schedule hB
  obtain ⟨hct, -, hb1⟩ := famCount_cross (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ hη0 hη₀ hcross
  have hev := famCount_even (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  rw [bandHalf] at hb1
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  obtain ⟨hlo, hhi⟩ := loAt_hiAt η₀ hev h4 hη0 hη₀
  have hκR : (4 : ℝ) ≤ κ := by exact_mod_cast h4
  have hFk : F.card ≤ κ := by omega
  have hb1R : (1 : ℝ) ≤ bandAt η₀ κ := by exact_mod_cast hb1
  have hv : voteSlack η₀ ≤ 1 / 5 := by rw [voteSlack, sig]; linarith
  have hb := bandAt_le η₀ κ hη₀
  have hlo0 : (0 : ℝ) ≤ loAt η₀ κ := Nat.cast_nonneg _
  have hsum : loAt η₀ κ + hiAt η₀ κ + 1 = κ := by
    have : ((loAt η₀ κ + hiAt η₀ κ + 1 : ℕ) : ℝ) = κ := by push_cast; rw [hlo, hhi]; ring
    exact_mod_cast this
  rw [hBlo, hBhi]
  refine ⟨fun hm => ?_, fun hm => ?_⟩
  · have h := (vote_tails O F p κ ((hiAt η₀ κ : ℝ) / κ) hFk (by positivity)
      (by rw [div_le_one (by linarith)]; rw [hhi]; nlinarith)).1 (loAt η₀ κ)
      (by rw [mul_div_cancel₀ _ (by linarith)]; exact hm.le)
      (by rw [mul_div_cancel₀ _ (by linarith), hhi, hlo]; linarith)
    exact h.trans hct
  · have h := (vote_tails O F p κ ((loAt η₀ κ : ℝ) / κ) hFk (by positivity)
      (by rw [div_le_one (by linarith)]; rw [hlo]; linarith)).2 (hiAt η₀ κ + 1)
      (by rw [mul_div_cancel₀ _ (by linarith)]; exact hm)
      (by rw [mul_div_cancel₀ _ (by linarith)]; push_cast; rw [hhi, hlo]; linarith)
    refine le_trans (measureReal_mono fun ω hω => hω) (le_trans h (le_trans ?_ hct))
    rw [binomSfGe_symm _ κ (hiAt η₀ κ + 1) (by omega),
      show κ + 1 - (hiAt η₀ κ + 1) = loAt η₀ κ + 1 by omega, crossTail]
    refine sub_le_sub_left (binomSfGe_mono (by positivity) ?_ ?_ κ (loAt η₀ κ + 1)) 1
    · have : (0 : ℝ) ≤ (loAt η₀ κ : ℝ) / κ := by positivity
      linarith
    · rw [le_sub_iff_add_le, ← add_div, div_le_one (by linarith)]
      have : ((loAt η₀ κ + hiAt η₀ κ + 1 : ℕ) : ℝ) = κ := by exact_mod_cast hsum
      push_cast at this
      linarith

/-- A count that clears `log (c/ε) / (2γ²)` kills the tail it was read off. -/
lemma tail_le_of_count {γ ε c : ℝ} {n : ℕ} (hγ : 0 < γ) (hε : 0 < ε) (hc : 0 < c)
    (hn : Real.log (c / ε) / (2 * γ ^ 2) ≤ (n : ℝ)) :
    c * Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ ε := by
  have hγ2 : (0 : ℝ) < 2 * γ ^ 2 := by positivity
  have h1 : Real.log (c / ε) ≤ 2 * (n : ℝ) * γ ^ 2 := by
    rw [div_le_iff₀ hγ2] at hn
    linarith
  have h2 : Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ ε / c := by
    rw [show (-2 * (n : ℝ) * γ ^ 2) = -(2 * (n : ℝ) * γ ^ 2) from by ring]
    refine exp_neg_le_of_log_le (by positivity) ?_
    rwa [one_div_div]
  calc c * Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ c * (ε / c) :=
        mul_le_mul_of_nonneg_left h2 hc.le
    _ = ε := by field_simp

/-- `tail_le_of_count` at an exponent rate `a` rather than `2γ²`. -/
lemma tail_le_of_rate {a ε c : ℝ} {n : ℕ} (ha : 0 < a) (hε : 0 < ε) (hc : 0 < c)
    (hn : Real.log (c / ε) / a ≤ (n : ℝ)) :
    c * Real.exp (-((n : ℝ) * a)) ≤ ε := by
  have h1 : Real.log (c / ε) ≤ (n : ℝ) * a := by
    rw [div_le_iff₀ ha] at hn; linarith
  have h2 : Real.exp (-((n : ℝ) * a)) ≤ ε / c :=
    exp_neg_le_of_log_le (by positivity) (by rwa [one_div_div])
  calc c * Real.exp (-((n : ℝ) * a)) ≤ c * (ε / c) := mul_le_mul_of_nonneg_left h2 hc.le
    _ = ε := by field_simp

/-- The pool is deep enough to hold accept-preserving suffixes.  This is the one event the
ladder shares, so it is charged once against `δ/4` rather than per rung. -/
lemma solved_findability (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ}
    (hδ : 0 < δ) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * (9 * pAP / 20) ^ 2) ≤ δ / 32 := by
  have ctap : Real.log (1 / (δ / 32)) / (2 * (9 * pAP / 20) ^ 2)
      ≤ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℕ) : ℝ) := by
    have hmono : Real.log (1 / (δ / 32))
        ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
      refine Real.log_le_log (by positivity) ?_
      rw [one_div_div, div_le_div_iff₀ hδ hδ]
      nlinarith
    refine le_trans (div_le_div_of_nonneg_right hmono (by positivity)) ?_
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊
        ≤ poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
      rw [poolCount]; omega
    exact_mod_cast hle
  have h := tail_le_of_count (γ := 9 * pAP / 20) (c := 1) (by positivity)
    (by positivity : (0 : ℝ) < δ / 32) (by norm_num) ctap
  linarith

/-! ### Part 1: whatever is returned is valid -/

/-- The family at a state is invalid: on some tame population `j` its cut is wrong on more of
`Dc j` than the gate holds that population to and `s`, or it leaves more than `u` undecided. -/
def FailAt (rule : Clusterer S) (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (Dc : J → Measure S) (εcov s u : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ tame, 1 - gateLimit uni εcov j - s
        ≤ (Dc j).real {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
          (oracleNoise x)}
      ∧ (Dc j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
          (oracleNoise x)} ≤ u}

lemma familyBy_card_le (O : Oracle μ S) (populations : Finset J) (B : State) (hk : 0 < B.k)
    (x : Run Ω S J) : (familyBy rule O.mq populations x B).card ≤ B.k := by
  have h1 := Finset.card_insert_le 1 (clusterBy rule O.mq populations x B)
  have h2 := clusterAt_card_add_one (rule := rule) O populations B x
  have h3 : (clusterFull rule O populations x B).card ≤ B.k := rule.card_le _ _ _ B.k hk
  unfold familyBy
  omega

open scoped Classical in
/-- A pool rich in accept-preserving draws holds one outside any set of at most `k` suffixes. -/
lemma exists_apRef (O : Oracle μ S) (M : ℕ) (pAP t : ℝ) (rs k : ℕ)
    (hcount : (k : ℝ) + 1 + rs ≤ M * (pAP - t)) (x : Run Ω S J)
    (hx : x ∉ apShort O M pAP t) (hgood : poolGood M rs x) (F : Finset S) (hF : F.card ≤ k) :
    ∃ v ∈ poolAt M x \ F, ∀ p, O.label (p * v) = O.label p := by
  classical
  set W : Finset S := (freshDraws ∅ (fun i => suffixDraw i x) M).filter
    (fun v => ∀ p, O.label (p * v) = O.label p) with hW
  have hfresh := ap_fresh_ge O M rs x hgood
  rw [← hW] at hfresh
  simp only [apShort, Set.mem_setOf_eq, not_le] at hx
  have hWk : F.card < W.card := by
    have : (k : ℝ) < W.card := by linarith
    have : k < W.card := by exact_mod_cast this
    omega
  obtain ⟨v, hvW, hvF⟩ := Finset.exists_mem_notMem_of_card_lt_card hWk
  refine ⟨v, Finset.mem_sdiff.2 ⟨Finset.mem_insert_of_mem (Finset.mem_filter.1 hvW).1, hvF⟩,
    (Finset.mem_filter.1 hvW).2⟩

lemma keptAt_of_good (populations : Finset J) (j : J) (m r : ℕ) (x : Run Ω S J)
    (htab : tabGood j m r x) (hcert : certGood populations j m r x) :
    keptAt populations j m x :=
  ⟨(freshDraws_of_good ∅ (fun i => prefixDraw j i x) m r htab).1,
    (certOf_of_good populations j m r x hcert).1⟩

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state.  A returned family invalid on some tame population failed the gate's
soundness there — a look's sides, or the passage to the population — or slipped past the FNR
test, which the certification draws see unless their sampler redrew too much. -/
theorem per_state_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations tame : Finset J) (uni : J) (huni : uni ∈ tame) (htame : tame ⊆ populations)
    (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf] (L : J → List S) (Dp Dc : J → Measure S)
    [∀ j, IsProbabilityMeasure (Dp j)] [∀ j, IsProbabilityMeasure (Dc j)]
    (hL : ∀ j ∈ tame, (L j).Nodup) (hPp : ∀ j ∈ tame, Pp j = prependMeasure (L j) (Dp j))
    (hPc : ∀ j ∈ tame, Pc j = iidStream (Dc j))
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (indecisionLimit εcov α u : ℝ)
    (hil : 0 < indecisionLimit) (hα0 : 0 < α) (hα1 : α ≤ 1) (B : State) (hk : 0 < B.k)
    (r : ℕ) (c : ℝ) (hc0 : 0 ≤ c) (hr : (r : ℝ) ≤ c * B.npref)
    (hu : 3 / 2 * indecisionLimit + c ≤ u)
    (ρ : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (Dc j).real {t} ≤ ρ ∧ (Dp j).real {t} ≤ ρ)
    (G : Set (Run Ω S J))
    (hG : ∀ x ∉ G, ∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p) :
    (streamMeasure μ Pp Pc Dsf).real ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni Dc εcov c u B) \ G)
      ≤ 7 * populations.card * gateLevel α populations B
        + ((tame.card : ℝ) * (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
            + rejTail (2 * B.npref + r)
              (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * ρ) r)
          + populations.card * Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2)) := by
  classical
  have hJ : (1 : ℝ) ≤ populations.card := by
    exact_mod_cast Finset.card_pos.2 ⟨uni, htame huni⟩
  have htc : (tame.card : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_le_card htame
  set a : ℝ := gateLevel α populations B with ha
  have ha0 : 0 < a := sizeLevel_pos (by positivity) _
  have ha1 : a ≤ 1 := sizeLevel_le (by positivity)
    (by rw [div_le_one (by positivity)]; nlinarith) _
  set gateJ : J → Set (Run Ω S J) := fun j =>
    (⋃ e ∈ Finset.range (B.npref + 1),
        sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e)))
      ∪ ⋃ N : ℕ, rawShortAt rule O populations (Dc j) j B (gateLimit uni εcov j + c) a (N + 1)
    with hgateJ
  set indJ : J → Set (Run Ω S J) := fun j =>
    undecidedShort rule O populations (Dc j) j B u (indecisionLimit / 2)
    with hindJ
  have hsub : ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni Dc εcov c u B) \ G)
      ⊆ (⋃ j ∈ tame, (gateJ j ∪ indJ j))
        ∪ {x : Run Ω S J | ¬ drawsGoodOn populations tame B.npref r x} := by
    rintro x ⟨⟨hret, hfail⟩, hxG⟩
    obtain ⟨hsize, hfnr, hgate⟩ := hret
    simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
    obtain ⟨j, hj, hfj⟩ := hfail
    by_cases hdg : drawsGoodOn populations tame B.npref r x
    swap
    · exact Or.inr hdg
    have hc2 := hdg.2 j hj
    have hcg := certGood_of_certGood2 populations j B.npref r x hc2
    have hkept := keptAt_of_good populations j B.npref r x (hdg.1 j hj) hcg
    by_cases hmass : 1 - gateLimit uni εcov j - c ≤ (Dc j).real {p | cutCorrect O B.lo B.hi
        (clusterBy rule O.mq populations x B) p (oracleNoise x)}
    · -- the cut is right enough, so the FNR test let through an undecided cut
      have hund := (not_and.1 hfj) hmass
      refine Or.inl (Set.mem_biUnion hj (Or.inr ?_))
      obtain ⟨hcardN, hcnt⟩ := certOf_of_good populations j B.npref r x hcg
      have hcard : ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
          = (B.npref : ℝ) := by exact_mod_cast hcardN
      refine ⟨(not_le.1 hund).le, ?_⟩
      have h := (hcnt (fun p => ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) p (oracleNoise x))).1
      have hf := hfnr j (htame hj) hkept
      rw [hcard] at hf
      have hm0 : (0 : ℝ) ≤ B.npref := Nat.cast_nonneg _
      nlinarith
    · -- the cut is wrong on more than the limit, and the gate passed it
      have hgt : gateLimit uni εcov j + c < (Dc j).real {p | ¬ cutCorrect O B.lo B.hi
          (clusterBy rule O.mq populations x B) p (oracleNoise x)} := by
        rw [show {p : S | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}
            = {p : S | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}ᶜ from rfl,
          measureReal_compl (measurableSet_of_countable _), probReal_univ]
        linarith [not_le.1 hmass]
      have hin := falseCertify_subset (rule := rule) O hηle populations (Dc j) j B
        (gateLimit uni εcov j) c a r hc0 hr ⟨hgate j (htame hj) hkept, hG x hxG, hgt, hc2⟩
      exact Or.inl (Set.mem_biUnion hj (Or.inl hin))
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have h2 : (streamMeasure μ Pp Pc Dsf).real ((⋃ j ∈ tame, (gateJ j ∪ indJ j))
        ∪ {x : Run Ω S J | ¬ drawsGoodOn populations tame B.npref r x})
      ≤ tame.card * (7 * a + Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2))
        + (tame.card : ℝ) * (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
            + rejTail (2 * B.npref + r)
              (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * ρ) r) := by
    refine le_trans (measureReal_union_le _ _) (add_le_add ?_
      (measureReal_drawsBadOn_le Pp Pc Dsf populations tame L Dp Dc hL hPp hPc B.npref r ρ hρ))
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun j hj => le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_))
    · exact measureReal_popGate_le hflat O populations Pp Pc Dsf hsuppP hsuppC hsuppSf j
        (htame hj) (Dc j) (hPc j hj) B ha0 ha1
    · exact measureReal_undecidedShort_le Pp Pc Dsf O populations j (Dc j) (hPc j hj) B _ _
        (by linarith) (by linarith)
  have he0 : 0 ≤ Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2) := Real.exp_nonneg _
  have hgl : 7 * (tame.card : ℝ) * a ≤ 7 * populations.card * a := by nlinarith
  have hel : (tame.card : ℝ) * Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2)
      ≤ populations.card * Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2) :=
    mul_le_mul_of_nonneg_right htc he0
  nlinarith

/-- The pool holds an accept-preserving draw beyond any family of the round's size and what
its sampler redraws, at the findability `9pAP/20` is read at. -/
lemma pool_holds_ref (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ} (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    ((famCount η₀ populations indecisionLimit εcov δ crossLimit + 1 : ℕ) : ℝ) + 1
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
      ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (pAP - 9 * pAP / 20) := by
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set L : ℝ := Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2) with hL
  have hc : 2 * ((κ : ℝ) + 1) / pAP + L ≤ (M : ℝ) := by
    rw [hMdef, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP), Nat.le_ceil L]
  have hlog : (4 : ℝ) ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
    have h1 : (128 : ℝ) ≤ 128 * (populations.card : ℝ) / δ := by
      rw [le_div_iff₀ hδ]; nlinarith
    have h2 : Real.log 128 ≤ Real.log (128 * (populations.card : ℝ) / δ) :=
      Real.log_le_log (by norm_num) h1
    have h3 : (4 : ℝ) ≤ Real.log 128 := by
      rw [Real.le_log_iff_exp_le (by norm_num)]
      have h := Real.exp_one_lt_d9
      have e : Real.exp 4 = Real.exp 1 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
      rw [e]
      have h0 : 0 < Real.exp 1 := Real.exp_pos 1
      have : Real.exp 1 ^ 4 ≤ (2.7182818286 : ℝ) ^ 4 := pow_le_pow_left₀ h0.le h.le 4
      nlinarith
    linarith
  have hLp : 2 ≤ pAP * L := by
    rw [hL, mul_div_assoc', le_div_iff₀ (by positivity)]
    nlinarith
  have hM : 2 * ((κ : ℝ) + 1) + 2 ≤ (M : ℝ) * pAP := by
    have := mul_le_mul_of_nonneg_left hc hpAP.le
    rw [mul_add, mul_div_cancel₀ _ hpAP.ne'] at this
    nlinarith
  have hrs : ((poolRej pAP M : ℕ) : ℝ) ≤ pAP * M / 20 := Nat.floor_le (by positivity)
  push_cast
  nlinarith

/-- The states of the schedule are told apart by their prefix counts. -/
lemma schedule_npref_inj {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} :
    Set.InjOn State.npref
      ↑(schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) := by
  intro B hB B' hB' h
  rw [Finset.mem_coe, schedule] at hB hB'
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  obtain ⟨i', -, rfl⟩ := Finset.mem_image.1 hB'
  simp only [solvedStateAt] at h ⊢
  rw [h]

/-- What the gate spends over the ladder's rungs is at most `α`. -/
lemma gateLevel_sum_le {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} (hα : 0 ≤ α)
    (hpop : populations.Nonempty) (s : Finset State)
    (hs : s ⊆ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) :
    ∑ B ∈ s, 7 * (populations.card : ℝ) * gateLevel α populations B ≤ α := by
  classical
  have hJ : (0 : ℝ) < populations.card := by exact_mod_cast Finset.card_pos.2 hpop
  have e : ∀ B : State, 7 * (populations.card : ℝ) * gateLevel α populations B
      = sizeLevel α B.npref := by
    intro B
    have h13 : (7 * (populations.card : ℝ)) ≠ 0 := by positivity
    unfold gateLevel sizeLevel
    field_simp
  rw [Finset.sum_congr rfl (fun B _ => e B)]
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz
  have hle : ∀ B ∈ s, B.npref ≤ N := by
    intro B hB
    have := hs hB
    rw [schedule] at this
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 this
    exact Nat.div_le_self _ _
  rw [← Finset.sum_image (f := fun n => sizeLevel α n)
    (fun B hB B' hB' h => schedule_npref_inj (hs hB) (hs hB') h)]
  have hsub : s.image State.npref ⊆ Finset.range (N + 1) := by
    intro n hn
    obtain ⟨B, hB, rfl⟩ := Finset.mem_image.1 hn
    exact Finset.mem_range.2 (Nat.lt_succ_of_le (hle B hB))
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun n _ _ => sizeLevel_nonneg hα n)) ?_
  · rw [sum_range_sizeLevel]
    have : 0 ≤ α * (1 / (((N + 1 : ℕ) : ℝ) + 1)) := by positivity
    linarith

open scoped Classical in
set_option maxHeartbeats 2000000 in
/-- Part 1 — whatever is returned is valid, whenever it is returned.

Except with probability `7δ/16 + α`, no rung the loop may stop at returns an invalid family.
The pool's findability and its redraws are one event for the whole ladder, at `δ/16`; each rung
carries its share of `3δ/8`, and the gate spends `α` over all of them. -/
theorem validity_of_returned {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations tame : Finset J) (uni : J) (huni : uni ∈ tame) (htame : tame ⊆ populations)
    (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf] (L : J → List S) (Dp Dc : J → Measure S)
    [∀ j, IsProbabilityMeasure (Dp j)] [∀ j, IsProbabilityMeasure (Dc j)]
    (hL : ∀ j ∈ tame, (L j).Nodup) (hPp : ∀ j ∈ tame, Pp j = prependMeasure (L j) (Dp j))
    (hPc : ∀ j ∈ tame, Pc j = iidStream (Dc j))
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α u : ℝ) (hindLim : 0 < indecisionLimit)
    (hu : 3 / 2 * indecisionLimit + rejFrac η₀ indecisionLimit εcov ≤ u)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (Dc j).real {t} ≤ 3 * ρ ∧ (Dp j).real {t} ≤ 3 * ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (pAP : ℝ) (hpAP0 : 0 < pAP) (hpAP1 : pAP ≤ 1)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcoll : ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP
          crossLimit) : ℕ) : ℝ) ^ 2 * ρsf
        / (2 * ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ
          pAP crossLimit) : ℝ) + 1))
      ≤ δ / 256) :
    (streamMeasure μ Pp Pc Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov
            δ α pAP crossLimit hsz ρ},
          retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
            ∩ FailAt rule O populations tame uni Dc εcov
                (rejFrac η₀ indecisionLimit εcov) u B.val) ≤ 7 * δ / 16 + α := by
  classical
  set iLc : ℝ := indecisionLimit with hiLc
  have hiLc0 : 0 < iLc := hindLim
  have hiLc1 : iLc ≤ indecisionLimit := le_rfl
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  have hJ : (1 : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_pos.2 hpop
  have hρ0 : 0 ≤ 3 * ρ := le_trans measureReal_nonneg (hρ uni huni 1).1
  set M : ℕ := poolCount η₀ populations iLc εcov δ pAP crossLimit with hMdef
  set rs : ℕ := poolRej pAP M with hrsdef
  set N : ℕ := prefCount η₀ populations iLc εcov δ α pAP crossLimit hsz with hNdef
  set V : ℕ := validCount η₀ populations iLc εcov δ pAP crossLimit with hVdef
  set G : Set (Run Ω S J) := apShort O M pAP (9 * pAP / 20) ∪ {x | ¬ poolGood M rs x} with hGdef
  set st := stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ with hst
  have hGle : (streamMeasure μ Pp Pc Dsf).real G ≤ δ / 16 := by
    refine le_trans (measureReal_union_le _ _) ?_
    have h1 := le_trans (measureReal_apShort_le Pp Pc Dsf O M pAP (9 * pAP / 20) hpAP0.le
      (by positivity) hpAPBound)
      (solved_findability (indecisionLimit := iLc) (crossLimit := crossLimit) η₀ populations
        (εcov := εcov) hδ hpAP0 hJ)
    have h2 := le_trans (measureReal_poolBad_le (μ := μ) Pp Pc Dsf M rs ρsf hρsf hρsf0) hcoll
    linarith
  have hper : ∀ B ∈ st, (streamMeasure μ Pp Pc Dsf).real
      ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni Dc εcov
            (rejFrac η₀ indecisionLimit εcov) u B) \ G)
      ≤ 7 * populations.card * gateLevel α populations B
        + (δ * (B.npref : ℝ) / (8 * N) + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0) := by
    intro B hB
    have hcap := capped_of_mem_stoppable hB
    have hsched := (Finset.mem_filter.1 hB).1
    have hBM : B.nsuff = M := nsuff_of_mem_schedule hsched
    have hk : 0 < B.k := by
      rw [schedule] at hsched
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hsched
      simp [solvedStateAt]
    have hBk : B.k = famCount η₀ populations iLc εcov δ crossLimit + 1 := by
      rw [schedule] at hsched
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hsched
      rfl
    set c : ℝ := rejFrac η₀ iLc εcov with hcdef
    have hc0 : 0 < c := rejFrac_pos η₀ hη₀ hεcov hiLc0
    obtain ⟨hcle, -⟩ := rejFrac_le (indecisionLimit := iLc) η₀ hη₀ hεcov hiLc0
    have hcutle := cutBudget_le η₀ iLc εcov
    set r : ℕ := rejCount η₀ iLc εcov B.npref with hrdef
    have hr : (r : ℝ) ≤ c * B.npref := Nat.floor_le (by positivity)
    have hG' : ∀ x ∉ G, ∃ v ∈ refsBy rule O.mq populations x B,
        ∀ p, O.label (p * v) = O.label p := by
      intro x hx
      simp only [hGdef, Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hx
      have hcount := pool_holds_ref (indecisionLimit := iLc) (crossLimit := crossLimit) η₀
        populations (εcov := εcov) hδ hδ1 hpAP0 hpAP1 hJ
      rw [← hMdef, ← hrsdef, ← hBk] at hcount
      unfold refsBy
      rw [hBM]
      exact exists_apRef O M pAP (9 * pAP / 20) rs B.k (by exact_mod_cast hcount) x hx.1 hx.2
        _ (familyBy_card_le O populations B hk x)
    have h := per_state_le (rule := rule) hflat O populations tame uni huni htame Pp Pc Dsf L Dp
      Dc hL hPp hPc hsuppP hsuppC hsuppSf hηle hη₀ indecisionLimit εcov α u hindLim hα0 hα1 B hk
      r c hc0.le hr hu (3 * ρ) hρ G hG'
    refine le_trans h (add_le_add le_rfl (le_trans ?_ hcap.share))
    -- the redraws, at the tame populations' count, and the FNR tail at the schedule's
    -- indecision rate
    have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
    have htc : (tame.card : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_le_card htame
    have hX0 : 0 ≤ rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * (3 * ρ)) r
        + rejTail (2 * B.npref + r)
          (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * (3 * ρ)) r :=
      add_nonneg (rejTail_nonneg _ (by positivity) _) (rejTail_nonneg _ (by positivity) _)
    have hT := mul_le_mul_of_nonneg_right htc hX0
    unfold stateFail
    rw [← hrdef]
    nlinarith [hT]
  have hsplit : (⋃ B : {B : State // B ∈ st},
        retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
          ∩ FailAt rule O populations tame uni Dc εcov
              (rejFrac η₀ indecisionLimit εcov) u B.val)
      ⊆ (⋃ B ∈ st, (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
          ∩ FailAt rule O populations tame uni Dc εcov
              (rejFrac η₀ indecisionLimit εcov) u B) \ G) ∪ G := by
    intro x hx
    obtain ⟨B, hB⟩ := Set.mem_iUnion.1 hx
    by_cases hg : x ∈ G
    · exact Or.inr hg
    · exact Or.inl (Set.mem_biUnion B.2 ⟨hB, hg⟩)
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hNpos : (0 : ℝ) < N := by exact_mod_cast prefCount_pos _ _ _ _ _ _
  have hsumN := stoppable_npref_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit)
    (hsz := hsz) η₀
    populations εcov δ α pAP ρ
  have hinv := schedule_inv_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit) η₀
    populations εcov δ α pAP V st (by rw [hst, stoppable]; exact Finset.filter_subset _ _)
  have hgate := gateLevel_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit)
    (η₀ := η₀) (εcov := εcov) (δ := δ) (pAP := pAP) hα0.le hpop st
    (by rw [hst, stoppable]; exact Finset.filter_subset _ _)
  have hite : ∀ B : State, (if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0)
      = δ / 16 * (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) := by
    intro B; split_ifs <;> ring
  have hmain : (streamMeasure μ Pp Pc Dsf).real (⋃ B ∈ st, (retBy rule O.mq populations uni η₀
        indecisionLimit εcov α B ∩ FailAt rule O populations tame uni Dc εcov
            (rejFrac η₀ indecisionLimit εcov) u B) \ G)
      ≤ α + 3 * δ / 8 := by
    calc _ ≤ ∑ B ∈ st, (streamMeasure μ Pp Pc Dsf).real ((retBy rule O.mq populations uni η₀
            indecisionLimit εcov α B ∩ FailAt rule O populations tame uni Dc εcov
                (rejFrac η₀ indecisionLimit εcov) u B)
              \ G) := measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ st, (7 * populations.card * gateLevel α populations B
            + (δ * (B.npref : ℝ) / (8 * N)
              + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0)) :=
          Finset.sum_le_sum hper
      _ = ∑ B ∈ st, 7 * (populations.card : ℝ) * gateLevel α populations B
            + ((δ / (8 * N)) * ∑ B ∈ st, (B.npref : ℝ)
              + δ / 16 * ∑ B ∈ st, (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0)) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
          congr 1
          congr 1
          · exact Finset.sum_congr rfl (fun B _ => by ring)
          · exact Finset.sum_congr rfl (fun B _ => hite B)
      _ ≤ α + ((δ / (8 * N)) * (2 * N) + δ / 16 * 2) := by
          gcongr
      _ = α + 3 * δ / 8 := by field_simp; ring
  linarith

/-! ### Part 2: the loop returns -/

/-! ### What the solved budget's fields buy

Each of these says that a field, read off its condition, meets it.  They are the analytic
content of the construction; everything else about `PassableAt` is arithmetic on the
definitions. -/

/-- The cut budget is at least a tenth of the scale. -/
lemma budgetScale_pos (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) : 0 < budgetScale η₀ indecisionLimit εcov :=
  lt_min (mul_pos (lt_min hε (by norm_num)) (pow_pos (gateA_pos hη0 hsig) 2)) hind

/-- The scale sits under the signal: `a²/2 ≤ sig` since `a ≤ 1 − η₀`. -/
lemma budgetScale_le_sig (η₀ indecisionLimit εcov : ℝ) (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    budgetScale η₀ indecisionLimit εcov ≤ sig η₀ := by
  have ha := gateA_pos hη0 hsig
  have hne : (1 - η₀) ≠ 0 := by intro h; linarith
  have e : (1 - η₀) * gateA η₀ = 1 - 2 * η₀ := by unfold gateA; field_simp
  have ha1 : gateA η₀ ≤ 1 - η₀ := by nlinarith
  have hmn : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
  have h : min εcov (1 / 2) * gateA η₀ ^ 2 ≤ sig η₀ := by
    rw [sig]
    rcases le_or_gt 0 (min εcov (1 / 2)) with h0 | h0
    · nlinarith [mul_le_mul_of_nonneg_left ha1 ha.le]
    · nlinarith [sq_nonneg (gateA η₀)]
  exact (min_le_left _ _).trans h

lemma cutBudget_ge (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    budgetScale η₀ indecisionLimit εcov / 10 ≤ cutBudget η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hB := budgetScale_pos η₀ indecisionLimit εcov hε hind hη0 hsig
  have hmn : 0 < min εcov (1 / 2) := lt_min hε (by norm_num)
  have h0 : budgetScale η₀ indecisionLimit εcov ≤ min εcov (1 / 2) * gateA η₀ ^ 2 :=
    min_le_left _ _
  have ha1 := gateA_le_one hη0 hsig
  have ha0 := gateA_pos hη0 hsig
  have h1 : budgetScale η₀ indecisionLimit εcov ≤ εcov := by
    have : min εcov (1 / 2) * gateA η₀ ^ 2 ≤ min εcov (1 / 2) :=
      mul_le_of_le_one_right hmn.le (by nlinarith)
    linarith [min_le_left εcov (1 / 2)]
  have h4 : budgetScale η₀ indecisionLimit εcov / 10
      ≤ 32 * gateKappa η₀ * min εcov (1 / 2) / 65 := by
    have hk := gateKappa_ge hη0 hsig
    have := mul_le_mul_of_nonneg_left hk hmn.le
    nlinarith
  have h2 := budgetScale_le_sig η₀ indecisionLimit εcov hη0 hsig
  have h3 : budgetScale η₀ indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  rw [cutBudget]
  refine le_min (by linarith) (le_min (by linarith) (le_min (by linarith) h4))

/-- The redraw fraction is at least a constant times the scale squared: the cut budget is a tenth
of the scale, and the flip fraction three fifths of the signal. -/
lemma rejFrac_ge (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    6 * budgetScale η₀ indecisionLimit εcov ^ 2 / 12800 ≤ rejFrac η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hB := budgetScale_pos η₀ indecisionLimit εcov hε hind hη0 hsig
  have hBs := budgetScale_le_sig η₀ indecisionLimit εcov hη0 hsig
  have hcut := cutBudget_ge η₀ indecisionLimit εcov hε hind hη0 hsig
  have hf : 6 * sig η₀ / 10 ≤ flipFrac η₀ := by
    rw [flipFrac, div_le_div_iff₀ (by norm_num) (by linarith)]
    nlinarith
  rw [rejFrac]
  have := mul_le_mul hcut (le_trans (by linarith : 6 * budgetScale η₀ indecisionLimit εcov / 10
    ≤ 6 * sig η₀ / 10) hf) (by positivity) (by linarith [hcut, cutBudget_pos η₀ hsig hε hind])
  nlinarith

/-- What the pool's collision cap buys: its redraws cost a hundred-and-twenty-eighth of the error
budget per population. -/
lemma poolDraws_le_of_cap (η₀ : ℝ) (populations : Finset J) {εcov δ pAP ρsf : ℝ}
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hδ : 0 < δ) (hρsf0 : 0 ≤ ρsf)
    (hρsf : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit) : ℕ) : ℝ)
        ^ 2 * ρsf
      / (2 * ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit) : ℝ)
        + 1))
      ≤ δ / (128 * ((populations.card : ℝ) + 1)) := by
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hrej : poolRej pAP M = ⌊pAP / 20 * M⌋₊ := by
    rw [poolRej]; congr 1; ring
  rw [hrej]
  have hrt := rejTerm_le (c := pAP / 20) (by positivity) (by linarith) M
  have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
  have hM0 : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hcap : ρsf * (2 ^ 12 * ((populations.card : ℝ) + 1) * (M : ℝ)) ≤ δ * pAP := by
    rcases Nat.eq_zero_or_pos M with hM | hM
    · rw [hM]; simp; positivity
    · have h := hρsf
      rw [poolCollisionCap, ← hMdef, le_div_iff₀ (by positivity)] at h
      exact h
  have hstep : ((M + ⌊pAP / 20 * M⌋₊ : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((⌊pAP / 20 * M⌋₊ : ℝ) + 1))
      ≤ ρsf / 2 * (2 * (M : ℝ) / (pAP / 20)) := by
    rw [show ((M + ⌊pAP / 20 * M⌋₊ : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((⌊pAP / 20 * M⌋₊ : ℝ) + 1))
        = ρsf / 2 * (((M + ⌊pAP / 20 * M⌋₊ : ℕ) : ℝ) ^ 2 / ((⌊pAP / 20 * M⌋₊ : ℝ) + 1)) by
      field_simp]
    gcongr
  refine le_trans hstep ?_
  rw [show ρsf / 2 * (2 * (M : ℝ) / (pAP / 20)) = 20 * (ρsf * M) / pAP by field_simp,
    div_le_div_iff₀ hpAP (by positivity)]
  nlinarith [hcap]

/-- The top rung's tails, at `δ/(128·|populations|)` and, for the screen's and the flip
understatement's unions over the pool, sixteen and eight times that. -/
lemma solved_tails (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (9 * pAP / 20) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ 2 * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1)
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
          * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
          * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-((prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * (cutBudget η₀ indecisionLimit εcov / 2112)))
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * (indecisionLimit / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ)) := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hf : 0 < flipFrac η₀ := flipFrac_pos η₀ hsig
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- each count clears the logarithm its tail asks for
  have ctap : Real.log (1 / ε₀) / (2 * (9 * pAP / 20) ^ 2) ≤ (M : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊ ≤ M := by
      rw [hMdef, poolCount]; omega
    exact_mod_cast hle
  have cscr : Real.log ((2 * ((M : ℝ) + 1)) / ε₀)
      / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤ (m : ℝ) := by
    have heq : (2 * ((M : ℝ) + 1)) / ε₀
        = 256 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (256 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cdirty : Real.log (((M : ℝ) + 1) / ε₀)
      / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
      ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)⌉₊ ≤
          m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cth2 : Real.log (128 * (populations.card : ℝ) / δ)
      * (2112 / cutBudget η₀ indecisionLimit εcov + 2 / indecisionLimit ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ)
        * (2112 / cutBudget η₀ indecisionLimit εcov + 2 / indecisionLimit ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have heq1 : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp
  have cth : Real.log (1 / ε₀) / (cutBudget η₀ indecisionLimit εcov / 2112) ≤ (m : ℝ) := by
    rw [heq1, div_div_eq_mul_div, mul_div_assoc]
    rcases le_or_gt 0 (Real.log (128 * (populations.card : ℝ) / δ)) with hlog0 | hlog0
    · have : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) * (2 / indecisionLimit ^ 2) := by
        positivity
      nlinarith [cth2]
    · have : Real.log (128 * (populations.card : ℝ) / δ) * (2112 / cutBudget η₀
        indecisionLimit εcov) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hlog0.le (by positivity)
      linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have cund : Real.log (1 / ε₀) / (2 * (indecisionLimit / 2) ^ 2) ≤ (m : ℝ) := by
    rw [heq1, show 2 * (indecisionLimit / 2) ^ 2 = indecisionLimit ^ 2 / 2 by ring,
      div_div_eq_mul_div, mul_div_assoc]
    rcases le_or_gt 0 (Real.log (128 * (populations.card : ℝ) / δ)) with hlog0 | hlog0
    · have : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) * (2112 / cutBudget η₀
        indecisionLimit εcov) := by positivity
      nlinarith [cth2]
    · have : Real.log (128 * (populations.card : ℝ) / δ) * (2 / indecisionLimit ^ 2) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg hlog0.le (by positivity)
      linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  -- the tails those counts kill
  have ttap : Real.exp (-2 * (M : ℝ) * (9 * pAP / 20) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := 9 * pAP / 20) (c := 1) (by positivity) hε₀ (by norm_num) ctap
    linarith
  have tscr : 2 * ((M : ℝ) + 1)
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
      ≤ ε₀ := tail_le_of_count (by positivity) (by positivity) (by positivity) cscr
  have tdirty : ((M : ℝ) + 1) * Real.exp (-2 * (m : ℝ)
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ :=
    tail_le_of_count (by positivity) (by positivity) (by linarith) cdirty
  have tdirty' : (M : ℝ) * Real.exp (-2 * (m : ℝ)
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tdirty
    exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)
  have tth : Real.exp (-((m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 2112))) ≤ ε₀ := by
    have h := tail_le_of_rate (a := cutBudget η₀ indecisionLimit εcov / 2112) (c := 1)
      (by positivity) hε₀ (by norm_num) cth
    linarith
  have tund : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := indecisionLimit / 2) (c := 1) (by positivity) hε₀
      (by norm_num) cund
    linarith
  exact ⟨ttap, tscr, tdirty', tth, tund⟩

/-- The pool holds two accept-preserving draws in expectation beyond what its sampler redraws. -/
lemma poolCount_found (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ} (hsig : η₀ < 1 / 2)
    (hpAP : 0 < pAP) :
    (2 : ℝ) + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
      ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) * (pAP / 2) := by
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hκ : (4 : ℝ) ≤ (κ : ℝ) := by exact_mod_cast famCount_ge_four η₀ populations εcov δ hsig
  have hc : 2 * ((κ : ℝ) + 1) / pAP ≤ (M : ℝ) := by
    rw [hMdef, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊]
  have hM : 2 * ((κ : ℝ) + 1) ≤ (M : ℝ) * pAP := by
    rw [div_le_iff₀ hpAP] at hc
    exact hc
  have hrs : ((poolRej pAP M : ℕ) : ℝ) ≤ pAP * M / 20 := Nat.floor_le (by positivity)
  nlinarith

/-- A tail a count `V` kills at `ε` is under `ε·V/m` at any count `m ≥ V`: past `V` it decays
faster than `V/m` does. -/
lemma tail_le_of_count_scaled {γ ε c V m : ℝ} (hγ : 0 < γ) (hε : 0 < ε) (hε6 : ε ≤ 1 / 6)
    (hc : 1 ≤ c) (hV0 : 0 < V) (hV : Real.log (c / ε) / (2 * γ ^ 2) ≤ V) (hVm : V ≤ m) :
    c * Real.exp (-2 * m * γ ^ 2) ≤ ε * V / m := by
  have hm0 : 0 < m := lt_of_lt_of_le hV0 hVm
  have ha : Real.log (c / ε) ≤ 2 * V * γ ^ 2 := by
    rw [div_le_iff₀ (by positivity)] at hV
    linarith
  have hlog1 : 1 ≤ Real.log (c / ε) := by
    have h6 : 6 ≤ c / ε := by rw [le_div_iff₀ hε]; linarith
    have h2 : Real.log ((2 : ℝ)⁻¹) ≤ (2 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
    have h3 : Real.log ((3 : ℝ)⁻¹) ≤ (3 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
    rw [Real.log_inv] at h2 h3
    have hl6 : Real.log 6 = Real.log 2 + Real.log 3 := by
      rw [show (6 : ℝ) = 2 * 3 by norm_num, Real.log_mul (by norm_num) (by norm_num)]
    have hmono : Real.log 6 ≤ Real.log (c / ε) := Real.log_le_log (by norm_num) h6
    linarith
  have h1 : c * Real.exp (-2 * V * γ ^ 2) ≤ ε := by
    have he : Real.exp (-2 * V * γ ^ 2) ≤ ε / c := by
      calc Real.exp (-2 * V * γ ^ 2) ≤ Real.exp (-Real.log (c / ε)) :=
            Real.exp_le_exp.2 (by linarith)
        _ = ε / c := by
            rw [Real.exp_neg, Real.exp_log (by positivity)]
            field_simp
    calc c * Real.exp (-2 * V * γ ^ 2) ≤ c * (ε / c) :=
          mul_le_mul_of_nonneg_left he (by linarith)
      _ = ε := by field_simp
  have h2 : Real.exp (-2 * (m - V) * γ ^ 2) ≤ V / m := by
    have hk : (m - V) / V ≤ 2 * (m - V) * γ ^ 2 := by
      rw [div_le_iff₀ hV0]
      nlinarith [sub_nonneg.2 hVm]
    have ht : m / V ≤ Real.exp ((m - V) / V) := by
      have h := Real.add_one_le_exp ((m - V) / V)
      have h' : (m - V) / V + 1 = m / V := by field_simp; ring
      linarith
    calc Real.exp (-2 * (m - V) * γ ^ 2) ≤ Real.exp (-((m - V) / V)) :=
          Real.exp_le_exp.2 (by linarith)
      _ = 1 / Real.exp ((m - V) / V) := by rw [Real.exp_neg, one_div]
      _ ≤ 1 / (m / V) := one_div_le_one_div_of_le (by positivity) ht
      _ = V / m := by field_simp
  have hsplit : c * Real.exp (-2 * m * γ ^ 2)
      = c * Real.exp (-2 * V * γ ^ 2) * Real.exp (-2 * (m - V) * γ ^ 2) := by
    rw [mul_assoc c, ← Real.exp_add]
    ring_nf
  rw [hsplit]
  calc c * Real.exp (-2 * V * γ ^ 2) * Real.exp (-2 * (m - V) * γ ^ 2) ≤ ε * (V / m) :=
        mul_le_mul h1 h2 (by positivity) hε.le
    _ = ε * V / m := by ring

/-- At atoms within the cap, each of a rung's two samplers is rejected more than `rejCount` of
its draws with chance at most `(m/N)·2^-(r+1)`: it lands on what is already held at most
`(|J| + 3)·m·3ρ` of the time, a `(m/N)/8` share of `rejFrac`. -/
lemma rung_tails_le (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (m : ℕ) (hm : m ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) :
    rejTail (m + rejCount η₀ indecisionLimit εcov m)
        (((m + rejCount η₀ indecisionLimit εcov m : ℕ) : ℝ) * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov m)
      ≤ (m : ℝ) / (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * (1 / 2) ^ (rejCount η₀ indecisionLimit εcov m + 1)
    ∧ rejTail (2 * m + rejCount η₀ indecisionLimit εcov m)
        (((populations.card * m + (2 * m + rejCount η₀ indecisionLimit εcov m) : ℕ) : ℝ)
          * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov m)
      ≤ (m : ℝ) / (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * (1 / 2) ^ (rejCount η₀ indecisionLimit εcov m + 1) := by
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  set r : ℕ := rejCount η₀ indecisionLimit εcov m with hrdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hind
  have hc4 : c ≤ 1 / 4 := by
    obtain ⟨hcle, -⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hsig hε hind
    have h2 := (cutBudget_le η₀ indecisionLimit εcov).2.1
    have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    linarith
  have hNpos : (0 : ℝ) < N := by
    rw [hNdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hmN : (m : ℝ) ≤ N := by exact_mod_cast hm
  have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg _
  have hP0 : (0 : ℝ) ≤ P := Nat.cast_nonneg _
  have hr1 : (r : ℝ) ≤ c * m := Nat.floor_le (by positivity)
  have hr2 : c * m < (r : ℝ) + 1 := Nat.lt_floor_add_one _
  have hρc : ρ * (64 * (P + 3) * N) ≤ c := by
    have h := le_trans hρcap (le_trans (min_le_left _ _) (min_le_left _ _))
    rwa [le_div_iff₀ (by positivity)] at h
  have hx0 : 0 ≤ (m : ℝ) / N := by positivity
  have hx1 : (m : ℝ) / N ≤ 1 := (div_le_one hNpos).2 hmN
  have hpow : ((m : ℝ) / N / 2) ^ (r + 1) ≤ (m : ℝ) / N * (1 / 2) ^ (r + 1) := by
    rw [div_eq_mul_one_div ((m : ℝ) / N) 2, mul_pow]
    exact mul_le_mul_of_nonneg_right (pow_le_of_le_one hx0 hx1 (by omega)) (by positivity)
  have hcert : rejTail (2 * m + r)
      (((populations.card * m + (2 * m + r) : ℕ) : ℝ) * (3 * ρ)) r
      ≤ (m : ℝ) / N * (1 / 2) ^ (r + 1) := by
    refine le_trans (rejTail_le_pow (by positivity) hx0 ?_) hpow
    push_cast
    rw [← hP]
    have hcm : c * m ≤ 1 / 4 * m := mul_le_mul_of_nonneg_right hc4 hm0
    have hA : 2 * (m : ℝ) + r ≤ 9 / 4 * m := by linarith
    have hB : P * m + (2 * m + r) ≤ (P + 3) * m := by linarith
    have hB0 : 0 ≤ P * m + (2 * m + r) := by positivity
    have h1 : (2 * (m : ℝ) + r) * ((P * m + (2 * m + r)) * (3 * ρ))
        ≤ 9 / 4 * m * ((P + 3) * m * (3 * ρ)) :=
      mul_le_mul hA (mul_le_mul_of_nonneg_right hB (by positivity)) (by positivity)
        (by positivity)
    have h3 : 54 * (P + 3) * ρ * (m : ℝ) ^ 2 ≤ c * m ^ 2 / N := by
      rw [le_div_iff₀ hNpos]
      have : 54 * (P + 3) * ρ * N ≤ c := by nlinarith
      nlinarith [sq_nonneg (m : ℝ)]
    have h4 : c * (m : ℝ) ^ 2 / N ≤ ((r : ℝ) + 1) * (m / N) := by
      rw [show c * (m : ℝ) ^ 2 / N = c * m * (m / N) by ring]
      exact mul_le_mul_of_nonneg_right hr2.le hx0
    nlinarith
  refine ⟨le_trans (rejTail_mono (by positivity) (by omega) ?_ r) hcert, hcert⟩
  refine mul_le_mul_of_nonneg_right ?_ (by positivity)
  exact_mod_cast (show m + r ≤ populations.card * m + (2 * m + r) by omega)

/-- A rung whose count clears `log(32|J|/δ)/(rejFrac·log 2)` pays for both samplers out of half
its proportional share. -/
lemma rung_redraws_le (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (m : ℕ) (hm : m ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov * m * Real.log 2) :
    (populations.card : ℝ) * (rejTail (m + rejCount η₀ indecisionLimit εcov m)
        (((m + rejCount η₀ indecisionLimit εcov m : ℕ) : ℝ) * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov m)
      + rejTail (2 * m + rejCount η₀ indecisionLimit εcov m)
        (((populations.card * m + (2 * m + rejCount η₀ indecisionLimit εcov m) : ℕ) : ℝ)
          * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov m))
      ≤ δ * m / (16 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ))
      := by
  obtain ⟨h1, h2⟩ := rung_tails_le (crossLimit := crossLimit) η₀ populations (δ := δ) (α := α)
    (pAP := pAP) hη0 hsig hε hind hρ0 hρcap m hm
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  set r : ℕ := rejCount η₀ indecisionLimit εcov m with hrdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hNpos : (0 : ℝ) < N := by
    rw [hNdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hind
  have hr2 : c * m < (r : ℝ) + 1 := Nat.lt_floor_add_one _
  have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hpow : 32 * P / δ ≤ (2 : ℝ) ^ (r + 1) := by
    have h : Real.log (32 * P / δ) ≤ Real.log ((2 : ℝ) ^ (r + 1)) := by
      rw [Real.log_pow]
      push_cast
      nlinarith [mul_le_mul_of_nonneg_right hr2.le hl2.le]
    exact (Real.log_le_log_iff (by positivity) (by positivity)).1 h
  have hhalf : ((1 : ℝ) / 2) ^ (r + 1) ≤ δ / (32 * P) := by
    rw [div_pow, one_pow, div_le_div_iff₀ (by positivity) (by positivity)]
    rw [div_le_iff₀ hδ] at hpow
    linarith
  have hx0 : 0 ≤ (m : ℝ) / N := by positivity
  have hsum := add_le_add h1 h2
  have hb : (m : ℝ) / N * (1 / 2) ^ (r + 1) ≤ (m : ℝ) / N * (δ / (32 * P)) :=
    mul_le_mul_of_nonneg_left hhalf hx0
  calc P * (_ + _) ≤ P * ((m : ℝ) / N * (δ / (32 * P)) + (m : ℝ) / N * (δ / (32 * P))) :=
        mul_le_mul_of_nonneg_left (by linarith) hcard.le
    _ = δ * m / (16 * N) := by field_simp; ring

/-- At the top count the rejections are rare enough, with room: `rejFrac·N·log 2` is many
times `log(128|J|/δ)`, the count's own tail. -/
lemma solved_rej_log (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.log (256 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov
        * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * Real.log 2 := by
  obtain ⟨-, tscr, -, -, -⟩ := solved_tails (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀
    populations hsig hε hind hδ hpAP (by linarith)
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set ff : ℝ := flipFrac η₀ with hffdef
  set P : ℝ := (populations.card : ℝ) with hP
  set sm : ℝ := screenMargin η₀ populations indecisionLimit εcov δ with hsmdef
  have hcb0 : 0 < cb := cutBudget_pos η₀ hsig hε hind
  have hff0 : 0 < ff := flipFrac_pos η₀ hsig
  have hs := sig_pos η₀ hsig
  have hcbs : cb ≤ 1 / 20 := by
    have h := (cutBudget_le η₀ indecisionLimit εcov).2.1
    rw [sig] at h; linarith
  have hffle : ff ≤ 7 / 20 := by
    rw [hffdef, flipFrac, sig, div_le_iff₀ (by linarith)]
    nlinarith
  have hsm0 : 0 < sm := screenMargin_pos η₀ populations hsig hε hind (by linarith)
  have hsm : sm ≤ cb * ff / 2 := by
    rw [hsmdef, screenMargin, flipBudget, ← hcb, ← hffdef, ← hP]
    have hsig1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    have h1 : (1 - η₀) * sig η₀ ≤ 1 / 2 := by nlinarith
    have h2 : 0 ≤ (1 - η₀) * sig η₀ := by nlinarith
    have h3 : 55 * cb * ff / (64 * P) ≤ 55 * cb * ff / 64 :=
      div_le_div_of_nonneg_left (by positivity) (by norm_num) (by linarith)
    have h4 : 0 ≤ 55 * cb * ff / (64 * P) := by positivity
    nlinarith
  set L : ℝ := Real.log (256 * P / δ) with hLdef
  have hL : L ≤ 2 * N * (sm / 2) ^ 2 := by
    have hM : (2 : ℝ) ≤ 2 * ((M : ℝ) + 1) := by
      have : (0 : ℝ) ≤ M := Nat.cast_nonneg _; linarith
    have he := Real.exp_nonneg (-2 * (N : ℝ) * (sm / 2) ^ 2)
    have h1 : 2 * Real.exp (-2 * N * (sm / 2) ^ 2) ≤ δ / (128 * P) :=
      le_trans (mul_le_mul_of_nonneg_right hM he) tscr
    have h1' : Real.exp (-2 * N * (sm / 2) ^ 2) ≤ δ / (256 * P) := by
      rw [show δ / (256 * P) = δ / (128 * P) / 2 by field_simp; ring]; linarith
    have h3 := Real.log_le_log (Real.exp_pos _) h1'
    rw [Real.log_exp, Real.log_div (by positivity) (by positivity)] at h3
    rw [hLdef, Real.log_div (by positivity) (by positivity)]
    linarith
  have hrej : rejFrac η₀ indecisionLimit εcov = cb * ff / 128 := by
    obtain ⟨-, h⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hsig hε hind
    rw [h, ← hcb, ← hffdef]; ring
  rw [hrej]
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  have hsm2 : (sm / 2) ^ 2 ≤ (cb * ff / 4) ^ 2 :=
    pow_le_pow_left₀ (by positivity) (by linarith) 2
  have hcf : cb * ff ≤ 7 / 400 := by nlinarith
  have hcf0 : 0 ≤ cb * ff := by positivity
  have h1 : 2 * (N : ℝ) * (sm / 2) ^ 2 ≤ 2 * N * (cb * ff / 4) ^ 2 :=
    mul_le_mul_of_nonneg_left hsm2 (by positivity)
  have h2 : 2 * (N : ℝ) * (cb * ff / 4) ^ 2 ≤ cb * ff / 128 * N * Real.log 2 := by
    have : (cb * ff) * (cb * ff) / 8 ≤ (cb * ff) / 128 * Real.log 2 := by
      have := mul_le_mul_of_nonneg_left (show cb * ff ≤ Real.log 2 / 16 by linarith) hcf0
      nlinarith
    nlinarith
  linarith

/-- The top rung carries its own share of the error budget, `δ/8`. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) :
    stateFail η₀ populations indecisionLimit εcov (3 * ρ) (solvedState η₀ populations
      indecisionLimit εcov δ α pAP crossLimit hsz)
      ≤ δ / 8 := by
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 0 < populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  obtain ⟨-, -, -, -, tund⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hlog := solved_rej_log (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀ populations
    hη0 hsig hε hind hδ hδ1 hpAP hcard1
  have hlog32 : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov * m * Real.log 2 :=
    le_trans (Real.log_le_log (by positivity)
      (div_le_div_of_nonneg_right (by linarith) hδ.le)) hlog
  have hdr := rung_redraws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hind hδ
    hcard hρ0 hρsmall m le_rfl hlog32
  rw [← hmdef, show δ * (m : ℝ) / (16 * (m : ℝ)) = δ / 16 by field_simp] at hdr
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  rw [stateFail, hBm, mul_add]
  nlinarith [mul_le_mul_of_nonneg_left tund hcard.le]

/-- Every rung at or above `validCount` whose rejections are rare enough carries its share: its
samplers' tails from the share proportional to its count, its undecided tail from the one
inversely proportional. -/
theorem rung_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (i : ℕ) (hi : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (hV : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit
      ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz / 2 ^ i)
    (hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov
        * ((prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz / 2 ^ i : ℕ)
          : ℝ) * Real.log 2) :
    solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
        (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz / 2 ^ i)
      ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  set m : ℕ := N / 2 ^ i with hmdef
  have hB : solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m
      ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz := by
    rw [schedule]
    exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩
  have hVpos : 0 < V := by rw [hVdef, validCount]; omega
  have hVR : (0 : ℝ) < (V : ℝ) := by exact_mod_cast hVpos
  have hVm : (V : ℝ) ≤ (m : ℝ) := by exact_mod_cast hV
  have hmR : (0 : ℝ) < (m : ℝ) := lt_of_lt_of_le hVR hVm
  have hmN' : m ≤ N := Nat.div_le_self N (2 ^ i)
  have hmN : (m : ℝ) ≤ (N : ℝ) := by exact_mod_cast hmN'
  have hNR : (0 : ℝ) < (N : ℝ) := lt_of_lt_of_le hmR hmN
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨lohi_of_mem_schedule hB,
    by show 0 < m; exact_mod_cast hmR, poolCount_found η₀ populations hsig hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov (3 * ρ)
      (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m)
    ≤ δ * (m : ℝ) / (8 * N) + if V ≤ m then δ * (V : ℝ) / (16 * m) else 0
  rw [if_pos hV, stateFail, show (solvedStateAt η₀ populations indecisionLimit εcov δ pAP
    crossLimit m).npref = m from rfl]
  have hdr := rung_redraws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hind hδ
    hcard hρ0 hρcap m hmN' hlog
  rw [← hNdef] at hdr
  have hcoll : δ * (m : ℝ) / (16 * N) ≤ δ * (m : ℝ) / (8 * N) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) (by linarith)
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  have hε₀3 : ε₀ ≤ 1 / 6 := by
    rw [hε₀def, div_le_iff₀ (by positivity)]
    have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
      have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
      exact_mod_cast h1
    nlinarith
  have hV4 : Real.log (128 * (populations.card : ℝ) / δ) / (2 * (indecisionLimit / 2) ^ 2)
      ≤ (V : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (indecisionLimit / 2) ^ 2)⌉₊
        ≤ V := by rw [hVdef, validCount]; omega
    exact_mod_cast this
  have hq3 : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp
  have t4 := tail_le_of_count_scaled (γ := indecisionLimit / 2) (c := 1) (by positivity) hε₀ hε₀3
    le_rfl hVR (by rw [hq3]; exact hV4) hVm
  rw [one_mul] at t4
  have htails : (populations.card : ℝ) * (ε₀ * V / m) ≤ δ * (V : ℝ) / (16 * m) := by
    rw [hε₀def]
    field_simp
    nlinarith [mul_pos hδ hVR]
  have hscale := mul_le_mul_of_nonneg_left t4 hcard.le
  rw [mul_add]
  linarith [hscale, hcoll, htails, hdr]

/-- The fewest prefixes a rung needs for its rejections to be rare enough at the cap. -/
noncomputable def rejCountMin (η₀ : ℝ) (populations : Finset J) (indecisionLimit εcov δ : ℝ) : ℕ :=
  ⌈Real.log (32 * (populations.card : ℝ) / δ)
    / (rejFrac η₀ indecisionLimit εcov * Real.log 2)⌉₊

/-- Some rung the loop may stop at has fewer than twice the larger of `validCount` and
`rejCountMin` prefixes: the lowest rung whose count is still at or above both, or any rung when
the top is already below them. -/
theorem exists_small_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (hne : (stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ).Nonempty) :
    ∃ B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ,
      (B.npref : ℝ) ≤ 2 * (max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
        (rejCountMin η₀ populations indecisionLimit εcov δ) : ℝ) := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set V : ℕ := max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
    (rejCountMin η₀ populations indecisionLimit εcov δ) with hVdef
  have hVpos : 0 < V := by
    have : 0 < validCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
      rw [validCount]; omega
    exact lt_of_lt_of_le this (le_max_left _ _)
  have hcast : ((max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
      (rejCountMin η₀ populations indecisionLimit εcov δ) : ℕ) : ℝ) = (V : ℝ) := rfl
  rw [show (max ((validCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℕ) : ℝ)
      ((rejCountMin η₀ populations indecisionLimit εcov δ : ℕ) : ℝ)) = (V : ℝ) by
    rw [hVdef]; push_cast; rfl]
  by_cases hNV : N < V
  · obtain ⟨B, hB⟩ := hne
    refine ⟨B, hB, ?_⟩
    have hsched := Finset.mem_of_mem_filter _ hB
    rw [schedule] at hsched
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 hsched
    show ((N / 2 ^ j : ℕ) : ℝ) ≤ 2 * (V : ℝ)
    have : N / 2 ^ j ≤ V := le_trans (Nat.div_le_self _ _) hNV.le
    have : ((N / 2 ^ j : ℕ) : ℝ) ≤ (V : ℝ) := by exact_mod_cast this
    linarith [(Nat.cast_nonneg V : (0 : ℝ) ≤ V)]
  push_neg at hNV
  have hex : ∃ i, N / 2 ^ (i + 1) < V := by
    refine ⟨N, ?_⟩
    have hlt : N < 2 ^ (N + 1) :=
      lt_of_lt_of_le Nat.lt_two_pow_self (Nat.pow_le_pow_right (by norm_num) (by omega))
    rw [Nat.div_eq_of_lt hlt]
    exact hVpos
  set i := Nat.find hex with hidef
  have hi1 : N / 2 ^ (i + 1) < V := Nat.find_spec hex
  have hi0 : V ≤ N / 2 ^ i := by
    rcases Nat.eq_zero_or_pos i with h0 | hpos
    · rw [h0, pow_zero, Nat.div_one]; exact hNV
    · have h := Nat.find_min hex (show i - 1 < i by omega)
      push_neg at h
      rwa [show i - 1 + 1 = i by omega] at h
  have hhalf : N / 2 ^ i / 2 = N / 2 ^ (i + 1) := by
    rw [Nat.div_div_eq_div_mul, pow_succ]
  have hlt : N / 2 ^ i < 2 * V := by omega
  have hiL : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz := by
    have hpow : 2 ^ i ≤ N := by
      have h1 : 1 ≤ N / 2 ^ i := le_trans hVpos hi0
      exact (Nat.one_le_div_iff (by positivity)).1 h1
    have := Nat.le_log_of_pow_le (by norm_num) hpow
    rw [ladderLen, ← hNdef]
    omega
  have hval : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit ≤ N / 2 ^ i :=
    le_trans (le_max_left _ _) hi0
  have hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov * ((N / 2 ^ i : ℕ) : ℝ) * Real.log 2 := by
    have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hsig hε hind
    have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    have hmin : rejCountMin η₀ populations indecisionLimit εcov δ ≤ N / 2 ^ i :=
      le_trans (le_max_right _ _) hi0
    have hceil : Real.log (32 * (populations.card : ℝ) / δ)
        / (rejFrac η₀ indecisionLimit εcov * Real.log 2) ≤ ((N / 2 ^ i : ℕ) : ℝ) :=
      le_trans (Nat.le_ceil _) (by rw [rejCountMin] at hmin; exact_mod_cast hmin)
    rw [div_le_iff₀ (by positivity)] at hceil
    linarith
  refine ⟨_, rung_mem_stoppable η₀ populations hη0 hsig hε hind hδ hδ1 hpAP hcard hρ0 hρcap i hiL
    hval hlog, ?_⟩
  show ((N / 2 ^ i : ℕ) : ℝ) ≤ 2 * (V : ℝ)
  exact_mod_cast hlt.le

/-- The top rung can always be stopped at: it carries the whole proportional share. -/
theorem solvedState_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz) :
    solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz
      ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ := by
  classical
  have hB := solvedState_mem_schedule (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) (hsz := hsz) η₀ populations εcov δ α pAP
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  have hmpos : 0 < m := prefCount_pos (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨lohi_of_mem_schedule hB, hmpos,
    poolCount_found η₀ populations hsig hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov (3 * ρ)
      (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    ≤ δ * (m : ℝ) / (8 * m) + _
  have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by field_simp
  rw [hval]
  exact le_add_of_le_of_nonneg (solved_share η₀ populations hsig hη0 hε hind hδ hδ1 hpAP hcard hρ0
    hρcap) (by split_ifs <;> positivity)

open scoped Classical in
/-- At a fixed table the state's two counts on a sample are within budget: on the prefixes the
family is light for `frac_selected_le` prices each, and the heavy ones are a fraction of their
own. -/
theorem light_counts_whp {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (lo hi : ℕ) (l lcut lmis : ℝ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (hone : (1 : S) ∈ cands) (hT1 : ∀ t ∈ T, (1 : S) ∉ t)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ lmis)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo (hi + 1) (insert 1 A₀) p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo hi A₀ p ω} ≤ E) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ ((((C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
            ∧ ((C.filter (fun p => ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
                ≤ (lcut + lmis) * (C.card : ℝ))}
      ≤ Real.exp (-((C.card : ℝ) * relRate l E))
        + Real.exp (-((C.card : ℝ) * relRate lmis E)) := by
  classical
  have hsplit : ∀ (ω : Ω) (Q : S → Prop) [DecidablePred Q],
      (C.filter (fun p => Q p)).card
        ≤ (C.filter (fun p => fam ω ∈ good p ∧ Q p)).card
          + (C.filter (fun p => fam ω ∉ good p)).card := by
    intro ω Q _
    refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
    intro p hp
    obtain ⟨hpC, hq⟩ := Finset.mem_filter.1 hp
    by_cases hg : fam ω ∈ good p
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
  have hsub : {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
      ∧ ¬ ((((C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
              ≤ 2 * l * (C.card : ℝ))
          ∧ ((C.filter (fun p => ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
              ≤ (lcut + lmis) * (C.card : ℝ))}
      ⊆ {ω | l * (C.card : ℝ)
            < ((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)}
        ∪ {ω | lmis * (C.card : ℝ)
              < ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)} := by
    rintro ω ⟨hheavy, hbad⟩
    by_contra hc
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_lt] at hc
    obtain ⟨hindL, hmisL⟩ := hc
    have hscale : lcut * (C.card : ℝ) ≤ l * (C.card : ℝ) :=
      mul_le_mul_of_nonneg_right hlcl (Nat.cast_nonneg _)
    have h1 := hsplit ω (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)
    have h2 := hsplit ω (fun p => ¬ cutCorrect O lo hi (fam ω) p ω)
    have h1' : (((C.filter (fun p => ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ))
        ≤ ((C.filter (fun p => fam ω ∈ good p
            ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
          + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast h1
    have h2' : (((C.filter (fun p => ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ))
        ≤ ((C.filter (fun p => fam ω ∈ good p
            ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
          + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast h2
    exact hbad ⟨by linarith, by linarith⟩
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · exact indecision_frac_le_seed hflat O P cands C hP hV hCPre hPC hone lo (hi + 1) T good t₀ ht₀
      hTC hT1 fam hfam hcongr E l hE (le_trans hElcut hlcl) hdec
  · exact miscut_frac_le hflat O P cands C hP hV hCPre hPC lo hi T good t₀ ht₀ hTC
      fam hfam hcongr E lmis hE hElm hcut

open scoped Classical in
lemma measurableSet_miscutAll (O : Oracle μ S) (lo hi : ℕ) (A₀ A : Finset S) (r : ℝ) :
    MeasurableSet {ω | (((A.filter (fun p => ¬ cutCorrect O lo hi A₀ p ω)).card : ℝ) ≤ r)} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => ¬ cutCorrect O lo hi A₀ p ω)
    (fun p _ => measurableSet_cutCorrect' O lo hi A₀ p) (fun U => ((U.card : ℝ) ≤ r)))

open scoped Classical in
/-- The draws were good and the family light on all but `lcut` of population `j`'s sample, and
still the sample leaves more than `2l` undecided or the cut is wrong on more than
`lcut + lmis`. -/
noncomputable def lightMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (j : J)
    (B : State) (l lcut lmis f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x)
      (certOf (prefixesAt populations B.npref x) j B.npref x)
    ∧ (certOf (prefixesAt populations B.npref x) j B.npref x).card = B.npref
    ∧ 0 < (certOf (prefixesAt populations B.npref x) j B.npref x).card
    ∧ (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
        ≤ lcut * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
    ∧ ¬ ((((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
            ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
              (oracleNoise x))).card : ℝ)
          ≤ 2 * l * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
        ∧ (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x))).card : ℝ)
          ≤ (lcut + lmis) * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ))}

open scoped Classical in
lemma measurableSet_lightMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (l lcut lmis f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (lightMiss rule O populations j B l lcut lmis f kmin kmax) := by
  classical
  set Φ : Finset S → Finset S → Finset S → Ω → Prop := fun P C A ω =>
    ((A.filter (fun p => ¬ famGood O f kmin kmax
        (clusterOf rule O B.sc B.scd P C B.k ω) p)).card : ℝ) ≤ lcut * (A.card : ℝ)
      ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
              (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) p ω)).card : ℝ)
            ≤ 2 * l * (A.card : ℝ)
          ∧ ((A.filter (fun p => ¬ cutCorrect O B.lo B.hi
              (clusterOf rule O B.sc B.scd P C B.k ω) p ω)).card : ℝ)
            ≤ (lcut + lmis) * (A.card : ℝ)) with hΦ
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
            ((A.filter (fun p => ¬ famGood O f kmin kmax A₀ p)).card : ℝ) ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ ((A.filter (fun p => ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
                  ≤ (lcut + lmis) * (A.card : ℝ))})
          (fun A₀ => ?_)
        by_cases hlight : ((A.filter (fun p =>
            ¬ famGood O f kmin kmax A₀ p)).card : ℝ) ≤ lcut * (A.card : ℝ)
        · have hrw : {ω : Ω |
              ((A.filter (fun p => ¬ famGood O f kmin kmax A₀ p)).card : ℝ) ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ ((A.filter (fun p => ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
                    ≤ (lcut + lmis) * (A.card : ℝ))}
              = ({ω : Ω | ((A.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)}
                ∩ {ω : Ω | ((A.filter (fun p => ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
                    ≤ (lcut + lmis) * (A.card : ℝ)})ᶜ := by
            ext ω
            simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_inter_iff, hlight, true_and]
          rw [hrw]
          exact ((measurableSet_indecisionCount O B.lo (B.hi + 1) (insert 1 A₀) A
            (2 * l * (A.card : ℝ))).inter
              (measurableSet_miscutAll O B.lo B.hi A₀ A ((lcut + lmis) * (A.card : ℝ)))).compl
        · have hz : {ω : Ω |
              ((A.filter (fun p => ¬ famGood O f kmin kmax A₀ p)).card : ℝ) ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ ((A.filter (fun p => ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
                    ≤ (lcut + lmis) * (A.card : ℝ))} = (∅ : Set Ω) := by
            ext ω
            simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
            rintro ⟨h, -⟩
            exact hlight h
          rw [hz]
          exact MeasurableSet.empty
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
  have hrw : lightMiss rule O populations j B l lcut lmis f kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          {x : Run Ω S J | (Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card)
            ∧ Φ P C A (oracleNoise x)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j B.npref x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), lightMiss, hΦ,
      ← clusterAt_eq_clusterOf O populations B x, and_assoc]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_certSet populations j B _ hR

open scoped Classical in
/-- The lift of `light_counts_whp`. -/
theorem measureReal_lightMiss_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (l lcut lmis f E : ℝ) (kmin kmax : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ lmis)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (streamMeasure μ Pp Pc Dsf).real (lightMiss rule O populations j B l lcut lmis f kmin kmax)
      ≤ Real.exp (-((B.npref : ℝ) * relRate l E))
        + Real.exp (-((B.npref : ℝ) * relRate lmis E)) := by
  classical
  set R : ℝ := Real.exp (-((B.npref : ℝ) * relRate l E))
    + Real.exp (-((B.npref : ℝ) * relRate lmis E)) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : streamMeasure μ Pp Pc Dsf (lightMiss rule O populations j B l lcut lmis f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine streamMeasure_slice_le Pp Pc Dsf _
      (measurableSet_lightMiss O populations j B l lcut lmis f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre Pp Pc Dsf Pre populations hsuppP,
      ae_cert_mem_Pre Pp Pc Dsf Pre populations hsuppC, ae_sfx_mem_Suf Pp Pc Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
    set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    set Ad : Finset S := freshDraws Pd (fun i => d.2 j i) B.npref with hAd
    have hPx : ∀ ω : Ω, prefixesAt populations B.npref ((ω, d) : Run Ω S J) = Pd := fun ω => rfl
    have hCx : ∀ ω : Ω, poolAt B.nsuff ((ω, d) : Run Ω S J) = Cd := fun ω => rfl
    have hAx : ∀ ω : Ω, certOf (prefixesAt populations B.npref ((ω, d) : Run Ω S J)) j B.npref
        ((ω, d) : Run Ω S J) = Ad := fun ω => rfl
    have hP : ∀ q ∈ Pd, q ∈ Pre :=
      table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hdP
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, rfl⟩ := mem_freshDraws hp
      exact hdC j hj i
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref ∧ 0 < Ad.card
    · obtain ⟨hdisj, hAdm, -⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B with hfamdef
      set T : Finset (Finset S) := (Cd.erase 1).powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
          hCx ω ▸ clusterAt_subset O populations B _ hv⟩)
      have hT1 : ∀ t ∈ T, (1 : S) ∉ t := fun t ht h1 =>
        (Finset.mem_erase.1 (Finset.mem_powerset.1 ht h1)).1 rfl
      have hmain := light_counts_whp hflat O Pd Cd Ad hP hV hA hdisj B.lo B.hi l lcut lmis
        T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => (Finset.mem_powerset.1 ht).trans (Finset.erase_subset _ _))
        (Finset.mem_insert_self _ _) hT1 fam hfamT
        (fun ω ω' h => clusterAt_congr O populations B d (by rw [hPx ω, hCx ω]; exact h))
        E hE hElcut hlcl hElm
        (fun p _ A₀ hA₀ hgp => hdec A₀ p (hT1 A₀ hA₀) (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ lightMiss rule O populations j B l lcut lmis f kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (Ad.card : ℝ)
            ∧ ¬ ((((Ad.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
                  ≤ 2 * l * (Ad.card : ℝ))
              ∧ ((Ad.filter (fun p => ¬ cutCorrect O B.lo B.hi (fam ω) p ω)).card : ℝ)
                  ≤ (lcut + lmis) * (Ad.card : ℝ))} := by
        rintro ω ⟨-, -, -, hlight, hbad⟩
        rw [hAx ω] at hlight hbad
        refine ⟨le_trans (le_of_eq ?_) hlight, hbad⟩
        refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
        intro p _
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
        exact Iff.rfl
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq (by rw [hcardR])))
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ lightMiss rule O populations j B l lcut lmis f kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        rw [hAx ω, hPx ω] at h1
        rw [hAx ω] at h2 h3
        exact hdraw ⟨h1, h2, h3⟩
      simp [hsec]
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (lightMiss rule O populations j B l lcut lmis f kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

lemma gateLevel_pos_le (populations : Finset J) {uni : J} (huni : uni ∈ populations) (B : State)
    {α : ℝ} (hα0 : 0 < α) (hα1 : α ≤ 1) :
    0 < gateLevel α populations B ∧ gateLevel α populations B ≤ 1 := by
  have hJ : (1 : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_pos.2 ⟨uni, huni⟩
  exact ⟨sizeLevel_pos (by positivity) _,
    sizeLevel_le (by positivity) (by rw [div_le_one (by positivity)]; nlinarith) _⟩

open scoped Classical in
/-- The round's own test at one population, its gate read at the population's first look. -/
noncomputable def retAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) (j : J) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
    ∧ (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card
    ∧ gateAt rule O.mq populations uni η₀ εcov α B j 0 x}

open scoped Classical in
/-- `retAt` at a population the round keeps. -/
noncomputable def retAtK (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) (j : J) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
    ∧ (keptAt populations j B.npref x →
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card
      ∧ gateAt rule O.mq populations uni η₀ εcov α B j 0 x)}

lemma retAt_subset_retAtK (O : Oracle μ S) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) (j : J) :
    retAt rule O populations uni η₀ indecisionLimit εcov α B j
      ⊆ retAtK rule O populations uni η₀ indecisionLimit εcov α B j :=
  fun _ h => ⟨h.1, fun _ => h.2⟩

lemma mem_retAll_of_retAtK (O : Oracle μ S) (populations : Finset J)
    (hpop : populations.Nonempty) (uni : J) (η₀ indecisionLimit εcov α : ℝ) (B : State)
    (x : Run Ω S J)
    (h : ∀ j ∈ populations, x ∈ retAtK rule O populations uni η₀ indecisionLimit εcov α B j) :
    x ∈ retAll rule O.mq populations uni η₀ indecisionLimit εcov α B := by
  obtain ⟨j₀, hj₀⟩ := hpop
  exact ⟨(h j₀ hj₀).1, fun j hj hk => ((h j hj).2 hk).1, fun j hj hk => ((h j hj).2 hk).2⟩

open scoped Classical in
/-- A sample found in time, of a population the family is clean for and whose draws are not
mostly flip-heavy, has at most `lcut` of it heavy: Markov over the family charges the flip mass
at the fraction the vote absorbs, and the redraws hide `c` of the heavy prefixes. -/
lemma heavy_le_of_clean (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    [IsProbabilityMeasure Dj] (j : J) (B : State) (x : Run Ω S J) {f Δp th c lcut : ℝ}
    {r kmin kmax : ℕ} (hf0 : 0 < f) (hΔp : 0 ≤ Δp) (hr : (r : ℝ) ≤ c * B.npref)
    (hheavy : Δp / f + th + c ≤ lcut) (hgood : certGood populations j B.npref r x)
    (hsize : kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)
    (hclean : ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp)
    (hhits : x ∉ heavyHits rule O populations Dj j B f (Δp / f) th) :
    (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
      ≤ lcut * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ) := by
  classical
  set F : Finset S := clusterBy rule O.mq populations x B with hFdef
  obtain ⟨hcardN, hcnt⟩ := certOf_of_good populations j B.npref r x hgood
  have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
  have hmass : Dj.real {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} ≤ Δp / f := by
    rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
    · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)} = (∅ : Set S) := by
        ext p
        simp only [Set.mem_empty_iff_false, iff_false, not_not]
        have hFe : F = ∅ := Finset.card_eq_zero.1 hF0
        simp [flipCount, hFe]
      rw [hz]
      simp only [measureReal_empty]
      exact div_nonneg hΔp hf0.le
    · have hsetle : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)}
          ⊆ {p | f * (F.card : ℝ) ≤ flipCount O F p} := by
        intro p hp
        have hlt := not_le.1 hp
        rw [Set.mem_setOf_eq, mul_comm]
        linarith
      exact le_trans (measureReal_mono hsetle (measure_ne_top _ _))
        (flipCount_mass_le O Dj F Δp f hf0 hFpos hclean)
  have hfew : (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
      ¬ (flipCount O F (certPrefix j i.val x) ≤ (F.card : ℝ) * f))).card : ℝ)
      < (B.npref : ℝ) * (Δp / f + th) := by
    by_contra h
    exact hhits ⟨hmass, not_lt.1 h⟩
  have hH := (hcnt (fun p => ¬ (flipCount O F p ≤ (F.card : ℝ) * f))).2
  have hle : (((certOf (prefixesAt populations B.npref x) j B.npref x).filter
      (fun p => ¬ famGood O f kmin kmax F p)).card : ℝ)
      = (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        ¬ (flipCount O F p ≤ (F.card : ℝ) * f))).card : ℝ) := by
    refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr (fun p _ => ?_))
    unfold famGood
    simp only [hsize.1, hsize.2, and_true]
  have hcard : ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
      = (B.npref : ℝ) := by exact_mod_cast hcardN
  rw [hle, hcard]
  have hb := mul_le_mul_of_nonneg_left hheavy hm0
  nlinarith

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- A filled sample outside the tables, light on all but `lcut` of it, passes the FNR test and
the gate unless a light count misses, a side is dead or the pool is short. -/
lemma retAt_of_light (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (j : J) (B : State) (x : Run Ω S J) (hmpos : 0 < B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lcut lmis f t : ℝ) (r rs kmin kmax : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hl2 : 2 * l ≤ indecisionLimit) (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 0 < B.k)
    (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + lmis) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hsize : kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)
    (hCn : (certOf (prefixesAt populations B.npref x) j B.npref x).card = B.npref)
    (hhv : (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card : ℝ)
      ≤ lcut * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ))
    (h8 : x ∉ lightMiss rule O populations j B l lcut lmis f kmin kmax)
    (h10 : x ∉ sidesDead rule O populations j B B.npref t) (h11 : poolGood B.nsuff rs x) :
    x ∈ retAt rule O populations uni η₀ indecisionLimit εcov α B j := by
  classical
  set n : ℕ := B.npref with hndef
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hmpos
  set F : Finset S := clusterBy rule O.mq populations x B with hFdef
  set ω : Ω := oracleNoise x with hω
  set T : Finset S := prefixesAt populations n x with hT
  set C : Finset S := certOf T j n x with hC
  set R : Finset S := refsBy rule O.mq populations x B with hR
  have hCpos : 0 < C.card := by rw [hCn]; exact hmpos
  have hl8 : ((C.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) p ω)).card : ℝ) ≤ 2 * l * (C.card : ℝ)
      ∧ ((C.filter (fun p => ¬ cutCorrect O B.lo B.hi F p ω)).card : ℝ)
        ≤ (lcut + lmis) * (C.card : ℝ) := by
    by_contra hcon
    exact h8 ⟨(disjoint_freshDraws _ _ _).symm, hCn, hCpos, hhv, hcon⟩
  rw [hCn] at hl8
  -- a reference
  have hRne : R.Nonempty := by
    have hfresh := (freshDraws_of_good ∅ (fun i => suffixDraw i x) B.nsuff rs h11).1
    have hpoolc : B.nsuff ≤ (poolAt B.nsuff x).card := by
      have := Finset.card_le_card (Finset.subset_insert (1 : S)
        (freshDraws ∅ (fun i => suffixDraw i x) B.nsuff))
      rw [hfresh] at this
      exact this
    have hfam : (familyBy rule O.mq populations x B).card ≤ B.k :=
      familyBy_card_le O populations B hkpos x
    have hlt : (familyBy rule O.mq populations x B).card < (poolAt B.nsuff x).card :=
      lt_of_le_of_lt hfam (lt_of_lt_of_le hpool hpoolc)
    obtain ⟨v, hv, hvF⟩ := Finset.exists_mem_notMem_of_card_lt_card hlt
    exact ⟨v, Finset.mem_sdiff.2 ⟨hv, hvF⟩⟩
  have hne1 : ∀ v ∈ R, ∀ p : S, p * v ≠ p := fun v hv p h =>
    (Finset.mem_sdiff.1 hv).2 (by
      have : v = 1 := mul_left_cancel (a := p) (by rw [h, mul_one])
      rw [this]; exact Finset.mem_insert_self _ _)
  -- the gate
  have hlive : sidesLive O B.lo B.hi F C R t t ω := by
    by_contra hcon; exact h10 hcon
  have hgateN := hgate n le_rfl (by omega)
  have hlim : min εcov (1 / 2) ≤ gateLimit uni εcov j := by
    unfold gateLimit
    split_ifs
    · exact min_le_left _ _
    · exact min_le_right _ _
  have hgate' : (lcut + lmis) * n / gateKappa η₀
        + 2 * ((gateDevT η₀ + gateDevU η₀) * t
          + gateDevD η₀ * hoeffDev n (sizeLevel (gateLevel α populations B) n))
      ≤ n * gateLimit uni εcov j - hoeffDev n (sizeLevel (gateLevel α populations B) n) := by
    have := mul_le_mul_of_nonneg_left hlim (Nat.cast_nonneg n : (0 : ℝ) ≤ _)
    simp only [sub_self, add_zero] at hgateN
    linarith
  have hga := gateLevel_pos_le populations huni B hα0 hα1
  have hcert := certifiesAt_of_live O hηle hη₀ B.lo B.hi hlohi F R C hRne
    (fun v hv p _ => hne1 v hv p) hmpos hCn
    (lim := gateLimit uni εcov j) (a := gateLevel α populations B) (t := t)
    (M := (lcut + lmis) * n) hga.1 hga.2 ht hlive hl8.2 hgate'
  refine ⟨?_, ?_, hcert⟩
  · have : B.k ≤ F.card + 1 := by omega
    exact this
  · rw [hCn]
    have := mul_le_mul_of_nonneg_right hl2 hnR.le
    linarith [hl8.1]

open scoped Classical in
/-- Where the round fails at one population: the certification sample not found in time, a
family of the wrong size or dirty on the population, too many heavy prefixes, a light count
missed, a dead side, or a short pool. -/
lemma notRetAt_subset (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (j : J) (Dj : Measure S) [IsProbabilityMeasure Dj] (B : State)
    (hmpos : 0 < B.npref) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α)
    (hα1 : α ≤ 1) (l lcut lmis f Δp th c t : ℝ) (r rs kmin kmax : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hl2 : 2 * l ≤ indecisionLimit) (hΔp : 0 ≤ Δp) (hf0 : 0 < f) (hr : (r : ℝ) ≤ c * B.npref)
    (hheavy : Δp / f + th + c ≤ lcut) (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 0 < B.k)
    (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + lmis) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N)) :
    {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ indecisionLimit εcov α B j}
      ⊆ {x | ¬ certGood populations j B.npref r x} ∪ stalled rule O populations B kmin kmax
        ∪ {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp}
        ∪ heavyHits rule O populations Dj j B f (Δp / f) th
        ∪ lightMiss rule O populations j B l lcut lmis f kmin kmax
        ∪ sidesDead rule O populations j B B.npref t ∪ {x | ¬ poolGood B.nsuff rs x} := by
  intro x hx
  by_contra hc
  simp only [Set.mem_union, not_or] at hc
  obtain ⟨⟨⟨⟨⟨⟨h1, h5⟩, h6⟩, h7⟩, h8⟩, h10⟩, h11⟩ := hc
  simp only [Set.mem_setOf_eq, not_not] at h1 h6 h11
  have h5' : kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax := by
    by_contra h; exact h5 h
  obtain ⟨hCn, -⟩ := certOf_of_good populations j B.npref r x h1
  have hhv := heavy_le_of_clean O populations Dj j B x hf0 hΔp hr hheavy h1 h5' h6 h7
  exact hx (retAt_of_light O populations uni huni j B x hmpos hηle hη₀ εcov α hα0 hα1 l lcut
    lmis f t r rs kmin kmax hkmin hl2 hlohi ht hkpos hpool hgate h5' hCn hhv h8 h10 h11)

open scoped Classical in
set_option maxHeartbeats 4000000 in
/-- Where the round can fail at one population, as `notRetAt_subset` lists it, priced. -/
theorem measureReal_failUnion_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0) (j : J)
    (hj : j ∈ populations) (Dj : Measure S) [IsProbabilityMeasure Dj] (hPc : Pc j = iidStream Dj)
    (B : State) (l lcut lmis f E Δp th t ρsf : ℝ) (r rs kmin kmax : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ lmis) (hth : 0 ≤ th)
    (ht : 0 ≤ t) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (Ecert : ℝ) (hcertBad : (streamMeasure μ Pp Pc Dsf).real
      {x : Run Ω S J | ¬ certGood populations j B.npref r x} ≤ Ecert)
    (Estall : ℝ) (hstall : (streamMeasure μ Pp Pc Dsf).real (stalled rule O populations B kmin kmax)
      ≤ Estall)
    (Edirty : ℝ) (hdirty : (streamMeasure μ Pp Pc Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (streamMeasure μ Pp Pc Dsf).real
        ({x | ¬ certGood populations j B.npref r x} ∪ stalled rule O populations B kmin kmax
          ∪ {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp}
          ∪ heavyHits rule O populations Dj j B f (Δp / f) th
          ∪ lightMiss rule O populations j B l lcut lmis f kmin kmax
          ∪ sidesDead rule O populations j B B.npref t ∪ {x | ¬ poolGood B.nsuff rs x})
      ≤ Ecert + Estall + Edirty
        + Real.exp (-((B.npref : ℝ) * relRate (Δp / f + th) (Δp / f)))
        + (Real.exp (-((B.npref : ℝ) * relRate l E))
          + Real.exp (-((B.npref : ℝ) * relRate lmis E)))
        + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
        + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) := by
  classical
  set E1 : Set (Run Ω S J) := {x | ¬ certGood populations j B.npref r x} with hE1
  set E5 : Set (Run Ω S J) := stalled rule O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O Dj v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits rule O populations Dj j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := lightMiss rule O populations j B l lcut lmis f kmin kmax with hE8
  set E10 : Set (Run Ω S J) := sidesDead rule O populations j B B.npref t with hE10
  set E11 : Set (Run Ω S J) := {x | ¬ poolGood B.nsuff rs x} with hE11
  set μr := streamMeasure μ Pp Pc Dsf with hμr
  have u : ∀ A B' : Set (Run Ω S J), μr.real (A ∪ B') ≤ μr.real A + μr.real B' :=
    fun A B' => measureReal_union_le A B'
  have b7 := measureReal_heavyHits_le (rule := rule) Pp Pc Dsf O populations j Dj hPc B f (Δp / f)
    th hth
  have b8 := measureReal_lightMiss_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf j hj B l lcut lmis f E kmin kmax hE hElcut hlcl hElm hdec hcut
  have b10 := measureReal_sidesDead_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf j hj B B.npref ht
  have b11 := measureReal_poolBad_le (μ := μ) Pp Pc Dsf B.nsuff rs ρsf hρsf hρsf0
  have := u E1 E5
  have := u (E1 ∪ E5) E6
  have := u (E1 ∪ E5 ∪ E6) E7
  have := u (E1 ∪ E5 ∪ E6 ∪ E7) E8
  have := u (E1 ∪ E5 ∪ E6 ∪ E7 ∪ E8) E10
  have := u (E1 ∪ E5 ∪ E6 ∪ E7 ∪ E8 ∪ E10) E11
  linarith

/-! ### What the gate's deviations ask of the prefix count -/

/-- `A·log(kn) ≤ n` once `n ≥ A(log(Ak) + 2 log log(Ak))`, given `log(Ak) ≥ 2`. -/
lemma mul_log_mul_le {A k n : ℝ} (hA : 0 < A) (hk : 0 < k) (hy : 2 ≤ Real.log (A * k))
    (hn : A * (Real.log (A * k) + 2 * Real.log (Real.log (A * k))) ≤ n) :
    A * Real.log (k * n) ≤ n := by
  set y : ℝ := Real.log (A * k) with hydef
  have hy0 : 0 < y := by linarith
  have hly0 : 0 ≤ Real.log y := Real.log_nonneg (by linarith)
  have hly : Real.log y ≤ y - 1 := Real.log_le_sub_one_of_pos hy0
  have hs : 0 < y + 2 * Real.log y := by linarith
  set n₀ : ℝ := A * (y + 2 * Real.log y) with hn₀
  have hAn₀ : A ≤ n₀ := by rw [hn₀]; nlinarith
  have hn₀pos : 0 < n₀ := by linarith
  have hnpos : 0 < n := by linarith
  have e1 : k * n = A * k * (y + 2 * Real.log y) * (n / n₀) := by
    rw [hn₀]; field_simp
  have e : Real.log (k * n) = y + Real.log (y + 2 * Real.log y) + Real.log (n / n₀) := by
    rw [e1, Real.log_mul (by positivity) (by positivity),
      Real.log_mul (by positivity) hs.ne']
  have h2 : Real.log (y + 2 * Real.log y) ≤ 2 * Real.log y := by
    have hsq : Real.log (y ^ 2) = 2 * Real.log y := by rw [Real.log_pow]; norm_num
    have hq := mul_nonneg (by linarith : 0 ≤ y - 1) (by linarith : 0 ≤ y - 2)
    exact (Real.log_le_log hs (by nlinarith)).trans_eq hsq
  have h3 : Real.log (n / n₀) ≤ n / n₀ - 1 := Real.log_le_sub_one_of_pos (by positivity)
  have h4 : A * (n / n₀) ≤ n - n₀ + A := by
    rw [mul_div_assoc', div_le_iff₀ hn₀pos]
    nlinarith [mul_nonneg (by linarith : 0 ≤ n - n₀) (by linarith : 0 ≤ n₀ - A)]
  have h5 : A * Real.log (k * n) ≤ A * (y + 2 * Real.log y + (n / n₀ - 1)) := by
    rw [e]
    exact mul_le_mul_of_nonneg_left (by linarith) hA.le
  have h6 : A * (y + 2 * Real.log y + (n / n₀ - 1)) = n₀ + A * (n / n₀) - A := by
    rw [hn₀]; ring
  linarith

lemma hoeffDev_le_sqrt {n : ℕ} {a Λ : ℝ} (hlog : Real.log (1 / a) ≤ Λ) :
    hoeffDev n a ≤ Real.sqrt (n * Λ / 2) := by
  unfold hoeffDev
  refine Real.sqrt_le_sqrt ?_
  have := mul_le_mul_of_nonneg_left hlog (Nat.cast_nonneg n : (0 : ℝ) ≤ n)
  linarith

/-- The levels the gate reads at a look of `N ≤ 2m` draws at a state of `m` prefixes: none below
`α/(7|J|(2m + 2)⁴)`. -/
lemma log_sizeLevel_le {α c : ℝ} (hα : 0 < α) (hc : 0 < c) {m N : ℕ} (hNm : N ≤ 2 * m) :
    Real.log (1 / sizeLevel (sizeLevel (α / (7 * c)) m) N)
      ≤ Real.log (7 * c / α) + 4 * Real.log (2 * m + 2) := by
  have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg _
  have hN : (N : ℝ) ≤ 2 * m := by exact_mod_cast hNm
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  have e : 1 / sizeLevel (sizeLevel (α / (7 * c)) m) N
      = 7 * c / α * (((m : ℝ) + 1) * ((m : ℝ) + 2) * (((N : ℝ) + 1) * ((N : ℝ) + 2))) := by
    unfold sizeLevel
    field_simp
  rw [e, Real.log_mul (by positivity) (by positivity)]
  have hle : ((m : ℝ) + 1) * ((m : ℝ) + 2) * (((N : ℝ) + 1) * ((N : ℝ) + 2)) ≤ (2 * m + 2) ^ 4 := by
    have h1 : (m : ℝ) + 1 ≤ 2 * m + 2 := by linarith
    have h2 : (m : ℝ) + 2 ≤ 2 * m + 2 := by linarith
    have h3 : (N : ℝ) + 1 ≤ 2 * m + 2 := by linarith
    have h4 : (N : ℝ) + 2 ≤ 2 * m + 2 := by linarith
    calc ((m : ℝ) + 1) * ((m : ℝ) + 2) * (((N : ℝ) + 1) * ((N : ℝ) + 2))
        ≤ (2 * m + 2) * (2 * m + 2) * ((2 * m + 2) * (2 * m + 2)) := by gcongr
      _ = (2 * m + 2) ^ 4 := by ring
  have := Real.log_le_log (by positivity) hle
  rw [Real.log_pow] at this
  push_cast at this
  linarith

lemma gateRoom_facts (η₀ indecisionLimit εcov : ℝ) (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀)
    (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    0 < gateRoom η₀ indecisionLimit εcov ∧ gateRoom η₀ indecisionLimit εcov ≤ 1 / 2
      ∧ (1 / 5 - 1 / 1024) * min εcov (1 / 2) ≤ gateRoom η₀ indecisionLimit εcov := by
  have hκ := gateKappa_pos hsig
  have hcut := cutBudget_pos η₀ hsig hε hind
  obtain ⟨h8, h10, -⟩ := cutBudget_le η₀ indecisionLimit εcov
  have hg := cutBudget_le_gate η₀ indecisionLimit εcov
  set cut := cutBudget η₀ indecisionLimit εcov
  set mn : ℝ := min εcov (1 / 2) with hmn
  have hmn0 : 0 < mn := lt_min hε (by norm_num)
  have hmn1 : mn ≤ 1 / 2 := min_le_right _ _
  have hcm : cut ≤ mn / 8 := by
    have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    rcases min_choice εcov (1 / 2) with h | h <;> rw [hmn, h] <;> linarith
  have hch : 13 * cut / (8 * gateKappa η₀) ≤ 4 / 5 * mn := by
    rw [div_le_iff₀ (by positivity)]
    nlinarith
  have hpos : 0 ≤ 13 * cut / (8 * gateKappa η₀) := by positivity
  unfold gateRoom
  refine ⟨by linarith, by linarith, by linarith⟩

set_option maxHeartbeats 2000000 in
/-- The top rung's gate clears its charge and its deviations at every count of draws its
redraws allow. -/
lemma solved_gate (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) (hα : 0 < α) (hα1 : α ≤ 1)
    (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    ∀ N : ℕ, (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref ≤ N →
      N ≤ (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref
        + rejCount η₀ indecisionLimit εcov
          (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref →
      (cutBudget η₀ indecisionLimit εcov + 5 / 8 * cutBudget η₀ indecisionLimit εcov)
          * (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref
          / gateKappa η₀
        + 2 * ((gateDevT η₀ + gateDevU η₀)
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit
              hsz).npref
              (δ / (32 * populations.card
                * (((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).nsuff
                  : ℝ) + 3)))
          + gateDevD η₀
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit
              hsz).npref
              (sizeLevel (gateLevel α populations
                (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz))
                (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref))
        + ((N : ℝ) - (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref)
      ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations
          (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)) N) := by
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).nsuff = M :=
    rfl
  have hga : ∀ N : ℕ, sizeLevel (gateLevel α populations
        (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)) N
      = sizeLevel (sizeLevel (α / (7 * (populations.card : ℝ))) m) N := fun N => rfl
  rw [hBm, hBM]
  simp only [hga]
  -- the two counts the gate's terms ask for
  have hc1 : ⌈25 / 2 * (gateDevT η₀ + gateDevU η₀) ^ 2
        * Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 3) / δ)
        / gateRoom η₀ indecisionLimit εcov ^ 2⌉₊ ≤ m := by
    rw [hmdef, prefCount, ← hMdef]; omega
  have hc2 : ⌈gateLogCount (50 / 9 * (gateDevD η₀ + 1) ^ 2 / gateRoom η₀ indecisionLimit εcov ^ 2)
        (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α)))⌉₊ ≤ m := by
    rw [hmdef, prefCount]; omega
  have hmR1 : (1 : ℝ) ≤ m := by
    have := prefCount_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      (hsz := hsz) η₀
      populations εcov δ α pAP
    exact_mod_cast this
  have hm0 : (0 : ℝ) < m := by linarith
  set c : ℝ := (populations.card : ℝ) with hcdef
  set cut : ℝ := cutBudget η₀ indecisionLimit εcov with hcutdef
  set κ : ℝ := gateKappa η₀ with hκdef
  set mn : ℝ := min εcov (1 / 2) with hmndef
  set gR : ℝ := gateRoom η₀ indecisionLimit εcov with hgRdef
  set W : ℝ := gateDevT η₀ + gateDevU η₀ with hWdef
  set D : ℝ := gateDevD η₀ with hDdef
  have hcut : 0 < cut := cutBudget_pos η₀ hsig hε hind
  have hκ : 0 < κ := gateKappa_pos hsig
  have hmn0 : 0 ≤ mn := le_min hε.le (by norm_num)
  obtain ⟨hgR0, hgR1, -⟩ := gateRoom_facts η₀ indecisionLimit εcov hsig hη0 hε hind
  rw [← hgRdef] at hgR0 hgR1
  have hgReq : gR * m = mn * m - 13 * cut / (8 * κ) * m - cut / 128 * m := by
    rw [hgRdef, gateRoom]; ring
  obtain ⟨hT0, hD0, hU0⟩ := gateDev_pos hη0 hsig
  have hW0 : 0 < W := by rw [hWdef]; linarith
  -- the references' and the seed's deviation, against two fifths of the room
  set L₁ : ℝ := Real.log (32 * c * ((M : ℝ) + 3) / δ) with hL₁
  have hP1 : 25 / 2 * W ^ 2 * L₁ ≤ m * gR ^ 2 := by
    have := le_trans (Nat.le_ceil _) (by exact_mod_cast hc1 :
      ((⌈25 / 2 * W ^ 2 * L₁ / gR ^ 2⌉₊ : ℕ) : ℝ) ≤ m)
    rwa [div_le_iff₀ (by positivity)] at this
  set t : ℝ := hoeffDev m (δ / (32 * c * ((M : ℝ) + 3))) with htdef
  have ht : t ≤ Real.sqrt (m * L₁ / 2) :=
    hoeffDev_le_sqrt (by rw [one_div_div])
  have hsqt : 5 * W * Real.sqrt (m * L₁ / 2) ≤ gR * m := by
    have h25 : (5 * W) ^ 2 * (m * L₁ / 2) ≤ (gR * m) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hP1 hm0.le
      linarith only [this]
    calc 5 * W * Real.sqrt (m * L₁ / 2) = Real.sqrt ((5 * W) ^ 2 * (m * L₁ / 2)) := by
          rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
      _ ≤ Real.sqrt ((gR * m) ^ 2) := Real.sqrt_le_sqrt h25
      _ = gR * m := Real.sqrt_sq (by positivity)
  have hWt : 2 * (W * t) ≤ 2 / 5 * (gR * m) := by
    have := mul_le_mul_of_nonneg_left ht hW0.le
    linarith
  -- the levels' deviations, against the rest
  set K : ℝ := 50 / 9 * (D + 1) ^ 2 / gR ^ 2 with hKdef
  set q : ℝ := 7 * c / α with hqdef
  set kk : ℝ := 4 * Real.sqrt (Real.sqrt q) with hkkdef
  have hq1 : 1 ≤ q := by rw [hqdef, le_div_iff₀ hα]; linarith
  have hkk4 : 4 ≤ kk := by
    have : 1 ≤ Real.sqrt (Real.sqrt q) := Real.one_le_sqrt.2 (Real.one_le_sqrt.2 hq1)
    rw [hkkdef]; linarith
  have hK4 : 4 ≤ K := by
    rw [hKdef, le_div_iff₀ (by positivity)]
    have h1 : gR ^ 2 ≤ 1 / 4 := by nlinarith only [hgR0, hgR1]
    have h2 : 1 ≤ (D + 1) ^ 2 := by nlinarith only [hD0]
    linarith only [h1, h2]
  have hy : 2 ≤ Real.log (4 * K * kk) := by
    have h64 : Real.log 64 ≤ Real.log (4 * K * kk) :=
      Real.log_le_log (by norm_num) (by
        have := mul_le_mul hK4 hkk4 (by norm_num) (by linarith)
        linarith only [this])
    have e64 : Real.log 64 = 6 * Real.log 2 := by
      rw [show (64 : ℝ) = 2 ^ 6 by norm_num, Real.log_pow]; norm_num
    linarith [Real.log_two_gt_d9]
  have hP2 : 4 * K * Real.log (kk * m) ≤ m := by
    refine mul_log_mul_le (by positivity) (by positivity) hy ?_
    exact le_trans (Nat.le_ceil _) (by exact_mod_cast hc2)
  set Λ : ℝ := Real.log q + 4 * Real.log (2 * m + 2) with hΛ
  have hΛkk : Λ ≤ 4 * Real.log (kk * m) := by
    have hl4 : Real.log (Real.sqrt (Real.sqrt q)) = Real.log q / 4 := by
      rw [Real.log_sqrt (Real.sqrt_nonneg _), Real.log_sqrt (by linarith)]; ring
    have hsq0 : 0 < Real.sqrt (Real.sqrt q) := by positivity
    have e : Real.log (kk * m) = Real.log 4 + Real.log q / 4 + Real.log m := by
      rw [hkkdef, Real.log_mul (by positivity) (by positivity),
        Real.log_mul (by norm_num) hsq0.ne', hl4]
    have h4m : Real.log (2 * (m : ℝ) + 2) ≤ Real.log 4 + Real.log m := by
      rw [← Real.log_mul (by norm_num) (by positivity)]
      exact Real.log_le_log (by linarith) (by linarith)
    rw [hΛ, e]; linarith
  have hKΛ : 50 * (D + 1) ^ 2 * Λ ≤ 9 * gR ^ 2 * m := by
    have h1 : K * Λ ≤ m := by
      have := mul_le_mul_of_nonneg_left hΛkk (by linarith : 0 ≤ K)
      linarith only [this, hP2]
    rw [hKdef] at h1
    have e : 50 / 9 * (D + 1) ^ 2 / gR ^ 2 * Λ * (9 * gR ^ 2) = 50 * (D + 1) ^ 2 * Λ := by
      field_simp
    have := mul_le_mul_of_nonneg_right h1 (by positivity : (0 : ℝ) ≤ 9 * gR ^ 2)
    linarith
  have hΛ0 : 0 ≤ Λ := by
    have : 0 ≤ Real.log q := Real.log_nonneg hq1
    have : 0 ≤ Real.log (2 * (m : ℝ) + 2) := Real.log_nonneg (by linarith)
    rw [hΛ]; linarith
  have hsl : ∀ N : ℕ, N ≤ 2 * m →
      Real.log (1 / sizeLevel (sizeLevel (α / (7 * c)) m) N) ≤ Λ := by
    intro N hN
    exact log_sizeLevel_le (m := m) hα (by positivity) hN
  set s : ℝ := Real.sqrt (m * Λ / 2) with hsdef
  have hdm := hoeffDev_le_sqrt (n := m) (hsl m (by omega))
  have hss : 10 * (D + 1) * s ≤ 3 * (gR * m) := by
    have h : (10 * (D + 1)) ^ 2 * (m * Λ / 2) ≤ (3 * (gR * m)) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hKΛ hm0.le
      linarith only [this]
    calc 10 * (D + 1) * s = Real.sqrt ((10 * (D + 1)) ^ 2 * (m * Λ / 2)) := by
          rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
      _ ≤ Real.sqrt ((3 * (gR * m)) ^ 2) := Real.sqrt_le_sqrt h
      _ = 3 * (gR * m) := Real.sqrt_sq (by positivity)
  intro N hN1 hN2
  have hr : ((rejCount η₀ indecisionLimit εcov m : ℕ) : ℝ) ≤ cut / 128 * m := by
    have h1 : ((rejCount η₀ indecisionLimit εcov m : ℕ) : ℝ)
        ≤ rejFrac η₀ indecisionLimit εcov * m := Nat.floor_le (by
      have := rejFrac_pos η₀ hsig hε hind; positivity)
    have h2 := (rejFrac_le η₀ hsig hε hind).1
    have := mul_le_mul_of_nonneg_right h2 hm0.le
    linarith only [h1, this]
  have hNR1 : (m : ℝ) ≤ N := by exact_mod_cast hN1
  have hNR2 : (N : ℝ) ≤ m + cut / 128 * m := by
    have : (N : ℝ) ≤ m + (rejCount η₀ indecisionLimit εcov m : ℕ) := by exact_mod_cast hN2
    linarith
  have hcut1 : cut ≤ 1 := by
    have := (cutBudget_le η₀ indecisionLimit εcov).2.1
    have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    linarith
  have hN2mR : (N : ℝ) ≤ 2 * m := by
    have := mul_le_mul_of_nonneg_right hcut1 hm0.le
    linarith only [hNR2, this, hm0]
  have hN2m : N ≤ 2 * m := by exact_mod_cast hN2mR
  have hdN : hoeffDev N (sizeLevel (sizeLevel (α / (7 * c)) m) N) ≤ 2 * s := by
    refine le_trans (hoeffDev_le_sqrt (hsl N hN2m)) ?_
    calc Real.sqrt (N * Λ / 2) ≤ Real.sqrt (2 ^ 2 * (m * Λ / 2)) :=
          Real.sqrt_le_sqrt (by
            have := mul_le_mul_of_nonneg_right hN2mR hΛ0
            have := mul_nonneg hm0.le hΛ0
            linarith only [this, ‹(N : ℝ) * Λ ≤ 2 * m * Λ›])
      _ = 2 * s := by rw [Real.sqrt_mul (by norm_num), Real.sqrt_sq (by norm_num)]
  have hDd : D * hoeffDev m (sizeLevel (sizeLevel (α / (7 * c)) m) m) ≤ D * s :=
    mul_le_mul_of_nonneg_left hdm hD0.le
  have hch : (cut + 5 / 8 * cut) * m / κ = 13 * cut / (8 * κ) * m := by
    field_simp; ring
  have hNmn : (m : ℝ) * mn ≤ N * mn := mul_le_mul_of_nonneg_right hNR1 hmn0
  have hsD : 0 ≤ s := Real.sqrt_nonneg _
  rw [hch]
  linarith

/-- What one round at the uniform pool can cost: the samplers redrawing too much, the family's
size and cleanliness, the sample's heavy fraction, the two light counts, the gate's liveness,
and the pool. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut th E γscr γdirty gdirty tap ρ ρsf t : ℝ) (r rs : ℕ) (B : State) : ℝ :=
  rejTail (2 * B.npref + r) (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * ρ) r
    + (((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
      + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
        + (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
          + 2 * ((B.nsuff : ℝ) + 1) * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
    + ((rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
          + 2 * ((B.nsuff : ℝ) + 1) * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
        + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
          + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))))
    + Real.exp (-((B.npref : ℝ) * relRate (lcut + th) lcut))
    + (Real.exp (-((B.npref : ℝ) * relRate l E))
      + Real.exp (-((B.npref : ℝ) * relRate (5 / 8 * lcut) E)))
    + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
    + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))

/-- What one round at a named population can cost: the family's size, a clean reference, the
screen, the hash's split of the window and of each member's flips on it, the uniform pool's
table and its share of the window, the two light counts, the gate's liveness, and the pool. -/
noncomputable def namedFail (l lcut E γscr γdirty tap ρ ρsf t L θ : ℝ) (r rs G : ℕ)
    (B : State) : ℝ :=
  (((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
      + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
        + (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
          + 2 * ((B.nsuff : ℝ) + 1) * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
    + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
      + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)))
    + 2 * ((B.nsuff : ℝ) + 1) * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
    + (((B.nsuff : ℝ) + 1) * Real.exp (-L) + 2 * Real.exp (-((G : ℝ) ^ 2 / B.npref)))
    + (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
      + rejTail (B.npref + r) (((2 * B.npref : ℕ) : ℝ) * ρ) (⌈θ * B.npref⌉₊ - 1))
    + (Real.exp (-((B.npref : ℝ) * relRate l E))
      + Real.exp (-((B.npref : ℝ) * relRate (5 / 8 * lcut) E)))
    + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
    + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))

lemma roundFail_nonneg (populations : Finset J) (l lcut th E γscr γdirty gdirty tap ρ ρsf t : ℝ)
    (r rs : ℕ) (B : State) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf) :
    0 ≤ roundFail populations l lcut th E γscr γdirty gdirty tap ρ ρsf t r rs B := by
  unfold roundFail
  repeat' (first | exact rejTail_nonneg _ (by positivity) _ | positivity | apply add_nonneg)

lemma namedFail_nonneg (l lcut E γscr γdirty tap ρ ρsf t L θ : ℝ) (r rs G : ℕ) (B : State)
    (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf) :
    0 ≤ namedFail l lcut E γscr γdirty tap ρ ρsf t L θ r rs G B := by
  unfold namedFail
  repeat' (first | exact rejTail_nonneg _ (by positivity) _ | positivity | apply add_nonneg)

/-- A state whose round can pass.  Every clause is an inequality among the state's budgets,
the oracle's rates, the suffix distribution's findability and the error budget; no probability
enters. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (hsz : ℕ) (B : State) : Prop :=
  ∃ (th tap q E γscr γdirty gdirty Δ lcut f c t L θ : ℝ) (r rs G : ℕ),
    0 < B.npref ∧ 2 ≤ B.k ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 < f ∧ 0 ≤ pAP ∧ 0 < lcut ∧ lcut ≤ indecisionLimit / 2
    -- the samplers redraw at most a `c` fraction of the prefix draws
    ∧ 0 ≤ c ∧ (r : ℝ) ≤ c * B.npref
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification samples hold few heavy prefixes: the uniform pool's by its own
    -- draws, a named population's by the hash's split of a window of its draws
    ∧ (((populations.card : ℝ) * Δ + c + gdirty) / f + th + c ≤ lcut)
    ∧ 0 ≤ L ∧ 0 < θ ∧ 2 ≤ B.npref ∧ hsz + G ≤ B.npref
    ∧ (((populations.card : ℝ) * Δ * B.npref
          + 2 * Real.sqrt (L * ((populations.card : ℝ) * Δ * B.npref)) + 2 * L) / f
        + hsz + 2 * G + θ * B.npref ≤ lcut * B.npref)
    -- the screen's rate sits above the clean one and below the dirty one
    ∧ 0 < B.scd
    ∧ ((B.scd : ℝ) * (2 * γscr) ≤ (B.sc : ℝ))
    ∧ ((B.sc : ℝ)
        ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γdirty))
    -- the pool holds a family, beyond what its sampler redraws, and a reference beside it
    ∧ ((B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - tap))
    ∧ B.k < B.nsuff
    -- the thresholds decide, and decide right, on a family of the round's size an `f`
    -- fraction of which flips, whose reads' mean that leaves `q` toward the prefix's side; the
    -- FNR reads the family with its seed, cut one count higher
    ∧ 0 ≤ q ∧ q ≤ 1 ∧ q ≤ 1 - O.η - O.gap * f ∧ O.η + O.gap * f ≤ 1 - q
    ∧ ((B.hi : ℝ) + 1 + 2 ≤ (B.k : ℝ) * q)
    ∧ ((B.k : ℝ) * (1 - q) + 2 ≤ (B.lo : ℝ) + 1)
    ∧ ((B.lo : ℝ) + 2 ≤ ((B.k - 1 : ℕ) : ℝ) * q)
    ∧ (((B.k - 1 : ℕ) : ℝ) * (1 - q) + 2 ≤ (B.hi : ℝ) + 1)
    ∧ B.hi + 1 ≤ B.k - 1
    ∧ B.lo ≤ B.hi
    -- the gate clears the charge of a family wrong on `lcut` heavy and `5/8·lcut` light
    -- prefixes, and its deviations
    ∧ 0 ≤ t
    ∧ (∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + 5 / 8 * lcut) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    -- the vote misfires, on either side and with or without its seed, at `E`, under half the
    -- cut budget
    ∧ 0 ≤ E ∧ E ≤ lcut / 2
    ∧ 1 - binomSfGe B.k q (B.hi + 1 + 1) ≤ E ∧ 1 - binomSfGe B.k q (B.k - B.lo) ≤ E
    ∧ 1 - binomSfGe (B.k - 1) q (B.lo + 1) ≤ E ∧ 1 - binomSfGe (B.k - 1) q (B.k - 1 - B.hi) ≤ E
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * (roundFail populations (indecisionLimit / 2) lcut th E γscr γdirty gdirty tap ρ ρsf t
            r rs B
          + namedFail (indecisionLimit / 2) lcut E γscr γdirty tap ρ ρsf t L θ r rs G B)
      ≤ δ / 2)

/-- At the top count the uniform pool's share of a window is a tail in `cutBudget/1024` of its
count, which clears `log(256|J|/δ)` with room. -/
lemma solved_hits_log (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.log (256 * (populations.card : ℝ) / δ)
      ≤ cutBudget η₀ indecisionLimit εcov / 1024
        * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * Real.log 2 := by
  obtain ⟨-, -, -, thh, -⟩ := solved_tails (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀
    populations hsig hε hind hδ hpAP (by linarith)
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hNdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set P : ℝ := (populations.card : ℝ) with hP
  have hcb0 : 0 < cb := cutBudget_pos η₀ hsig hε hind
  have hl128 : Real.log (128 * P / δ) ≤ N * (cb / 2112) := by
    have h := Real.log_le_log (Real.exp_pos _) thh
    rw [Real.log_exp, Real.log_div (by positivity) (by positivity)] at h
    rw [Real.log_div (by positivity) (by positivity)]
    linarith
  have hl1 : Real.log 128 ≤ Real.log (128 * P / δ) := by
    refine Real.log_le_log (by norm_num) ?_
    rw [le_div_iff₀ hδ]; nlinarith
  have h128 : (4 : ℝ) ≤ Real.log 128 := by
    rw [show (128 : ℝ) = 2 ^ 7 by norm_num, Real.log_pow]
    have := Real.log_two_gt_d9
    push_cast; linarith
  have h256 : Real.log (256 * P / δ) = Real.log 2 + Real.log (128 * P / δ) := by
    rw [← Real.log_mul (by norm_num) (by positivity)]
    congr 1; ring
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have hl2' : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  rw [h256]
  have e : cb / 1024 * N = 2112 / 1024 * (N * (cb / 2112)) := by ring
  rw [e]
  nlinarith

set_option maxHeartbeats 2000000 in
/-- The round's failure, over every population, fits the error budget. -/
lemma solved_roundFail (η₀ : ℝ) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (hρsfsmall : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (populations.card : ℝ)
        * (roundFail populations (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (cutBudget η₀ indecisionLimit εcov / 32)
            (misfireTail η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32)
            (9 * pAP / 20) ρ ρsf
            (hoeffDev (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
              (δ / (32 * populations.card
                * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))))
            (rejCount η₀ indecisionLimit εcov
              (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz))
            (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
            (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
          + namedFail (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (misfireTail η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (9 * pAP / 20) ρ ρsf
            (hoeffDev (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
              (δ / (32 * populations.card
                * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))))
            (Real.log (128 * (populations.card : ℝ)
              * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
            (cutBudget η₀ indecisionLimit εcov / 1024)
            (rejCount η₀ indecisionLimit εcov
              (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz))
            (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
            ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
              * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
            (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz))
      ≤ δ / 2 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  obtain ⟨ttap, tscr, tdirty', tth, -⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz).nsuff = M :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- the vote's misfire rate sits under half the cut budget
  have hEc : misfireTail η₀ κ ≤ cutBudget η₀ indecisionLimit εcov / 2 :=
    (famCount_misfire η₀ populations hη0 hsig hε hind).1
  -- the light counts' rates, against the heavy hits'
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcbdef
  have hE0 : 0 ≤ misfireTail η₀ κ := misfireTail_nonneg η₀ κ hsig hη0
  have hr1 : cb / 2112 ≤ relRate cb (cb / 2) := by
    unfold relRate
    rw [div_le_div_iff₀ (by norm_num) (by linarith)]
    nlinarith
  have hrl : cb / 2112 ≤ relRate (indecisionLimit / 2) (misfireTail η₀ κ) :=
    le_trans hr1 (relRate_anti hE0 hEc (by linarith) hcutlim)
  have hrm : cb / 2112 ≤ relRate (5 / 8 * cb) (misfireTail η₀ κ) := by
    have h2 : cb / 2112 ≤ relRate (5 / 8 * cb) (cb / 2) := by
      unfold relRate
      rw [div_le_div_iff₀ (by norm_num) (by linarith)]
      nlinarith
    exact le_trans h2 (relRate_anti hE0 hEc (by linarith) le_rfl)
  have hexp : ∀ r : ℝ, cb / 2112 ≤ r → Real.exp (-((m : ℝ) * r)) ≤ ε₀ := fun r hr =>
    le_trans (Real.exp_le_exp.2 (neg_le_neg (mul_le_mul_of_nonneg_left hr hmR))) tth
  have tl : Real.exp (-((m : ℝ) * relRate (indecisionLimit / 2) (misfireTail η₀ κ))) ≤ ε₀ :=
    hexp _ hrl
  have tlm : Real.exp (-((m : ℝ) * relRate (5 / 8 * cb) (misfireTail η₀ κ))) ≤ ε₀ :=
    hexp _ hrm
  have tth' : Real.exp (-((m : ℝ) * relRate (cb + cb / 32) cb)) ≤ ε₀ := by
    refine hexp _ (le_of_eq ?_)
    unfold relRate
    field_simp
    ring
  -- the gate's liveness, at `4ε₀/(M + 3)` a side
  have hb0 : 0 < δ / (32 * (populations.card : ℝ) * ((M : ℝ) + 3)) := by positivity
  have hb1 : δ / (32 * (populations.card : ℝ) * ((M : ℝ) + 3)) ≤ 1 := by
    have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
      have : 1 ≤ populations.card := by exact_mod_cast hcard
      exact_mod_cast this
    rw [div_le_one (by positivity)]
    have := mul_le_mul hc1 (show (1 : ℝ) ≤ (M : ℝ) + 3 by linarith) zero_le_one (by linarith)
    linarith only [this, hδ1]
  have tlive : ((M : ℝ) + 3) * Real.exp (-2 * hoeffDev m
      (δ / (32 * (populations.card : ℝ) * ((M : ℝ) + 3))) ^ 2 / m) = 4 * ε₀ := by
    have h := exp_hoeffDev m (by exact_mod_cast hmpos) hb0 hb1
    rw [show -2 * hoeffDev m (δ / (32 * (populations.card : ℝ) * ((M : ℝ) + 3))) ^ 2 / m
        = -2 * (m : ℝ) * (hoeffDev m (δ / (32 * (populations.card : ℝ) * ((M : ℝ) + 3))) / m)
          ^ 2 by field_simp, h, hε₀def]
    field_simp
    norm_num
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  -- the redraws
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hind
  obtain ⟨hcle, hceq⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hsig hε hind
  set r : ℕ := rejCount η₀ indecisionLimit εcov m with hrdef
  set rs : ℕ := poolRej pAP M with hrsdef
  have hpool := poolDraws_le_of_cap η₀ populations hpAP hpAP1 hδ hρsf0 hρsfsmall
  rw [← hMdef, ← hrsdef] at hpool
  rw [roundFail, namedFail, hBm, hBM]
  have hJ1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast this
  -- the certification sample's redraws, the tables', and the uniform pool's share of a window
  set tab : ℝ := rejTail (m + r) (((m + r : ℕ) : ℝ) * ρ) r with htab
  set cert : ℝ := rejTail (2 * m + r) (((populations.card * m + (2 * m + r) : ℕ) : ℝ) * ρ) r
    with hcert
  set hits : ℝ := rejTail (m + r) (((2 * m : ℕ) : ℝ) * ρ)
    (⌈cutBudget η₀ indecisionLimit εcov / 1024 * m⌉₊ - 1) with hhits
  have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hhalf_of : ∀ k : ℕ, Real.log (256 * (populations.card : ℝ) / δ) ≤ k * Real.log 2 →
      ((1 : ℝ) / 2) ^ k ≤ δ / (256 * (populations.card : ℝ)) := by
    intro k hk
    have hpow : 256 * (populations.card : ℝ) / δ ≤ (2 : ℝ) ^ k := by
      have h : Real.log (256 * (populations.card : ℝ) / δ) ≤ Real.log ((2 : ℝ) ^ k) := by
        rw [Real.log_pow]; exact hk
      exact (Real.log_le_log_iff (by positivity) (by positivity)).1 h
    rw [div_pow, one_pow, div_le_div_iff₀ (by positivity) (by positivity)]
    rw [div_le_iff₀ hδ] at hpow
    linarith
  have hlog := solved_rej_log (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀ populations
    hη0 hsig hε hind hδ hδ1 hpAP hJ1
  rw [← hmdef] at hlog
  have hr2 : c * m < (r : ℝ) + 1 := Nat.lt_floor_add_one _
  have hrlog : Real.log (256 * (populations.card : ℝ) / δ) ≤ ((r + 1 : ℕ) : ℝ) * Real.log 2 := by
    push_cast
    nlinarith [mul_le_mul_of_nonneg_right hr2.le hl2.le]
  have hhalf := hhalf_of (r + 1) hrlog
  obtain ⟨ht3, hc3⟩ := rung_tails_le (crossLimit := crossLimit) η₀ populations (δ := δ) (α := α)
    (pAP := pAP) hη0 hsig hε hind hρ0 hρsmall m le_rfl
  rw [← hmdef, div_self (ne_of_gt hmpos), one_mul] at ht3 hc3
  have htab_le : tab ≤ δ / (256 * (populations.card : ℝ)) :=
    le_trans (rejTail_mono (by positivity) le_rfl (by nlinarith) r) (le_trans ht3 hhalf)
  have hcert_le : cert ≤ δ / (256 * (populations.card : ℝ)) :=
    le_trans (rejTail_mono (by positivity) le_rfl (by nlinarith) r) (le_trans hc3 hhalf)
  have hhits_le : hits ≤ δ / (256 * (populations.card : ℝ)) := by
    set θm : ℝ := cutBudget η₀ indecisionLimit εcov / 1024 * m with hθm
    have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
    have hθ0 : 0 < θm := by positivity
    have hk1 : 1 ≤ ⌈θm⌉₊ := Nat.one_le_iff_ne_zero.2 (Nat.ceil_pos.2 hθ0).ne'
    have hkR : (((⌈θm⌉₊ - 1 : ℕ) : ℝ) + 1) = (⌈θm⌉₊ : ℝ) := by
      rw [Nat.cast_sub hk1]; push_cast; ring
    have hθle : θm ≤ (⌈θm⌉₊ : ℝ) := Nat.le_ceil _
    have hρc : ρ * (64 * ((populations.card : ℝ) + 3) * m) ≤ c := by
      have h := le_trans hρsmall (le_trans (min_le_left _ _) (min_le_left _ _))
      rw [← hmdef, ← hcdef, le_div_iff₀ (by positivity)] at h
      exact h
    have hr1 : (r : ℝ) ≤ c * m := Nat.floor_le (by positivity)
    have hc4 : c ≤ 1 / 4 := by
      have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
      linarith [hcle, (cutBudget_le η₀ indecisionLimit εcov).2.1]
    have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg _
    have h1 : ρ * m ≤ c / 256 := by
      have hk : (256 : ℝ) ≤ 64 * ((populations.card : ℝ) + 3) := by linarith
      have : ρ * (256 * m) ≤ ρ * (64 * ((populations.card : ℝ) + 3) * m) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hk hm0) hρ0
      rw [le_div_iff₀ (by norm_num)]
      linarith only [this, hρc]
    have h2 : (m : ℝ) + r ≤ 5 / 4 * m := by
      have : c * m ≤ 1 / 4 * m := mul_le_mul_of_nonneg_right hc4 hm0
      linarith only [this, hr1]
    have h3 : (8 : ℝ) * (((m + r : ℕ) : ℝ) * (((2 * m : ℕ) : ℝ) * ρ))
        ≤ 20 * m * (ρ * m) := by
      push_cast
      have h2ρ : (0 : ℝ) ≤ 2 * m * ρ := by positivity
      have := mul_le_mul_of_nonneg_right h2 h2ρ
      nlinarith only [this]
    have h4 : 20 * (m : ℝ) * (ρ * m) ≤ 20 * m * (c / 256) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h5 : 20 * (m : ℝ) * (c / 256) ≤ θm := by
      rw [hθm]
      have : 20 * (m : ℝ) * (c / 256) ≤ 20 * m * (cutBudget η₀ indecisionLimit εcov / 128 / 256) :=
        mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hcle (by norm_num)) (by positivity)
      have e : 20 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 128 / 256)
          = 20 / 32 * (cutBudget η₀ indecisionLimit εcov / 1024 * m) := by ring
      have hθ0' : 0 ≤ cutBudget η₀ indecisionLimit εcov / 1024 * m := by positivity
      linarith only [this, e, hθ0']
    have hpow := rejTail_le_pow (n := m + r) (a := ((2 * m : ℕ) : ℝ) * ρ) (x := 1)
      (r := ⌈θm⌉₊ - 1) (by positivity) zero_le_one (by
        rw [hkR, mul_one]
        linarith only [h3, h4, h5, hθle])
    rw [show ⌈θm⌉₊ - 1 + 1 = ⌈θm⌉₊ by omega] at hpow
    have hhl := solved_hits_log (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀ populations
      hη0 hsig hε hind hδ hδ1 hpAP hJ1
    rw [← hmdef, ← hθm] at hhl
    exact le_trans hpow (hhalf_of ⌈θm⌉₊ (le_trans hhl (mul_le_mul_of_nonneg_right hθle hl2.le)))
  have hρterms : (populations.card : ℝ) * (cert + 2 * tab + (2 * tab + hits)) ≤ δ / 16 := by
    have hsum : cert + 2 * tab + (2 * tab + hits) ≤ 6 * (δ / (256 * (populations.card : ℝ))) := by linarith
    calc (populations.card : ℝ) * (cert + 2 * tab + (2 * tab + hits)) ≤ (populations.card : ℝ) * (6 * (δ / (256 * (populations.card : ℝ)))) :=
          mul_le_mul_of_nonneg_left hsum hcard.le
      _ = 6 * δ / 256 := by field_simp
      _ ≤ δ / 16 := by linarith
  have hP : (populations.card : ℝ) * (3 * (((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))))
      ≤ δ / 32 := by
    have h2 := mul_le_mul_of_nonneg_left hpool
      (by positivity : (0 : ℝ) ≤ 3 * (populations.card : ℝ))
    have e : 3 * (populations.card : ℝ) * (δ / (128 * ((populations.card : ℝ) + 1)))
        ≤ δ / 32 := by
      rw [show 3 * (populations.card : ℝ) * (δ / (128 * ((populations.card : ℝ) + 1)))
          = 3 * δ / 128 * ((populations.card : ℝ) / ((populations.card : ℝ) + 1)) by
            field_simp]
      have : (populations.card : ℝ) / ((populations.card : ℝ) + 1) ≤ 1 :=
        (div_le_one (by positivity)).2 (by linarith)
      have := mul_le_mul_of_nonneg_left this (by positivity : (0 : ℝ) ≤ 3 * δ / 128)
      linarith only [this, hδ]
    linarith
  -- the hash's two tails
  set L : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ) with hLdef
  set l1 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) with hl1def
  set G : ℕ := ⌈Real.sqrt ((m : ℝ) * l1)⌉₊ with hGdef
  have tbal : ((M : ℝ) + 1) * Real.exp (-L) = ε₀ := by
    rw [hLdef, Real.exp_neg, Real.exp_log (by positivity), hε₀def]
    field_simp
  have tcard : Real.exp (-((G : ℝ) ^ 2 / m)) ≤ ε₀ := by
    have hl1 : 0 ≤ l1 := Real.log_nonneg (by
      rw [le_div_iff₀ hδ]; nlinarith)
    have hG : Real.sqrt ((m : ℝ) * l1) ≤ (G : ℝ) := Nat.le_ceil _
    have hG2 : (m : ℝ) * l1 ≤ (G : ℝ) ^ 2 := by
      have := pow_le_pow_left₀ (Real.sqrt_nonneg _) hG 2
      rwa [Real.sq_sqrt (by positivity)] at this
    have hle : l1 ≤ (G : ℝ) ^ 2 / m := by rw [le_div_iff₀ hmpos]; linarith
    calc Real.exp (-((G : ℝ) ^ 2 / m)) ≤ Real.exp (-l1) := Real.exp_le_exp.2 (by linarith)
      _ = ε₀ := by
          rw [hl1def, Real.exp_neg, Real.exp_log (by positivity), hε₀def]
          field_simp
  have tscr' : 2 * ((M : ℝ) + 1)
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤
        ε₀ := tscr
  set pool : ℝ := ((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) with hpooldef
  refine le_trans (mul_le_mul_of_nonneg_left (show _ ≤ 40 * ε₀
      + (cert + 2 * tab + (2 * tab + hits)) + 2 * (3 * pool) from ?_) hcard.le) ?_
  · linarith [ttap, tscr', tdirty', tth', tl, tlm, tlive, tbal, tcard]
  · rw [mul_add, mul_add]
    have h1 : (populations.card : ℝ) * (40 * ε₀) = 40 * δ / 128 := by
      rw [show (populations.card : ℝ) * (40 * ε₀) = 40 * ((populations.card : ℝ) * ε₀) by ring,
        hcε]
      ring
    have h3 : (populations.card : ℝ) * (2 * (3 * pool))
        = 2 * ((populations.card : ℝ) * (3 * pool)) := by ring
    rw [h1, h3]
    linarith [hρterms, hP]

set_option maxHeartbeats 1000000 in
/-- At the top count a named population's heavy prefixes fit the cut budget: its members' table
flips, the hash's deviation on them and on the window's size, the harvested strings, and the
uniform pool's share of the window. -/
lemma solved_heavyNamed (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    2 ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz
    ∧ hsz + ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
        * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
      ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz
    ∧ (((populations.card : ℝ)
            * (flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀)
            * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
          + 2 * Real.sqrt (Real.log (128 * (populations.card : ℝ)
              * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
            * ((populations.card : ℝ)
              * (flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀
                / flipFrac η₀)
              * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)))
          + 2 * Real.log (128 * (populations.card : ℝ)
              * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
          / flipFracOf O η₀
        + hsz + 2 * ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov δ α pAP
            crossLimit hsz : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
        + cutBudget η₀ indecisionLimit εcov / 1024
          * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)
      ≤ cutBudget η₀ indecisionLimit εcov
          * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz : ℝ)) := by
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set ff : ℝ := flipFrac η₀ with hffdef
  set f : ℝ := flipFracOf O η₀ with hfdef
  set L : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ) with hLdef
  set l1 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) with hl1def
  set G : ℕ := ⌈Real.sqrt ((m : ℝ) * l1)⌉₊ with hGdef
  have hcb0 : 0 < cb := cutBudget_pos η₀ hη₀ hε hind
  have hff0 : 0 < ff := flipFrac_pos η₀ hη₀
  have hffle : ff ≤ 7 / 20 := by
    rw [hffdef, flipFrac, sig, div_le_iff₀ (by linarith)]
    nlinarith
  have hfge : ff ≤ f := flipFrac_le_flipFracOf O hηle hη₀
  have hf0 : 0 < f := lt_of_lt_of_le hff0 hfge
  have hcb16 : cb ≤ 1 / 16 := by
    have h := (cutBudget_le η₀ indecisionLimit εcov).2.1
    rw [sig] at h
    linarith
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hl1 : 1 ≤ l1 := by
    rw [hl1def, Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    rw [le_div_iff₀ hδ]
    nlinarith
  have hl1L : l1 ≤ L := Real.log_le_log (by positivity)
    (by rw [div_le_div_iff_of_pos_right hδ]; nlinarith)
  have hL0 : 0 ≤ L := by linarith
  -- the counts the schedule solves for
  have c1 : 2900 * L / (cb * ff) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈2900 * L / (cb * ff)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef, ← hcb, ← hffdef, ← hLdef]; omega
    exact_mod_cast hle
  have c2 : L / (2 * (3 * cb * ff / 32) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈L / (2 * (3 * cb * ff / 32) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef, ← hcb, ← hffdef, ← hLdef]; omega
    exact_mod_cast hle
  have c3 : 128 * (hsz : ℝ) / cb ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈128 * (hsz : ℝ) / cb⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hcb]; omega
    exact_mod_cast hle
  rw [div_le_iff₀ (by positivity)] at c1 c2 c3
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- each piece
  have hX : (populations.card : ℝ) * (flipBudget η₀ populations indecisionLimit εcov δ * f / ff)
      = 55 / 64 * cb * f := by
    rw [flipBudget, ← hcb, ← hffdef]
    field_simp
  rw [hX]
  have hcm : 2900 * L ≤ f * cb * (m : ℝ) := by
    have := mul_le_mul_of_nonneg_left hfge (mul_nonneg hcb0.le hmR)
    nlinarith
  have hT2 : Real.sqrt (L * (55 / 64 * cb * f * m)) ≤ f * cb * m / 58 := by
    rw [show f * cb * (m : ℝ) / 58 = Real.sqrt ((f * cb * m / 58) ^ 2) by
      rw [Real.sqrt_sq (by positivity)]]
    refine Real.sqrt_le_sqrt ?_
    have h55 : L * (55 / 64 * cb * f * m) = (55 / 64) * (L * (cb * f * m)) := by ring
    rw [h55]
    have hcfm : 0 ≤ cb * f * (m : ℝ) := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hcm hcfm]
  have hT3 : 2 * L ≤ f * cb * m / 1450 := by linarith
  have hG1 : (G : ℝ) ≤ Real.sqrt ((m : ℝ) * l1) + 1 := (Nat.ceil_lt_add_one (by positivity)).le
  have hT5 : Real.sqrt ((m : ℝ) * l1) ≤ 3 * cb * m / 64 := by
    rw [show 3 * cb * (m : ℝ) / 64 = Real.sqrt ((3 * cb * m / 64) ^ 2) by
      rw [Real.sqrt_sq (by positivity)]]
    refine Real.sqrt_le_sqrt ?_
    have hff2 : ff ^ 2 ≤ 1 / 8 := by nlinarith
    have hml : (m : ℝ) * l1 ≤ (m : ℝ) * L := mul_le_mul_of_nonneg_left hl1L hmR
    have hLm : L ≤ (m : ℝ) * (2 * (3 * cb * ff / 32) ^ 2) := c2
    have : (m : ℝ) * L ≤ (m : ℝ) * ((m : ℝ) * (2 * (3 * cb * ff / 32) ^ 2)) :=
      mul_le_mul_of_nonneg_left hLm hmR
    have hb : (m : ℝ) * ((m : ℝ) * (2 * (3 * cb * ff / 32) ^ 2)) ≤ (3 * cb * m / 64) ^ 2 := by
      have e : (m : ℝ) * ((m : ℝ) * (2 * (3 * cb * ff / 32) ^ 2))
          = (3 * cb * m / 64) ^ 2 * (8 * ff ^ 2) := by ring
      rw [e]
      have : 8 * ff ^ 2 ≤ 1 := by linarith
      have h0 : 0 ≤ (3 * cb * m / 64) ^ 2 := sq_nonneg _
      nlinarith
    linarith
  have hcm1 : 8192 ≤ cb * (m : ℝ) := by
    have : 2900 * L ≤ cb * ff * m := by linarith
    have h1 : cb * ff * (m : ℝ) ≤ cb * m * (7 / 20) := by nlinarith
    nlinarith
  have hsz' : (hsz : ℝ) ≤ cb * m / 128 := by linarith
  refine ⟨?_, ?_, ?_⟩
  · have : (2 : ℝ) ≤ m := by nlinarith
    exact_mod_cast this
  · have : (hsz : ℝ) + G ≤ m := by nlinarith
    exact_mod_cast this
  · rw [add_div, add_div]
    have e1 : 55 / 64 * cb * f * (m : ℝ) / f = 55 / 64 * cb * m := by field_simp
    have e2 : 2 * Real.sqrt (L * (55 / 64 * cb * f * m)) / f ≤ cb * m / 29 := by
      rw [div_le_iff₀ hf0]; nlinarith
    have e3 : 2 * L / f ≤ cb * m / 1450 := by
      rw [div_le_iff₀ hf0]; nlinarith
    rw [e1]
    nlinarith

set_option maxHeartbeats 4000000 in
/-- The computed ladder holds a state whose round can pass: the top rung, every clause
discharged from the closed forms that define it. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J)
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ0 : 0 ≤ ρ) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz)
    (hρsfcap : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    ∃ B : State, B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz ρ
      ∧ PassableAt O η₀ populations Dsf indecisionLimit εcov α δ ρ ρsf pAP hsz B := by
  classical
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hη0 : 0 ≤ O.η := eta_nonneg O
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hη₀ hεcov hindLim
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hη₀ hεcov hindLim hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hindLim hcard
  have hv := voteSlack_pos η₀ hη₀
  have hη0' : 0 ≤ η₀ := le_trans hη0 hηle
  have hff := flipFrac_pos η₀ hη₀
  have hfo := flipFracOf_pos O hηle hη₀
  have hfle := flipFrac_le_flipFracOf O hηle hη₀
  obtain ⟨hmisκ, h5⟩ := famCount_misfire (crossLimit := crossLimit) η₀ populations (δ := δ) hη0'
    hη₀ hεcov hindLim
  have hev := famCount_even (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  obtain ⟨hlo, hhi⟩ := loAt_hiAt η₀ hev h4 hη0' hη₀
  have hb := bandAt_le η₀ κ hη₀
  have hb0 : (0 : ℝ) ≤ bandAt η₀ κ := Nat.cast_nonneg _
  have hκR : (4 : ℝ) ≤ (κ : ℝ) := by exact_mod_cast h4
  have hsum : loAt η₀ κ + hiAt η₀ κ + 1 = κ := by
    have : ((loAt η₀ κ + hiAt η₀ κ + 1 : ℕ) : ℝ) = κ := by push_cast; rw [hlo, hhi]; ring
    exact_mod_cast this
  have hlohi : loAt η₀ κ ≤ hiAt η₀ κ := by unfold loAt hiAt; omega
  have hv5 : voteSlack η₀ ≤ 1 / 5 := by rw [voteSlack, sig]; linarith
  have hq0 : (0 : ℝ) ≤ 1 / 2 + voteSlack η₀ := by linarith
  have hq1 : 1 / 2 + voteSlack η₀ ≤ 1 := by linarith
  have hmisle : 1 - binomSfGe κ (1 / 2 + voteSlack η₀) (loAt η₀ κ + 1) ≤ misfireTail η₀ κ := by
    rw [misfireTail]
    refine sub_le_sub_left (le_trans (binomSfGe_succ_le hq0 hq1 (hiAt η₀ κ + 1)) ?_) 1
    exact binomSfGe_antitone' hq0 hq1 (by omega)
  have hg := gap_mul_flipFracOf O η₀ (lt_of_le_of_lt hηle hη₀)
  have hκpos : 0 < κ := by omega
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit hsz with hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  have hBkR : (B.k : ℝ) = κ + 1 := by rw [hBk]; push_cast; ring
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := rfl
  have hBlo : B.lo = loAt η₀ κ := rfl
  have hBhi : B.hi = hiAt η₀ κ := rfl
  have hBscd : B.scd = ⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈63 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  have hMceil : 2 * ((κ : ℝ) + 1) / pAP ≤ ((B.nsuff : ℕ) : ℝ) := by
    rw [hBM, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊]
  have hcutle := cutBudget_le η₀ indecisionLimit εcov
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast this
  have hgate := solved_gate (crossLimit := crossLimit) (hsz := hsz) (pAP := pAP) (δ := δ) η₀
    populations hη₀
    hη0' hεcov
    hindLim hαpos (by linarith) hcard1
  rw [← hBdef] at hgate
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  have hheavyN := solved_heavyNamed (crossLimit := crossLimit) (hsz := hsz) (α := α) (pAP := pAP)
    O η₀ populations hηle hη₀ hη0' hεcov hindLim hδ hδ1 hcard1
  rw [← hmdef] at hheavyN
  clear_value B κ m
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hη₀ hεcov hindLim
  obtain ⟨hcle, hceq⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hη₀ hεcov hindLim
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_⟩⟩,
    cutBudget η₀ indecisionLimit εcov / 32, 9 * pAP / 20, 1 / 2 + voteSlack η₀, misfireTail η₀ κ,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32,
    flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀,
    cutBudget η₀ indecisionLimit εcov, flipFracOf O η₀, rejFrac η₀ indecisionLimit εcov,
    hoeffDev m (δ / (32 * populations.card
      * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))),
    Real.log (128 * (populations.card : ℝ)
      * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ),
    cutBudget η₀ indecisionLimit εcov / 1024,
    rejCount η₀ indecisionLimit εcov m,
    poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    ⌈Real.sqrt ((m : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_⟩
  -- in the schedule
  · rw [hBdef]
    exact solvedState_mem_schedule η₀ populations εcov δ α pAP
  -- Capped
  · rw [hBlo, hBhi]
    exact hlohi
  · rw [hBm]; exact hmpos
  · rw [hBM]
    exact poolCount_found η₀ populations hη₀ hpAPPositive
  · have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by
      rw [mul_comm (8 : ℝ) (m : ℝ), mul_comm δ (m : ℝ),
        mul_div_mul_left _ _ (ne_of_gt hmR)]
    rw [hBm, ← hmdef, hval, hBdef]
    exact le_add_of_le_of_nonneg (solved_share η₀ populations hη₀ hη0' hεcov hindLim hδ hδ1
      hpAPPositive hcard hρ0 hρcap) (by split_ifs <;> positivity)
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · exact hindLim
  · exact hind1
  · linarith
  · linarith
  · linarith [hγ.le]
  · linarith [hγ.le]
  · have := flipFrac_pos η₀ hη₀
    positivity
  · positivity
  · exact hfo
  · linarith
  · exact hcut
  · exact (cutBudget_le η₀ indecisionLimit εcov).2.2
  · exact hc0.le
  · rw [hBm]
    exact Nat.floor_le (by positivity)
  · exact hpAPBound
  · exact hρsf
  -- the family's flips fit the cut budget
  · have hne : ((populations.card : ℝ)) ≠ 0 := ne_of_gt hcard
    have hsplit : ((populations.card : ℝ)
          * (flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀)
          + rejFrac η₀ indecisionLimit εcov
        + 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) / flipFracOf O η₀
        = (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ / flipFrac η₀
          + rejFrac η₀ indecisionLimit εcov / flipFracOf O η₀
          + 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32 / flipFracOf O η₀ := by
      rw [add_div, add_div]
      congr 2
      field_simp
    have hid : (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
        / flipFrac η₀ = cutBudget η₀ indecisionLimit εcov * (55 / 64) := by
      rw [flipBudget]
      field_simp
    have hcf : rejFrac η₀ indecisionLimit εcov / flipFracOf O η₀
        ≤ cutBudget η₀ indecisionLimit εcov / 128 := by
      calc rejFrac η₀ indecisionLimit εcov / flipFracOf O η₀
          ≤ rejFrac η₀ indecisionLimit εcov / flipFrac η₀ :=
            div_le_div_of_nonneg_left hc0.le hff hfle
        _ = cutBudget η₀ indecisionLimit εcov / 128 := by
            rw [hceq]
            field_simp
    have h2 : 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32 / flipFracOf O η₀
        ≤ 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32 / flipFrac η₀ :=
      div_le_div_of_nonneg_left (by positivity) hff hfle
    have h3 : 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32 / flipFrac η₀
        = 3 * cutBudget η₀ indecisionLimit εcov / 32 := by
      field_simp
    rw [hsplit, hid]
    linarith
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
    exact hheavyN.2.2
  -- the screen's rate
  · rw [hBscd]
    omega
  · rw [hBsc, hBscd]
    have h2 : (2 : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2)
        = screenMargin η₀ populations indecisionLimit εcov δ := by ring
    rw [h2]
    exact Nat.le_ceil _
  · rw [hBdef]
    refine le_trans (rung_facts populations (δ := δ) (pAP := pAP) (prefCount η₀ populations
      indecisionLimit εcov δ α pAP crossLimit hsz) hη₀ hεcov hindLim hcard).2.2
      (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
    have h := mul_le_mul_of_nonneg_left (flipFracOf_sep O hηle hη₀)
      (div_nonneg hΔ.le hff.le :
        (0 : ℝ) ≤ flipBudget η₀ populations indecisionLimit εcov δ / flipFrac η₀)
    have e1 : flipBudget η₀ populations indecisionLimit εcov δ / flipFrac η₀
        * (flipFrac η₀ * ((1 - η₀) * (1 - 2 * η₀)))
        = flipBudget η₀ populations indecisionLimit εcov δ * ((1 - η₀) * (1 - 2 * η₀)) := by
      field_simp
    have e2 : flipBudget η₀ populations indecisionLimit εcov δ / flipFrac η₀
        * (flipFracOf O η₀ * (O.gap * (1 - 2 * O.η)))
        = flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀
          * (O.gap * (1 - 2 * O.η)) := by ring
    rw [e1, e2] at h
    linarith
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
  · rw [hg, sig]; linarith
  · rw [hg, sig]; linarith
  · rw [hBhi, hBkR, hhi]; nlinarith
  · rw [hBlo, hBkR, hlo]; nlinarith
  · rw [hBlo, hBkappa, hlo]; nlinarith
  · rw [hBhi, hBkappa, hhi]; nlinarith
  · rw [hBhi, hBkappa]; omega
  · rw [hBlo, hBhi]; exact hlohi
  -- the gate
  · exact Real.sqrt_nonneg _
  · rw [hBm]
    rw [hBm, hBM] at hgate
    exact hgate
  -- the misfire
  · exact misfireTail_nonneg η₀ κ hη₀ hη0'
  · exact hmisκ
  · rw [hBk, hBhi]; exact le_rfl
  · rw [hBk, hBlo, show κ + 1 - loAt η₀ κ = hiAt η₀ κ + 1 + 1 by omega]; exact le_rfl
  · rw [hBkappa, hBlo]; exact hmisle
  · rw [hBkappa, hBhi, show κ - hiAt η₀ κ = loAt η₀ κ + 1 by omega]; exact hmisle
  · have h := solved_roundFail (crossLimit := crossLimit) (α := α) (ρ := ρ) (ρsf := ρsf)
      η₀ populations hη₀ hη0' hεcov hδ hδ1 hpAPPositive hpAP1 hindLim
      hcard hρ0 hρsf0 hρcap hρsfcap
    rw [← hBdef, ← hmdef, ← hκdef] at h
    exact h

#print axioms validity_of_returned

end Loop



end OrthoDFA
