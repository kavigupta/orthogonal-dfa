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

section Loop

variable {D : J → Measure S} {Dsf : Measure S}
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
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (rs rt : ℕ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real (stalled rule O populations B (B.k - 1) (B.k - 1))
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + (((B.npref + rt : ℕ) : ℝ) ^ 2 * ρ / (2 * ((rt : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations D Dsf hsupp hsuppSf B j₀ hj₀ γ pAP t ρsf ρ
      rs rt hγ hpAP0 ht hpAPBound hscd hsig hsc hcount hρsf hρsf0 hρj hρ0)

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
          - 2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2))
    ∧ Real.exp (-2
        * (((solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi).k
            - 1 : ℕ) : ℝ)
        * (voteSlack η₀ / 2) ^ 2) ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  classical
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hind hcard
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set B : State := solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit mi
    with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBscd : B.scd = ⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  clear_value B κ
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hBk]; omega
  · rw [hBscd]; omega
  · rw [hBsc, hBscd]
    have hone : (15 : ℝ) ≤ 2 * ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ +
      1 : ℕ) : ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ := by
      have hcl : (15 : ℝ) / (2 * screenMargin η₀ populations indecisionLimit εcov δ)
          ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ) := by
        refine le_trans (Nat.le_ceil _) ?_
        push_cast
        linarith
      rw [div_le_iff₀ (by positivity)] at hcl
      linarith
    have hceilub : (⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
      ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ : ℝ)
        ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ)
          * screenMargin η₀ populations indecisionLimit εcov δ + 1 :=
      le_of_lt (Nat.ceil_lt_add_one (by positivity))
    have hFeq : flipBudget η₀ populations indecisionLimit εcov δ * ((1 - η₀) * (1 - 2 * η₀))
        = 32 / 15 * screenMargin η₀ populations indecisionLimit εcov δ := by
      rw [screenMargin, hsval]; ring
    rw [hFeq]
    have hS0 : (0 : ℝ) ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 :
      ℕ) : ℝ) :=
      Nat.cast_nonneg _
    nlinarith [hceilub, hone, hγ.le, hS0]
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail η₀ populations hη₀ hεcov hind

/-- The band is wide enough for `cross_of_mem_schedule` to read the crossing bound off. -/
lemma band_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    B.lo ≤ B.hi ∧ 2 ≤ B.k
      ∧ ((B.k : ℝ) - 1) * Real.log (1 / crossLimit) ≤ 2 * ((B.hi : ℝ) - B.lo) ^ 2 := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  have hv := voteSlack_pos η₀ hη₀
  have hκpos := famCount_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hb := bandHalf_wide (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη0 hη₀
  have hL := famCount_cross (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ hη₀
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set c : ℕ := ⌈(κ : ℝ) / 2⌉₊ with hcdef
  set B : State := solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBlo : B.lo = c - b - 1 := rfl
  have hBhi : B.hi = c + b := rfl
  clear_value B b κ c
  rw [hBk, hBlo, hBhi]
  refine ⟨by omega, by omega, ?_⟩
  rw [Nat.sub_sub, Nat.cast_sub hb1]
  push_cast
  have hκ0 : (0 : ℝ) ≤ κ := Nat.cast_nonneg _
  have hy : 0 ≤ (κ : ℝ) * voteSlack η₀ / 2 := by positivity
  have hsq := pow_le_pow_left₀ hy hb 2
  nlinarith [mul_le_mul_of_nonneg_left hL hκ0]

/-- A vote over a family no larger than a rung's own, whose mean lies outside the rung's band,
reaches the band's far side at most `crossLimit` of the time. -/
theorem cross_of_mem_schedule (O : Oracle μ S) {populations : Finset J} {εcov δ α pAP : ℝ}
    {B : State} (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2) (hcross : 0 < crossLimit)
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (F : Finset S) (hF : F.card + 1 ≤ B.k) (p : S) :
    (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
    ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit) := by
  obtain ⟨hlohi, hk, hband⟩ := band_of_mem_schedule hη0 hη₀ hB
  have hnpos : (0 : ℝ) < ((B.k - 1 : ℕ) : ℝ) := by
    have : 0 < B.k - 1 := by omega
    exact_mod_cast this
  have hnR : ((B.k - 1 : ℕ) : ℝ) = (B.k : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega)]; simp
  have hexp : Real.exp (-2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)) ≤ crossLimit := by
    rw [show -2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)
        = -(2 * ((B.hi : ℝ) - B.lo) ^ 2 / ((B.k - 1 : ℕ) : ℝ)) by ring]
    refine exp_neg_le_of_log_le hcross ?_
    rw [le_div_iff₀ hnpos, hnR]
    linarith
  obtain ⟨h1, h2⟩ := vote_cross_le O F p hlohi (n := B.k - 1) (by omega) (by omega)
  exact ⟨fun h => (h1 h).trans hexp, fun h => (h2 h).trans hexp⟩

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

/-- The family at a state is invalid: on some tame population its cut is wrong on more than the
gate holds that population to and `s`, or it leaves more than `2·indecisionLimit` undecided. -/
def FailAt (rule : Clusterer S) (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (D : J → Measure S) (indecisionLimit εcov s : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ tame, 1 - gateLimit uni εcov j - s
        ≤ (D j).real {p | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
          (oracleNoise x)}
      ∧ (D j).real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
          (oracleNoise x)} ≤ 2 * indecisionLimit}

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
    (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (indecisionLimit εcov α : ℝ)
    (hil : 0 < indecisionLimit) (hα0 : 0 < α) (hα1 : α ≤ 1) (B : State) (hk : 0 < B.k)
    (r : ℕ) (c : ℝ) (hc0 : 0 ≤ c) (hr : (r : ℝ) ≤ c * B.npref) (hc : c ≤ indecisionLimit / 2)
    (ρ : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ) (hρ0 : 0 ≤ ρ)
    (G : Set (Run Ω S J))
    (hG : ∀ x ∉ G, ∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p) :
    (runMeasure μ D Dsf).real ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni D indecisionLimit εcov c B) \ G)
      ≤ 7 * populations.card * gateLevel α populations B
        + ((tame.card : ℝ) * (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
            + (((2 * B.npref + r : ℕ) : ℝ) * ((populations.card : ℝ) * B.npref * ρ)
              + ((2 * B.npref + r : ℕ) : ℝ) ^ 2 * ρ / 2) / ((r : ℝ) + 1))
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
      ∪ ⋃ N : ℕ, rawShortAt rule O populations (D j) j B (gateLimit uni εcov j + c) a (N + 1)
    with hgateJ
  set indJ : J → Set (Run Ω S J) := fun j =>
    undecidedShort rule O populations (D j) j B (2 * indecisionLimit) (indecisionLimit / 2)
    with hindJ
  have hsub : ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni D indecisionLimit εcov c B) \ G)
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
    by_cases hmass : 1 - gateLimit uni εcov j - c ≤ (D j).real {p | cutCorrect O B.lo B.hi
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
      have hgt : gateLimit uni εcov j + c < (D j).real {p | ¬ cutCorrect O B.lo B.hi
          (clusterBy rule O.mq populations x B) p (oracleNoise x)} := by
        rw [show {p : S | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}
            = {p : S | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}ᶜ from rfl,
          measureReal_compl (measurableSet_of_countable _), probReal_univ]
        linarith [not_le.1 hmass]
      have hin := falseCertify_subset (rule := rule) O hηle populations (D j) j B
        (gateLimit uni εcov j) c a r hc0 hr ⟨hgate j (htame hj) hkept, hG x hxG, hgt, hc2⟩
      exact Or.inl (Set.mem_biUnion hj (Or.inl hin))
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have h2 : (runMeasure μ D Dsf).real ((⋃ j ∈ tame, (gateJ j ∪ indJ j))
        ∪ {x : Run Ω S J | ¬ drawsGoodOn populations tame B.npref r x})
      ≤ tame.card * (7 * a + Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2))
        + (tame.card : ℝ) * (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
            + (((2 * B.npref + r : ℕ) : ℝ) * ((populations.card : ℝ) * B.npref * ρ)
              + ((2 * B.npref + r : ℕ) : ℝ) ^ 2 * ρ / 2) / ((r : ℝ) + 1)) := by
    refine le_trans (measureReal_union_le _ _) (add_le_add ?_
      (measureReal_drawsBadOn_le D Dsf populations tame B.npref r ρ hρ hρ0))
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun j hj => le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_))
    · exact measureReal_popGate_le hflat O populations D Dsf hsupp hsuppSf j (htame hj) B ha0 ha1
    · exact measureReal_undecidedShort_le D Dsf O populations j B _ _ (by linarith)
        (by linarith)
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
      ↑(schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit) := by
  intro B hB B' hB' h
  rw [Finset.mem_coe, schedule] at hB hB'
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  obtain ⟨i', -, rfl⟩ := Finset.mem_image.1 hB'
  simp only [solvedStateAt] at h ⊢
  rw [h]

/-- What the gate spends over the ladder's rungs is at most `α`. -/
lemma gateLevel_sum_le {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} (hα : 0 ≤ α)
    (hpop : populations.Nonempty) (s : Finset State)
    (hs : s ⊆ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
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
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit
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

Except with probability `δ/2 + α`, no rung the loop may stop at returns an invalid family.  The
pool's findability and its redraws are one event for the whole ladder; each rung carries its
share of `δ/2`, and the gate spends `α` over all of them. -/
theorem validity_of_returned {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations tame : Finset J) (uni : J) (huni : uni ∈ tame) (htame : tame ⊆ populations)
    (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α : ℝ) (hindLim : 0 < indecisionLimit)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (pAP : ℝ) (hpAP0 : 0 < pAP) (hpAP1 : pAP ≤ 1)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcoll : ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP
          crossLimit) : ℕ) : ℝ) ^ 2 * ρsf
        / (2 * ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ
          pAP crossLimit) : ℝ) + 1))
      ≤ δ / 16) :
    (runMeasure μ D Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov
            δ α pAP crossLimit ρ},
          retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
            ∩ FailAt rule O populations tame uni D indecisionLimit εcov
                (rejFrac η₀ indecisionLimit εcov) B.val) ≤ δ / 2 + α := by
  classical
  set iLc : ℝ := indecisionLimit with hiLc
  have hiLc0 : 0 < iLc := hindLim
  have hiLc1 : iLc ≤ indecisionLimit := le_rfl
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  have hJ : (1 : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_pos.2 hpop
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ uni huni 1)
  set M : ℕ := poolCount η₀ populations iLc εcov δ pAP crossLimit with hMdef
  set rs : ℕ := poolRej pAP M with hrsdef
  set N : ℕ := prefCount η₀ populations iLc εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations iLc εcov δ pAP crossLimit with hVdef
  set G : Set (Run Ω S J) := apShort O M pAP (9 * pAP / 20) ∪ {x | ¬ poolGood M rs x} with hGdef
  set st := stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ with hst
  have hGle : (runMeasure μ D Dsf).real G ≤ δ / 8 := by
    refine le_trans (measureReal_union_le _ _) ?_
    have h1 := le_trans (measureReal_apShort_le D Dsf O M pAP (9 * pAP / 20) hpAP0.le
      (by positivity) hpAPBound)
      (solved_findability (indecisionLimit := iLc) (crossLimit := crossLimit) η₀ populations
        (εcov := εcov) hδ hpAP0 hJ)
    have h2 := le_trans (measureReal_poolBad_le (μ := μ) D Dsf M rs ρsf hρsf hρsf0) hcoll
    linarith
  have hper : ∀ B ∈ st, (runMeasure μ D Dsf).real
      ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni D indecisionLimit εcov
            (rejFrac η₀ indecisionLimit εcov) B) \ G)
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
    have h := per_state_le (rule := rule) hflat O populations tame uni huni htame D Dsf hsupp
      hsuppSf hηle hη₀ indecisionLimit εcov α hindLim hα0 hα1 B hk r c hc0.le hr
      (by linarith [hcutle.2.2]) ρ hρ hρ0 G hG'
    refine le_trans h (add_le_add le_rfl (le_trans ?_ hcap.share))
    -- the redraws, against `(3|J| + 4)·m·ρ/c`, and the FNR tail at the schedule's indecision rate
    have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
    have hc4 : c ≤ 1 / 4 := by
      have h2 := hcutle.2.1
      rw [sig] at h2
      linarith
    have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
    have htc : (tame.card : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_le_card htame
    have hcost := redrawCost_le hc0 hc4 B.npref hJ0 hρ0
    have hrr : (⌊c * (B.npref : ℝ)⌋₊ : ℕ) = r := rfl
    rw [hrr] at hcost
    have hcost0 : 0 ≤ (3 * (populations.card : ℝ) + 4) * B.npref * ρ / c := by positivity
    have hcostT := mul_le_mul htc hcost (by positivity) hJ0
    unfold stateFail
    have e : (3 * (populations.card : ℝ) + 4) * (B.npref : ℝ) * ρ / c
        = (3 * (populations.card : ℝ) + 4) * B.npref * ρ / c := rfl
    nlinarith [hcostT]
  have hsplit : (⋃ B : {B : State // B ∈ st},
        retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
          ∩ FailAt rule O populations tame uni D indecisionLimit εcov
              (rejFrac η₀ indecisionLimit εcov) B.val)
      ⊆ (⋃ B ∈ st, (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
          ∩ FailAt rule O populations tame uni D indecisionLimit εcov
              (rejFrac η₀ indecisionLimit εcov) B) \ G) ∪ G := by
    intro x hx
    obtain ⟨B, hB⟩ := Set.mem_iUnion.1 hx
    by_cases hg : x ∈ G
    · exact Or.inr hg
    · exact Or.inl (Set.mem_biUnion B.2 ⟨hB, hg⟩)
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hNpos : (0 : ℝ) < N := by exact_mod_cast prefCount_pos _ _ _ _ _ _
  have hsumN := stoppable_npref_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit) η₀
    populations εcov δ α pAP ρ
  have hinv := schedule_inv_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit) η₀
    populations εcov δ α pAP V st (by rw [hst, stoppable]; exact Finset.filter_subset _ _)
  have hgate := gateLevel_sum_le (indecisionLimit := iLc) (crossLimit := crossLimit)
    (η₀ := η₀) (εcov := εcov) (δ := δ) (pAP := pAP) hα0.le hpop st
    (by rw [hst, stoppable]; exact Finset.filter_subset _ _)
  have hite : ∀ B : State, (if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0)
      = δ / 16 * (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) := by
    intro B; split_ifs <;> ring
  have hmain : (runMeasure μ D Dsf).real (⋃ B ∈ st, (retBy rule O.mq populations uni η₀
        indecisionLimit εcov α B ∩ FailAt rule O populations tame uni D indecisionLimit εcov
            (rejFrac η₀ indecisionLimit εcov) B) \ G)
      ≤ α + 3 * δ / 8 := by
    calc _ ≤ ∑ B ∈ st, (runMeasure μ D Dsf).real ((retBy rule O.mq populations uni η₀
            indecisionLimit εcov α B ∩ FailAt rule O populations tame uni D indecisionLimit εcov
                (rejFrac η₀ indecisionLimit εcov) B)
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
of the scale, and the flip fraction seven tenths of the signal. -/
lemma rejFrac_ge (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    7 * budgetScale η₀ indecisionLimit εcov ^ 2 / 12800 ≤ rejFrac η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hB := budgetScale_pos η₀ indecisionLimit εcov hε hind hη0 hsig
  have hBs := budgetScale_le_sig η₀ indecisionLimit εcov hη0 hsig
  have hcut := cutBudget_ge η₀ indecisionLimit εcov hε hind hη0 hsig
  have hf : 7 * sig η₀ / 10 ≤ flipFrac η₀ := by
    rw [flipFrac, div_le_div_iff₀ (by norm_num) (by linarith)]
    nlinarith
  rw [rejFrac]
  have := mul_le_mul hcut (le_trans (by linarith : 7 * budgetScale η₀ indecisionLimit εcov / 10
    ≤ 7 * sig η₀ / 10) hf) (by positivity) (by linarith [hcut, cutBudget_pos η₀ hsig hε hind])
  nlinarith

/-- What the collision cap buys: a state's redraws, at `|J|(4|J| + 4)ρ/rejFrac` per draw, are a
sixty-fourth of the error budget spread over the top count. -/
lemma draws_le_of_cap (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hρ0 : 0 ≤ ρ)
    (hρ : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    (populations.card : ℝ) * (4 * (populations.card : ℝ) + 4) * ρ
        / rejFrac η₀ indecisionLimit εcov
      ≤ δ / (64 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)) := by
  set N : ℝ := (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) with hN
  set B : ℝ := budgetScale η₀ indecisionLimit εcov with hBdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  set Jc : ℝ := (populations.card : ℝ) with hJc
  have hs := sig_pos η₀ hsig
  have hB : 0 < B := budgetScale_pos η₀ indecisionLimit εcov hε hind hη0 hsig
  have hN0 : 0 < N := by
    rw [hN]
    exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      η₀ populations εcov δ α pAP
  have hJ0 : 0 ≤ Jc := Nat.cast_nonneg _
  have hc := rejFrac_ge η₀ indecisionLimit εcov hε hind hη0 hsig
  rw [← hBdef, ← hcdef] at hc
  have hc0 : 0 < c := lt_of_lt_of_le (by positivity) hc
  have hcap : ρ * (2 ^ 20 * (Jc + 1) ^ 2 * N) ≤ δ * B ^ 2 := by
    have h := hρ
    rw [collisionCap, ← hN, ← hBdef, ← hJc, le_div_iff₀ (by positivity)] at h
    exact h
  rw [div_le_div_iff₀ hc0 (by positivity)]
  have hJ : Jc * (4 * Jc + 4) ≤ 4 * (Jc + 1) ^ 2 := by nlinarith
  have step1 : Jc * (4 * Jc + 4) * ρ * (64 * N) ≤ 4 * (Jc + 1) ^ 2 * ρ * (64 * N) := by
    have := mul_le_mul_of_nonneg_right hJ (by positivity : (0 : ℝ) ≤ ρ * (64 * N))
    nlinarith
  have step2 : 4 * (Jc + 1) ^ 2 * ρ * (64 * N) * 2 ^ 20 ≤ 256 * δ * B ^ 2 := by
    nlinarith [hcap]
  have step3 : 256 * δ * B ^ 2 ≤ δ * c * 2 ^ 20 := by
    nlinarith [mul_le_mul_of_nonneg_left hc hδ.le]
  nlinarith [step1, step2, step3]

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

/-- The top rung's tails, each at `δ/(128·|populations|)`. -/
lemma solved_tails (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (9 * pAP / 20) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) ^ 2
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
          * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
          * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
        * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ)) := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hf : 0 < flipFrac η₀ := flipFrac_pos η₀ hsig
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
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
  have cscr : Real.log ((((M : ℝ) + 2) ^ 2) / ε₀)
      / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤ (m : ℝ) := by
    have heq : (((M : ℝ) + 2) ^ 2) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ)
        / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cdirty : Real.log (((M : ℝ) + 1) / ε₀)
      / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)
      ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2)⌉₊ ≤
          m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cth : Real.log (1 / ε₀) / (2 * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2) ≤ (m : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ)
        / (2 * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  -- the tails those counts kill
  have ttap : Real.exp (-2 * (M : ℝ) * (9 * pAP / 20) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := 9 * pAP / 20) (c := 1) (by positivity) hε₀ (by norm_num) ctap
    linarith
  have tscr : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
      ≤ ε₀ := tail_le_of_count (by positivity) hε₀ (by positivity) cscr
  have tdirty : ((M : ℝ) + 1) * Real.exp (-2 * (m : ℝ)
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ :=
    tail_le_of_count (by positivity) hε₀ (by linarith) cdirty
  have tdirty' : (M : ℝ) * Real.exp (-2 * (m : ℝ)
      * (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tdirty
    exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)
  have tth : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 32) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := cutBudget η₀ indecisionLimit εcov / 32) (c := 1) (by
      positivity) hε₀
      (by norm_num) cth
    linarith
  exact ⟨ttap, tscr, tdirty', tth⟩

lemma famCount_ge_four (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    4 ≤ famCount η₀ populations indecisionLimit εcov δ crossLimit := by
  have hv := voteSlack_pos η₀ hsig
  have h1 : 0 < ⌈1 / voteSlack η₀⌉₊ := Nat.ceil_pos.2 (by positivity)
  rw [famCount]
  omega

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

/-- The redraws a rung's draws cost, against the share proportional to its count. -/
lemma rung_draws_le (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    {m : ℝ} (hm : 0 ≤ m) :
    (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 4) * m * ρ
        / rejFrac η₀ indecisionLimit εcov)
      ≤ δ / (64 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)) * m := by
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hsig hε hind
  have hdraws := draws_le_of_cap η₀ populations hη0 hsig hε hind hδ hρ0 hρcap
  have e1 : (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 4) * m * ρ
      / rejFrac η₀ indecisionLimit εcov)
      ≤ (populations.card : ℝ) * (4 * (populations.card : ℝ) + 4) * ρ
        / rejFrac η₀ indecisionLimit εcov * m := by
    rw [show (populations.card : ℝ) * (4 * (populations.card : ℝ) + 4) * ρ
        / rejFrac η₀ indecisionLimit εcov * m
        = (populations.card : ℝ) * ((4 * (populations.card : ℝ) + 4) * m * ρ
          / rejFrac η₀ indecisionLimit εcov) by ring]
    gcongr
    linarith
  exact le_trans e1 (mul_le_mul_of_nonneg_right hdraws hm)

/-- The top rung carries its own share of the error budget, `δ/8`. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    stateFail η₀ populations indecisionLimit εcov ρ (solvedState η₀ populations
      indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 8 := by
  have hcle := cutBudget_le η₀ indecisionLimit εcov
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  obtain ⟨-, -, -, tth⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref = m :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have tund : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hdr := rung_draws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hind hδ hρ0
    hρsmall hmR
  rw [← hmdef, show δ / (64 * (m : ℝ)) * (m : ℝ) = δ / 64 by field_simp] at hdr
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  rw [stateFail, hBm, mul_add]
  nlinarith [mul_le_mul_of_nonneg_left tund hcard.le]

/-- Every rung at or above `validCount` carries its share: its collision terms from the share
proportional to its count, its tail from the one inversely proportional. -/
theorem rung_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ) (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (i : ℕ) (hi : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hV : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit
      ≤ prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i) :
    solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
        (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit / 2 ^ i)
      ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  set m : ℕ := N / 2 ^ i with hmdef
  have hB : solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m
      ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
    rw [schedule]
    exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩
  have hVpos : 0 < V := by rw [hVdef, validCount]; omega
  have hVR : (0 : ℝ) < (V : ℝ) := by exact_mod_cast hVpos
  have hVm : (V : ℝ) ≤ (m : ℝ) := by exact_mod_cast hV
  have hmR : (0 : ℝ) < (m : ℝ) := lt_of_lt_of_le hVR hVm
  have hmN : (m : ℝ) ≤ (N : ℝ) := by exact_mod_cast Nat.div_le_self N (2 ^ i)
  have hNR : (0 : ℝ) < (N : ℝ) := lt_of_lt_of_le hmR hmN
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨(band_of_mem_schedule hη0 hsig hB).1,
    by show 0 < m; exact_mod_cast hmR, poolCount_found η₀ populations hsig hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov ρ
      (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m)
    ≤ δ * (m : ℝ) / (8 * N) + if V ≤ m then δ * (V : ℝ) / (16 * m) else 0
  rw [if_pos hV, stateFail]
  show (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 4) * (m : ℝ) * ρ
      / rejFrac η₀ indecisionLimit εcov
    + Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2))
    ≤ δ * (m : ℝ) / (8 * N) + δ * (V : ℝ) / (16 * m)
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  have hdr := rung_draws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hind hδ hρ0
    hρcap hmR.le
  rw [← hNdef] at hdr
  have hcoll : δ / (64 * (N : ℝ)) * (m : ℝ) ≤ δ * (m : ℝ) / (8 * N) := by
    rw [div_mul_eq_mul_div, div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_pos hδ hmR, hNR]
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  have hε₀3 : ε₀ ≤ 1 / 6 := by
    rw [hε₀def, div_le_iff₀ (by positivity)]
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

/-- Some rung the loop may stop at has fewer than twice `validCount` prefixes: the lowest rung
whose count is still at or above it, or any rung when the top is already below it. -/
theorem exists_small_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ) (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hne : (stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ).Nonempty) :
    ∃ B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ,
      (B.npref : ℝ) ≤ 2 * (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  have hVpos : 0 < V := by rw [hVdef, validCount]; omega
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
  have hiL : i < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
    have hpow : 2 ^ i ≤ N := by
      have h1 : 1 ≤ N / 2 ^ i := le_trans hVpos hi0
      exact (Nat.one_le_div_iff (by positivity)).1 h1
    have := Nat.le_log_of_pow_le (by norm_num) hpow
    rw [ladderLen, ← hNdef]
    omega
  refine ⟨_, rung_mem_stoppable η₀ populations hη0 hsig hε hind hδ hδ1 hpAP hcard hρ0 hρcap i hiL
    hi0, ?_⟩
  show ((N / 2 ^ i : ℕ) : ℝ) ≤ 2 * (V : ℝ)
  exact_mod_cast hlt.le

/-- The top rung can always be stopped at: it carries the whole proportional share. -/
theorem solvedState_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ) (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit
      ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ := by
  classical
  have hB := solvedState_mem_schedule (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  have hmpos : 0 < m := prefCount_pos (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨(band_of_mem_schedule hη0 hsig hB).1, hmpos,
    poolCount_found η₀ populations hsig hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov ρ
      (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    ≤ δ * (m : ℝ) / (8 * m) + _
  have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by field_simp
  rw [hval]
  exact le_add_of_le_of_nonneg (solved_share η₀ populations hsig hη0 hε hind hδ hpAP hcard hρ0
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
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2)
        + Real.exp (-2 * (C.card : ℝ) * (lmis - E) ^ 2) := by
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
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (l lcut lmis f E : ℝ) (kmin kmax : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ lmis)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real (lightMiss rule O populations j B l lcut lmis f kmin kmax)
      ≤ Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
        + Real.exp (-2 * (B.npref : ℝ) * (lmis - E) ^ 2) := by
  classical
  set R : ℝ := Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
    + Real.exp (-2 * (B.npref : ℝ) * (lmis - E) ^ 2) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : runMeasure μ D Dsf (lightMiss rule O populations j B l lcut lmis f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_lightMiss O populations j B l lcut lmis f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
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
  calc (runMeasure μ D Dsf (lightMiss rule O populations j B l lcut lmis f kmin kmax)).toReal
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

lemma mem_retAll_of_retAt (O : Oracle μ S) (populations : Finset J) (hpop : populations.Nonempty)
    (uni : J) (η₀ indecisionLimit εcov α : ℝ) (B : State) (x : Run Ω S J)
    (h : ∀ j ∈ populations, x ∈ retAt rule O populations uni η₀ indecisionLimit εcov α B j) :
    x ∈ retAll rule O.mq populations uni η₀ indecisionLimit εcov α B := by
  obtain ⟨j₀, hj₀⟩ := hpop
  exact ⟨(h j₀ hj₀).1, fun j hj _ => (h j hj).2.1, fun j hj _ => (h j hj).2.2⟩

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
set_option maxHeartbeats 4000000 in
/-- The round returns at one population.  Everything outside the light miss counts and the
gate's liveness is a fact about the draws: a certification sample not found in time, a family of
the wrong size or dirty on population `j` or the uniform pool, too many heavy prefixes, a uniform
pool short of either class, or a pool too short to leave a reference. -/
theorem measureReal_notRetAt_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0) (j : J)
    (hj : j ∈ populations) (B : State) (hmpos : 0 < B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lcut lmis f E Δp th c t ρsf : ℝ) (r rs kmin kmax : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ lmis)
    (hl2 : 2 * l ≤ indecisionLimit)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (hf0 : 0 < f) (hr : (r : ℝ) ≤ c * B.npref)
    (hheavy : Δp / f + th + c ≤ lcut) (hlohi : B.lo ≤ B.hi)
    (ht : 0 ≤ t) (hkpos : 0 < B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + lmis) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (Ecert : ℝ) (hcertBad : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ certGood populations j B.npref r x} ≤ Ecert)
    (Estall : ℝ) (hstall : (runMeasure μ D Dsf).real (stalled rule O populations B kmin kmax)
      ≤ Estall)
    (Edirty : ℝ) (hdirty : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ indecisionLimit εcov α B j}
      ≤ Ecert + Estall + Edirty
        + Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
        + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
          + Real.exp (-2 * (B.npref : ℝ) * (lmis - E) ^ 2))
        + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
        + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) := by
  classical
  set n : ℕ := B.npref with hndef
  set E1 : Set (Run Ω S J) := {x | ¬ certGood populations j n r x} with hE1
  set E5 : Set (Run Ω S J) := stalled rule O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits rule O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := lightMiss rule O populations j B l lcut lmis f kmin kmax with hE8
  set E10 : Set (Run Ω S J) := sidesDead rule O populations j B n t with hE10
  set E11 : Set (Run Ω S J) := {x | ¬ poolGood B.nsuff rs x} with hE11
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hmpos
  have hsub : {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ indecisionLimit εcov α B j}
      ⊆ E1 ∪ E5 ∪ E6 ∪ E7 ∪ E8 ∪ E10 ∪ E11 := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, not_or] at hc
    obtain ⟨⟨⟨⟨⟨⟨h1, h5⟩, h6⟩, h7⟩, h8⟩, h10⟩, h11⟩ := hc
    simp only [hE1, hE6, hE11, Set.mem_setOf_eq, not_not] at h1 h6 h11
    have h5' : kmin ≤ (clusterBy rule O.mq populations x B).card
        ∧ (clusterBy rule O.mq populations x B).card ≤ kmax := by
      by_contra h; exact h5 h
    set F : Finset S := clusterBy rule O.mq populations x B with hFdef
    set ω : Ω := oracleNoise x with hω
    set T : Finset S := prefixesAt populations n x with hT
    set C : Finset S := certOf T j n x with hC
    set R : Finset S := refsBy rule O.mq populations x B with hR
    obtain ⟨hCn, -⟩ := certOf_of_good populations j n r x h1
    have hCpos : 0 < C.card := by rw [hCn]; exact hmpos
    -- the light counts
    have hhv := heavy_le_of_clean O populations (D j) j B x hf0 hΔp hr hheavy h1 h5' h6 h7
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
    refine hx ⟨?_, ?_, hcert⟩
    · have : B.k ≤ F.card + 1 := by omega
      exact this
    · rw [hCn]
      have := mul_le_mul_of_nonneg_right hl2 hnR.le
      linarith [hl8.1]
  set μr := runMeasure μ D Dsf with hμr
  have u : ∀ A B' : Set (Run Ω S J), μr.real (A ∪ B') ≤ μr.real A + μr.real B' :=
    fun A B' => measureReal_union_le A B'
  have b7 := measureReal_heavyHits_le (rule := rule) D Dsf O populations j B f (Δp / f) th hth
  have b8 := measureReal_lightMiss_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj
    B l lcut lmis f E kmin kmax hE hElcut hlcl hElm hdec hcut
  have b10 := measureReal_sidesDead_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf j hj
    B n ht
  have b11 := measureReal_poolBad_le (μ := μ) D Dsf B.nsuff rs ρsf hρsf hρsf0
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
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
    ∀ N : ℕ, (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref ≤ N →
      N ≤ (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref
        + rejCount η₀ indecisionLimit εcov
          (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref →
      (cutBudget η₀ indecisionLimit εcov + 5 / 8 * cutBudget η₀ indecisionLimit εcov)
          * (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref
          / gateKappa η₀
        + 2 * ((gateDevT η₀ + gateDevU η₀)
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref
              (δ / (128 * populations.card
                * (((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).nsuff
                  : ℝ) + 3)))
          + gateDevD η₀
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref
              (sizeLevel (gateLevel α populations
                (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit))
                (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref))
        + ((N : ℝ) - (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref)
      ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations
          (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit)) N) := by
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref = m := rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).nsuff = M := rfl
  have hga : ∀ N : ℕ, sizeLevel (gateLevel α populations
        (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit)) N
      = sizeLevel (sizeLevel (α / (7 * (populations.card : ℝ))) m) N := fun N => rfl
  rw [hBm, hBM]
  simp only [hga]
  -- the two counts the gate's terms ask for
  have hc1 : ⌈25 / 2 * (gateDevT η₀ + gateDevU η₀) ^ 2
        * Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 3) / δ)
        / gateRoom η₀ indecisionLimit εcov ^ 2⌉₊ ≤ m := by
    rw [hmdef, prefCount, ← hMdef]; omega
  have hc2 : ⌈gateLogCount (50 / 9 * (gateDevD η₀ + 1) ^ 2 / gateRoom η₀ indecisionLimit εcov ^ 2)
        (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α)))⌉₊ ≤ m := by
    rw [hmdef, prefCount]; omega
  have hmR1 : (1 : ℝ) ≤ m := by
    have := prefCount_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit) η₀
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
  set L₁ : ℝ := Real.log (128 * c * ((M : ℝ) + 3) / δ) with hL₁
  have hP1 : 25 / 2 * W ^ 2 * L₁ ≤ m * gR ^ 2 := by
    have := le_trans (Nat.le_ceil _) (by exact_mod_cast hc1 :
      ((⌈25 / 2 * W ^ 2 * L₁ / gR ^ 2⌉₊ : ℕ) : ℝ) ≤ m)
    rwa [div_le_iff₀ (by positivity)] at this
  set t : ℝ := hoeffDev m (δ / (128 * c * ((M : ℝ) + 3))) with htdef
  have ht : t ≤ Real.sqrt (m * L₁ / 2) :=
    hoeffDev_le_sqrt (by rw [one_div_div])
  have hsqt : 5 * W * Real.sqrt (m * L₁ / 2) ≤ gR * m := by
    have h25 : (5 * W) ^ 2 * (m * L₁ / 2) ≤ (gR * m) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hP1 hm0.le
      nlinarith
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
    nlinarith
  have hy : 2 ≤ Real.log (4 * K * kk) := by
    have h64 : Real.log 64 ≤ Real.log (4 * K * kk) :=
      Real.log_le_log (by norm_num) (by nlinarith)
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
    have h1 : K * Λ ≤ m := by nlinarith
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
      nlinarith
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
    nlinarith
  have hNR1 : (m : ℝ) ≤ N := by exact_mod_cast hN1
  have hNR2 : (N : ℝ) ≤ m + cut / 128 * m := by
    have : (N : ℝ) ≤ m + (rejCount η₀ indecisionLimit εcov m : ℕ) := by exact_mod_cast hN2
    linarith
  have hcut1 : cut ≤ 1 := by
    have := (cutBudget_le η₀ indecisionLimit εcov).2.1
    have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    linarith
  have hN2mR : (N : ℝ) ≤ 2 * m := by nlinarith
  have hN2m : N ≤ 2 * m := by exact_mod_cast hN2mR
  have hdN : hoeffDev N (sizeLevel (sizeLevel (α / (7 * c)) m) N) ≤ 2 * s := by
    refine le_trans (hoeffDev_le_sqrt (hsl N hN2m)) ?_
    calc Real.sqrt (N * Λ / 2) ≤ Real.sqrt (2 ^ 2 * (m * Λ / 2)) :=
          Real.sqrt_le_sqrt (by nlinarith)
      _ = 2 * s := by rw [Real.sqrt_mul (by norm_num), Real.sqrt_sq (by norm_num)]
  have hDd : D * hoeffDev m (sizeLevel (sizeLevel (α / (7 * c)) m) m) ≤ D * s :=
    mul_le_mul_of_nonneg_left hdm hD0.le
  have hch : (cut + 5 / 8 * cut) * m / κ = 13 * cut / (8 * κ) * m := by
    field_simp; ring
  have hNmn : (m : ℝ) * mn ≤ N * mn := mul_le_mul_of_nonneg_right hNR1 hmn0
  have hsD : 0 ≤ s := Real.sqrt_nonneg _
  rw [hch]
  linarith

/-- What one round at one population can cost: the samplers redrawing too much, the family's
size and cleanliness, the sample's heavy fraction, the two light counts, the gate's liveness,
and the pool. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut th E γscr γdirty gdirty tap ρ ρsf t : ℝ) (r rs : ℕ) (B : State) : ℝ :=
  ((B.npref + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
    + (((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
      + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
        + (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
    + ((((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
        + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
          + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))))
    + Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
    + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
      + Real.exp (-2 * (B.npref : ℝ) * (5 / 8 * lcut - E) ^ 2))
    + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
    + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))

/-- A state whose round can pass.  Every clause is an inequality among the state's budgets,
the oracle's rates, the suffix distribution's findability and the error budget; no probability
enters. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (B : State) : Prop :=
  ∃ (th tap γdec γscr γdirty gdirty Δ lcut f c t : ℝ) (r rs : ℕ),
    0 < B.npref ∧ 2 ≤ B.k ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γdec ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 < f ∧ 0 ≤ pAP ∧ 0 < lcut ∧ lcut ≤ indecisionLimit / 2
    -- the samplers redraw at most a `c` fraction of the prefix draws
    ∧ 0 ≤ c ∧ (r : ℝ) ≤ c * B.npref
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification samples hold few heavy prefixes
    ∧ (((populations.card : ℝ) * Δ + c + gdirty) / f + th + c ≤ lcut)
    -- the screen's rate sits above the clean one and below the dirty one
    ∧ 0 < B.scd
    ∧ ((B.scd : ℝ) * (2 * γscr) ≤ (B.sc : ℝ))
    ∧ ((B.sc : ℝ)
        ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γdirty))
    -- the pool holds a family, beyond what its sampler redraws, and a reference beside it
    ∧ ((B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - tap))
    ∧ B.k < B.nsuff
    -- the thresholds decide, and decide right, on a family of the round's size an `f`
    -- fraction of which flips; the FNR reads the family with its seed, cut one count higher
    ∧ ((B.hi : ℝ) + 1 ≤ (B.k : ℝ) * ((1 - O.η - O.gap * f) - γdec))
    ∧ ((B.k : ℝ) * ((O.η + O.gap * f) + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * ((O.η + O.gap * f) + γdec) ≤ (B.hi : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η - O.gap * f) - γdec))
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
    -- the vote misfires at under half the cut budget
    ∧ (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2) ≤ lcut / 2)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) lcut th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf t
            r rs B
      ≤ δ / 2)

set_option maxHeartbeats 2000000 in
/-- The round's failure, over every population, fits the error budget. -/
lemma solved_roundFail (η₀ : ℝ) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfsmall : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (cutBudget η₀ indecisionLimit εcov / 32)
            (Real.exp (-2
              * (((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).k
                  - 1 : ℕ) : ℝ)
              * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32)
            (9 * pAP / 20) ρ ρsf
            (hoeffDev (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)
              (δ / (128 * populations.card
                * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))))
            (rejCount η₀ indecisionLimit εcov
              (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit))
            (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
            (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 2 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  obtain ⟨ttap, tscr, tdirty', tth⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref = m :=
    rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).nsuff = M :=
    rfl
  have hBκ : ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).k - 1 : ℕ)
      = κ := by
    show κ + 1 - 1 = κ
    omega
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- the vote's misfire rate sits under half the cut budget
  have hEc : Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 :=
    famCount_tail_seed η₀ populations hsig hε hind
  have hE0 : 0 ≤ Real.exp (-2 * (κ : ℝ)
      * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2) := Real.exp_nonneg _
  have tl : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
        ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tlm : Real.exp (-2 * (m : ℝ) * (5 / 8 * cutBudget η₀ indecisionLimit εcov
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
        ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  -- the gate's liveness, at `ε₀/(M + 3)` a side
  have hb0 : 0 < δ / (128 * (populations.card : ℝ) * ((M : ℝ) + 3)) := by positivity
  have hb1 : δ / (128 * (populations.card : ℝ) * ((M : ℝ) + 3)) ≤ 1 := by
    have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
      have : 1 ≤ populations.card := by exact_mod_cast hcard
      exact_mod_cast this
    rw [div_le_one (by positivity)]
    nlinarith
  have tlive : ((M : ℝ) + 3) * Real.exp (-2 * hoeffDev m
      (δ / (128 * (populations.card : ℝ) * ((M : ℝ) + 3))) ^ 2 / m) = ε₀ := by
    have h := exp_hoeffDev m (by exact_mod_cast hmpos) hb0 hb1
    rw [show -2 * hoeffDev m (δ / (128 * (populations.card : ℝ) * ((M : ℝ) + 3))) ^ 2 / m
        = -2 * (m : ℝ) * (hoeffDev m (δ / (128 * (populations.card : ℝ) * ((M : ℝ) + 3))) / m)
          ^ 2 by field_simp, h, hε₀def]
    field_simp
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  -- the redraws
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hind
  obtain ⟨hcle, -⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hsig hε hind
  set r : ℕ := rejCount η₀ indecisionLimit εcov m with hrdef
  set rs : ℕ := poolRej pAP M with hrsdef
  have hrt : ((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1) ≤ 2 * (m : ℝ) / c :=
    rejTerm_le hc0 (by
      have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
      linarith [hcle, (cutBudget_le η₀ indecisionLimit εcov).2.1]) m
  have hdraws := draws_le_of_cap η₀ populations hη0 hsig hε hind hδ hρ0 hρsmall
  rw [← hmdef, ← hcdef] at hdraws
  have hpool := poolDraws_le_of_cap η₀ populations hpAP hpAP1 hδ hρsf0 hρsfsmall
  rw [← hMdef, ← hrsdef] at hpool
  have hA : ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ
        / ((r : ℝ) + 1)
      + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
      ≤ 2 * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c) := by
    rw [show ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ
          / ((r : ℝ) + 1)
        + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
        = (3 * (populations.card : ℝ) / 2 + 3 / 2) * ρ
          * (((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1)) by
      field_simp
      ring]
    calc (3 * (populations.card : ℝ) / 2 + 3 / 2) * ρ * (((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1))
        ≤ (3 * (populations.card : ℝ) / 2 + 3 / 2) * ρ * (2 * (m : ℝ) / c) := by gcongr
      _ ≤ 2 * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c) := by
          rw [show 2 * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c)
              = (3 * (populations.card : ℝ) + 3) * ρ * (2 * (m : ℝ) / c) by ring]
          gcongr <;> linarith
  have hB : (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c)
      ≤ δ / 64 := by
    have e : (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c)
        = (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ / c * (m : ℝ) := by ring
    rw [e]
    have hdraws3 : (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ / c
        ≤ δ / (64 * (m : ℝ)) := by
      refine le_trans ?_ hdraws
      have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
      exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left (by linarith) hJ0) hρ0) hc0.le
    calc (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ / c * (m : ℝ)
        ≤ δ / (64 * (m : ℝ)) * (m : ℝ) := mul_le_mul_of_nonneg_right hdraws3 hmpos.le
      _ = δ / 64 := by field_simp
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
      nlinarith
    linarith
  rw [roundFail, hBm, hBM, hBκ]
  have tscr' : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤
        ε₀ := tscr
  refine le_trans (mul_le_mul_of_nonneg_left (show _ ≤ 20 * ε₀
      + (((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
        + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
        + 3 * (((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)))) from ?_) hcard.le) ?_
  · linarith [ttap, tscr', tdirty', tth, tl, tlm, tlive]
  · have hA' := mul_le_mul_of_nonneg_left hA hcard.le
    nlinarith [hcε, hB, hP, hA']

set_option maxHeartbeats 4000000 in
/-- The computed ladder holds a state whose round can pass: the top rung, every clause
discharged from the closed forms that define it. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfcap : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    ∃ B : State, B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ
      ∧ PassableAt O η₀ populations Dsf indecisionLimit εcov α δ ρ ρsf pAP B := by
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
  have hb := bandHalf_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hb1 := bandHalf_lt (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη0' hη₀
  set b : ℕ := bandHalf η₀ populations indecisionLimit εcov δ crossLimit with hbdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hκpos : 0 < κ := famCount_pos η₀ populations εcov δ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit with hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  set x : ℝ := (κ : ℝ) / 2 with hxdef
  have hxpos : 0 < x := by
    rw [hxdef]
    exact div_pos hκR (by norm_num)
  have hxexact : (⌈x⌉₊ : ℝ) = x := by rw [hxdef, hκdef]; exact famCount_half η₀ populations εcov δ
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  obtain ⟨hlowShift, hhiShift⟩ := vote_shifts O hηle hη₀ (κ : ℝ)
  have hb0 : (0 : ℝ) ≤ (b : ℝ) := Nat.cast_nonneg _
  have hκv : (0 : ℝ) ≤ (κ : ℝ) * voteSlack η₀ := mul_nonneg hκ0 hv.le
  -- the fields, named before the definitions are made opaque
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := rfl
  have hBlo : B.lo = ⌈x⌉₊ - b - 1 := rfl
  have hBhi : B.hi = ⌈x⌉₊ + b := rfl
  have hBscd : B.scd = ⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  have hMceil : 2 * ((κ : ℝ) + 1) / pAP ≤ ((B.nsuff : ℕ) : ℝ) := by
    rw [hBM, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊]
  have hdS : decSlack η₀ populations indecisionLimit εcov δ crossLimit
      = voteSlack η₀ / 2 - 1 / (2 * ((κ : ℝ) + 1)) := by rw [decSlack, hκdef]
  have hdS0 := decSlack_nonneg (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  have hdSle := decSlack_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hdSmul : (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit
      ≤ (κ : ℝ) * (voteSlack η₀ / 2) := mul_le_mul_of_nonneg_left hdSle hκ0
  obtain ⟨hlowShift1, hhiShift1⟩ := vote_shifts O hηle hη₀ ((κ : ℝ) + 1)
  have hdSk : ((κ : ℝ) + 1) * (1 / (2 * ((κ : ℝ) + 1))) = 1 / 2 := by field_simp
  have hcutle := cutBudget_le η₀ indecisionLimit εcov
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast this
  have hgate := solved_gate (crossLimit := crossLimit) (pAP := pAP) (δ := δ) η₀ populations hη₀
    hη0' hεcov
    hindLim hαpos (by linarith) hcard1
  rw [← hBdef] at hgate
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  clear_value B b κ m x
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hη₀ hεcov hindLim
  obtain ⟨hcle, hceq⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hη₀ hεcov hindLim
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_⟩⟩,
    cutBudget η₀ indecisionLimit εcov / 32, 9 * pAP / 20,
    decSlack η₀ populations indecisionLimit εcov δ crossLimit,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32,
    flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀,
    cutBudget η₀ indecisionLimit εcov, flipFracOf O η₀, rejFrac η₀ indecisionLimit εcov,
    hoeffDev m (δ / (128 * populations.card
      * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))),
    rejCount η₀ indecisionLimit εcov m,
    poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  -- in the schedule
  · rw [hBdef]
    exact solvedState_mem_schedule η₀ populations εcov δ α pAP
  -- Capped
  · rw [hBlo, hBhi]
    omega
  · rw [hBm]; exact hmpos
  · rw [hBM]
    exact poolCount_found η₀ populations hη₀ hpAPPositive
  · have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by
      rw [mul_comm (8 : ℝ) (m : ℝ), mul_comm δ (m : ℝ),
        mul_div_mul_left _ _ (ne_of_gt hmR)]
    rw [hBm, ← hmdef, hval, hBdef]
    exact le_add_of_le_of_nonneg (solved_share η₀ populations hη₀ hη0' hεcov hindLim hδ
      hpAPPositive hcard hρ0 hρcap) (by split_ifs <;> positivity)
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · exact hindLim
  · exact hind1
  · linarith
  · linarith
  · exact hdS0
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
      indecisionLimit εcov δ α pAP crossLimit) hη₀ hεcov hindLim hcard).2.2.1
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
    nlinarith
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    have : ((κ + 1 : ℕ) : ℝ) < (B.nsuff : ℝ) := by push_cast; nlinarith
    exact_mod_cast this
  -- the thresholds decide, and decide right
  · rw [hBhi, hBk, hdS]
    push_cast
    linarith [hxexact, hhiShift1, hb, hv, hdSk]
  · rw [hBlo, hBk, hdS, Nat.sub_sub, Nat.cast_sub hb1]
    push_cast
    linarith [hxexact, hlowShift1, hb, hv, hdSk]
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega]
    push_cast
    linarith [hxexact, hlowShift, hdSmul]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.sub_sub, Nat.cast_sub hb1]
    push_cast
    linarith [hxexact, hhiShift, hdSmul]
  -- the gate
  · rw [hBlo, hBhi]
    omega
  · exact Real.sqrt_nonneg _
  · rw [hBm]
    rw [hBm, hBM] at hgate
    exact hgate
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail_seed η₀ populations hη₀ hεcov hindLim
  · have h := solved_roundFail (crossLimit := crossLimit) (α := α) (ρ := ρ) (ρsf := ρsf)
      η₀ populations hη₀ hη0' hεcov hδ hδ1 hpAPPositive hpAP1 hindLim
      hcard hρ0 hρsf0 hρcap hρsfcap
    rw [← hBdef, ← hmdef] at h
    exact h

set_option maxHeartbeats 2000000 in
/-- Part 2 — the loop returns: except with probability `δ/2`, the top rung passes at every
population, its gate read at the first look.  `iLtest` is the indecision rate the FNR test is
run at, at least the one the schedule is solved at. -/
theorem loop_terminates {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit iLtest εcov α : ℝ) (ρ pAP δ : ℝ)
    (hsig : O.η < 1 / 2) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2) (htest : indecisionLimit ≤ iLtest)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsf : collisionMass Dsf
        ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ},
      x ∉ retAll rule O.mq populations uni η₀ iLtest εcov α B.val} ≤ δ / 2 := by
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  obtain ⟨B, hB, hpass⟩ := exists_passable (crossLimit := crossLimit) O populations D Dsf
    indecisionLimit εcov α δ ρ (collisionMass Dsf) pAP hsig hpop hηle hη₀ hεcov hε1 hδ hδ1
    hαpos hα hindLim hind1 hpAPPositive hpAPBound hρ hρ0 le_rfl
    (tsum_nonneg (fun a => sq_nonneg _)) hρcap hρsf
  obtain ⟨th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, f, c, t, r, rs, hmpos, hkpos, hindLim',
    hind1', hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hf0, hpAP0, hlcut, hlcl,
    hc0, hr, hpAPBound', hρsf', hheavy, hscd, hscLow, hscHigh, hcount, hpool,
    hhiUp, hloUp, hhiLo, hloLo, hlohi, ht, hgate, hEhalf,
    hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hl2 : 2 * l ≤ iLtest := by rw [hl]; linarith
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hsub : {x : Run Ω S J | ∀ B' : {B : State //
          B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ},
        x ∉ retAll rule O.mq populations uni η₀ iLtest εcov α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ iLtest εcov α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_setOf_eq, not_and] at hc
    refine hx ⟨B, hB⟩ (mem_retAll_of_retAt O populations hpop uni η₀ iLtest εcov α B x
      (fun j hj => ?_))
    by_contra hcj
    exact hc j hj hcj
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hstall := measureReal_stalled_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf B
    (by omega) uni huni γscr pAP tap (collisionMass Dsf) ρ rs r hγscr hpAP0 htap hpAPBound' hscd
    hsig.le hscLow hcount hρsf' hρsf0 (hρ uni huni) hρ0
  have hdirtyAt : ∀ j ∈ populations, (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
        flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + c + gdirty}
      ≤ (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
        + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
          + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * collisionMass Dsf / (2 * ((rs : ℝ) + 1))) := by
    intro j hj
    have hdirtyI := measureReal_dirtyMember_le (rule := rule) hflat O populations D Dsf hsupp
      hsuppSf j hj B hsig.le hmpos Δ γdirty gdirty c ρ r hΔ hγdirty hgdirty hc0 hρ0 hr (hρ j hj)
      hscd hscHigh
    have hnoclean := measureReal_noCleanRef_le D Dsf O populations B pAP tap
      (collisionMass Dsf) rs hpAP0 htap hpAPBound' le_rfl hρsf0
      (by
        have hk2 : (2 : ℝ) ≤ (B.k : ℝ) := by exact_mod_cast hkpos
        linarith [hcount, hk2])
    have hcov : {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + c + gdirty}
        ⊆ ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
              ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
            ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
              flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + c + gdirty})
          ∪ {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
              ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} := by
      intro x hx
      by_cases hc : ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0
      · exact Or.inl ⟨hc, hx⟩
      · exact Or.inr hc
    refine le_trans (measureReal_mono hcov (measure_ne_top _ _)) ?_
    refine le_trans (measureReal_union_le _ _) ?_
    gcongr
  have hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E := by
    intro F p h1 hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    have hcardI : ((insert 1 F).card : ℝ) = (B.k : ℝ) := by
      rw [Finset.card_insert_of_notMem h1, hcardF, hκ]
      have : 1 ≤ B.k := by omega
      push_cast [this]
      ring
    have hcle : (F.card : ℝ) ≤ ((insert 1 F).card : ℝ) := by
      exact_mod_cast Finset.card_le_card (Finset.subset_insert _ _)
    have hflI : flipCount O (insert 1 F) p ≤ ((insert 1 F).card : ℝ) * f :=
      le_trans (flipCount_insert_one O F p)
        (le_trans hfl (mul_le_mul_of_nonneg_right hcle hf0.le))
    refine le_trans (decided_whp O (insert 1 F) p B.lo (B.hi + 1) f γdec hγdec hsig.le hflI
      ?_ ?_) ?_
    · rw [hcardI]; push_cast; exact hhiUp
    · rw [hcardI]; exact hloUp
    · rw [hE, hcardI]
      refine Real.exp_le_exp.2 ?_
      have hk : (κ : ℝ) ≤ (B.k : ℝ) := by rw [hκ]; exact_mod_cast Nat.sub_le _ _
      nlinarith [sq_nonneg γdec]
  have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E := by
    intro F p hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    refine le_trans (cutCorrect_whp O F p B.lo B.hi f γdec hsig.le hfl hγdec ?_ ?_) ?_
    · rw [hcardF]; linarith [hhiLo]
    · rw [hcardF]; exact hloLo.le
    · rw [hE, hcardF]
  have hper : ∀ j ∈ populations,
      (runMeasure μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ iLtest εcov α B j}
        ≤ roundFail populations l lcut th E γscr γdirty gdirty tap ρ (collisionMass Dsf) t
          r rs B := by
    intro j hj
    have hcertBad := measureReal_certNotGood_le (μ := μ) D Dsf populations j B.npref r ρ hρ
      (hρ j hj) hρ0
    have hmain := measureReal_notRetAt_le (indecisionLimit := iLtest) hflat O populations uni huni
      D Dsf hsupp hsuppSf j hj B hmpos hηle hη₀ εcov α hαpos (by linarith)
      l lcut (5 / 8 * lcut) f E ((populations.card : ℝ) * Δ + c + gdirty) th c t
      (collisionMass Dsf) r rs κ κ (by omega) (Real.exp_nonneg _) (by linarith)
      (by rw [hl]; exact hlcl) (by linarith) hl2
      (by positivity) hth hf0 hr hheavy hlohi ht (by omega) hpool hgate
      hρsf' hρsf0 _ hcertBad _ hstall _ (hdirtyAt j hj) hdec hcut
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  have hsum : ∑ j ∈ populations, (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ iLtest εcov α B j}
      ≤ (populations.card : ℝ)
          * roundFail populations l lcut th E γscr γdirty gdirty tap ρ (collisionMass Dsf) t
            r rs B := by
    refine le_trans (Finset.sum_le_sum hper) (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul]
  linarith [hsum, hbudget]

#print axioms validity_of_returned
#print axioms loop_terminates

end Loop

/-- `validity_of_returned` (whatever the loop returns is good, whenever it is returned, except
w.p. `δ/2 + α`) and `loop_terminates` (it returns, except w.p. `δ/2`), glued by
`sound_and_terminating`. -/
theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O rule populations tame uni D Dsf _ _ Pre Suf η₀ indecisionLimit εcov α δ
    ρ pAP crossLimit k
    hηle hη₀ huni htame hflat hsupp hsuppSf hρ hpAPPositive hpAPBound hindLim hind1
    hαpos hα hεcov hε1 hδ _hbudget hρcap hρsf
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ - α ≤ 0) measureReal_nonneg
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ uni huni 1)
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  have hcoll := le_trans (poolDraws_le_of_cap (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit)
      η₀ populations hpAPPositive hpAP1 hδ hρsf0 hρsf)
    (show δ / (128 * ((populations.card : ℝ) + 1)) ≤ δ / 16 by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith)
  have hvalid := validity_of_returned (rule := rule) (crossLimit := crossLimit) hflat O populations
    tame uni huni htame D Dsf hsupp hsuppSf indecisionLimit εcov α hindLim hηle hη₀ hαpos
    (by linarith) ρ (collisionMass Dsf) hρ le_rfl hρsf0 hεcov δ hδ hδ1 pAP hpAPPositive hpAP1
    hpAPBound hcoll
  have hc0 : 0 ≤ rejFrac η₀ indecisionLimit εcov :=
    (rejFrac_pos η₀ hη₀ hεcov hindLim).le
  set st := stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ with hst
  set Bad : Set (Run Ω S J) := ⋃ B : {B : State // B ∈ st},
      retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
        ∩ FailAt rule O populations tame uni D indecisionLimit εcov
          (rejFrac η₀ indecisionLimit εcov) B.val with hBad
  have hgood : ∀ x ∉ Bad, ∀ B : {B : State // B ∈ st},
      x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val →
      ∀ j ∈ tame, 1 - gateLimit uni εcov j - 2 * rejFrac η₀ indecisionLimit εcov
          ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
              (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
        ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
              (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
            ≤ 2 * indecisionLimit := by
    intro x hx B hB j hj
    have hnf : x ∉ FailAt rule O populations tame uni D indecisionLimit εcov
        (rejFrac η₀ indecisionLimit εcov) B.val := fun hf =>
      hx (Set.mem_iUnion.2 ⟨B, hB, hf⟩)
    simp only [FailAt, Set.mem_setOf_eq, not_not] at hnf
    obtain ⟨hcov, hund⟩ := hnf j hj
    refine ⟨le_trans (by linarith) (le_trans hcov (measureReal_mono (fun p hp =>
      cutCorrect_insert_one O _ _ _ (Finset.notMem_erase _ _) p _ hp) (measure_ne_top _ _))),
      hund⟩
  by_cases hall : ∀ j ∈ populations, j ∈ tame
  · have hρc : ∀ j ∈ populations, collisionMass (D j) ≤ ρ := fun j hj =>
      collisionMass_le_of_atoms (D j) (hρ j (hall j hj))
    have hterm := loop_terminates (rule := rule) (crossLimit := crossLimit) hflat O populations uni
      (htame huni) D Dsf hsupp hsuppSf indecisionLimit indecisionLimit εcov α ρ pAP δ hsig hηle
      hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1 le_rfl hpAPPositive hpAPBound hρc hρ0 hρcap
      hρsf
    have h := sound_and_terminating (runMeasure μ D Dsf)
      (fun B : {B : State // B ∈ st} =>
        retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
          ∩ FailAt rule O populations tame uni D indecisionLimit εcov
            (rejFrac η₀ indecisionLimit εcov) B.val)
      (fun B : {B : State // B ∈ st} =>
        retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val) (δ / 2 + α) (δ / 2)
      hvalid
      (le_trans (measureReal_mono (fun x hx B hB => hx B
          (retAll_subset_retBy rule O.mq η₀ indecisionLimit εcov α B.val hB))
          (measure_ne_top _ _)) hterm)
    rw [show (1 : ℝ) - (δ / 2 + α + δ / 2) = 1 - δ - α by ring] at h
    refine le_trans h (measureReal_mono ?_ (measure_ne_top _ _))
    intro x hx
    simp only [Set.mem_setOf_eq, Set.mem_inter_iff] at hx ⊢
    obtain ⟨hex, hall'⟩ := hx
    have hxB : x ∉ Bad := by
      intro hb
      obtain ⟨B, hf⟩ := Set.mem_iUnion.1 hb
      exact hall' B hf
    exact ⟨fun _ => hex, hgood x hxB⟩
  · have hBadle : (runMeasure μ D Dsf).real Bad ≤ δ / 2 + α := hvalid
    have h1 : (1 : ℝ) ≤ (runMeasure μ D Dsf).real Bad + (runMeasure μ D Dsf).real Badᶜ := by
      have := measureReal_union_le (μ := runMeasure μ D Dsf) Bad Badᶜ
      rwa [Set.union_compl_self, probReal_univ] at this
    have h2 := measureReal_mono (μ := runMeasure μ D Dsf)
      (show Badᶜ ⊆ {x | ((∀ j ∈ populations, j ∈ tame) → ∃ B : {B : State // B ∈ st},
          x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val)
        ∧ ∀ B : {B : State // B ∈ st},
          x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val →
          ∀ j ∈ tame, 1 - gateLimit uni εcov j - 2 * rejFrac η₀ indecisionLimit εcov
              ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                  (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
                ≤ 2 * indecisionLimit} from
        fun x hx => ⟨fun h => absurd h hall, hgood x hx⟩) (measure_ne_top _ _)
    linarith

#print axioms clustering_correct

end OrthoDFA
