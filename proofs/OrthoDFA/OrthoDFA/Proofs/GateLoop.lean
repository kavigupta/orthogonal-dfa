import OrthoDFA.Proofs.MiscutGate
import OrthoDFA.Proofs.CountRule
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
variable {εs : ℝ}

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
/-- The liveness half's obligation for the count rule, whose screen admits every candidate:
the family has the round's size unless the pool's sampler redraws too often. -/
theorem measureReal_stalled_nearest_le (O : Oracle μ S) (populations : Finset J)
    (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf] (B : State) (hkpos : 0 < B.k) (hscreen : B.scd ≤ B.sc)
    (hkM : B.k ≤ B.nsuff) (ρsf : ℝ) (rs : ℕ) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf) :
    (streamMeasure μ Pp Pc Dsf).real
        (stalled nearestClusterer O populations B (B.k - 1) (B.k - 1))
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) := by
  refine le_trans (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _))
    (measureReal_poolBad_le Pp Pc Dsf B.nsuff rs ρsf hρsf hρsf0)
  have h := stalled_subset (rule := nearestClusterer) O populations B hkpos hx
  simp only [Set.mem_setOf_eq] at h ⊢
  intro hgood
  apply h
  rw [screenedAt, screened_eq_of_le O.mq hscreen]
  have hfresh := (freshDraws_of_good ∅ (fun i => suffixDraw i x) B.nsuff rs hgood).1
  have : B.nsuff ≤ (poolAt B.nsuff x).card := by
    have := Finset.card_le_card (Finset.subset_insert (1 : S)
      (freshDraws ∅ (fun i => suffixDraw i x) B.nsuff))
    rw [hfresh] at this
    exact this
  omega

/-! ### The band -/

lemma band_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    B.k = famCount η₀ populations indecisionLimit εcov δ crossLimit + 1
      ∧ B.lo = loAt η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit)
      ∧ B.hi = hiAt η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit) := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  exact ⟨rfl, rfl, rfl⟩

lemma lohi_of_mem_schedule {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    B.lo ≤ B.hi := by
  obtain ⟨-, hlo, hhi⟩ := band_of_mem_schedule hB
  rw [hlo, hhi]
  unfold loAt hiAt
  omega

/-- A vote over a family no larger than a rung's own, whose mean lies outside the rung's band,
reaches the band's far side at most `crossLimit` of the time. -/
theorem cross_of_mem_schedule (O : Oracle μ S) {populations : Finset J} {εcov δ α pAP : ℝ}
    {B : State} (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2) (hcross : 0 < crossLimit)
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (F : Finset S) (hF : F.card + 1 ≤ B.k) (p : S) :
    (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
    ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit) := by
  obtain ⟨hBk, hBlo, hBhi⟩ := band_of_mem_schedule hB
  obtain ⟨hct, hb1⟩ := famCount_cross (indecisionLimit := indecisionLimit)
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
  have hv : voteSlack η₀ ≤ 2 / 5 := by rw [voteSlack, sig]; linarith
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

/-- The pool's count clears the log its relative tail at `9pAP/20` asks for. -/
lemma poolCount_tail (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ} (hpAP : 0 < pAP) :
    Real.log (128 / δ) / ((9 * pAP / 20) ^ 2 / (2 * pAP))
      ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
  have e : Real.log (128 / δ) / ((9 * pAP / 20) ^ 2 / (2 * pAP))
      = 800 * Real.log (128 / δ) / (81 * pAP) := by
    field_simp; ring
  rw [e]
  refine le_trans (Nat.le_ceil _) ?_
  have hle : ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊
      ≤ poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
    rw [poolCount]; omega
  exact_mod_cast hle

/-- The pool is deep enough to hold accept-preserving suffixes.  This is the one event the
ladder shares, so it is charged once against `δ/4` rather than per rung. -/
lemma solved_findability (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ}
    (hδ : 0 < δ) (hpAP : 0 < pAP) :
    Real.exp (-((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * ((9 * pAP / 20) ^ 2 / (2 * pAP)))) ≤ δ / 32 := by
  have ctap : Real.log (1 / (δ / 32)) / ((9 * pAP / 20) ^ 2 / (2 * pAP))
      ≤ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℕ) : ℝ) := by
    have hmono : Real.log (1 / (δ / 32)) ≤ Real.log (128 / δ) := by
      refine Real.log_le_log (by positivity) ?_
      rw [one_div_div, div_le_div_iff₀ hδ hδ]
      nlinarith
    exact le_trans (div_le_div_of_nonneg_right hmono (by positivity))
      (poolCount_tail η₀ populations hpAP)
  have h := tail_le_of_rate (c := 1) (by positivity)
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
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    ((famCount η₀ populations indecisionLimit εcov δ crossLimit + 1 : ℕ) : ℝ) + 1
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
      ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * (pAP - 9 * pAP / 20) := by
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set L : ℝ := 800 * Real.log (128 / δ) / (81 * pAP) with hL
  have hc : 2 * ((κ : ℝ) + 1) / pAP + L ≤ (M : ℝ) := by
    rw [hMdef, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP), Nat.le_ceil L]
  have hlog : (4 : ℝ) ≤ Real.log (128 / δ) := by
    have h1 : (128 : ℝ) ≤ 128 / δ := by
      rw [le_div_iff₀ hδ]; nlinarith
    have h2 : Real.log 128 ≤ Real.log (128 / δ) :=
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
      ↑(schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) := by
  intro B hB B' hB' h
  rw [Finset.mem_coe, schedule] at hB hB'
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  obtain ⟨i', -, rfl⟩ := Finset.mem_image.1 hB'
  simp only [solvedStateAt] at h ⊢
  rw [h]

/-- What the gate spends over the ladder's rungs is at most `α`. -/
lemma gateLevel_sum_le {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} (hα : 0 ≤ α)
    (hpop : populations.Nonempty) (s : Finset State)
    (hs : s ⊆ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
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
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
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
    (hu : 3 / 2 * indecisionLimit + rejFrac η₀ indecisionLimit εcov εs ≤ u)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (Dc j).real {t} ≤ 3 * ρ ∧ (Dp j).real {t} ≤ 3 * ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (hεs : 0 < εs) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (pAP : ℝ) (hpAP0 : 0 < pAP) (hpAP1 : pAP ≤ 1)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcoll : ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit
        + poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP
          crossLimit) : ℕ) : ℝ) ^ 2 * ρsf
        / (2 * ((poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ
          pAP crossLimit) : ℝ) + 1))
      ≤ δ / 256) :
    (streamMeasure μ Pp Pc Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov εs
            δ α pAP crossLimit hsz ρ},
          retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
            ∩ FailAt rule O populations tame uni Dc εcov
                (rejFrac η₀ indecisionLimit εcov εs) u B.val) ≤ 7 * δ / 16 + α := by
  classical
  set iLc : ℝ := indecisionLimit with hiLc
  have hiLc0 : 0 < iLc := hindLim
  have hiLc1 : iLc ≤ indecisionLimit := le_rfl
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  have hρ0 : 0 ≤ 3 * ρ := le_trans measureReal_nonneg (hρ uni huni 1).1
  set M : ℕ := poolCount η₀ populations iLc εcov δ pAP crossLimit with hMdef
  set rs : ℕ := poolRej pAP M with hrsdef
  set N : ℕ := prefCount η₀ populations iLc εcov εs δ α pAP crossLimit hsz with hNdef
  set V : ℕ := validCount η₀ populations iLc εcov δ pAP crossLimit with hVdef
  set G : Set (Run Ω S J) := apShort O M pAP (9 * pAP / 20) ∪ {x | ¬ poolGood M rs x} with hGdef
  set st := stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ with hst
  have hGle : (streamMeasure μ Pp Pc Dsf).real G ≤ δ / 16 := by
    refine le_trans (measureReal_union_le _ _) ?_
    have h1 := le_trans (measureReal_apShort_le Pp Pc Dsf O M pAP (9 * pAP / 20) hpAP0.le
      (by positivity) hpAPBound)
      (solved_findability (indecisionLimit := iLc) (crossLimit := crossLimit) η₀ populations
        (εcov := εcov) hδ hpAP0)
    have h2 := le_trans (measureReal_poolBad_le (μ := μ) Pp Pc Dsf M rs ρsf hρsf hρsf0) hcoll
    linarith
  have hper : ∀ B ∈ st, (streamMeasure μ Pp Pc Dsf).real
      ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations tame uni Dc εcov
            (rejFrac η₀ indecisionLimit εcov εs) u B) \ G)
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
    set c : ℝ := rejFrac η₀ iLc εcov εs with hcdef
    have hc0 : 0 < c := rejFrac_pos η₀ hη₀ hεcov hεs hiLc0
    set r : ℕ := rejCount η₀ iLc εcov εs B.npref with hrdef
    have hr : (r : ℝ) ≤ c * B.npref := Nat.floor_le (by positivity)
    have hG' : ∀ x ∉ G, ∃ v ∈ refsBy rule O.mq populations x B,
        ∀ p, O.label (p * v) = O.label p := by
      intro x hx
      simp only [hGdef, Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hx
      have hcount := pool_holds_ref (indecisionLimit := iLc) (crossLimit := crossLimit) η₀
        populations (εcov := εcov) hδ hδ1 hpAP0 hpAP1
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
              (rejFrac η₀ indecisionLimit εcov εs) u B.val)
      ⊆ (⋃ B ∈ st, (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
          ∩ FailAt rule O populations tame uni Dc εcov
              (rejFrac η₀ indecisionLimit εcov εs) u B) \ G) ∪ G := by
    intro x hx
    obtain ⟨B, hB⟩ := Set.mem_iUnion.1 hx
    by_cases hg : x ∈ G
    · exact Or.inr hg
    · exact Or.inl (Set.mem_biUnion B.2 ⟨hB, hg⟩)
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hNpos : (0 : ℝ) < N := by exact_mod_cast prefCount_pos _ _ _ _ _ _
  have hsumN := stoppable_npref_sum_le (εs := εs) (indecisionLimit := iLc)
      (crossLimit := crossLimit)
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
            (rejFrac η₀ indecisionLimit εcov εs) u B) \ G)
      ≤ α + 3 * δ / 8 := by
    calc _ ≤ ∑ B ∈ st, (streamMeasure μ Pp Pc Dsf).real ((retBy rule O.mq populations uni η₀
            indecisionLimit εcov α B ∩ FailAt rule O populations tame uni Dc εcov
                (rejFrac η₀ indecisionLimit εcov εs) u B)
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
    budgetScale η₀ indecisionLimit εcov / 32 ≤ cutBudget η₀ indecisionLimit εcov := by
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
  have h2 := budgetScale_le_sig η₀ indecisionLimit εcov hη0 hsig
  have h3 : budgetScale η₀ indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  rw [cutBudget]
  refine le_min (by linarith) (le_min (by linarith) (le_min (by linarith) (by nlinarith)))

/-- The count rule's budget is a thirty-second of the cut budget: `a ≥ 2·sig`. -/
lemma shiftBudget_le_cutBudget (η₀ indecisionLimit εcov : ℝ) (hind : 0 < indecisionLimit)
    (hε : 0 < εcov) (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    32 * shiftBudget η₀ indecisionLimit εcov ≤ cutBudget η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hs2 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have hβ := shiftScale_pos hε hind
  have hβm : shiftScale indecisionLimit εcov ≤ min εcov (1 / 2) := min_le_left _ _
  have hβu : shiftScale indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  have hmε : min εcov (1 / 2) ≤ εcov := min_le_left _ _
  have hm2 : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
  have ha : 2 * sig η₀ ≤ gateA η₀ := by
    have e : (1 - η₀) * gateA η₀ = 1 - 2 * η₀ := by
      have hne : (1 - η₀) ≠ 0 := by intro h; linarith
      unfold gateA; field_simp
    rw [sig]; nlinarith [gateA_pos hη0 hsig]
  have hsβ : sig η₀ ^ 2 * shiftScale indecisionLimit εcov ≤ shiftScale indecisionLimit εcov :=
    mul_le_of_le_one_left hβ.le (by nlinarith)
  rw [shiftBudget, cutBudget]
  refine le_min (by nlinarith) (le_min (by nlinarith) (le_min (by nlinarith) ?_))
  have h4 : 4 * sig η₀ ^ 2 ≤ gateA η₀ ^ 2 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h4 (le_trans hβ.le hβm)]

lemma two_sig_le_gateA {η₀ : ℝ} (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) : 2 * sig η₀ ≤ gateA η₀ := by
  have e : (1 - η₀) * gateA η₀ = 1 - 2 * η₀ := by
    have hne : (1 - η₀) ≠ 0 := by intro h; linarith
    unfold gateA; field_simp
  rw [sig]; nlinarith [gateA_pos hη0 hsig]

/-- The redraws are a hundred-and-twenty-eighth of the cut budget. -/
lemma rejFrac_le_cut (η₀ indecisionLimit εcov εs : ℝ) (hind : 0 < indecisionLimit)
    (hε : 0 < εcov) (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    rejFrac η₀ indecisionLimit εcov εs ≤ cutBudget η₀ indecisionLimit εcov / 128 := by
  obtain ⟨hcm, hcu, -⟩ := rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀
    (εcov := εcov)
  have hs := sig_pos η₀ hsig
  have hs2 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have ha := two_sig_le_gateA hη0 hsig
  have hmn0 : 0 < min εcov (1 / 2) := lt_min hε (by norm_num)
  have hm1 : min εcov (1 / 2) ≤ εcov := min_le_left _ _
  have hm2 : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
  have hs2' : sig η₀ ^ 2 ≤ 1 / 4 := by nlinarith
  rw [cutBudget, le_div_iff₀ (by norm_num), le_min_iff, le_min_iff, le_min_iff]
  refine ⟨by nlinarith, by nlinarith, by nlinarith, ?_⟩
  have h4 : 4 * sig η₀ ^ 2 ≤ gateA η₀ ^ 2 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h4 hmn0.le]

/-- The redraw fraction is at least a constant times the signal times the smaller of the
coverage times the signal and the FNR test's limit, or a quarter of `εs`. -/
lemma rejFrac_ge_lin (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hε1 : εcov ≤ 1)
    (hind : 0 < indecisionLimit) (hsig : η₀ < 1 / 2) :
    min (sig η₀ * min (εcov * sig η₀) indecisionLimit / 2048) (εs / 4)
      ≤ rejFrac η₀ indecisionLimit εcov εs := by
  have hs := sig_pos η₀ hsig
  have h1 : min (εcov * sig η₀) indecisionLimit ≤ εcov * sig η₀ := min_le_left _ _
  have h2 : min (εcov * sig η₀) indecisionLimit ≤ indecisionLimit := min_le_right _ _
  have hm : εcov / 2 ≤ min εcov (1 / 2) := le_min (by linarith) (by linarith)
  rw [rejFrac]
  refine min_le_min (le_min ?_ ?_) le_rfl
  · have := mul_le_mul_of_nonneg_left h1 hs.le
    have := mul_le_mul_of_nonneg_left hm (sq_nonneg (sig η₀))
    nlinarith
  · have := mul_le_mul_of_nonneg_left h2 hs.le
    have : 0 ≤ sig η₀ * indecisionLimit := by positivity
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

/-- The top rung's tails, at `δ/(128·|populations|)`; the pool's, which every population shares,
at `δ/128`. -/
lemma solved_tails (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
        * ((9 * pAP / 20) ^ 2 / (2 * pAP))))
        ≤ δ / 128
    ∧ Real.exp (-((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * (shiftBudget η₀ indecisionLimit εcov / 1056)))
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * (indecisionLimit / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ)) := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < shiftBudget η₀ indecisionLimit εcov := shiftBudget_pos η₀ hsig hε hind
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- each count clears the logarithm its tail asks for
  have ctap : Real.log (1 / (δ / 128)) / ((9 * pAP / 20) ^ 2 / (2 * pAP)) ≤ (M : ℝ) := by
    rw [one_div_div]
    exact poolCount_tail η₀ populations hpAP
  have cth2 : Real.log (128 * (populations.card : ℝ) / δ)
      * (1056 / shiftBudget η₀ indecisionLimit εcov + 2 / indecisionLimit ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ)
        * (1056 / shiftBudget η₀ indecisionLimit εcov + 2 / indecisionLimit ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have heq1 : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp
  have cth : Real.log (1 / ε₀) / (shiftBudget η₀ indecisionLimit εcov / 1056) ≤ (m : ℝ) := by
    rw [heq1, div_div_eq_mul_div, mul_div_assoc]
    rcases le_or_gt 0 (Real.log (128 * (populations.card : ℝ) / δ)) with hlog0 | hlog0
    · have : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) * (2 / indecisionLimit ^ 2) := by
        positivity
      nlinarith [cth2]
    · have : Real.log (128 * (populations.card : ℝ) / δ) * (1056 / shiftBudget η₀
        indecisionLimit εcov) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hlog0.le (by positivity)
      linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have cund : Real.log (1 / ε₀) / (2 * (indecisionLimit / 2) ^ 2) ≤ (m : ℝ) := by
    rw [heq1, show 2 * (indecisionLimit / 2) ^ 2 = indecisionLimit ^ 2 / 2 by ring,
      div_div_eq_mul_div, mul_div_assoc]
    rcases le_or_gt 0 (Real.log (128 * (populations.card : ℝ) / δ)) with hlog0 | hlog0
    · have : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) * (1056 / shiftBudget η₀
        indecisionLimit εcov) := by positivity
      nlinarith [cth2]
    · have : Real.log (128 * (populations.card : ℝ) / δ) * (2 / indecisionLimit ^ 2) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg hlog0.le (by positivity)
      linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  -- the tails those counts kill
  have ttap : Real.exp (-((M : ℝ) * ((9 * pAP / 20) ^ 2 / (2 * pAP)))) ≤ δ / 128 := by
    have h := tail_le_of_rate (c := 1) (by positivity)
      (by positivity : (0 : ℝ) < δ / 128) (by norm_num) ctap
    linarith
  have tth : Real.exp (-((m : ℝ) * (shiftBudget η₀ indecisionLimit εcov / 1056))) ≤ ε₀ := by
    have h := tail_le_of_rate (a := shiftBudget η₀ indecisionLimit εcov / 1056) (c := 1)
      (by positivity) hε₀ (by norm_num) cth
    linarith
  have tund : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := indecisionLimit / 2) (c := 1) (by positivity) hε₀
      (by norm_num) cund
    linarith
  exact ⟨ttap, tth, tund⟩

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
        ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊]
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
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (m : ℕ) (hm : m ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    rejTail (m + rejCount η₀ indecisionLimit εcov εs m)
        (((m + rejCount η₀ indecisionLimit εcov εs m : ℕ) : ℝ) * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov εs m)
      ≤ (m : ℝ) / (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * (1 / 2) ^ (rejCount η₀ indecisionLimit εcov εs m + 1)
    ∧ rejTail (2 * m + rejCount η₀ indecisionLimit εcov εs m)
        (((populations.card * m + (2 * m + rejCount η₀ indecisionLimit εcov εs m) : ℕ) : ℝ)
          * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov εs m)
      ≤ (m : ℝ) / (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * (1 / 2) ^ (rejCount η₀ indecisionLimit εcov εs m + 1) := by
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hcdef
  set r : ℕ := rejCount η₀ indecisionLimit εcov εs m with hrdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hεs hind
  have hc4 : c ≤ 1 / 4 := by
    have hcle := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀ (εcov := εcov)).1
    have h1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    have h2 : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
    have h3 := sig_pos η₀ hsig
    have h4 : 0 ≤ min εcov (1 / 2) := le_min hε.le (by norm_num)
    nlinarith [mul_le_mul h1 h1 h3.le (by norm_num : (0 : ℝ) ≤ 1 / 2)]
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
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (m : ℕ) (hm : m ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov εs * m * Real.log 2) :
    (populations.card : ℝ) * (rejTail (m + rejCount η₀ indecisionLimit εcov εs m)
        (((m + rejCount η₀ indecisionLimit εcov εs m : ℕ) : ℝ) * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov εs m)
      + rejTail (2 * m + rejCount η₀ indecisionLimit εcov εs m)
        (((populations.card * m + (2 * m + rejCount η₀ indecisionLimit εcov εs m) : ℕ) : ℝ)
          * (3 * ρ))
        (rejCount η₀ indecisionLimit εcov εs m))
      ≤ δ * m / (16 * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ))
      := by
  obtain ⟨h1, h2⟩ := rung_tails_le (crossLimit := crossLimit) η₀ populations (δ := δ) (α := α)
    (pAP := pAP) hη0 hsig hε hεs hind hρ0 hρcap m hm
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hcdef
  set r : ℕ := rejCount η₀ indecisionLimit εcov εs m with hrdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hNpos : (0 : ℝ) < N := by
    rw [hNdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hεs hind
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
times `log(256|J|/δ)`, the count's own tail. -/
lemma solved_rej_log (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.log (256 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov εs
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * Real.log 2 := by
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hsig hε hind
  have hs := sig_pos η₀ hsig
  have hs2 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  -- a named population's count, `64·log(128|J|(M + 1)/δ)/shiftBudget`, is many times this
  set L' : ℝ := Real.log (128 * P * ((M : ℝ) + 1) / δ) with hL'def
  have c1 : 64 * L' / Δ ≤ (N : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * L' / Δ⌉₊ ≤ N := by
      rw [hNdef, prefCount, ← hMdef, ← hΔdef, ← hP, ← hL'def]; omega
    exact_mod_cast hle
  rw [div_le_iff₀ hΔ0] at c1
  have hM1 : (1 : ℝ) ≤ M := by
    have h : 0 < ⌈2 * ((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
        / pAP⌉₊ := Nat.ceil_pos.2 (by positivity)
    have : 1 ≤ M := by rw [hMdef, poolCount]; omega
    exact_mod_cast this
  have hLL : Real.log (256 * P / δ) ≤ L' :=
    Real.log_le_log (by positivity) (by
      rw [div_le_div_iff_of_pos_right hδ]; nlinarith)
  have hL0 : 0 ≤ Real.log (256 * P / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  -- and the redraws' term, `6·log(256|J|/δ)/εs`, clears it when `rejFrac = εs/4`
  have c2 : 6 * Real.log (256 * P / δ) / εs ≤ (N : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈6 * Real.log (256 * P / δ) / εs⌉₊ ≤ N := by
      rw [hNdef, prefCount, ← hP]; omega
    exact_mod_cast hle
  rw [div_le_iff₀ hεs] at c2
  -- the first two branches are each at least a quarter of the count rule's budget
  have hβm : shiftScale indecisionLimit εcov ≤ min εcov (1 / 2) := min_le_left _ _
  have hβu : shiftScale indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  have hβ0 := shiftScale_pos hε hind
  have hb1 : Δ / 4 ≤ sig η₀ ^ 2 * min εcov (1 / 2) / 1024 := by
    rw [hΔdef, shiftBudget]
    nlinarith [mul_le_mul_of_nonneg_left hβm (sq_nonneg (sig η₀))]
  have hb2 : Δ / 4 ≤ sig η₀ * indecisionLimit / 128 := by
    rw [hΔdef, shiftBudget]
    have h1 : sig η₀ ^ 2 * shiftScale indecisionLimit εcov ≤ sig η₀ * indecisionLimit := by
      have : sig η₀ ^ 2 ≤ sig η₀ := by nlinarith
      exact mul_le_mul this hβu hβ0.le hs.le
    have : 0 ≤ sig η₀ * indecisionLimit := by positivity
    linarith
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
  rw [rejFrac, min_mul_of_nonneg _ _ hN0, min_mul_of_nonneg _ _ (by linarith : (0 : ℝ) ≤ Real.log 2),
    min_mul_of_nonneg _ _ hN0, min_mul_of_nonneg _ _ (by linarith : (0 : ℝ) ≤ Real.log 2)]
  have hq : 16 * L' ≤ Δ / 4 * N := by nlinarith
  refine le_min (le_min ?_ ?_) ?_
  · have h := mul_le_mul_of_nonneg_right hb1 hN0
    nlinarith [mul_le_mul_of_nonneg_left hl2.le (by linarith : (0 : ℝ) ≤ Δ / 4 * N)]
  · have h := mul_le_mul_of_nonneg_right hb2 hN0
    nlinarith [mul_le_mul_of_nonneg_left hl2.le (by linarith : (0 : ℝ) ≤ Δ / 4 * N)]
  · have hsN : 0 ≤ εs * N := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hl2.le hsN]

/-- The top rung carries its own share of the error budget, `δ/8`. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    stateFail η₀ populations indecisionLimit εcov εs (3 * ρ) (solvedState η₀ populations
      indecisionLimit εcov εs δ α pAP crossLimit hsz)
      ≤ δ / 8 := by
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 0 < populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  obtain ⟨-, -, tund⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  have hBm :
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hlog := solved_rej_log (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀ populations
    hη0 hsig hε hεs hind hδ hδ1 hpAP hcard1
  have hlog32 : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov εs * m * Real.log 2 :=
    le_trans (Real.log_le_log (by positivity)
      (div_le_div_of_nonneg_right (by linarith) hδ.le)) hlog
  have hdr := rung_redraws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hεs hind hδ
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
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (i : ℕ) (hi : i < ladderLen η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hV : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit
      ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz / 2 ^ i)
    (hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov εs
        * ((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz / 2 ^ i : ℕ)
          : ℝ) * Real.log 2) :
    solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
        (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz / 2 ^ i)
      ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set V : ℕ := validCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hVdef
  set m : ℕ := N / 2 ^ i with hmdef
  have hB : solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m
      ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
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
  show stateFail η₀ populations indecisionLimit εcov εs (3 * ρ)
      (solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit m)
    ≤ δ * (m : ℝ) / (8 * N) + if V ≤ m then δ * (V : ℝ) / (16 * m) else 0
  rw [if_pos hV, stateFail, show (solvedStateAt η₀ populations indecisionLimit εcov δ pAP
    crossLimit m).npref = m from rfl]
  have hdr := rung_redraws_le (crossLimit := crossLimit) η₀ populations hη0 hsig hε hεs hind hδ
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
noncomputable def rejCountMin (η₀ : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ : ℝ) : ℕ :=
  ⌈Real.log (32 * (populations.card : ℝ) / δ)
    / (rejFrac η₀ indecisionLimit εcov εs * Real.log 2)⌉₊

/-- Some rung the loop may stop at has fewer than twice the larger of `validCount` and
`rejCountMin` prefixes: the lowest rung whose count is still at or above both, or any rung when
the top is already below them. -/
theorem exists_small_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hne : (stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ).Nonempty) :
    ∃ B ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ,
      (B.npref : ℝ) ≤ 2 * (max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
        (rejCountMin η₀ populations indecisionLimit εcov εs δ) : ℝ) := by
  classical
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set V : ℕ := max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
    (rejCountMin η₀ populations indecisionLimit εcov εs δ) with hVdef
  have hVpos : 0 < V := by
    have : 0 < validCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
      rw [validCount]; omega
    exact lt_of_lt_of_le this (le_max_left _ _)
  have hcast : ((max (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
      (rejCountMin η₀ populations indecisionLimit εcov εs δ) : ℕ) : ℝ) = (V : ℝ) := rfl
  rw [show (max ((validCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℕ) : ℝ)
      ((rejCountMin η₀ populations indecisionLimit εcov εs δ : ℕ) : ℝ)) = (V : ℝ) by
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
  have hiL : i < ladderLen η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
    have hpow : 2 ^ i ≤ N := by
      have h1 : 1 ≤ N / 2 ^ i := le_trans hVpos hi0
      exact (Nat.one_le_div_iff (by positivity)).1 h1
    have := Nat.le_log_of_pow_le (by norm_num) hpow
    rw [ladderLen, ← hNdef]
    omega
  have hval : validCount η₀ populations indecisionLimit εcov δ pAP crossLimit ≤ N / 2 ^ i :=
    le_trans (le_max_left _ _) hi0
  have hlog : Real.log (32 * (populations.card : ℝ) / δ)
      ≤ rejFrac η₀ indecisionLimit εcov εs * ((N / 2 ^ i : ℕ) : ℝ) * Real.log 2 := by
    have hc0 : 0 < rejFrac η₀ indecisionLimit εcov εs := rejFrac_pos η₀ hsig hε hεs hind
    have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    have hmin : rejCountMin η₀ populations indecisionLimit εcov εs δ ≤ N / 2 ^ i :=
      le_trans (le_max_right _ _) hi0
    have hceil : Real.log (32 * (populations.card : ℝ) / δ)
        / (rejFrac η₀ indecisionLimit εcov εs * Real.log 2) ≤ ((N / 2 ^ i : ℕ) : ℝ) :=
      le_trans (Nat.le_ceil _) (by rw [rejCountMin] at hmin; exact_mod_cast hmin)
    rw [div_le_iff₀ (by positivity)] at hceil
    linarith
  refine ⟨_, rung_mem_stoppable η₀ populations hη0 hsig hε hεs hind hδ hδ1 hpAP hcard hρ0 hρcap
    i hiL hval hlog, ?_⟩
  show ((N / 2 ^ i : ℕ) : ℝ) ≤ 2 * (V : ℝ)
  exact_mod_cast hlt.le

/-- The top rung can always be stopped at: it carries the whole proportional share. -/
theorem solvedState_mem_stoppable (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (0 : ℝ) < (populations.card : ℝ))
    (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) :
    solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
      ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ := by
  classical
  have hB := solvedState_mem_schedule (εs := εs) (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) (hsz := hsz) η₀ populations εcov δ α pAP
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  have hmpos : 0 < m := prefCount_pos (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  rw [stoppable]
  refine Finset.mem_filter.2 ⟨hB, ⟨lohi_of_mem_schedule hB, hmpos,
    poolCount_found η₀ populations hsig hpAP, ?_⟩⟩
  show stateFail η₀ populations indecisionLimit εcov εs (3 * ρ)
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    ≤ δ * (m : ℝ) / (8 * m) + _
  have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by field_simp
  rw [hval]
  exact le_add_of_le_of_nonneg (solved_share η₀ populations hsig hη0 hε hεs hind hδ hδ1 hpAP hcard
    hρ0 hρcap) (by split_ifs <;> positivity)

open scoped Classical in
/-- At a fixed table the state's two light counts on a sample are within budget: the prefixes the
family is usable at that its vote leaves undecided, and those it cuts wrongly, each priced by
`frac_selected_le`. -/
theorem light_counts_whp {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (lo hi : ℕ) (l lmis : ℝ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (hone : (1 : S) ∈ cands) (hT1 : ∀ t ∈ T, (1 : S) ∉ t)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l) (hElm : E ≤ lmis)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo (hi + 1) (insert 1 A₀) p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo hi A₀ p ω} ≤ E) :
    μ.real {ω | ¬ ((((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
              ≤ l * (C.card : ℝ))
            ∧ ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
              ≤ lmis * (C.card : ℝ))}
      ≤ Real.exp (-((C.card : ℝ) * relRate l E))
        + Real.exp (-((C.card : ℝ) * relRate lmis E)) := by
  classical
  have hsub : {ω | ¬ ((((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
              ≤ l * (C.card : ℝ))
            ∧ ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)
              ≤ lmis * (C.card : ℝ))}
      ⊆ {ω | l * (C.card : ℝ)
            < ((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq lo (hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)}
        ∪ {ω | lmis * (C.card : ℝ)
              < ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo hi (fam ω) p ω)).card : ℝ)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_and_or, not_le] at hω
    exact hω
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · exact indecision_frac_le_seed hflat O P cands C hP hV hCPre hPC hone lo (hi + 1) T good t₀ ht₀
      hTC hT1 fam hfam hcongr E l hE hEl hdec
  · exact miscut_frac_le hflat O P cands C hP hV hCPre hPC lo hi T good t₀ ht₀ hTC
      fam hfam hcongr E lmis hE hElm hcut

open scoped Classical in
/-- The draws were good, and still population `j`'s sample leaves more than `l` of it undecided,
or cuts more than `lmis` of it wrongly, among the prefixes the family is usable at. -/
noncomputable def lightMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (j : J)
    (B : State) (l lmis : ℝ) (f : S → ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x)
      (certOf (prefixesAt populations B.npref x) j B.npref x)
    ∧ (certOf (prefixesAt populations B.npref x) j B.npref x).card = B.npref
    ∧ 0 < (certOf (prefixesAt populations B.npref x) j B.npref x).card
    ∧ ¬ ((((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
            famGood O f kmin kmax (clusterBy rule O.mq populations x B) p
              ∧ ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
                (oracleNoise x))).card : ℝ)
          ≤ l * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
        ∧ (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
            famGood O f kmin kmax (clusterBy rule O.mq populations x B) p
              ∧ ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
                (oracleNoise x))).card : ℝ)
          ≤ lmis * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ))}

open scoped Classical in
lemma measurableSet_lightMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (l lmis : ℝ) (f : S → ℝ) (kmin kmax : ℕ) :
    MeasurableSet (lightMiss rule O populations j B l lmis f kmin kmax) := by
  classical
  set Φ : Finset S → Finset S → Finset S → Ω → Prop := fun P C A ω =>
    ¬ (((A.filter (fun p => famGood O f kmin kmax (clusterOf rule O B.sc B.scd P C B.k ω) p
            ∧ ¬ decided O.mq B.lo (B.hi + 1)
              (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) p ω)).card : ℝ)
          ≤ l * (A.card : ℝ)
        ∧ ((A.filter (fun p => famGood O f kmin kmax (clusterOf rule O B.sc B.scd P C B.k ω) p
            ∧ ¬ cutCorrect O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω) p ω)).card : ℝ)
          ≤ lmis * (A.card : ℝ)) with hΦ
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
            ¬ (((A.filter (fun p => famGood O f kmin kmax A₀ p
                  ∧ ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card : ℝ)
                ≤ l * (A.card : ℝ)
              ∧ ((A.filter (fun p => famGood O f kmin kmax A₀ p
                  ∧ ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
                ≤ lmis * (A.card : ℝ))})
          (fun A₀ => ?_)
        have hd : MeasurableSet[noiseAlg O Set.univ] {ω | ((A.filter (fun p =>
            famGood O f kmin kmax A₀ p ∧ ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card
              : ℝ) ≤ l * (A.card : ℝ)} := by
          refine measurableSet_filter_pred' O
            (fun p ω => famGood O f kmin kmax A₀ p ∧ ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)
            (fun p _ => ?_) (fun U => ((U.card : ℝ) ≤ l * (A.card : ℝ)))
          by_cases hg : famGood O f kmin kmax A₀ p
          · simpa [hg] using measurableSet_decided' O B.lo (B.hi + 1) (insert 1 A₀) p
          · simp [hg]
        have hc : MeasurableSet[noiseAlg O Set.univ] {ω | ((A.filter (fun p =>
            famGood O f kmin kmax A₀ p ∧ ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ)
              ≤ lmis * (A.card : ℝ)} := by
          refine measurableSet_filter_pred' O
            (fun p ω => famGood O f kmin kmax A₀ p ∧ ¬ cutCorrect O B.lo B.hi A₀ p ω)
            (fun p _ => ?_) (fun U => ((U.card : ℝ) ≤ lmis * (A.card : ℝ)))
          by_cases hg : famGood O f kmin kmax A₀ p
          · simpa [hg] using measurableSet_cutCorrect' O B.lo B.hi A₀ p
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
  have hrw : lightMiss rule O populations j B l lmis f kmin kmax
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
    (l lmis E : ℝ) (f : S → ℝ) (kmin kmax : ℕ)
    (hE : 0 ≤ E) (hEl : E ≤ l) (hElm : E ≤ lmis)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f p →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f p →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (streamMeasure μ Pp Pc Dsf).real (lightMiss rule O populations j B l lmis f kmin kmax)
      ≤ Real.exp (-((B.npref : ℝ) * relRate l E))
        + Real.exp (-((B.npref : ℝ) * relRate lmis E)) := by
  classical
  set R : ℝ := Real.exp (-((B.npref : ℝ) * relRate l E))
    + Real.exp (-((B.npref : ℝ) * relRate lmis E)) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : streamMeasure μ Pp Pc Dsf (lightMiss rule O populations j B l lmis f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine streamMeasure_slice_le Pp Pc Dsf _
      (measurableSet_lightMiss O populations j B l lmis f kmin kmax) _ ?_
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
        flipCount O t p ≤ (t.card : ℝ) * f p ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
          hCx ω ▸ clusterAt_subset O populations B _ hv⟩)
      have hT1 : ∀ t ∈ T, (1 : S) ∉ t := fun t ht h1 =>
        (Finset.mem_erase.1 (Finset.mem_powerset.1 ht h1)).1 rfl
      have hmain := light_counts_whp hflat O Pd Cd Ad hP hV hA hdisj B.lo B.hi l lmis
        T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => (Finset.mem_powerset.1 ht).trans (Finset.erase_subset _ _))
        (Finset.mem_insert_self _ _) hT1 fam hfamT
        (fun ω ω' h => clusterAt_congr O populations B d (by rw [hPx ω, hCx ω]; exact h))
        E hE hEl hElm
        (fun p _ A₀ hA₀ hgp => hdec A₀ p (hT1 A₀ hA₀) (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
      have hgoodiff : ∀ ω p, fam ω ∈ good p ↔ famGood O f kmin kmax (fam ω) p := by
        intro ω p
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
        exact Iff.rfl
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ lightMiss rule O populations j B l lmis f kmin kmax}
          ⊆ {ω : Ω | ¬ ((((Ad.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (fam ω)) p ω)).card : ℝ)
              ≤ l * (Ad.card : ℝ))
            ∧ ((Ad.filter (fun p => fam ω ∈ good p
                ∧ ¬ cutCorrect O B.lo B.hi (fam ω) p ω)).card : ℝ)
              ≤ lmis * (Ad.card : ℝ))} := by
        rintro ω ⟨-, -, -, hbad⟩
        rw [hAx ω] at hbad
        simp only [hgoodiff]
        exact hbad
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq (by rw [hcardR])))
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ lightMiss rule O populations j B l lmis f kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        rw [hAx ω, hPx ω] at h1
        rw [hAx ω] at h2 h3
        exact hdraw ⟨h1, h2, h3⟩
      simp [hsec]
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (lightMiss rule O populations j B l lmis f kmin kmax)).toReal
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
/-- A sample found in time, of a population whose `bad` prefixes have mass at most `q` and whose
draws do not hold too many of them, holds at most `m(q + th) + r` of them: the redraws hide
`r`. -/
lemma heavy_le_of_clean (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    [IsProbabilityMeasure Dj] (j : J) (B : State) (x : Run Ω S J) (bad : Finset S → S → Prop)
    {q th : ℝ} {r : ℕ} (hgood : certGood populations j B.npref r x)
    (hmass : Dj.real {p | bad (clusterBy rule O.mq populations x B) p} ≤ q)
    (hhits : x ∉ heavyHits rule O populations Dj j B bad q th) :
    (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        bad (clusterBy rule O.mq populations x B) p)).card : ℝ)
      ≤ B.npref * (q + th) + r := by
  classical
  obtain ⟨-, hcnt⟩ := certOf_of_good populations j B.npref r x hgood
  have hfew : (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
      bad (clusterBy rule O.mq populations x B) (certPrefix j i.val x))).card : ℝ)
      < (B.npref : ℝ) * (q + th) := by
    by_contra h
    exact hhits ⟨hmass, not_lt.1 h⟩
  have hH := (hcnt (fun p => bad (clusterBy rule O.mq populations x B) p)).2
  linarith

/-- A class's heavy mass at the round's table budget `Δ`: Markov at the class's `σ_b` of the shift
the family carries on it, the members' share `Δ`, the redraws' `c·w_b` and the understatement
`g`. -/
noncomputable def clsQ (O : Oracle μ S) (η₀ Δ c g : ℝ) (b : Bool) : ℝ :=
  (Δ + c * clsShift O b + g) / clsSig O η₀ b

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- A filled sample outside the tables passes the FNR test and the gate unless a light count
misses, a side is dead or the pool is short, once each class's bad prefixes `M b` fit the FNR
test's share `lF` and, charged at that class's rate, the gate's share `lG`. -/
lemma retAt_of_light (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (j : J) (B : State) (x : Run Ω S J) (hmpos : 0 < B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lF lG lmis t : ℝ) (M : Bool → ℝ) (r rs kmin kmax : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hlF : 2 * lF + l ≤ indecisionLimit) (hlmis : 0 ≤ lmis) (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t)
    (hkpos : 0 < B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      4 * (2 * lG + 2 * lmis / gateA η₀ ^ 2) * B.npref
          + 4 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hsize : kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax)
    (hCn : (certOf (prefixesAt populations B.npref x) j B.npref x).card = B.npref)
    (hM : ∀ b, (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        clsBadP O η₀ b (clusterBy rule O.mq populations x B) p)).card : ℝ) ≤ M b)
    (hMF : ∀ b, M b ≤ lF * B.npref)
    (hMG : ∀ b, M b * sideCharge O η₀ (!b) ≤ lG * B.npref)
    (h8 : x ∉ lightMiss rule O populations j B l lmis (fracAt O η₀) kmin kmax)
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
  set good : S → Prop := fun p => famGood O (fracAt O η₀) kmin kmax F p with hgood
  have hCpos : 0 < C.card := by rw [hCn]; exact hmpos
  have hl8 : ((C.filter (fun p => good p ∧ ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) p ω)).card : ℝ) ≤ l * (C.card : ℝ)
      ∧ ((C.filter (fun p => good p ∧ ¬ cutCorrect O B.lo B.hi F p ω)).card : ℝ)
        ≤ lmis * (C.card : ℝ) := by
    by_contra hcon
    exact h8 ⟨(disjoint_freshDraws _ _ _).symm, hCn, hCpos, hcon⟩
  rw [hCn] at hl8
  -- off the usable prefixes a prefix is bad for its own class
  have hbad : ∀ p, ¬ good p → clsBadP O η₀ true F p ∨ clsBadP O η₀ false F p := by
    intro p hp
    have hfl : ¬ flipCount O F p ≤ F.card * fracAt O η₀ p := fun h => hp ⟨h, hsize⟩
    rcases O.label_bit p with hl | hl
    · exact Or.inr ⟨by simpa [sideClass] using hl, hfl⟩
    · exact Or.inl ⟨by simpa [sideClass] using hl, hfl⟩
  -- the FNR test
  have hundec : ((C.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
      (familyBy rule O.mq populations x B) p ω)).card : ℝ) ≤ indecisionLimit * n := by
    have hsub : C.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (familyBy rule O.mq populations x B) p ω)
        ⊆ (C.filter (fun p => good p ∧ ¬ decided O.mq B.lo (B.hi + 1)
            (familyBy rule O.mq populations x B) p ω))
          ∪ ((C.filter (fun p => clsBadP O η₀ true F p))
            ∪ (C.filter (fun p => clsBadP O η₀ false F p))) := by
      intro p hp
      obtain ⟨hpC, hq⟩ := Finset.mem_filter.1 hp
      by_cases hg : good p
      · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
      · rcases hbad p hg with h | h
        · exact Finset.mem_union_right _ (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, h⟩))
        · exact Finset.mem_union_right _ (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, h⟩))
    have hc := le_trans (Finset.card_le_card hsub)
      (le_trans (Finset.card_union_le _ _) (Nat.add_le_add_left (Finset.card_union_le _ _) _))
    have hcR : ((C.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) p ω)).card : ℝ)
        ≤ ((C.filter (fun p => good p ∧ ¬ decided O.mq B.lo (B.hi + 1)
            (familyBy rule O.mq populations x B) p ω)).card : ℝ)
          + (((C.filter (fun p => clsBadP O η₀ true F p)).card : ℝ)
            + ((C.filter (fun p => clsBadP O η₀ false F p)).card : ℝ)) := by
      exact_mod_cast hc
    have h1 := le_trans (hM true) (hMF true)
    have h2 := le_trans (hM false) (hMF false)
    have := mul_le_mul_of_nonneg_right hlF hnR.le
    linarith
  -- each class's miscut: its bad prefixes, and the usable ones the vote misfires on
  have hmis : ∀ b, ((C.filter (fun p => O.label p = sideClass b
      ∧ ¬ cutCorrect O B.lo B.hi F p ω)).card : ℝ) ≤ M b + lmis * n := by
    intro b
    have hsub : C.filter (fun p => O.label p = sideClass b ∧ ¬ cutCorrect O B.lo B.hi F p ω)
        ⊆ (C.filter (fun p => clsBadP O η₀ b F p))
          ∪ (C.filter (fun p => good p ∧ ¬ cutCorrect O B.lo B.hi F p ω)) := by
      intro p hp
      obtain ⟨hpC, hl, hq⟩ := Finset.mem_filter.1 hp
      by_cases hg : good p
      · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
      · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hl,
          fun h => hg ⟨h, hsize⟩⟩)
    have hc := le_trans (Finset.card_le_card hsub) (Finset.card_union_le _ _)
    have hcR : ((C.filter (fun p => O.label p = sideClass b
        ∧ ¬ cutCorrect O B.lo B.hi F p ω)).card : ℝ)
        ≤ ((C.filter (fun p => clsBadP O η₀ b F p)).card : ℝ)
          + ((C.filter (fun p => good p ∧ ¬ cutCorrect O B.lo B.hi F p ω)).card : ℝ) := by
      exact_mod_cast hc
    linarith [hM b, hl8.2]
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
  have ha : 0 < gateA η₀ := gateA_pos (le_trans (eta_nonneg O) hηle) hη₀
  have hcharge : ∀ b, (M b + lmis * n) * sideCharge O η₀ (!b)
      ≤ lG * n + lmis / gateA η₀ ^ 2 * n := by
    intro b
    have h1 := sideCharge_le_gateA O hηle hη₀ (!b)
    have h2 : lmis * n * sideCharge O η₀ (!b) ≤ lmis * n * (1 / gateA η₀ ^ 2) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have e : lmis * n * (1 / gateA η₀ ^ 2) = lmis / gateA η₀ ^ 2 * n := by ring
    rw [add_mul]
    linarith [hMG b]
  have hgate' : 4 * ((M false + lmis * n) * sideCharge O η₀ true
        + (M true + lmis * n) * sideCharge O η₀ false)
        + 4 * ((gateDevT η₀ + gateDevU η₀) * t
          + gateDevD η₀ * hoeffDev n (sizeLevel (gateLevel α populations B) n))
      ≤ n * gateLimit uni εcov j - hoeffDev n (sizeLevel (gateLevel α populations B) n) := by
    have := mul_le_mul_of_nonneg_left hlim (Nat.cast_nonneg n : (0 : ℝ) ≤ _)
    simp only [sub_self, add_zero] at hgateN
    have hf := hcharge false
    have ht' := hcharge true
    simp only [Bool.not_false, Bool.not_true] at hf ht'
    have hsum : 4 * ((M false + lmis * n) * sideCharge O η₀ true
        + (M true + lmis * n) * sideCharge O η₀ false)
        ≤ 4 * (2 * lG + 2 * lmis / gateA η₀ ^ 2) * n := by
      have e : 4 * (2 * lG + 2 * lmis / gateA η₀ ^ 2) * n
          = 4 * ((lG * n + lmis / gateA η₀ ^ 2 * n) + (lG * n + lmis / gateA η₀ ^ 2 * n)) := by
        ring
      rw [e]; linarith
    linarith
  have hga := gateLevel_pos_le populations huni B hα0 hα1
  have hcert := certifiesAt_of_live_rates O hηle hη₀ B.lo B.hi F R C hRne
    (fun v hv p _ => hne1 v hv p) hmpos hCn (lim := gateLimit uni εcov j)
    (a := gateLevel α populations B) (t := t) (fun b => M b + lmis * n) hga.1 hga.2 ht hlive
    hmis hgate'
  refine ⟨?_, ?_, hcert⟩
  · have : B.k ≤ F.card + 1 := by omega
    exact this
  · rw [hCn]
    exact hundec

open scoped Classical in
/-- Where the round fails at the uniform pool: the certification sample not found in time, a
family of the wrong size, its shift on a class not read off its draws, too many of a class's
heavy prefixes on the sample, a light count missed, a dead side, or a short pool. -/
lemma notRetAt_subset (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (j : J) (Dj : Measure S) [IsProbabilityMeasure Dj] (B : State)
    (hmpos : 0 < B.npref) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α)
    (hα1 : α ≤ 1) (l lF lG lmis Δ c g t : ℝ) (r rs kmin kmax : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hlF : 2 * lF + l ≤ indecisionLimit) (hlmis : 0 ≤ lmis) (hΔ : 0 ≤ Δ) (hc : 0 ≤ c)
    (hg : 0 ≤ g) (hr : (r : ℝ) ≤ c * B.npref)
    (hUF : ∀ b, clsQ O η₀ Δ c g b * (33 / 32) + c ≤ lF)
    (hUG : ∀ b, sideCharge O η₀ (!b) * (clsQ O η₀ Δ c g b * (33 / 32) + c) ≤ lG)
    (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 0 < B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      4 * (2 * lG + 2 * lmis / gateA η₀ ^ 2) * B.npref
          + 4 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N)) :
    {x : Run Ω S J | x ∉ retAt rule O populations uni η₀ indecisionLimit εcov α B j}
      ⊆ {x | ¬ certGood populations j B.npref r x} ∪ stalled rule O populations B kmin kmax
        ∪ ({x | ¬ ∑ v ∈ clusterBy rule O.mq populations x B, wMass O Dj (clsW O true) v
              ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * (Δ + c * clsShift O true + g)}
          ∪ {x | ¬ ∑ v ∈ clusterBy rule O.mq populations x B, wMass O Dj (clsW O false) v
              ≤ ((clusterBy rule O.mq populations x B).card : ℝ)
                * (Δ + c * clsShift O false + g)})
        ∪ (heavyHits rule O populations Dj j B (clsBadP O η₀ true) (clsQ O η₀ Δ c g true)
              (clsQ O η₀ Δ c g true / 32)
          ∪ heavyHits rule O populations Dj j B (clsBadP O η₀ false) (clsQ O η₀ Δ c g false)
              (clsQ O η₀ Δ c g false / 32))
        ∪ lightMiss rule O populations j B l lmis (fracAt O η₀) kmin kmax
        ∪ sidesDead rule O populations j B B.npref t ∪ {x | ¬ poolGood B.nsuff rs x} := by
  classical
  intro x hx
  by_contra hcon
  simp only [Set.mem_union, not_or] at hcon
  obtain ⟨⟨⟨⟨⟨⟨h1, h5⟩, ⟨h6t, h6f⟩⟩, ⟨h7t, h7f⟩⟩, h8⟩, h10⟩, h11⟩ := hcon
  simp only [Set.mem_setOf_eq, not_not] at h1 h6t h6f h11
  have h5' : kmin ≤ (clusterBy rule O.mq populations x B).card
      ∧ (clusterBy rule O.mq populations x B).card ≤ kmax := by
    by_contra h; exact h5 h
  obtain ⟨hCn, -⟩ := certOf_of_good populations j B.npref r x h1
  have hmR : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
  have hcnt : ∀ b, x ∉ heavyHits rule O populations Dj j B (clsBadP O η₀ b)
      (clsQ O η₀ Δ c g b) (clsQ O η₀ Δ c g b / 32) →
      ∑ v ∈ clusterBy rule O.mq populations x B, wMass O Dj (clsW O b) v
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * (Δ + c * clsShift O b + g) →
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
        clsBadP O η₀ b (clusterBy rule O.mq populations x B) p)).card : ℝ)
        ≤ B.npref * (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) + r := by
    intro b hh hd
    have hw := (clsShift_pos O hηle hη₀ b).le
    exact heavy_le_of_clean O populations Dj j B x _ h1
      (clsBad_mass_le O hηle hη₀ Dj _ b _ (by positivity) hd) hh
  have hQ0 : ∀ b, 0 ≤ clsQ O η₀ Δ c g b := fun b => by
    have := (clsShift_pos O hηle hη₀ b).le
    have := (clsSig_pos O hηle hη₀ b).le
    unfold clsQ; positivity
  have hMF : ∀ b, B.npref * (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) + r
      ≤ lF * B.npref := by
    intro b
    have := mul_le_mul_of_nonneg_left (hUF b) hmR
    nlinarith
  have hMG : ∀ b, (B.npref * (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) + r)
      * sideCharge O η₀ (!b) ≤ lG * B.npref := by
    intro b
    have hch := sideCharge_nonneg O η₀ (!b)
    have h1 : B.npref * (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) + r
        ≤ B.npref * (clsQ O η₀ Δ c g b * (33 / 32) + c) := by nlinarith
    have h2 := mul_le_mul_of_nonneg_right h1 hch
    have h3 := mul_le_mul_of_nonneg_left (hUG b) hmR
    nlinarith
  refine hx (retAt_of_light O populations uni huni j B x hmpos hηle hη₀ εcov α hα0 hα1 l lF lG
    lmis t (fun b => B.npref * (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) + r) r rs kmin kmax
    hkmin hlF hlmis hlohi ht hkpos hpool hgate h5' hCn (fun b => ?_) hMF hMG h8 h10 h11)
  cases b
  · exact hcnt false h7f h6f
  · exact hcnt true h7t h6t

open scoped Classical in
set_option maxHeartbeats 4000000 in
/-- Where the round can fail at one population, as `notRetAt_subset` lists it, priced. -/
theorem measureReal_failUnion_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0) (j : J)
    (hj : j ∈ populations) (B : State) (l lmis E t ρsf : ℝ) (f : S → ℝ) (r rs kmin kmax : ℕ)
    (hE : 0 ≤ E) (hEl : E ≤ l) (hElm : E ≤ lmis)
    (ht : 0 ≤ t) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (Ecert : ℝ) (hcertBad : (streamMeasure μ Pp Pc Dsf).real
      {x : Run Ω S J | ¬ certGood populations j B.npref r x} ≤ Ecert)
    (Estall : ℝ) (hstall : (streamMeasure μ Pp Pc Dsf).real (stalled rule O populations B kmin kmax)
      ≤ Estall)
    (Edset : Set (Run Ω S J)) (Edirty : ℝ)
    (hdirty : (streamMeasure μ Pp Pc Dsf).real Edset ≤ Edirty)
    (Ehset : Set (Run Ω S J)) (Eheavy : ℝ)
    (hheavy : (streamMeasure μ Pp Pc Dsf).real Ehset ≤ Eheavy)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f p →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f p →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (streamMeasure μ Pp Pc Dsf).real
        ({x | ¬ certGood populations j B.npref r x} ∪ stalled rule O populations B kmin kmax
          ∪ Edset ∪ Ehset
          ∪ lightMiss rule O populations j B l lmis f kmin kmax
          ∪ sidesDead rule O populations j B B.npref t ∪ {x | ¬ poolGood B.nsuff rs x})
      ≤ Ecert + Estall + Edirty + Eheavy
        + (Real.exp (-((B.npref : ℝ) * relRate l E))
          + Real.exp (-((B.npref : ℝ) * relRate lmis E)))
        + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
        + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) := by
  classical
  set E1 : Set (Run Ω S J) := {x | ¬ certGood populations j B.npref r x} with hE1
  set E5 : Set (Run Ω S J) := stalled rule O populations B kmin kmax with hE5
  set E8 : Set (Run Ω S J) := lightMiss rule O populations j B l lmis f kmin kmax with hE8
  set E10 : Set (Run Ω S J) := sidesDead rule O populations j B B.npref t with hE10
  set E11 : Set (Run Ω S J) := {x | ¬ poolGood B.nsuff rs x} with hE11
  set μr := streamMeasure μ Pp Pc Dsf with hμr
  have u : ∀ A B' : Set (Run Ω S J), μr.real (A ∪ B') ≤ μr.real A + μr.real B' :=
    fun A B' => measureReal_union_le A B'
  have b8 := measureReal_lightMiss_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf j hj B l lmis E f kmin kmax hE hEl hElm hdec hcut
  have b10 := measureReal_sidesDead_le (rule := rule) hflat O populations Pp Pc Dsf hsuppP hsuppC
    hsuppSf j hj B B.npref ht
  have b11 := measureReal_poolBad_le (μ := μ) Pp Pc Dsf B.nsuff rs ρsf hρsf hρsf0
  have := u E1 E5
  have := u (E1 ∪ E5) Edset
  have := u (E1 ∪ E5 ∪ Edset) Ehset
  have := u (E1 ∪ E5 ∪ Edset ∪ Ehset) E8
  have := u (E1 ∪ E5 ∪ Edset ∪ Ehset ∪ E8) E10
  have := u (E1 ∪ E5 ∪ Edset ∪ Ehset ∪ E8 ∪ E10) E11
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

lemma gateRoom_facts (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) :
    0 < gateRoom η₀ indecisionLimit εcov ∧ gateRoom η₀ indecisionLimit εcov ≤ 1 / 4
      ∧ min εcov (1 / 2) / 2 ≤ gateRoom η₀ indecisionLimit εcov := by
  have hmn0 : 0 < min εcov (1 / 2) := lt_min hε (by norm_num)
  have hmn1 : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
  unfold gateRoom
  refine ⟨by linarith, by linarith, le_rfl⟩

set_option maxHeartbeats 2000000 in
/-- The top rung's gate clears its charge and its deviations at every count of draws its
redraws allow. -/
lemma solved_gate (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hεs : 0 < εs)
    (hind : 0 < indecisionLimit) (hα : 0 < α) (hα1 : α ≤ 1)
    (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    ∀ N : ℕ, (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref ≤ N →
      N ≤ (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref
        + rejCount η₀ indecisionLimit εcov εs
          (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref →
      4 * (2 * (min εcov (1 / 2) / 32) + 2 * (5 / 8 * cutBudget η₀ indecisionLimit εcov)
            / gateA η₀ ^ 2)
          * (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref
        + 4 * ((gateDevT η₀ + gateDevU η₀)
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
              hsz).npref
              (δ / (32 * populations.card
                * (((solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
                    hsz).nsuff
                  : ℝ) + 3)))
          + gateDevD η₀
            * hoeffDev (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
              hsz).npref
              (sizeLevel (gateLevel α populations
                (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz))
                (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref))
        + ((N : ℝ) - (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
            hsz).npref)
      ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations
          (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)) N) := by
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hBm :
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hBM :
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).nsuff = M :=
    rfl
  have hga : ∀ N : ℕ, sizeLevel (gateLevel α populations
        (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)) N
      = sizeLevel (sizeLevel (α / (7 * (populations.card : ℝ))) m) N := fun N => rfl
  rw [hBm, hBM]
  simp only [hga]
  -- the two counts the gate's terms ask for
  have hc1 : ⌈50 * (gateDevT η₀ + gateDevU η₀) ^ 2
        * Real.log (32 * (populations.card : ℝ) * ((M : ℝ) + 3) / δ)
        / gateRoom η₀ indecisionLimit εcov ^ 2⌉₊ ≤ m := by
    rw [hmdef, prefCount, ← hMdef]; omega
  have hc2 : ⌈gateLogCount (200 / 9 * (gateDevD η₀ + 1) ^ 2 / gateRoom η₀ indecisionLimit εcov ^ 2)
        (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α)))⌉₊ ≤ m := by
    rw [hmdef, prefCount]; omega
  have hmR1 : (1 : ℝ) ≤ m := by
    have := prefCount_pos (εs := εs) (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      (hsz := hsz) η₀
      populations εcov δ α pAP
    exact_mod_cast this
  have hm0 : (0 : ℝ) < m := by linarith
  set c : ℝ := (populations.card : ℝ) with hcdef
  set cut : ℝ := cutBudget η₀ indecisionLimit εcov with hcutdef
  set a : ℝ := gateA η₀ with hadef
  set mn : ℝ := min εcov (1 / 2) with hmndef
  set gR : ℝ := gateRoom η₀ indecisionLimit εcov with hgRdef
  set W : ℝ := gateDevT η₀ + gateDevU η₀ with hWdef
  set D : ℝ := gateDevD η₀ with hDdef
  have hcut : 0 < cut := cutBudget_pos η₀ hsig hε hind
  have ha : 0 < a := gateA_pos hη0 hsig
  have hmn0 : 0 ≤ mn := le_min hε.le (by norm_num)
  obtain ⟨hgR0, hgR1, -⟩ := gateRoom_facts η₀ indecisionLimit εcov hε
  rw [← hgRdef] at hgR0 hgR1
  have hgReq : gR = mn / 2 := by rw [hgRdef, gateRoom]
  obtain ⟨hT0, hD0, hU0⟩ := gateDev_pos hη0 hsig
  have hW0 : 0 < W := by rw [hWdef]; linarith
  -- the references' and the seed's deviation, against two fifths of the room
  set L₁ : ℝ := Real.log (32 * c * ((M : ℝ) + 3) / δ) with hL₁
  have hP1 : 50 * W ^ 2 * L₁ ≤ m * gR ^ 2 := by
    have := le_trans (Nat.le_ceil _) (by exact_mod_cast hc1 :
      ((⌈50 * W ^ 2 * L₁ / gR ^ 2⌉₊ : ℕ) : ℝ) ≤ m)
    rwa [div_le_iff₀ (by positivity)] at this
  set t : ℝ := hoeffDev m (δ / (32 * c * ((M : ℝ) + 3))) with htdef
  have ht : t ≤ Real.sqrt (m * L₁ / 2) :=
    hoeffDev_le_sqrt (by rw [one_div_div])
  have hsqt : 10 * W * Real.sqrt (m * L₁ / 2) ≤ gR * m := by
    have h25 : (10 * W) ^ 2 * (m * L₁ / 2) ≤ (gR * m) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hP1 hm0.le
      linarith only [this]
    calc 10 * W * Real.sqrt (m * L₁ / 2) = Real.sqrt ((10 * W) ^ 2 * (m * L₁ / 2)) := by
          rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
      _ ≤ Real.sqrt ((gR * m) ^ 2) := Real.sqrt_le_sqrt h25
      _ = gR * m := Real.sqrt_sq (by positivity)
  have hWt : 4 * (W * t) ≤ 2 / 5 * (gR * m) := by
    have := mul_le_mul_of_nonneg_left ht hW0.le
    linarith
  -- the levels' deviations, against the rest
  set K : ℝ := 200 / 9 * (D + 1) ^ 2 / gR ^ 2 with hKdef
  set q : ℝ := 7 * c / α with hqdef
  set kk : ℝ := 4 * Real.sqrt (Real.sqrt q) with hkkdef
  have hq1 : 1 ≤ q := by rw [hqdef, le_div_iff₀ hα]; linarith
  have hkk4 : 4 ≤ kk := by
    have : 1 ≤ Real.sqrt (Real.sqrt q) := Real.one_le_sqrt.2 (Real.one_le_sqrt.2 hq1)
    rw [hkkdef]; linarith
  have hK4 : 4 ≤ K := by
    rw [hKdef, le_div_iff₀ (by positivity)]
    have h1 : gR ^ 2 ≤ 1 / 16 := by nlinarith only [hgR0, hgR1]
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
  have hKΛ : 200 * (D + 1) ^ 2 * Λ ≤ 9 * gR ^ 2 * m := by
    have h1 : K * Λ ≤ m := by
      have := mul_le_mul_of_nonneg_left hΛkk (by linarith : 0 ≤ K)
      linarith only [this, hP2]
    rw [hKdef] at h1
    have e : 200 / 9 * (D + 1) ^ 2 / gR ^ 2 * Λ * (9 * gR ^ 2) = 200 * (D + 1) ^ 2 * Λ := by
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
  have hss : 20 * (D + 1) * s ≤ 3 * (gR * m) := by
    have h : (20 * (D + 1)) ^ 2 * (m * Λ / 2) ≤ (3 * (gR * m)) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hKΛ hm0.le
      linarith only [this]
    calc 20 * (D + 1) * s = Real.sqrt ((20 * (D + 1)) ^ 2 * (m * Λ / 2)) := by
          rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
      _ ≤ Real.sqrt ((3 * (gR * m)) ^ 2) := Real.sqrt_le_sqrt h
      _ = 3 * (gR * m) := Real.sqrt_sq (by positivity)
  intro N hN1 hN2
  have hs1 : sig η₀ ^ 2 ≤ 1 := by
    have : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
    have := sig_pos η₀ hsig
    nlinarith
  have hr : ((rejCount η₀ indecisionLimit εcov εs m : ℕ) : ℝ) ≤ mn / 1024 * m := by
    have h1 : ((rejCount η₀ indecisionLimit εcov εs m : ℕ) : ℝ)
        ≤ rejFrac η₀ indecisionLimit εcov εs * m := Nat.floor_le (by
      have := rejFrac_pos η₀ hsig hε hεs hind; positivity)
    have h2 := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀ (εcov := εcov)).1
    have h3 : sig η₀ ^ 2 * mn / 1024 ≤ mn / 1024 := by
      have := mul_le_mul_of_nonneg_right hs1 hmn0; linarith
    have := mul_le_mul_of_nonneg_right (h2.trans h3) hm0.le
    linarith only [h1, this]
  have hNR1 : (m : ℝ) ≤ N := by exact_mod_cast hN1
  have hNR2 : (N : ℝ) ≤ m + mn / 1024 * m := by
    have : (N : ℝ) ≤ m + (rejCount η₀ indecisionLimit εcov εs m : ℕ) := by exact_mod_cast hN2
    linarith
  have hN2mR : (N : ℝ) ≤ 2 * m := by
    have : mn / 1024 * m ≤ m := by
      have := mul_le_mul_of_nonneg_right (show mn / 1024 ≤ 1 by linarith [min_le_right εcov (1 / 2)])
        hm0.le
      linarith
    linarith only [hNR2, this]
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
  have hcuta : 5 / 8 * cut / a ^ 2 ≤ 5 / 256 * mn := by
    have h := cutBudget_le_gate η₀ indecisionLimit εcov
    rw [← hcutdef, ← hadef, ← hmndef] at h
    rw [div_le_iff₀ (by positivity)]
    nlinarith
  have hch : 4 * (2 * (mn / 32) + 2 * (5 / 8 * cut) / a ^ 2) * m ≤ (mn / 4 + 5 / 32 * mn) * m := by
    have e : 2 * (5 / 8 * cut) / a ^ 2 = 2 * (5 / 8 * cut / a ^ 2) := by ring
    rw [e]
    nlinarith
  have hNmn : (m : ℝ) * mn ≤ N * mn := mul_le_mul_of_nonneg_right hNR1 hmn0
  have hsD : 0 ≤ s := Real.sqrt_nonneg _
  rw [hgReq] at hWt hss
  nlinarith

/-- What one round at the uniform pool can cost, the count rule's table event apart: the
samplers redrawing too much, the family's size, its shift on each class
(`measureReal_famShift_le`: the table redrawing, the pool redrawing or short of accept-preserving
rows, a member's first draws understating its shift), each class's heavy fraction on the sample,
the two light counts, the gate's liveness, and the pool. -/
noncomputable def roundFail (populations : Finset J)
    (l lmis h E L tap pAP ρ ρsf t : ℝ) (r rs : ℕ) (B : State) : ℝ :=
  rejTail (2 * B.npref + r) (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * ρ) r
    + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
    + 2 * (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
        + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + Real.exp (-((B.nsuff : ℝ) * (tap ^ 2 / (2 * pAP)))) + (B.nsuff : ℝ) * Real.exp (-L))
    + 2 * Real.exp (-((B.npref : ℝ) * h))
    + (Real.exp (-((B.npref : ℝ) * relRate l E))
      + Real.exp (-((B.npref : ℝ) * relRate lmis E)))
    + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))
    + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))

/-- What one round at a named population can cost, the count rule's table event and the pool's
apart: the hash's split of the window and of each member's flips on it on each class, the uniform
pool's table and its share of the window, the two light counts, and the gate's liveness. -/
noncomputable def namedFail (l lmis E ρ t L θ : ℝ) (r G : ℕ) (B : State) : ℝ :=
  (2 * (((B.nsuff : ℝ) + 1) * Real.exp (-L)) + 2 * Real.exp (-((G : ℝ) ^ 2 / B.npref)))
    + (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
      + rejTail (B.npref + r) (((2 * B.npref : ℕ) : ℝ) * ρ) (⌈θ * B.npref⌉₊ - 1))
    + (Real.exp (-((B.npref : ℝ) * relRate l E))
      + Real.exp (-((B.npref : ℝ) * relRate lmis E)))
    + 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / B.npref))

/-- What the pool's events, which every population shares, can cost: the family's size, the pool
short of accept-preserving rows, and its sampler redrawing too much. -/
noncomputable def poolFail (tap pAP ρsf : ℝ) (rs : ℕ) (B : State) : ℝ :=
  2 * (((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)))
    + Real.exp (-((B.nsuff : ℝ) * (tap ^ 2 / (2 * pAP))))

lemma roundFail_nonneg (populations : Finset J) (l lmis h E L tap pAP ρ ρsf t : ℝ)
    (r rs : ℕ) (B : State) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf) :
    0 ≤ roundFail populations l lmis h E L tap pAP ρ ρsf t r rs B := by
  unfold roundFail
  repeat' (first
    | exact rejTail_nonneg _ (by positivity) _
    | positivity
    | apply add_nonneg
    | apply mul_nonneg)

lemma namedFail_nonneg (l lmis E ρ t L θ : ℝ) (r G : ℕ) (B : State) (hρ0 : 0 ≤ ρ) :
    0 ≤ namedFail l lmis E ρ t L θ r G B := by
  unfold namedFail
  repeat' (first | exact rejTail_nonneg _ (by positivity) _ | positivity | apply add_nonneg)

/-- A named population's heavy prefixes of class `b` on its sample: on the window, Markov at the
class's fraction of the members' flips on the class, which the hash holds to their table flips
`Δ/w_b` each and its deviation; beyond the window, every prefix. -/
noncomputable def namedQ (O : Oracle μ S) (η₀ Δ L θ : ℝ) (hsz G m : ℕ) (b : Bool) : ℝ :=
  ((Δ / clsShift O b) * m + 2 * Real.sqrt (L * ((Δ / clsShift O b) * m)) + 2 * L)
      / clsFrac O η₀ b
    + hsz + 2 * G + θ * m

/-- A state whose round can pass.  Every clause is an inequality among the state's budgets,
the oracle's rates, the suffix distribution's findability and the error budget; no probability
enters.  The count rule's heavy rows are charged at `θ = 15/16` of the table budget `Δ`, in the
count rule's own shift units, and each class's heavy prefixes at that class's rates. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (hsz : ℕ) (B : State) : Prop :=
  ∃ (tap q E Δ g u x a e1 e2 lF lG lmis c t L θ : ℝ) (r rs G : ℕ),
    0 < B.npref ∧ 2 ≤ B.k ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ 0 ≤ tap ∧ 0 < Δ ∧ 0 < g ∧ 0 ≤ pAP ∧ 0 ≤ lmis
    -- the samplers redraw at most a `c` fraction of the prefix draws
    ∧ 0 ≤ c ∧ (r : ℝ) ≤ c * B.npref
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- each class's heavy prefixes fit the FNR test's share and, charged at the class's rates,
    -- the gate's: the uniform pool's by its own draws, a named population's by the hash's split
    -- of a window of its draws
    ∧ (∀ b, clsQ O η₀ Δ c g b * (33 / 32) + c ≤ lF)
    ∧ (∀ b, sideCharge O η₀ (!b) * (clsQ O η₀ Δ c g b * (33 / 32) + c) ≤ lG)
    ∧ 0 ≤ L ∧ 0 < θ ∧ 2 ≤ B.npref ∧ hsz + G ≤ B.npref
    ∧ (∀ b, namedQ O η₀ Δ L θ hsz G B.npref b ≤ lF * B.npref)
    ∧ (∀ b, sideCharge O η₀ (!b) * namedQ O η₀ Δ L θ hsz G B.npref b ≤ lG * B.npref)
    ∧ 2 * lF + indecisionLimit / 2 ≤ indecisionLimit
    -- the screen admits every candidate, and the count rule picks the family
    ∧ B.scd ≤ B.sc
    ∧ 0 ≤ u ∧ 2 ≤ x
    ∧ (u + x) * Real.sqrt ((populations.card : ℝ) * B.npref) ≤ 15 / 16 * Δ * B.npref
    ∧ a ≤ (B.nsuff : ℝ) * (pAP - tap) - rs - 1 ∧ ((B.k - 1 : ℕ) : ℝ) ≤ a
    ∧ 2 * L * (Δ + c + g) ≤ B.npref * g ^ 2
    ∧ a * Real.exp (-(2 * u ^ 2)) / (a - ((B.k - 1 : ℕ) : ℝ) + 1) ≤ e1
    ∧ ((B.nsuff : ℝ) + 1) * (15 / 16) / ((1 - 15 / 16) * ((B.k - 1 : ℕ) : ℝ))
        * Real.exp (-(2 / 5 * x ^ 2)) ≤ e2
    -- the pool holds a family, beyond what its sampler redraws, and a reference beside it
    ∧ ((B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - tap))
    ∧ B.k < B.nsuff
    -- the thresholds decide, and decide right, on a family of the round's size flipping at most
    -- `fracAt` at each prefix, whose reads' mean that leaves `q` toward the prefix's side; the
    -- FNR reads the family with its seed, cut one count higher
    ∧ 0 ≤ q ∧ q ≤ 1
    ∧ (∀ p, 0 ≤ fracAt O η₀ p ∧ q ≤ 1 - O.rate p - O.gap * fracAt O η₀ p
      ∧ O.rate p + O.gap * fracAt O η₀ p ≤ 1 - q)
    ∧ ((B.hi : ℝ) + 1 + 2 ≤ (B.k : ℝ) * q)
    ∧ ((B.k : ℝ) * (1 - q) + 2 ≤ (B.lo : ℝ) + 1)
    ∧ ((B.lo : ℝ) + 2 ≤ ((B.k - 1 : ℕ) : ℝ) * q)
    ∧ (((B.k - 1 : ℕ) : ℝ) * (1 - q) + 2 ≤ (B.hi : ℝ) + 1)
    ∧ B.hi + 1 ≤ B.k - 1
    ∧ B.lo ≤ B.hi
    -- the gate clears the charge of a family within its shares, of the vote's misfires on
    -- the usable prefixes, and its deviations
    ∧ 0 ≤ t
    ∧ (∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      4 * (2 * lG + 2 * lmis / gateA η₀ ^ 2) * B.npref
          + 4 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    -- the vote misfires, on either side and with or without its seed, at `E`
    ∧ 0 ≤ E ∧ E ≤ lmis ∧ E ≤ indecisionLimit / 2
    ∧ 1 - binomSfGe B.k q (B.hi + 1 + 1) ≤ E ∧ 1 - binomSfGe B.k q (B.k - B.lo) ≤ E
    ∧ 1 - binomSfGe (B.k - 1) q (B.lo + 1) ≤ E ∧ 1 - binomSfGe (B.k - 1) q (B.k - 1 - B.hi) ≤ E
    -- and the whole round, the count rule's table event and the pool's once and every
    -- population, fits in the budget
    ∧ (e1 + e2 + poolFail tap pAP ρsf rs B
        + roundFail populations (indecisionLimit / 2) lmis (g / 1056) E L tap pAP ρ ρsf t r rs B
        + (populations.card : ℝ) * namedFail (indecisionLimit / 2) lmis E ρ t L θ r G B
      ≤ δ / 2)

/-- At the top count the uniform pool's share of a window is a tail in `cutBudget/1024` of its
count, which clears `log(256|J|/δ)` with room: the cut budget is `32·shiftBudget` or more, and
the count clears `1056·log(128|J|/δ)/shiftBudget`. -/
lemma solved_hits_log (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.log (256 * (populations.card : ℝ) / δ)
      ≤ cutBudget η₀ indecisionLimit εcov / 1024
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
        * Real.log 2 := by
  set N : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hNdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hsig hε hind
  have hΔcb : 32 * Δ ≤ cb := shiftBudget_le_cutBudget η₀ indecisionLimit εcov hind hε hη0 hsig
  have hl1 : 0 ≤ Real.log (128 * P / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  have c1 : Real.log (128 * P / δ) * (1056 / Δ + 2 / indecisionLimit ^ 2) ≤ (N : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * P / δ) * (1056 / Δ + 2 / indecisionLimit ^ 2)⌉₊ ≤ N := by
      rw [hNdef, prefCount, ← hΔdef, ← hP]; omega
    exact_mod_cast hle
  have c1' : 1056 * Real.log (128 * P / δ) ≤ Δ * N := by
    have h0 : 0 ≤ Real.log (128 * P / δ) * (2 / indecisionLimit ^ 2) := by positivity
    have e : Real.log (128 * P / δ) * (1056 / Δ + 2 / indecisionLimit ^ 2)
        = 1056 * Real.log (128 * P / δ) / Δ + Real.log (128 * P / δ) * (2 / indecisionLimit ^ 2) := by
      ring
    rw [e] at c1
    have : 1056 * Real.log (128 * P / δ) / Δ ≤ N := by linarith
    rwa [div_le_iff₀ hΔ0, mul_comm (N : ℝ)] at this
  have hl1' : Real.log 128 ≤ Real.log (128 * P / δ) := by
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
  have hcbN : 33 * Real.log (128 * P / δ) ≤ cb / 1024 * N := by
    have := mul_le_mul_of_nonneg_right hΔcb hN0
    nlinarith
  rw [h256]
  nlinarith [mul_le_mul_of_nonneg_right hcbN (by linarith : (0 : ℝ) ≤ Real.log 2)]

set_option maxHeartbeats 2000000 in
/-- The round's failure, the pool's once and over every population, fits the error budget. -/
lemma solved_roundFail (η₀ : ℝ) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hεs : 0 < εs) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hρsfsmall : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    poolFail (9 * pAP / 20) pAP ρsf
        (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
        (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
      + roundFail populations (indecisionLimit / 2) (5 / 8 * cutBudget η₀ indecisionLimit εcov)
          (shiftBudget η₀ indecisionLimit εcov / 1056)
          (misfireTail η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit))
          (Real.log (128 * (populations.card : ℝ)
            * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
          (9 * pAP / 20) pAP ρ ρsf
          (hoeffDev (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
            (δ / (32 * populations.card
              * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))))
          (rejCount η₀ indecisionLimit εcov εs
            (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz))
          (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
          (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
      + (populations.card : ℝ)
        * namedFail (indecisionLimit / 2) (5 / 8 * cutBudget η₀ indecisionLimit εcov)
            (misfireTail η₀ (famCount η₀ populations indecisionLimit εcov δ crossLimit)) ρ
            (hoeffDev (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
              (δ / (32 * populations.card
                * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3))))
            (Real.log (128 * (populations.card : ℝ)
              * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
            (cutBudget η₀ indecisionLimit εcov / 1024)
            (rejCount η₀ indecisionLimit εcov εs
              (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz))
            ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
                hsz : ℝ)
              * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
            (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
      ≤ 3 * δ / 8 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  obtain ⟨ttap, tth, -⟩ :=
    solved_tails (α := α) (crossLimit := crossLimit) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  have hBm :
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).npref = m :=
    rfl
  have hBM :
      (solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz).nsuff = M :=
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
  have hΔcb : 32 * shiftBudget η₀ indecisionLimit εcov ≤ cb :=
    shiftBudget_le_cutBudget η₀ indecisionLimit εcov hind hε hη0 hsig
  have hΔ0 := shiftBudget_pos η₀ hsig hε hind
  have hexp : ∀ r : ℝ, cb / 2112 ≤ r → Real.exp (-((m : ℝ) * r)) ≤ ε₀ := fun r hr =>
    le_trans (Real.exp_le_exp.2 (neg_le_neg (mul_le_mul_of_nonneg_left
      (le_trans (by linarith : shiftBudget η₀ indecisionLimit εcov / 1056 ≤ cb / 2112) hr) hmR)))
      tth
  have tl : Real.exp (-((m : ℝ) * relRate (indecisionLimit / 2) (misfireTail η₀ κ))) ≤ ε₀ :=
    hexp _ hrl
  have tlm : Real.exp (-((m : ℝ) * relRate (5 / 8 * cb) (misfireTail η₀ κ))) ≤ ε₀ :=
    hexp _ hrm
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
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hcdef
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hεs hind
  have hcle := rejFrac_le_cut η₀ indecisionLimit εcov εs hind hε hη0 hsig
  set r : ℕ := rejCount η₀ indecisionLimit εcov εs m with hrdef
  set rs : ℕ := poolRej pAP M with hrsdef
  have hpool := poolDraws_le_of_cap η₀ populations hpAP hpAP1 hδ hρsf0 hρsfsmall
  rw [← hMdef, ← hrsdef] at hpool
  rw [poolFail, roundFail, namedFail, hBm, hBM]
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
    hη0 hsig hε hεs hind hδ hδ1 hpAP hJ1
  rw [← hmdef] at hlog
  have hr2 : c * m < (r : ℝ) + 1 := Nat.lt_floor_add_one _
  have hrlog : Real.log (256 * (populations.card : ℝ) / δ) ≤ ((r + 1 : ℕ) : ℝ) * Real.log 2 := by
    push_cast
    nlinarith [mul_le_mul_of_nonneg_right hr2.le hl2.le]
  have hhalf := hhalf_of (r + 1) hrlog
  obtain ⟨ht3, hc3⟩ := rung_tails_le (crossLimit := crossLimit) η₀ populations (δ := δ) (α := α)
    (pAP := pAP) hη0 hsig hε hεs hind hρ0 hρsmall m le_rfl
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
    have hhl := solved_hits_log (εs := εs) (α := α) (crossLimit := crossLimit) (hsz := hsz) η₀
        populations
      hη0 hsig hε hind hδ hδ1 hpAP hJ1
    rw [← hmdef, ← hθm] at hhl
    exact le_trans hpow (hhalf_of ⌈θm⌉₊ (le_trans hhl (mul_le_mul_of_nonneg_right hθle hl2.le)))
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
  have tmass : (M : ℝ) * Real.exp (-L) ≤ ε₀ := by
    have h0 : 0 ≤ Real.exp (-L) := Real.exp_nonneg _
    nlinarith [tbal]
  set pool : ℝ := ((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) with hpooldef
  have hpool0 : 0 ≤ pool := by positivity
  have htab0 : 0 ≤ tab := rejTail_nonneg _ (by positivity) _
  have hhits0 : 0 ≤ hits := rejTail_nonneg _ (by positivity) _
  have hcert0 : 0 ≤ cert := rejTail_nonneg _ (by positivity) _
  -- the population-free terms against `δ/256` and `δ/128`, the named ones' sum against `δ/128`
  have h256 : ∀ x : ℝ, x ≤ δ / (256 * (populations.card : ℝ)) → x ≤ δ / 256 := fun x hx =>
    le_trans hx (div_le_div_of_nonneg_left hδ.le (by norm_num) (by linarith))
  have hsum256 : ∀ x : ℝ, x ≤ δ / (256 * (populations.card : ℝ))
      → (populations.card : ℝ) * x ≤ δ / 256 := fun x hx => by
    have h := mul_le_mul_of_nonneg_left hx hcard.le
    rwa [show (populations.card : ℝ) * (δ / (256 * (populations.card : ℝ))) = δ / 256 by
      field_simp] at h
  have hε₀le : ε₀ ≤ δ / 128 := div_le_div_of_nonneg_left hδ.le (by norm_num) (by linarith)
  have hpool' : pool ≤ δ / 256 :=
    le_trans hpool (div_le_div_of_nonneg_left hδ.le (by norm_num) (by linarith))
  have hcert' := h256 cert hcert_le
  have htab' := h256 tab htab_le
  have hJtab := hsum256 tab htab_le
  have hJhits := hsum256 hits hhits_le
  refine le_trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (show _ ≤ 14 * ε₀ + (tab + hits) from ?_) hcard.le)) ?_
  · linarith [tl, tlm, tlive, tbal, tcard]
  · have h1 : (populations.card : ℝ) * (14 * ε₀ + (tab + hits))
        = 14 * δ / 128 + ((populations.card : ℝ) * tab + (populations.card : ℝ) * hits) := by
      rw [mul_add, mul_add,
        show (populations.card : ℝ) * (14 * ε₀) = 14 * ((populations.card : ℝ) * ε₀) by ring, hcε]
      ring
    linarith [ttap, tmass, tth, tl, tlm, tlive]

set_option maxHeartbeats 1000000 in
/-- A class's share at a named population, in the reals: Markov on the window at a fraction
`f ≥ sig/5` with `σ = f·w ≥ (2/5)·sig²`, the hash's deviation, and the prefixes beyond the window,
against the FNR test's share and, at a charge of at most `5/(2 sig²)·σ` and `1/a²`, the gate's.
The table budget `Δ` reads `L ≤ Δm/64` off the count. -/
lemma named_share_le {s β εu mn Δ w f sc a cb L m X : ℝ} (hs : 0 < s) (hs1 : s ≤ 1 / 2)
    (hβ : 0 < β) (hβu : β ≤ εu) (hβm : β ≤ mn) (hΔ : Δ = s ^ 2 * β / 256)
    (hw0 : 0 < w) (hw1 : w ≤ 1) (hf : s / 5 ≤ f) (hσ : 2 / 5 * s ^ 2 ≤ f * w)
    (hsc0 : 0 ≤ sc) (hsc : sc ≤ 5 / (2 * s ^ 2) * (f * w)) (ha : 0 < a) (hsca : sc ≤ 1 / a ^ 2)
    (hL : 0 ≤ L) (hm : 0 ≤ m) (hm64 : 64 * L ≤ Δ * m)
    (hX0 : 0 ≤ X) (hX : X ≤ 1029 / 10000 * (cb * m)) (hcbu : cb ≤ εu / 2)
    (hcba : cb ≤ a ^ 2 * mn / 32) :
    ((Δ / w) * m + 2 * Real.sqrt (L * ((Δ / w) * m)) + 2 * L) / f + X ≤ εu / 8 * m
    ∧ sc * (((Δ / w) * m + 2 * Real.sqrt (L * ((Δ / w) * m)) + 2 * L) / f + X)
      ≤ mn / 32 * m := by
  have hf0 : 0 < f := lt_of_lt_of_le (by positivity) hf
  have hfw : 0 < f * w := mul_pos hf0 hw0
  have hΔ0 : 0 ≤ Δ := by rw [hΔ]; positivity
  set R : ℝ := Real.sqrt (L * ((Δ / w) * m)) with hRdef
  have hR0 : 0 ≤ R := Real.sqrt_nonneg _
  have hR2 : R ^ 2 = L * ((Δ / w) * m) := Real.sq_sqrt (by positivity)
  -- `w·R ≤ Δm/8`, from `L ≤ Δm/64`
  have hwR : w * R ≤ Δ * m / 8 := by
    have h1 : (w * R) ^ 2 = w * (L * (Δ * m)) := by
      rw [mul_pow, hR2]; field_simp
    have hLΔ : L * (Δ * m) ≤ (Δ * m) ^ 2 / 64 := by nlinarith
    have h2 : w * (L * (Δ * m)) ≤ (Δ * m / 8) ^ 2 := by
      have : w * (L * (Δ * m)) ≤ 1 * (L * (Δ * m)) :=
        mul_le_mul_of_nonneg_right hw1 (by positivity)
      nlinarith
    rw [← h1] at h2
    exact le_of_pow_le_pow_left₀ two_ne_zero (by positivity) h2
  have e1 : (Δ / w) * m / f = Δ * m / (f * w) := by field_simp
  have e2 : R / f = w * R / (f * w) := by field_simp
  have hsplit : ((Δ / w) * m + 2 * R + 2 * L) / f = Δ * m / (f * w) + 2 * (w * R / (f * w))
      + 2 * L / f := by
    field_simp
  -- the three window terms, at the signal's floors
  have t1 : Δ * m / (f * w) ≤ 5 / (2 * s ^ 2) * (Δ * m) := by
    rw [div_le_iff₀ hfw]
    have := mul_le_mul_of_nonneg_left hσ (by positivity : (0 : ℝ) ≤ 5 / (2 * s ^ 2) * (Δ * m))
    have e : 5 / (2 * s ^ 2) * (Δ * m) * (2 / 5 * s ^ 2) = Δ * m := by field_simp
    linarith
  have t2 : w * R / (f * w) ≤ 5 / (2 * s ^ 2) * (Δ * m / 8) := by
    rw [div_le_iff₀ hfw]
    have := mul_le_mul_of_nonneg_left hσ (by positivity : (0 : ℝ) ≤ 5 / (2 * s ^ 2) * (Δ * m / 8))
    have e : 5 / (2 * s ^ 2) * (Δ * m / 8) * (2 / 5 * s ^ 2) = Δ * m / 8 := by field_simp
    linarith
  have t3 : 2 * L / f ≤ 10 * (Δ * m / 64) / s := by
    rw [div_le_div_iff₀ hf0 hs]
    nlinarith
  have hDm : Δ * m = s ^ 2 * (β * m) / 256 := by rw [hΔ]; ring
  have hβm0 : 0 ≤ β * m := by positivity
  have hs2 : 0 < s ^ 2 := by positivity
  have u1 : 5 / (2 * s ^ 2) * (Δ * m) = 5 / 512 * (β * m) := by rw [hDm]; field_simp; ring
  have u2 : 5 / (2 * s ^ 2) * (Δ * m / 8) = 5 / 4096 * (β * m) := by rw [hDm]; field_simp; ring
  have u3 : 10 * (Δ * m / 64) / s = 10 / 16384 * (s * (β * m)) := by rw [hDm]; field_simp; ring
  have hβε : β * m ≤ εu * m := mul_le_mul_of_nonneg_right hβu hm
  have hβmn : β * m ≤ mn * m := mul_le_mul_of_nonneg_right hβm hm
  have hsβ : s * (β * m) ≤ β * m := mul_le_of_le_one_left hβm0 (by linarith)
  have hcbm : cb * m ≤ εu / 2 * m := mul_le_mul_of_nonneg_right hcbu hm
  constructor
  · rw [hsplit]
    nlinarith
  · -- each term at its charge
    have hscs : 0 ≤ 5 / (2 * s ^ 2) := by positivity
    have g1 : sc * (Δ * m / (f * w)) ≤ 5 / (2 * s ^ 2) * (Δ * m) := by
      have := mul_le_mul_of_nonneg_right hsc (by positivity : (0 : ℝ) ≤ Δ * m / (f * w))
      have e : 5 / (2 * s ^ 2) * (f * w) * (Δ * m / (f * w)) = 5 / (2 * s ^ 2) * (Δ * m) := by
        field_simp
      linarith
    have g2 : sc * (w * R / (f * w)) ≤ 5 / (2 * s ^ 2) * (Δ * m / 8) := by
      have := mul_le_mul_of_nonneg_right hsc (by positivity : (0 : ℝ) ≤ w * R / (f * w))
      have e : 5 / (2 * s ^ 2) * (f * w) * (w * R / (f * w)) = 5 / (2 * s ^ 2) * (w * R) := by
        field_simp
      have := mul_le_mul_of_nonneg_left hwR hscs
      linarith
    have g3 : sc * (2 * L / f) ≤ 5 / (2 * s ^ 2) * (2 * (Δ * m / 64)) := by
      have := mul_le_mul_of_nonneg_right hsc (by positivity : (0 : ℝ) ≤ 2 * L / f)
      have e : 5 / (2 * s ^ 2) * (f * w) * (2 * L / f) = 5 / (2 * s ^ 2) * (2 * (w * L)) := by
        field_simp
      have hwL : w * L ≤ Δ * m / 64 := by nlinarith
      have := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hwL (by norm_num : (0 : ℝ) ≤ 2))
        hscs
      linarith
    have g4 : sc * X ≤ 1029 / 10000 * (mn / 32 * m) := by
      have h1 : sc * X ≤ 1 / a ^ 2 * X := mul_le_mul_of_nonneg_right hsca hX0
      have h2 : 1 / a ^ 2 * X ≤ 1 / a ^ 2 * (1029 / 10000 * (cb * m)) :=
        mul_le_mul_of_nonneg_left hX (by positivity)
      have h3 : 1 / a ^ 2 * (cb * m) ≤ mn / 32 * m := by
        rw [one_div, ← div_eq_inv_mul, div_le_iff₀ (by positivity)]
        nlinarith
      nlinarith
    have u4 : 5 / (2 * s ^ 2) * (2 * (Δ * m / 64)) = 5 / 16384 * (β * m) := by
      rw [hDm]; field_simp; ring
    rw [hsplit]
    have e : sc * (Δ * m / (f * w) + 2 * (w * R / (f * w)) + 2 * L / f + X)
        = sc * (Δ * m / (f * w)) + 2 * (sc * (w * R / (f * w))) + sc * (2 * L / f) + sc * X := by
      ring
    rw [e]
    nlinarith

set_option maxHeartbeats 1000000 in
/-- A class's share at the uniform pool, in the reals: Markov at `σ ≥ (2/5)·sig²` of the table
budget `Δ`, the redraws' `c·w` and the understatement `Δ`, against the FNR test's share and, at a
charge of at most `5/(2 sig²)·σ` and `1/a²`, the gate's. -/
lemma uni_share_le {s β εu mn Δ c w σ sc a : ℝ} (hs : 0 < s) (hs1 : s ≤ 1 / 2) (hβ : 0 < β)
    (hβu : β ≤ εu) (hβm : β ≤ mn) (hΔ : Δ = s ^ 2 * β / 256) (hc0 : 0 ≤ c)
    (hcm : c ≤ s ^ 2 * mn / 1024) (hcu : c ≤ s * εu / 128)
    (hw0 : 0 < w) (hw1 : w ≤ 1) (hσ : 2 / 5 * s ^ 2 ≤ σ) (hwσ : w / σ ≤ 5 / s)
    (hsc0 : 0 ≤ sc) (hsc : sc ≤ 5 / (2 * s ^ 2) * σ) (hsca : sc ≤ 1 / a ^ 2) (ha : 2 * s ≤ a) :
    (Δ + c * w + Δ) / σ * (33 / 32) + c ≤ εu / 8
    ∧ sc * ((Δ + c * w + Δ) / σ * (33 / 32) + c) ≤ mn / 32 := by
  have hσ0 : 0 < σ := lt_of_lt_of_le (by positivity) hσ
  have hs2 : 0 < s ^ 2 := by positivity
  have hΔ0 : 0 ≤ Δ := by rw [hΔ]; positivity
  have hsplit : (Δ + c * w + Δ) / σ = 2 * Δ / σ + c * (w / σ) := by field_simp; ring
  have t1 : 2 * Δ / σ ≤ 5 * β / 256 := by
    rw [div_le_iff₀ hσ0, hΔ]
    have := mul_le_mul_of_nonneg_left hσ (by positivity : (0 : ℝ) ≤ 5 * β / 256)
    nlinarith
  have t2 : c * (w / σ) ≤ c * (5 / s) := mul_le_mul_of_nonneg_left hwσ hc0
  have hc5 : c * (5 / s) ≤ 5 * εu / 128 := by
    rw [mul_div_assoc', div_le_iff₀ hs]; nlinarith
  have hcε : c ≤ εu / 256 := by nlinarith
  have hmn : 0 ≤ mn := le_trans hβ.le hβm
  constructor
  · rw [hsplit]; nlinarith
  · have hscQ : sc * ((Δ + c * w + Δ) / σ) ≤ 5 / (2 * s ^ 2) * (2 * Δ + c) := by
      have h1 := mul_le_mul_of_nonneg_right hsc (by positivity : (0 : ℝ) ≤ (Δ + c * w + Δ) / σ)
      have e : 5 / (2 * s ^ 2) * σ * ((Δ + c * w + Δ) / σ) = 5 / (2 * s ^ 2) * (2 * Δ + c * w) := by
        field_simp; ring
      have h2 : 5 / (2 * s ^ 2) * (2 * Δ + c * w) ≤ 5 / (2 * s ^ 2) * (2 * Δ + c) :=
        mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
      linarith
    have hval : 5 / (2 * s ^ 2) * (2 * Δ + c) ≤ 5 * β / 256 + 5 * mn / 2048 := by
      rw [hΔ]
      have e : 5 / (2 * s ^ 2) * (2 * (s ^ 2 * β / 256) + c) = 5 * β / 256 + 5 / (2 * s ^ 2) * c := by
        field_simp
      rw [e]
      have : 5 / (2 * s ^ 2) * c ≤ 5 * mn / 2048 := by
        rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]; nlinarith
      linarith
    have hscc : sc * c ≤ mn / 4096 := by
      have h1 : sc * c ≤ 1 / a ^ 2 * c := mul_le_mul_of_nonneg_right hsca hc0
      have ha0 : 0 < a := by linarith
      have h2 : 1 / a ^ 2 * c ≤ 1 / (4 * s ^ 2) * c := by
        refine mul_le_mul_of_nonneg_right ?_ hc0
        rw [div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith
      have h3 : 1 / (4 * s ^ 2) * c ≤ mn / 4096 := by
        rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]; nlinarith
      linarith
    have e : sc * ((Δ + c * w + Δ) / σ * (33 / 32) + c)
        = 33 / 32 * (sc * ((Δ + c * w + Δ) / σ)) + sc * c := by ring
    rw [e]
    nlinarith

/-- The top rung clears the count rule's signal: a heavy row's shift, at `15/16` of the table
budget, stands `(u + x)√(|J| m)` above the clean rows. -/
lemma solved_signal (η₀ : ℝ) (populations : Finset J) (hpop : populations.Nonempty)
    {εcov δ α pAP : ℝ} (hη₀ : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    (cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit)
        * Real.sqrt ((populations.card : ℝ)
          * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ))
      ≤ 15 / 16 * shiftBudget η₀ indecisionLimit εcov
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) := by
  have hmc : (populations.card : ℝ) * (cleanLevel η₀ populations indecisionLimit εcov δ pAP
        crossLimit + heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit) ^ 2
        / (15 / 16 * shiftBudget η₀ indecisionLimit εcov) ^ 2
      ≤ (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have : ⌈(populations.card : ℝ) * (cleanLevel η₀ populations indecisionLimit εcov δ pAP
        crossLimit + heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit) ^ 2
        / (15 / 16 * shiftBudget η₀ indecisionLimit εcov) ^ 2⌉₊
        ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
      rw [prefCount]; omega
    exact_mod_cast this
  set m : ℝ := (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) with hm
  set ux : ℝ := cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit
    + heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit with hux
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔ
  set Jc : ℝ := (populations.card : ℝ) with hJc
  have hJ0 : (0 : ℝ) ≤ Jc := Nat.cast_nonneg _
  have hm0 : 0 ≤ m := Nat.cast_nonneg _
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hη₀ hε hind
  have hux0 : 0 ≤ ux := by
    have := Real.sqrt_nonneg (Real.log (apFloor pAP (poolCount η₀ populations indecisionLimit εcov
      δ pAP crossLimit) / ((apFloor pAP (poolCount η₀ populations indecisionLimit εcov δ pAP
        crossLimit) - famCount η₀ populations indecisionLimit εcov δ crossLimit + 1)
          * (δ / 16))) / 2)
    have : (2 : ℝ) ≤ heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit :=
      le_max_left _ _
    rw [hux]; unfold cleanLevel; linarith
  have hsq : (ux * Real.sqrt (Jc * m)) ^ 2 ≤ (15 / 16 * Δ * m) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by positivity)]
    rw [div_le_iff₀ (by positivity)] at hmc
    nlinarith [hmc]
  exact le_of_pow_le_pow_left₀ two_ne_zero (by positivity) hsq

set_option maxHeartbeats 1000000 in
/-- At the top count a named population's heavy prefixes of each class fit the FNR test's share
and, charged at the class's rates, the gate's: its members' table shift, the hash's deviation on
it and on the window's size, the harvested strings, and the uniform pool's share of the window. -/
lemma solved_heavyNamed (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    2 ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
    ∧ hsz + ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
        hsz : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
      ≤ prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz
    ∧ (∀ b, namedQ O η₀ (shiftBudget η₀ indecisionLimit εcov)
        (Real.log (128 * (populations.card : ℝ)
          * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
        (cutBudget η₀ indecisionLimit εcov / 1024) hsz
        ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
          hsz : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
        (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) b
      ≤ indecisionLimit / 8
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ))
    ∧ (∀ b, sideCharge O η₀ (!b) * namedQ O η₀ (shiftBudget η₀ indecisionLimit εcov)
        (Real.log (128 * (populations.card : ℝ)
          * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ))
        (cutBudget η₀ indecisionLimit εcov / 1024) hsz
        ⌈Real.sqrt ((prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
          hsz : ℝ) * Real.log (128 * (populations.card : ℝ) / δ))⌉₊
        (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz) b
      ≤ min εcov (1 / 2) / 32
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)) := by
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set cb : ℝ := cutBudget η₀ indecisionLimit εcov with hcb
  set Δ : ℝ := shiftBudget η₀ indecisionLimit εcov with hΔdef
  set L : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ) with hLdef
  set l1 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) with hl1def
  set G : ℕ := ⌈Real.sqrt ((m : ℝ) * l1)⌉₊ with hGdef
  have hcb0 : 0 < cb := cutBudget_pos η₀ hη₀ hε hind
  have hΔ0 : 0 < Δ := shiftBudget_pos η₀ hη₀ hε hind
  have hΔcb : 32 * Δ ≤ cb := shiftBudget_le_cutBudget η₀ indecisionLimit εcov hind hε hη0 hη₀
  have hcb16 : cb ≤ 1 / 16 := by
    have h := (cutBudget_le η₀ indecisionLimit εcov).2.1
    rw [sig] at h
    linarith
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hl1 : 4 ≤ l1 := by
    have h1 : Real.log 128 ≤ l1 := Real.log_le_log (by norm_num) (by
      rw [le_div_iff₀ hδ]; nlinarith [hδ1])
    have h128 : (4 : ℝ) ≤ Real.log 128 := by
      rw [show (128 : ℝ) = 2 ^ 7 by norm_num, Real.log_pow]
      have := Real.log_two_gt_d9
      push_cast; linarith
    linarith
  have hl1L : l1 ≤ L := Real.log_le_log (by positivity)
    (by rw [div_le_div_iff_of_pos_right hδ]; nlinarith)
  have hL0 : 0 ≤ L := by linarith
  -- the counts the schedule solves for
  have c1 : 64 * L / Δ ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * L / Δ⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef, ← hΔdef, ← hLdef]; omega
    exact_mod_cast hle
  have c2 : 4096 * l1 / (9 * cb ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈4096 * l1 / (9 * cb ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hcb, ← hl1def]; omega
    exact_mod_cast hle
  have c3 : 128 * (hsz : ℝ) / cb ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈128 * (hsz : ℝ) / cb⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hcb]; omega
    exact_mod_cast hle
  rw [div_le_iff₀ hΔ0] at c1
  rw [div_le_iff₀ (by positivity)] at c2 c3
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hG1 : (G : ℝ) ≤ Real.sqrt ((m : ℝ) * l1) + 1 := (Nat.ceil_lt_add_one (by positivity)).le
  have hT5 : Real.sqrt ((m : ℝ) * l1) ≤ 3 * cb * m / 64 := by
    rw [show 3 * cb * (m : ℝ) / 64 = Real.sqrt ((3 * cb * m / 64) ^ 2) by
      rw [Real.sqrt_sq (by positivity)]]
    refine Real.sqrt_le_sqrt ?_
    have hlm : 4096 * l1 ≤ (m : ℝ) * (9 * cb ^ 2) := c2
    have : (m : ℝ) * (4096 * l1) ≤ (m : ℝ) * ((m : ℝ) * (9 * cb ^ 2)) :=
      mul_le_mul_of_nonneg_left hlm hmR
    have e : (m : ℝ) * ((m : ℝ) * (9 * cb ^ 2)) = 4096 * (3 * cb * m / 64) ^ 2 := by
      ring
    nlinarith
  have hcm1 : 8192 ≤ cb * (m : ℝ) := by
    have := mul_le_mul_of_nonneg_right hΔcb hmR
    nlinarith
  have hsz' : (hsz : ℝ) ≤ cb * m / 128 := by linarith
  have hX0 : (0 : ℝ) ≤ (hsz : ℝ) + 2 * G + cb / 1024 * m := by positivity
  have hX : (hsz : ℝ) + 2 * G + cb / 1024 * m ≤ 1029 / 10000 * (cb * m) := by nlinarith
  have hs := sig_pos η₀ hη₀
  have hs1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have hβ := shiftScale_pos hε hind
  have hshare : ∀ b, ((Δ / clsShift O b) * m + 2 * Real.sqrt (L * ((Δ / clsShift O b) * m))
        + 2 * L) / clsFrac O η₀ b + ((hsz : ℝ) + 2 * G + cb / 1024 * m)
      ≤ indecisionLimit / 8 * m
    ∧ sideCharge O η₀ (!b) * (((Δ / clsShift O b) * m
        + 2 * Real.sqrt (L * ((Δ / clsShift O b) * m)) + 2 * L) / clsFrac O η₀ b
        + ((hsz : ℝ) + 2 * G + cb / 1024 * m))
      ≤ min εcov (1 / 2) / 32 * m := by
    intro b
    have hσ := clsSig_ge O hηle hη₀ b
    rw [← clsFrac_mul_clsShift O η₀ (gap_pos O (lt_of_le_of_lt hηle hη₀)) b] at hσ
    have hsc := sideCharge_le_clsSig O hηle hη₀ b
    rw [← clsFrac_mul_clsShift O η₀ (gap_pos O (lt_of_le_of_lt hηle hη₀)) b] at hsc
    exact named_share_le hs hs1 hβ (min_le_right _ _) (min_le_left _ _) rfl
      (clsShift_pos O hηle hη₀ b) (clsShift_le_one O hηle hη₀ b) (clsFrac_ge O hηle hη₀ b) hσ
      (sideCharge_nonneg O η₀ _) hsc (gateA_pos hη0 hη₀) (sideCharge_le_gateA O hηle hη₀ _)
      hL0 hmR (by linarith [c1]) hX0 hX (cutBudget_le η₀ indecisionLimit εcov).2.2
      (cutBudget_le_gate η₀ indecisionLimit εcov)
  refine ⟨?_, ?_, fun b => ?_, fun b => ?_⟩
  · have : (2 : ℝ) ≤ m := by nlinarith
    exact_mod_cast this
  · have : (hsz : ℝ) + G ≤ m := by nlinarith
    exact_mod_cast this
  · have e : namedQ O η₀ Δ L (cb / 1024) hsz G m b
        = ((Δ / clsShift O b) * m + 2 * Real.sqrt (L * ((Δ / clsShift O b) * m))
          + 2 * L) / clsFrac O η₀ b + ((hsz : ℝ) + 2 * G + cb / 1024 * m) := by
      unfold namedQ; ring
    rw [e]; exact (hshare b).1
  · have e : namedQ O η₀ Δ L (cb / 1024) hsz G m b
        = ((Δ / clsShift O b) * m + 2 * Real.sqrt (L * ((Δ / clsShift O b) * m))
          + 2 * L) / clsFrac O η₀ b + ((hsz : ℝ) + 2 * G + cb / 1024 * m) := by
      unfold namedQ; ring
    rw [e]; exact (hshare b).2

/-- The count rule's two tails at the top rung, each at `δ/16`. -/
lemma solved_countTails (η₀ : ℝ) (populations : Finset J) (hpop : populations.Nonempty)
    {εcov δ pAP : ℝ} (hδ : 0 < δ) (hpAP : 0 < pAP) (hsig : η₀ < 1 / 2) :
    apFloor pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
        * Real.exp (-(2 * cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit ^ 2))
        / (apFloor pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit)
          - (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
      ≤ δ / 16
    ∧ ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) * (15 / 16)
        / ((1 - 15 / 16) * (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ))
        * Real.exp (-(2 / 5 * heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit
          ^ 2))
      ≤ δ / 16 := by
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκdef
  set ε : ℝ := δ / 16 with hεdef
  have hε : 0 < ε := by positivity
  have hκ4' : 4 ≤ κ := famCount_ge_four (indecisionLimit := indecisionLimit)
    (crossLimit := crossLimit) η₀ populations εcov δ hsig
  have hκ4 : (4 : ℝ) ≤ κ := by exact_mod_cast hκ4'
  -- the pool's clean rows beyond the family
  have hMκ : 2 * ((κ : ℝ) + 1) / pAP ≤ (M : ℝ) := by
    rw [hMdef, poolCount]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊]
  set a : ℝ := apFloor pAP M with hadef
  have haκ : (κ : ℝ) ≤ a := by
    rw [hadef, apFloor]
    rw [div_le_iff₀ hpAP] at hMκ
    nlinarith
  have had : 0 < a - κ + 1 := by linarith
  have ha0 : 0 < a := by linarith
  constructor
  · set X : ℝ := a / ((a - κ + 1) * ε) with hX
    have hX0 : 0 < X := by positivity
    by_cases hX1 : 1 ≤ X
    · have hl : 0 ≤ Real.log X := Real.log_nonneg hX1
      have hu2 : cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit ^ 2
          = Real.log X / 2 := by
        rw [cleanLevel, ← hMdef, ← hκdef, ← hadef, ← hεdef, ← hX, Real.sq_sqrt (by positivity)]
      rw [hu2, show -(2 * (Real.log X / 2)) = -Real.log X by ring, Real.exp_neg,
        Real.exp_log hX0, hX]
      rw [show a * (a / ((a - κ + 1) * ε))⁻¹ / (a - κ + 1) = ε by field_simp]
    · push_neg at hX1
      have hu0 : 0 ≤ cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit ^ 2 :=
        sq_nonneg _
      have he : Real.exp (-(2 * cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit
          ^ 2)) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
      have : a / (a - κ + 1) < ε := by
        rw [hX, div_lt_one (by positivity)] at hX1
        rw [div_lt_iff₀ had]; linarith
      calc a * Real.exp (-(2 * cleanLevel η₀ populations indecisionLimit εcov δ pAP crossLimit
            ^ 2)) / (a - κ + 1) ≤ a * 1 / (a - κ + 1) := by gcongr
        _ ≤ ε := by rw [mul_one]; exact this.le
  · set Y : ℝ := 15 * ((M : ℝ) + 1) / ((κ : ℝ) * ε) with hY
    have hY0 : 0 < Y := by positivity
    have hcoef : ((M : ℝ) + 1) * (15 / 16) / ((1 - 15 / 16) * (κ : ℝ)) = Y * ε := by
      rw [hY]; field_simp; ring
    rw [hcoef]
    by_cases hY1 : 1 ≤ Y
    · have hl : 0 ≤ Real.log Y := Real.log_nonneg hY1
      have hx2 : 5 / 2 * Real.log Y
          ≤ heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit ^ 2 := by
        have h := le_max_right (2 : ℝ) (Real.sqrt (5 / 2 * Real.log Y))
        have h0 : 0 ≤ Real.sqrt (5 / 2 * Real.log Y) := Real.sqrt_nonneg _
        have := pow_le_pow_left₀ h0 h 2
        rw [Real.sq_sqrt (by positivity)] at this
        simpa [heavyLevel, ← hMdef, ← hκdef, ← hεdef, ← hY] using this
      have he : Real.exp (-(2 / 5 * heavyLevel η₀ populations indecisionLimit εcov δ pAP
          crossLimit ^ 2)) ≤ Y⁻¹ := by
        rw [← Real.exp_log hY0, ← Real.exp_neg]
        exact Real.exp_le_exp.2 (by nlinarith)
      calc Y * ε * Real.exp (-(2 / 5 * heavyLevel η₀ populations indecisionLimit εcov δ pAP
            crossLimit ^ 2)) ≤ Y * ε * Y⁻¹ := by gcongr
        _ = ε := by field_simp
    · push_neg at hY1
      have he : Real.exp (-(2 / 5 * heavyLevel η₀ populations indecisionLimit εcov δ pAP
          crossLimit ^ 2)) ≤ 1 := Real.exp_le_one_iff.2 (by
            have := sq_nonneg (heavyLevel η₀ populations indecisionLimit εcov δ pAP crossLimit)
            linarith)
      calc Y * ε * Real.exp (-(2 / 5 * heavyLevel η₀ populations indecisionLimit εcov δ pAP
            crossLimit ^ 2)) ≤ Y * ε * 1 := by gcongr
        _ ≤ ε := by nlinarith

set_option maxHeartbeats 4000000 in
/-- The computed ladder holds a state whose round can pass: the top rung, every clause
discharged from the closed forms that define it. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J)
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hεs : 0 < εs) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ0 : 0 ≤ ρ) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (hρsfcap : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    ∃ B : State, B ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz ρ
      ∧ PassableAt O η₀ populations Dsf indecisionLimit εcov α δ ρ ρsf pAP hsz B := by
  classical
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hη0 : 0 ≤ O.η := eta_nonneg O
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hη₀ hεcov hindLim
  have hv := voteSlack_pos η₀ hη₀
  have hη0' : 0 ≤ η₀ := le_trans hη0 hηle
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
  have hv5 : voteSlack η₀ ≤ 2 / 5 := by rw [voteSlack, sig]; linarith
  have hq0 : (0 : ℝ) ≤ 1 / 2 + voteSlack η₀ := by linarith
  have hq1 : 1 / 2 + voteSlack η₀ ≤ 1 := by linarith
  have hmisle : 1 - binomSfGe κ (1 / 2 + voteSlack η₀) (loAt η₀ κ + 1) ≤ misfireTail η₀ κ := by
    rw [misfireTail]
    refine sub_le_sub_left (le_trans (binomSfGe_succ_le hq0 hq1 (hiAt η₀ κ + 1)) ?_) 1
    exact binomSfGe_antitone' hq0 hq1 (by omega)
  have hκpos : 0 < κ := by omega
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedState η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz with
      hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  have hBkR : (B.k : ℝ) = κ + 1 := by rw [hBk]; push_cast; ring
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := rfl
  have hBlo : B.lo = loAt η₀ κ := rfl
  have hBhi : B.hi = hiAt η₀ κ := rfl
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
  have hgate := solved_gate (crossLimit := crossLimit) (hsz := hsz) (pAP := pAP) (δ := δ) η₀
    populations hη₀
    hη0' hεcov hεs
    hindLim hαpos (by linarith) hcard1
  rw [← hBdef] at hgate
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  have hheavyN := solved_heavyNamed (εs := εs) (crossLimit := crossLimit) (hsz := hsz) (α := α)
      (pAP := pAP)
    O η₀ populations hηle hη₀ hη0' hεcov hindLim hδ hδ1 hcard1
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
  clear_value B κ m
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov εs := rejFrac_pos η₀ hη₀ hεcov hεs hindLim
  obtain ⟨hcm, hcu, -⟩ := rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀
    (εcov := εcov)
  have hΔ0 := shiftBudget_pos η₀ hη₀ hεcov hindLim
  have hs1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have hβ := shiftScale_pos hεcov hindLim
  have hgap := gap_pos O hsig
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_⟩⟩,
    9 * pAP / 20, 1 / 2 + voteSlack η₀, misfireTail η₀ κ,
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
    ?_, ?_, ?_, ?_, ?_, ?_⟩
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
    exact le_add_of_le_of_nonneg (solved_share η₀ populations hη₀ hη0' hεcov hεs hindLim hδ hδ1
      hpAPPositive hcard hρ0 hρcap) (by split_ifs <;> positivity)
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · exact hindLim
  · exact hind1
  · positivity
  · exact hΔ0
  · exact hΔ0
  · linarith
  · positivity
  · exact hc0.le
  · rw [hBm]
    exact Nat.floor_le (by positivity)
  · exact hpAPBound
  · exact hρsf
  -- each class's heavy prefixes at the uniform pool
  · intro b
    have hσ := clsSig_ge O hηle hη₀ b
    exact (uni_share_le hs hs1 hβ (min_le_right _ _) (min_le_left _ _) rfl hc0.le hcm hcu
      (clsShift_pos O hηle hη₀ b) (clsShift_le_one O hηle hη₀ b) hσ
      (clsShift_div_clsSig_le O hηle hη₀ b) (sideCharge_nonneg O η₀ _)
      (sideCharge_le_clsSig O hηle hη₀ b) (sideCharge_le_gateA O hηle hη₀ _)
      (two_sig_le_gateA hη0' hη₀)).1
  · intro b
    have hσ := clsSig_ge O hηle hη₀ b
    exact (uni_share_le hs hs1 hβ (min_le_right _ _) (min_le_left _ _) rfl hc0.le hcm hcu
      (clsShift_pos O hηle hη₀ b) (clsShift_le_one O hηle hη₀ b) hσ
      (clsShift_div_clsSig_le O hηle hη₀ b) (sideCharge_nonneg O η₀ _)
      (sideCharge_le_clsSig O hηle hη₀ b) (sideCharge_le_gateA O hηle hη₀ _)
      (two_sig_le_gateA hη0' hη₀)).2
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
  · intro p
    have h := rate_add_gap_fracAt O hηle hη₀ p
    exact ⟨fracAt_nonneg O hηle hη₀ p, by linarith, by linarith⟩
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
  · linarith
  · linarith [hcutle.2.2]
  · rw [hBk, hBhi]; exact le_rfl
  · rw [hBk, hBlo, show κ + 1 - loAt η₀ κ = hiAt η₀ κ + 1 + 1 by omega]; exact le_rfl
  · rw [hBkappa, hBlo]; exact hmisle
  · rw [hBkappa, hBhi, show κ - hiAt η₀ κ = loAt η₀ κ + 1 by omega]; exact hmisle
  · have h := solved_roundFail (crossLimit := crossLimit) (α := α) (ρ := ρ) (ρsf := ρsf)
      η₀ populations hη₀ hη0' hεcov hεs hδ hδ1 hpAPPositive hpAP1 hindLim
      hcard hρ0 hρsf0 hρcap hρsfcap
    rw [← hBdef, ← hmdef, ← hκdef] at h
    linarith

#print axioms validity_of_returned

end Loop

end OrthoDFA
