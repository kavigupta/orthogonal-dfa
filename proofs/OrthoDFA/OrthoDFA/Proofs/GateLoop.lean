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
/-- The round's own test at one population: the family's size, the FNR count and the gate. -/
noncomputable def retAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit α : ℝ) (B : State) (j : J) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule O.mq populations x B).card + 1
    ∧ (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (familyBy rule O.mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
    ∧ admitted O.mq B.lo B.hi α (clusterBy rule O.mq populations x B)
        (certOf (prefixesAt populations B.npref x) j B.npref x) (oracleNoise x)}

lemma mem_ret_of_retAt (O : Oracle μ S) (populations : Finset J) (hpop : populations.Nonempty)
    (indecisionLimit α : ℝ) (B : State) (x : Run Ω S J)
    (h : ∀ j ∈ populations, x ∈ retAt rule O populations indecisionLimit α B j) :
    x ∈ retAll rule O.mq populations indecisionLimit α B := by
  obtain ⟨j₀, hj₀⟩ := hpop
  exact ⟨(h j₀ hj₀).1, fun j hj => (h j hj).2.1, fun j hj => (h j hj).2.2⟩

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

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- The round returns at one population.  Everything outside `retMiss` is a fact about
the draws: the certification sample is not found in time, the family is dirty and the draws see
it, or the cluster has the wrong size, which is the liveness half's business and is carried as
`Estall`. -/
theorem measureReal_notRetAt_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (hmpos : 0 < B.npref) (hsig : O.η ≤ 1 / 2)
    (α τ l lcut f E Δp th c : ℝ) (r : ℕ) (kmin kmax nlo : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ)
    (hl1 : 2 * l ≤ 1) (hnloB : (nlo : ℝ) ≤ (1 - 2 * l) * (B.npref : ℝ))
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (hf0 : 0 < f)
    (hr : (r : ℝ) ≤ c * B.npref)
    (hheavy : Δp / f + th + c ≤ lcut)
    (Ecert : ℝ) (hcertBad : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ certGood populations j B.npref r x} ≤ Ecert)
    (Estall : ℝ) (hstall : (runMeasure μ D Dsf).real (stalled rule O populations B kmin kmax) ≤ Estall)
    (Edirty : ℝ) (hdirty : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E)
    (hga : ∀ n c : ℕ, nlo ≤ n → n ≤ c → c ≤ B.npref →
      (n : ℝ) * (1 / 2 + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (c : ℝ)))
    (hnlo0 : 0 < nlo) (hα : Real.exp (-2 * (nlo : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations (2 * l) α B j}
      ≤ Ecert
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by
  classical
  set E1 : Set (Run Ω S J) := {x | ¬ certGood populations j B.npref r x} with hE1
  set E5 : Set (Run Ω S J) := stalled rule O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits rule O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := retMiss rule O populations j B α l lcut f kmin kmax with hE8
  have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
  have hsub : {x : Run Ω S J | x ∉ retAt rule O populations (2 * l) α B j}
      ⊆ E1 ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8))) := by
    intro x hx
    by_cases h1 : certGood populations j B.npref r x
    · by_cases h5 : kmin ≤ (clusterBy rule O.mq populations x B).card
          ∧ (clusterBy rule O.mq populations x B).card ≤ kmax
      · by_cases h6 : ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D j) v ≤ Δp
        · -- the certification sample is good and the family is clean
          obtain ⟨hcardN, hcnt⟩ := certOf_of_good populations j B.npref r x h1
          have hcard : ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
              = (B.npref : ℝ) := by exact_mod_cast hcardN
          have hCpos : 0 < (certOf (prefixesAt populations B.npref x) j B.npref x).card := by
            rw [hcardN]; exact hmpos
          -- Markov over the family: the mean flip fraction is at most `Δp`, so the mass
          -- where an `f` fraction flips is `Δp / f`, with no family-size factor.
          have hmass : (D j).real {p | ¬ (flipCount O (clusterBy rule O.mq populations x B) p
              ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)}
              ≤ Δp / f := by
            set F : Finset S := clusterBy rule O.mq populations x B with hFdef
            rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
            · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)}
                  = (∅ : Set S) := by
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
              refine le_trans (measureReal_mono hsetle (measure_ne_top _ _)) ?_
              exact flipCount_mass_le O (D j) F Δp f hf0 hFpos
                (fun v hv => h6 v hv)
          by_cases h7 : (B.npref : ℝ) * (Δp / f + th)
              ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
                ¬ (flipCount O (clusterBy rule O.mq populations x B) (certPrefix j i.val x)
                  ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)
          · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩)))
          · refine Or.inr (Or.inr (Or.inr (Or.inr
              ⟨disjoint_freshDraws _ _ _ |>.symm, hcardN, hCpos, ?_, ?_⟩)))
            · have hlt := not_le.1 h7
              have hH := (hcnt (fun p => ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).2
              have hle : (((certOf (prefixesAt populations B.npref x) j B.npref x).filter
                  (fun p => ¬ famGood O f kmin kmax (clusterBy rule O.mq populations x B) p)).card
                    : ℝ)
                  = (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
                    ¬ (flipCount O (clusterBy rule O.mq populations x B) p
                      ≤ ((clusterBy rule O.mq populations x B).card : ℝ)
                        * f))).card : ℝ) := by
                refine congrArg (fun t : Finset S => (t.card : ℝ))
                  (Finset.filter_congr (fun p _ => ?_))
                unfold famGood
                simp only [h5.1, h5.2, and_true, true_and]
              rw [hle, hcard]
              have hb := mul_le_mul_of_nonneg_left hheavy hm0
              have hcm : (r : ℝ) ≤ c * (B.npref : ℝ) := hr
              nlinarith
            · intro hgood
              refine hx ⟨?_, hgood⟩
              omega
        · exact Or.inr (Or.inr (Or.inl h6))
      · exact Or.inr (Or.inl h5)
    · exact Or.inl h1
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | x ∉ retAt rule O populations (2 * l) α B j}
      ≤ (runMeasure μ D Dsf).real (E1 ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8)))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runMeasure μ D Dsf).real E1
        + ((runMeasure μ D Dsf).real E5 + ((runMeasure μ D Dsf).real E6
          + ((runMeasure μ D Dsf).real E7 + (runMeasure μ D Dsf).real E8))) := by
        have h78 := measureReal_union_le (μ := runMeasure μ D Dsf) E7 E8
        have h678 := measureReal_union_le (μ := runMeasure μ D Dsf) E6 (E7 ∪ E8)
        have h5678 := measureReal_union_le (μ := runMeasure μ D Dsf) E5 (E6 ∪ (E7 ∪ E8))
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) E1
          (E5 ∪ (E6 ∪ (E7 ∪ E8)))
        linarith
    _ ≤ Ecert
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) :=
        add_le_add hcertBad (add_le_add hstall (add_le_add hdirty (add_le_add
          (measureReal_heavyHits_le D Dsf O populations j B f (Δp / f) th hth)
          (measureReal_retMiss_le hflat O populations D Dsf hsupp hsuppSf j hj B α τ l lcut f E
            kmin kmax nlo hE hElcut hlcl hτ hl1 hnloB hsig hdec hcut hga hnlo0 hα))))

lemma exp_tail_anti {m a b : ℝ} (hm : 0 ≤ m) (ha : 0 ≤ a) (hab : a ≤ b) :
    Real.exp (-2 * m * b ^ 2) ≤ Real.exp (-2 * m * a ^ 2) := by
  refine Real.exp_le_exp.2 ?_
  have h := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ ha hab 2) hm
  nlinarith

/-! ### How `per_state_le` gets its bound

Validity is read off how the family was built, not off the gate.  Where the noise rate
depends on the class, a correct cut's agreement with the seed's column sits anywhere from
`1 − η` to `1` according to the population's mix of classes, so no null separates it from a
cut drifted on an `εcov` fraction.  What does bound the drift is the screen: it keeps no
candidate that flips more than `validFlip`, scaled by `validFracOf/validFrac`, of the table,
Markov over the family turns that
into a bound on the mass where many members flip, and the vote is right elsewhere.

`measureReal_validFail_le` carries that through the certification sample, which the family
never saw: a cut wrong on `εcov` of the population is wrong on `15εcov/16` of the sample
(`hitShort`), few of those can be prefixes the family flips (`heavyHits`), and few of the
rest can be misread by a family that barely flips there (`validMiss`, Hoeffding). -/

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

/-- The family at a state is invalid: on some population its cut is wrong on more than the gate
holds that population to, or it leaves more than `2·indecisionLimit` undecided. -/
def FailAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (D : J → Measure S) (indecisionLimit εcov : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - gateLimit uni εcov j
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

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state.  A returned family invalid on some population failed the gate's
soundness there — its calibration sample's sides, a look's sides, or the passage to the
population — or slipped past the FNR test, which the certification draws see unless their
sampler redrew too much. -/
theorem per_state_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (indecisionLimit εcov α : ℝ)
    (hil : 0 < indecisionLimit) (hα0 : 0 < α) (hα1 : α ≤ 1) (B : State) (hk : 0 < B.k)
    (r : ℕ) (c : ℝ) (hr : (r : ℝ) ≤ c * B.npref) (hc : c ≤ indecisionLimit / 2)
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (G : Set (Run Ω S J))
    (hG : ∀ x ∉ G, ∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p) :
    (runMeasure μ D Dsf).real ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations uni D indecisionLimit εcov B) \ G)
      ≤ 13 * populations.card * gateLevel α populations B
        + ((populations.card : ℝ) * ((populations.card : ℝ) + 1)
            * ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / ((r : ℝ) + 1)
          + populations.card * Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2)) := by
  classical
  have hJ : (1 : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_pos.2 ⟨uni, huni⟩
  set a : ℝ := gateLevel α populations B with ha
  have ha0 : 0 < a := sizeLevel_pos (by positivity) _
  have ha1 : a ≤ 1 := sizeLevel_le (by positivity)
    (by rw [div_le_one (by positivity)]; nlinarith) _
  set calBad : Set (Run Ω S J) := sidesBad rule O populations uni B B.npref (sizeLevel a B.npref)
    with hcal
  set gateJ : J → Set (Run Ω S J) := fun j =>
    (⋃ e ∈ Finset.range (B.npref + 1),
        sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e)))
      ∪ ⋃ N : ℕ, rawShortAt rule O populations (D j) j B (gateLimit uni εcov j) a (N + 1)
    with hgateJ
  set indJ : J → Set (Run Ω S J) := fun j =>
    undecidedShort rule O populations (D j) j B (2 * indecisionLimit) (indecisionLimit / 2)
    with hindJ
  have hsub : ((retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∩ FailAt rule O populations uni D indecisionLimit εcov B) \ G)
      ⊆ calBad ∪ ((⋃ j ∈ populations, (gateJ j ∪ indJ j))
        ∪ {x : Run Ω S J | ¬ drawsGood populations B.npref r x}) := by
    rintro x ⟨⟨hret, hfail⟩, hxG⟩
    obtain ⟨hsize, hfnr, hgate⟩ := hret
    simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
    obtain ⟨j, hj, hfj⟩ := hfail
    by_cases hmass : 1 - gateLimit uni εcov j ≤ (D j).real {p | cutCorrect O B.lo B.hi
        (clusterBy rule O.mq populations x B) p (oracleNoise x)}
    · -- the cut is right enough, so the FNR test let through an undecided cut
      have hund := (not_and.1 hfj) hmass
      by_cases hdg : drawsGood populations B.npref r x
      · refine Or.inr (Or.inl (Set.mem_biUnion hj (Or.inr ?_)))
        obtain ⟨hcardN, hcnt⟩ := certOf_of_good populations j B.npref r x (hdg.2 j hj)
        have hcard : ((certOf (prefixesAt populations B.npref x) j B.npref x).card : ℝ)
            = (B.npref : ℝ) := by exact_mod_cast hcardN
        refine ⟨(not_le.1 hund).le, ?_⟩
        have h := (hcnt (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (familyBy rule O.mq populations x B) p (oracleNoise x))).1
        have hf := hfnr j hj
        rw [hcard] at hf
        have hm0 : (0 : ℝ) ≤ B.npref := Nat.cast_nonneg _
        nlinarith
      · exact Or.inr (Or.inr hdg)
    · -- the cut is wrong on more than the limit, and the gate passed it
      have hgt : gateLimit uni εcov j < (D j).real {p | ¬ cutCorrect O B.lo B.hi
          (clusterBy rule O.mq populations x B) p (oracleNoise x)} := by
        rw [show {p : S | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}
            = {p : S | cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
              (oracleNoise x)}ᶜ from rfl,
          measureReal_compl (measurableSet_of_countable _), probReal_univ]
        linarith [not_le.1 hmass]
      have hin := falseCertify_subset (rule := rule) O hηle hη₀ populations (D j) uni j B
        (gateLimit uni εcov j) a ⟨hgate j hj, hG x hxG, hgt⟩
      rcases hin with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl (Set.mem_biUnion hj (Or.inl h)))
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 : (runMeasure μ D Dsf).real calBad ≤ 6 * a :=
    le_trans (measureReal_sidesBad_le hflat O populations D Dsf hsupp hsuppSf uni huni B B.npref
      (sizeLevel_pos ha0 _) (sizeLevel_le ha0.le ha1 _))
      (by have := sizeLevel_le_self ha0.le B.npref; linarith)
  have h2 : (runMeasure μ D Dsf).real ((⋃ j ∈ populations, (gateJ j ∪ indJ j))
        ∪ {x : Run Ω S J | ¬ drawsGood populations B.npref r x})
      ≤ populations.card * (7 * a + Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2))
        + (populations.card : ℝ) * ((populations.card : ℝ) + 1)
            * ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / ((r : ℝ) + 1) := by
    refine le_trans (measureReal_union_le _ _) (add_le_add ?_
      (measureReal_drawsBad_le D Dsf populations B.npref r ρ hρ hρ0))
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun j hj => le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_))
    · exact measureReal_popGate_le hflat O populations D Dsf hsupp hsuppSf j hj B ha0 ha1
    · exact measureReal_undecidedShort_le D Dsf O populations j B _ _ (by linarith)
        (by linarith)
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
    ∑ B ∈ s, 13 * (populations.card : ℝ) * gateLevel α populations B ≤ α := by
  classical
  have hJ : (0 : ℝ) < populations.card := by exact_mod_cast Finset.card_pos.2 hpop
  have e : ∀ B : State, 13 * (populations.card : ℝ) * gateLevel α populations B
      = sizeLevel α B.npref := by
    intro B
    have h13 : (13 * (populations.card : ℝ)) ≠ 0 := by positivity
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
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α pBal : ℝ) (hindLim : 0 < indecisionLimit)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hα0 : 0 < α) (hα1 : α ≤ 1) (hpBal : 0 < pBal)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (pAP : ℝ) (hpAP0 : 0 < pAP) (hpAP1 : pAP ≤ 1)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcoll : ((poolCount η₀ populations (cutLimit η₀ indecisionLimit pBal) εcov δ pAP crossLimit
        + poolRej pAP (poolCount η₀ populations (cutLimit η₀ indecisionLimit pBal) εcov δ pAP
          crossLimit) : ℕ) : ℝ) ^ 2 * ρsf
        / (2 * ((poolRej pAP (poolCount η₀ populations (cutLimit η₀ indecisionLimit pBal) εcov δ
          pAP crossLimit) : ℝ) + 1))
      ≤ δ / 16) :
    (runMeasure μ D Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ populations (cutLimit η₀ indecisionLimit pBal) εcov
            δ α pAP crossLimit ρ},
          retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
            ∩ FailAt rule O populations uni D indecisionLimit εcov B.val) ≤ δ / 2 + α := by
  classical
  set iLc : ℝ := cutLimit η₀ indecisionLimit pBal with hiLc
  have hiLc0 : 0 < iLc := lt_min hindLim (by have := sig_pos η₀ hη₀; rw [sig] at this; positivity)
  have hiLc1 : iLc ≤ indecisionLimit := min_le_left _ _
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  have hJ : (1 : ℝ) ≤ populations.card := by exact_mod_cast Finset.card_pos.2 hpop
  have hρ0 : 0 ≤ ρ := le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ uni huni)
  set M : ℕ := poolCount η₀ populations iLc εcov δ pAP crossLimit with hMdef
  set rs : ℕ := poolRej pAP M with hrsdef
  set N : ℕ := prefCount η₀ populations iLc εcov δ α pAP crossLimit with hNdef
  set V : ℕ := validCount η₀ populations iLc εcov δ pAP crossLimit with hVdef
  set G : Set (Run Ω S J) := apShort O M pAP (9 * pAP / 20) ∪ {x | ¬ poolGood M rs x} with hGdef
  set st := stoppable η₀ populations iLc εcov δ α pAP crossLimit ρ with hst
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
        ∩ FailAt rule O populations uni D indecisionLimit εcov B) \ G)
      ≤ 13 * populations.card * gateLevel α populations B
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
    have h := per_state_le (rule := rule) hflat O populations uni huni D Dsf hsupp hsuppSf hηle
      hη₀ indecisionLimit εcov α hindLim hα0 hα1 B hk r c hr (by linarith [hcutle.2.2]) ρ hρ hρ0
      G hG'
    refine le_trans h (add_le_add le_rfl (le_trans ?_ hcap.share))
    -- the redraws, from `(m + r)²/(r + 1)` to the linear `2m/c`, and the FNR tail at the
    -- schedule's indecision rate
    have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
    have hc4 : c ≤ 1 / 4 := by
      have h2 := hcutle.2.1
      rw [sig] at h2
      linarith
    have hrt := rejTerm_le hc0 hc4 B.npref
    have hn : ((B.npref + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1) ≤ 2 * (B.npref : ℝ) / c := hrt
    have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
    have hdraw1 : (populations.card : ℝ) * ((populations.card : ℝ) + 1)
          * ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / ((r : ℝ) + 1)
        ≤ (populations.card : ℝ) * ((populations.card : ℝ) + 1) * ρ
          * (2 * (B.npref : ℝ) / c) := by
      rw [show (populations.card : ℝ) * ((populations.card : ℝ) + 1)
          * ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / ((r : ℝ) + 1)
          = (populations.card : ℝ) * ((populations.card : ℝ) + 1) * ρ
            * (((B.npref + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1)) by ring]
      gcongr
    have hexp : Real.exp (-2 * (B.npref : ℝ) * (indecisionLimit / 2) ^ 2)
        ≤ Real.exp (-2 * (B.npref : ℝ) * (iLc / 2) ^ 2) :=
      exp_tail_anti (Nat.cast_nonneg _) (by positivity) (by linarith)
    have hlin : (populations.card : ℝ) * ((populations.card : ℝ) + 1) * ρ
          * (2 * (B.npref : ℝ) / c)
        ≤ (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (B.npref : ℝ) * ρ / c) := by
      have : 0 ≤ (populations.card : ℝ) * (B.npref : ℝ) * ρ / c := by positivity
      have e : (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (B.npref : ℝ) * ρ / c)
          - (populations.card : ℝ) * ((populations.card : ℝ) + 1) * ρ * (2 * (B.npref : ℝ) / c)
          = (populations.card : ℝ) * (B.npref : ℝ) * ρ / c := by field_simp; ring
      linarith
    unfold stateFail
    nlinarith [mul_le_mul_of_nonneg_left hexp hJ0]
  have hsplit : (⋃ B : {B : State // B ∈ st},
        retBy rule O.mq populations uni η₀ indecisionLimit εcov α B.val
          ∩ FailAt rule O populations uni D indecisionLimit εcov B.val)
      ⊆ (⋃ B ∈ st, (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
          ∩ FailAt rule O populations uni D indecisionLimit εcov B) \ G) ∪ G := by
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
        indecisionLimit εcov α B ∩ FailAt rule O populations uni D indecisionLimit εcov B) \ G)
      ≤ α + 3 * δ / 8 := by
    calc _ ≤ ∑ B ∈ st, (runMeasure μ D Dsf).real ((retBy rule O.mq populations uni η₀
            indecisionLimit εcov α B ∩ FailAt rule O populations uni D indecisionLimit εcov B)
              \ G) := measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ st, (13 * populations.card * gateLevel α populations B
            + (δ * (B.npref : ℝ) / (8 * N)
              + if V ≤ B.npref then δ * (V : ℝ) / (16 * B.npref) else 0)) :=
          Finset.sum_le_sum hper
      _ = ∑ B ∈ st, 13 * (populations.card : ℝ) * gateLevel α populations B
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


/-- What one round at one population can cost: the samplers redrawing too much, the family's
size and cleanliness, the sample's heavy fraction, and the round's own two tests. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut τ th E γscr γdirty gdirty tap ρ ρsf : ℝ) (r rs n₀ : ℕ) (B : State) : ℝ :=
  ((B.npref + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
    + ((((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
          + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
            + (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
        + (((((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
              + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
            + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
              + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))))
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
              + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2))))))

/-- A state whose round can pass.  Every clause is an inequality among the state's
budgets, the oracle's rates, the suffix distribution's findability and the error budget —
no probability enters, and nothing here is a free parameter of the algorithm.  Reaching
such a state is what the computed ladder is for.

The gate's sample is the decided part of the certification sample, which the FNR test keeps
at `(1 − indecisionLimit)·npref` at least, so its tail is read at that count.  Nothing here asks
a population to carry both labels: the gate pools its two sides. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (B : State) : Prop :=
  ∃ (τ th tap γdec γscr γdirty gdirty Δ lcut f c : ℝ) (r rs : ℕ),
    0 < B.npref ∧ 2 ≤ B.k ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ εcov ≤ 1
    ∧ 0 ≤ τ ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γdec ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 < f ∧ 0 ≤ pAP ∧ 0 < lcut ∧ lcut ≤ indecisionLimit / 2
    -- the samplers redraw at most a `c` fraction of the prefix draws
    ∧ 0 ≤ c ∧ (r : ℝ) ≤ c * B.npref
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification sample holds few heavy prefixes: Markov over the family, so the
    -- flip mass is charged at the fraction the vote absorbs and not at the family's size; the
    -- table's redraws hide `c` of a member's flips, and the sample's redraws `c` of its heavy
    -- prefixes
    ∧ (((populations.card : ℝ) * Δ + c + gdirty) / f + th + c ≤ lcut)
    -- the screen's rate sits above the clean one and below the dirty one
    ∧ 0 < B.scd
    ∧ ((B.scd : ℝ) * (2 * γscr) ≤ (B.sc : ℝ))
    ∧ ((B.sc : ℝ)
        ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γdirty))
    -- the pool holds a family, beyond what its sampler redraws
    ∧ ((B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - tap))
    -- the thresholds decide, and decide right, on a family of the round's size an `f`
    -- fraction of which flips, so both means move toward the centre by up to `gap·f`;
    -- the FNR reads the family with its seed, cut one count higher
    ∧ ((B.hi : ℝ) + 1 ≤ (B.k : ℝ) * ((1 - O.η - O.gap * f) - γdec))
    ∧ ((B.k : ℝ) * ((O.η + O.gap * f) + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * ((O.η + O.gap * f) + γdec) ≤ (B.hi : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η - O.gap * f) - γdec))
    -- the gate's margin clears its null on every decided count the FNR test allows
    ∧ (∀ n c : ℕ, ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ ≤ n → n ≤ c → c ≤ B.npref →
        (n : ℝ) * (1 / 2 + τ + τ)
          ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (c : ℝ)))
    ∧ 0 < ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊
    ∧ (Real.exp (-2 * (⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℝ) * τ ^ 2) ≤ α)
    -- the vote misfires at under half the cut budget, so the sample's count of misfires
    -- concentrates below the budget
    ∧ (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2) ≤ lcut / 2)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) lcut τ th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf r rs
            ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B
      ≤ δ / 2)

lemma mul_self_add_le_cube {x : ℝ} (hx : 0 ≤ x) : x * (x + 3) ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x * (x + 3) = x ^ 3 + 8 * x ^ 2 + 24 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma le_cube_of_nonneg {x : ℝ} (hx : 0 ≤ x) : x ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x = x ^ 3 + 9 * x ^ 2 + 26 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

/-! ### What the solved budget's fields buy

Each of these says that a field, read off its condition, meets it.  They are the analytic
content of the construction; everything else about `PassableAt` is arithmetic on the
definitions. -/

/-- The cut budget is at least a tenth of the scale. -/
lemma cutBudget_ge (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hsig : η₀ < 1 / 2) :
    budgetScale η₀ indecisionLimit εcov / 10 ≤ cutBudget η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hB : 0 < budgetScale η₀ indecisionLimit εcov := lt_min hε (lt_min hs hind)
  have h1 : budgetScale η₀ indecisionLimit εcov ≤ εcov := min_le_left _ _
  have h2 : budgetScale η₀ indecisionLimit εcov ≤ sig η₀ :=
    (min_le_right _ _).trans (min_le_left _ _)
  have h3 : budgetScale η₀ indecisionLimit εcov ≤ indecisionLimit :=
    (min_le_right _ _).trans (min_le_right _ _)
  rw [cutBudget]
  refine le_min (by linarith) (le_min (by linarith) (by linarith))

/-- The redraw fraction is at least a constant times the scale squared: the cut budget is a tenth
of the scale, and the flip fraction seven tenths of the signal. -/
lemma rejFrac_ge (η₀ indecisionLimit εcov : ℝ) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) :
    7 * budgetScale η₀ indecisionLimit εcov ^ 2 / 12800 ≤ rejFrac η₀ indecisionLimit εcov := by
  have hs := sig_pos η₀ hsig
  have hB : 0 < budgetScale η₀ indecisionLimit εcov := lt_min hε (lt_min hs hind)
  have hBs : budgetScale η₀ indecisionLimit εcov ≤ sig η₀ :=
    (min_le_right _ _).trans (min_le_left _ _)
  have hcut := cutBudget_ge η₀ indecisionLimit εcov hε hind hsig
  have hf : 7 * sig η₀ / 10 ≤ flipFrac η₀ := by
    rw [flipFrac, div_le_div_iff₀ (by norm_num) (by linarith)]
    nlinarith
  rw [rejFrac]
  have := mul_le_mul hcut (le_trans (by linarith : 7 * budgetScale η₀ indecisionLimit εcov / 10
    ≤ 7 * sig η₀ / 10) hf) (by positivity) (by linarith [hcut, cutBudget_pos η₀ hsig hε hind])
  nlinarith

/-- What the collision cap buys: a state's redraws, at `|J|(3|J| + 3)ρ/rejFrac` per draw, are a
sixty-fourth of the error budget spread over the top count. -/
lemma draws_le_of_cap (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hη0 : 0 ≤ η₀) (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hρ0 : 0 ≤ ρ)
    (hρ : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
        / rejFrac η₀ indecisionLimit εcov
      ≤ δ / (64 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)) := by
  set N : ℝ := (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) with hN
  set B : ℝ := budgetScale η₀ indecisionLimit εcov with hBdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  set Jc : ℝ := (populations.card : ℝ) with hJc
  have hs := sig_pos η₀ hsig
  have hB : 0 < B := lt_min hε (lt_min hs hind)
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
  have hJ : Jc * (3 * Jc + 3) ≤ 3 * (Jc + 1) ^ 2 := by nlinarith
  have step1 : Jc * (3 * Jc + 3) * ρ * (64 * N) ≤ 3 * (Jc + 1) ^ 2 * ρ * (64 * N) := by
    have := mul_le_mul_of_nonneg_right hJ (by positivity : (0 : ℝ) ≤ ρ * (64 * N))
    nlinarith
  have step2 : 3 * (Jc + 1) ^ 2 * ρ * (64 * N) * 2 ^ 20 ≤ 192 * δ * B ^ 2 := by
    nlinarith [hcap]
  have step3 : 192 * δ * B ^ 2 ≤ δ * c * 2 ^ 20 := by
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

/-- The gate's floor clears its own tail at the error rate `α`. -/
lemma solved_alpha (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hα : 0 < α)
    (hα1 : α < 1 / 2) (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-2
        * ((⌊(1 - indecisionLimit) * ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref : ℝ)⌋₊ : ℕ) : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ α := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hτ : (0 : ℝ) < sig η₀ * εcov / 4 := by positivity
  have hτ2 : (0 : ℝ) < (sig η₀ * εcov / 4) ^ 2 := pow_pos hτ 2
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the prefix count clears the gate's tail
  have hmlog : 64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  -- and the floor keeps a fraction of it
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have hsize : 2 ≤ εcov * (m : ℝ) / 32 := by
    rw [div_le_iff₀ hε] at hsizeR
    linarith
  have hgmin : εcov * (m : ℝ) / 64
      ≤ ((⌊(1 - indecisionLimit) * ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref : ℝ)⌋₊ : ℕ) : ℝ) := by
    show εcov * (m : ℝ) / 64 ≤ ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
    have hlt := Nat.lt_floor_add_one ((1 - indecisionLimit) * (m : ℝ))
    rw [div_le_iff₀ hε] at hsizeR
    nlinarith [mul_le_mul_of_nonneg_right hε1 hmR]
  rw [show (-2 * ((⌊(1 - indecisionLimit) * ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref : ℝ)⌋₊ : ℕ) : ℝ)
        * (sig η₀ * εcov / 4) ^ 2)
      = -(2 * ((⌊(1 - indecisionLimit) * ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref : ℝ)⌋₊ : ℕ) : ℝ)
        * (sig η₀ * εcov / 4) ^ 2) from by ring]
  refine exp_neg_le_of_log_le hα ?_
  have hlog0 : 0 ≤ Real.log (1 / α) := Real.log_nonneg (by rw [le_div_iff₀ hα]; linarith)
  rw [div_le_iff₀ (by positivity)] at hmlog
  nlinarith [mul_le_mul_of_nonneg_right hgmin hτ2.le, hmR, hlog0]

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

set_option maxHeartbeats 1000000 in
/-- The round's failure, over every population, fits the error budget. -/
lemma solved_roundFail (η₀ : ℝ) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hind : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsfsmall : ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (sig η₀ * εcov / 4)
            (cutBudget η₀ indecisionLimit εcov / 32)
            (Real.exp (-2
              * (((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).k
                  - 1 : ℕ) : ℝ)
              * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32)
            (9 * pAP / 20) ρ ρsf
            (rejCount η₀ indecisionLimit εcov
              (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit))
            (poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit))
            ⌊(1 - indecisionLimit)
              * ((solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref :
                  ℝ)⌋₊
            (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 2 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  have hτ : (0 : ℝ) < sig η₀ * εcov / 4 := by positivity
  obtain ⟨ttap, tscr, tdirty', tth⟩ :=
    solved_tails (α := α) η₀ populations hsig hε hind hδ hpAP hcard
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
  -- the gate's floor, and the tail it kills
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have hgmin : εcov * (m : ℝ) / 64
      ≤ ((⌊εcov * (m : ℝ) / 32⌋₊ : ℕ) : ℝ) := by
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    rw [div_le_iff₀ hε] at hsizeR
    linarith
  have cgate : Real.log (2 / ε₀) / (2 * (sig η₀ * εcov / 4) ^ 2)
      ≤ ((⌊εcov * (m : ℝ) / 32⌋₊ : ℕ) : ℝ) := by
    have heq : (2 : ℝ) / ε₀ = 256 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    have hmlog : 64 * Real.log (256 * (populations.card : ℝ) / δ)
        / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
      refine le_trans (Nat.le_ceil _) ?_
      have hle : ⌈64 * Real.log (256 * (populations.card : ℝ) / δ)
          / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
        rw [hmdef, prefCount]; omega
      exact_mod_cast hle
    have hlog0 : 0 ≤ Real.log (256 * (populations.card : ℝ) / δ) := by
      refine Real.log_nonneg ?_
      have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
        have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
        exact_mod_cast h1
      rw [le_div_iff₀ hδ]
      linarith
    rw [div_le_iff₀ (by positivity)] at hmlog ⊢
    linarith [mul_le_mul_of_nonneg_right hgmin
      (by positivity : (0 : ℝ) ≤ 2 * (sig η₀ * εcov / 4) ^ 2)]
  have tgate : 2 * Real.exp (-2
      * ((⌊εcov * (m : ℝ) / 32⌋₊ : ℕ) : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ ε₀ :=
    tail_le_of_count hτ hε₀ (by norm_num) cgate
  -- the gate's tail, at the decided count
  have hfloorge : ((⌊εcov * (m : ℝ) / 32⌋₊ : ℕ) : ℝ)
      ≤ ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ) := by
    have hgap : (0 : ℝ) ≤ 1 - indecisionLimit - εcov / 32 := by linarith
    have hmono : ⌊εcov * (m : ℝ) / 32⌋₊ ≤ ⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ :=
      Nat.floor_mono (by nlinarith [mul_nonneg hgap hmR])
    exact_mod_cast hmono
  have tgate' : 2 * Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tgate
    have hexp : Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
          * (sig η₀ * εcov / 4) ^ 2)
        ≤ Real.exp (-2
            * ((⌊εcov * (m : ℝ) / 32⌋₊ : ℕ) : ℝ)
          * (sig η₀ * εcov / 4) ^ 2) := by
      refine Real.exp_le_exp.2 ?_
      nlinarith [hfloorge, sq_nonneg (sig η₀ * εcov / 4)]
    linarith
  -- the vote's misfire rate sits under half the cut budget, so both count tails are no worse
  -- than the threshold's
  have hEc : Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 :=
    famCount_tail_seed η₀ populations hsig hε hind
  have hE0 : 0 ≤ Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2) := Real.exp_nonneg _
  have tl : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tlc : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov
      - Real.exp (-2 * (κ : ℝ) * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  -- the redraws: the prefix samplers' at `(3|J| + 3)·ρ·m/c`, the pool's at its own cap
  set c : ℝ := rejFrac η₀ indecisionLimit εcov with hcdef
  have hc0 : 0 < c := rejFrac_pos η₀ hsig hε hind
  obtain ⟨hcle, -⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hsig hε hind
  set r : ℕ := rejCount η₀ indecisionLimit εcov m with hrdef
  set rs : ℕ := poolRej pAP M with hrsdef
  have hrt : ((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1) ≤ 2 * (m : ℝ) / c :=
    rejTerm_le hc0 (by linarith [hcle, (cutBudget_le η₀ indecisionLimit εcov).1]) m
  have hdraws := draws_le_of_cap η₀ populations hη0 hsig hε hind hδ hρ0 hρsmall
  rw [← hmdef, ← hcdef] at hdraws
  have hpool := poolDraws_le_of_cap η₀ populations hpAP hpAP1 hδ hρsf0 hρsfsmall
  rw [← hMdef, ← hrsdef] at hpool
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hA : ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
      + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
      ≤ (3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c := by
    rw [show ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
        + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
        = (3 * (populations.card : ℝ) + 3) / 2 * ρ * (((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1)) by
      field_simp
      ring]
    calc (3 * (populations.card : ℝ) + 3) / 2 * ρ * (((m + r : ℕ) : ℝ) ^ 2 / ((r : ℝ) + 1))
        ≤ (3 * (populations.card : ℝ) + 3) / 2 * ρ * (2 * (m : ℝ) / c) := by gcongr
      _ = (3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c := by ring
  have hB : (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c)
      ≤ δ / 64 := by
    have e : (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * ρ * (m : ℝ) / c)
        = (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ / c * (m : ℝ) := by ring
    rw [e]
    calc (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ / c * (m : ℝ)
        ≤ δ / (64 * (m : ℝ)) * (m : ℝ) := mul_le_mul_of_nonneg_right hdraws hmpos.le
      _ = δ / 64 := by field_simp
  have hP : (populations.card : ℝ) * (2 * (((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))))
      ≤ δ / 64 := by
    have h2 := mul_le_mul_of_nonneg_left hpool (by positivity : (0 : ℝ) ≤ 2 * (populations.card : ℝ))
    have e : 2 * (populations.card : ℝ) * (δ / (128 * ((populations.card : ℝ) + 1)))
        ≤ δ / 64 := by
      rw [show 2 * (populations.card : ℝ) * (δ / (128 * ((populations.card : ℝ) + 1)))
          = δ / 64 * ((populations.card : ℝ) / ((populations.card : ℝ) + 1)) by field_simp; ring]
      have : (populations.card : ℝ) / ((populations.card : ℝ) + 1) ≤ 1 :=
        (div_le_one (by positivity)).2 (by linarith)
      nlinarith
    linarith
  rw [roundFail, hBm, hBM, hBκ]
  have tscr' : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤
        ε₀ := tscr
  refine le_trans (mul_le_mul_of_nonneg_left (show _ ≤ 10 * ε₀
      + (((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1)
        + 2 * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
        + 2 * (((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)))) from ?_) hcard.le) ?_
  · linarith [ttap, tscr', tdirty', tth, tl, tlc, tgate']
  · have hA' := mul_le_mul_of_nonneg_left hA hcard.le
    nlinarith [hcε, hB, hP, hA']

set_option maxHeartbeats 1000000 in
/-- The top rung carries its own share of the error budget, `δ/8`.  Validity's margins are all
wider than the round's, so each of its tails is under the round's own. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ : ℝ}
    (hsig : η₀ < 1 / 2) (hη0 : 0 ≤ η₀) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    stateFail η₀ populations indecisionLimit εcov ρ (solvedState η₀ populations
      indecisionLimit εcov δ α pAP crossLimit)
      ≤ δ / 8 := by
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcle := cutBudget_le η₀ indecisionLimit εcov
  obtain ⟨-, tscr, tdirty', tth⟩ :=
    solved_tails (α := α) η₀ populations hsig hε hind hδ hpAP hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).npref = m :=
    rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit).nsuff = M :=
    rfl
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hf := flipFrac_pos η₀ hsig
  have hs := sig_pos η₀ hsig
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  have hflipv := flipBudget_le_validFlip (indecisionLimit := indecisionLimit) (δ := δ)
    η₀ populations hsig hε.le hcard
  -- each of validity's margins is at least the round's that `solved_tails` killed
  have hmscr : screenMargin η₀ populations indecisionLimit εcov δ / 2
      ≤ validMargin η₀ populations εcov := by
    have hX : (0 : ℝ) ≤ (1 - η₀) * sig η₀ := mul_nonneg (by linarith) hs.le
    rw [screenMargin, validMargin]
    nlinarith [mul_le_mul_of_nonneg_right hflipv hX,
      flipBudget_pos (δ := δ) η₀ populations hsig hε hind hcard,
      mul_nonneg (validFlip_pos η₀ populations hsig hε hcard).le hX]
  have hmdirty : 3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32
      ≤ εcov * validFrac η₀ / 8 := by
    nlinarith [mul_le_mul_of_nonneg_right hcle.1 hf.le,
      mul_le_mul_of_nonneg_left (flipFrac_le_validFrac η₀ hsig) hε.le]
  have tscr' : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2) ≤ ε₀ :=
    le_trans (mul_le_mul_of_nonneg_left (exp_tail_anti hmR (by positivity) hmscr)
      (by positivity)) tscr
  have tdirty'' : (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2) ≤ ε₀ :=
    le_trans (mul_le_mul_of_nonneg_left (exp_tail_anti hmR (by positivity) hmdirty)
      (Nat.cast_nonneg _)) tdirty'
  have t32 : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tund : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  -- the redraws, against a sixty-fourth of the budget
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hsig hε hind
  have hdraws := draws_le_of_cap η₀ populations hη0 hsig hε hind hδ hρ0 hρsmall
  rw [← hmdef] at hdraws
  have hmpos : (0 : ℝ) < (m : ℝ) := by
    rw [hmdef]; exact_mod_cast prefCount_pos (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hJ0 : (0 : ℝ) ≤ (populations.card : ℝ) := Nat.cast_nonneg _
  have hdr : (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
      / rejFrac η₀ indecisionLimit εcov) ≤ δ / 64 := by
    have e1 : (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
        / rejFrac η₀ indecisionLimit εcov)
        ≤ (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ) := by
      rw [show (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ)
          = (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
            / rejFrac η₀ indecisionLimit εcov) by ring]
      gcongr
      linarith
    refine le_trans e1 ?_
    calc (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ)
        ≤ δ / (64 * (m : ℝ)) * (m : ℝ) := mul_le_mul_of_nonneg_right hdraws hmpos.le
      _ = δ / 64 := by field_simp
  rw [stateFail, hBm, hBM]
  have hsum : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
        + ((((M : ℝ) + 2) ^ 2
              * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
            + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
              + Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2))))
      ≤ 6 * ε₀ := by
    linarith [tscr', tdirty'', t32, tund]
  have hscale := mul_le_mul_of_nonneg_left hsum hJ0
  rw [mul_add]
  have : (populations.card : ℝ) * (6 * ε₀) = 6 * ((populations.card : ℝ) * ε₀) := by ring
  linarith [hdr, hscale, hcε]

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

set_option maxHeartbeats 1000000 in
/-- Every rung at or above `validCount` carries its share: its collision terms from the share
proportional to its count, its tails from the one inversely proportional. -/
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
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit with hMdef
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
  show (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
      / rejFrac η₀ indecisionLimit εcov
    + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
      + ((((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
          + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
        + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2))))))
    ≤ δ * (m : ℝ) / (8 * N) + δ * (V : ℝ) / (16 * m)
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  -- the redraws, against the proportional share
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hsig hε hind
  have hdraws := draws_le_of_cap η₀ populations hη0 hsig hε hind hδ hρ0 hρcap
  rw [← hNdef] at hdraws
  have hcoll : (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
      / rejFrac η₀ indecisionLimit εcov) ≤ δ * (m : ℝ) / (8 * N) := by
    have e1 : (populations.card : ℝ) * ((2 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
        / rejFrac η₀ indecisionLimit εcov)
        ≤ (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ) := by
      rw [show (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ)
          = (populations.card : ℝ) * ((3 * (populations.card : ℝ) + 3) * (m : ℝ) * ρ
            / rejFrac η₀ indecisionLimit εcov) by ring]
      gcongr
      linarith
    refine le_trans e1 ?_
    calc (populations.card : ℝ) * (3 * (populations.card : ℝ) + 3) * ρ
          / rejFrac η₀ indecisionLimit εcov * (m : ℝ)
        ≤ δ / (64 * N) * (m : ℝ) := mul_le_mul_of_nonneg_right hdraws hmR.le
      _ ≤ δ * (m : ℝ) / (8 * N) := by
          rw [div_mul_eq_mul_div, div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [mul_pos hδ hmR, hNR]
  -- the tails, against the inverse share
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  have hε₀3 : ε₀ ≤ 1 / 6 := by
    rw [hε₀def, div_le_iff₀ (by positivity)]
    nlinarith
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  have hf := validFrac_pos η₀ hsig
  have hvm : 0 < validMargin η₀ populations εcov := by
    rw [validMargin]
    have := validFlip_pos η₀ populations hsig hε hcard
    have := sig_pos η₀ hsig
    have : (0 : ℝ) < 1 - η₀ := by linarith
    positivity
  set T1 : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ)
    / (2 * validMargin η₀ populations εcov ^ 2) with hT1
  set T2 : ℝ := Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
    / (2 * (εcov * validFrac η₀ / 8) ^ 2) with hT2
  set T3 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) / (2 * (εcov / 32) ^ 2) with hT3
  set T4 : ℝ := Real.log (128 * (populations.card : ℝ) / δ) / (2 * (indecisionLimit / 2) ^ 2)
    with hT4
  have hVsum : (V : ℝ) = (⌈T1⌉₊ : ℝ) + (⌈T2⌉₊ : ℝ) + (⌈T3⌉₊ : ℝ) + (⌈T4⌉₊ : ℝ) + 1 := by
    rw [hVdef, validCount, ← hMdef]
    push_cast
    ring
  have hcl : ∀ x : ℝ, (0 : ℝ) ≤ (⌈x⌉₊ : ℝ) := fun x => Nat.cast_nonneg _
  have hV1 : T1 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T2, hcl T3, hcl T4])
  have hV2 : T2 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T3, hcl T4])
  have hV3 : T3 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T2, hcl T4])
  have hV4 : T4 ≤ (V : ℝ) := le_trans (Nat.le_ceil _)
    (by rw [hVsum]; linarith [hcl T1, hcl T2, hcl T3])
  have hq1 : ((M : ℝ) + 2) ^ 2 / ε₀ = 128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ := by
    rw [hε₀def]; field_simp
  have hq2 : ((M : ℝ) + 1) / ε₀ = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
    rw [hε₀def]; field_simp
  have hq3 : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
    rw [hε₀def]; field_simp
  have t1 := tail_le_of_count_scaled (c := ((M : ℝ) + 2) ^ 2) hvm hε₀ hε₀3 (by nlinarith) hVR
    (by rw [hq1]; exact hV1) hVm
  have t2 := tail_le_of_count_scaled (γ := εcov * validFrac η₀ / 8) (c := (M : ℝ) + 1)
    (by positivity) hε₀ hε₀3
    (by linarith) hVR (by rw [hq2]; exact hV2) hVm
  have t3 := tail_le_of_count_scaled (γ := εcov / 32) (c := 1) (by positivity) hε₀ hε₀3 le_rfl hVR
    (by rw [hq3]; exact hV3) hVm
  have t4 := tail_le_of_count_scaled (γ := indecisionLimit / 2) (c := 1) (by positivity) hε₀ hε₀3
    le_rfl hVR
    (by rw [hq3]; exact hV4) hVm
  have t2' : (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2)
      ≤ ε₀ * V / m :=
    le_trans (mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)) t2
  have t3' : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2) ≤ ε₀ * V / m := by simpa using t3
  have t4' : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2) ≤ ε₀ * V / m := by
    simpa using t4
  have htails : (populations.card : ℝ) * (6 * (ε₀ * V / m)) ≤ δ * (V : ℝ) / (16 * m) := by
    rw [hε₀def]
    field_simp
    nlinarith [mul_pos hδ hVR]
  have hsum : Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
      + ((((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (m : ℝ) * validMargin η₀ populations εcov ^ 2)
          + (M : ℝ) * Real.exp (-2 * (m : ℝ) * (εcov * validFrac η₀ / 8) ^ 2))
        + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
          + (Real.exp (-2 * (m : ℝ) * (εcov / 32) ^ 2)
            + Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2) ^ 2))))
      ≤ 6 * (ε₀ * V / m) := by
    linarith [t1, t2', t3', t4']
  have hscale := mul_le_mul_of_nonneg_left hsum hcard.le
  rw [mul_add]
  linarith [hscale, hcoll, htails]

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

set_option maxHeartbeats 1000000 in
/-- The computed ladder holds a state whose round can pass.

`PassableAt` is arithmetic: every clause is an inequality among the state's budgets, the
oracle's rates, the suffix distribution's findability and the error budget.  The witness is
`solvedState`, the ladder's top rung, and every clause is discharged from the closed forms
that define it — no reachability is assumed.

The order the constants come out in: the miscut budget
`lcut = cutBudget η₀ indecisionLimit εcov` under the indecision limit, the flip budget
`Δ` under `lcut` at the flip fraction `flipFracOf` the vote absorbs, which `flipBudget` fixes the
ratio of, the screen's two margins at
`screenMargin` (so the rate window is non-empty), then the prefix count large enough for
every exponential — the share's own among them, and `α` at the gate's own tail
`exp(−2·nlo·τ²)` at the decided count `nlo` —
then the pool at `k/(pAP − t)`.  The samplers' redraws enter linearly in the draws, which is
what `collisionCap` and `poolCollisionCap` bound. -/
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
      ∧ PassableAt O η₀ populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP B := by
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
  -- the gate's floor, from both sides
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast this
  have hsize : 2 ≤ εcov * (m : ℝ) / 32 := by
    rw [div_le_iff₀ hεcov] at hsizeR
    linarith
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
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊]
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
  clear_value B b κ m x
  have hc0 : 0 < rejFrac η₀ indecisionLimit εcov := rejFrac_pos η₀ hη₀ hεcov hindLim
  obtain ⟨hcle, hceq⟩ := rejFrac_le (indecisionLimit := indecisionLimit) η₀ hη₀ hεcov hindLim
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_⟩⟩,
    sig η₀ * εcov / 4, cutBudget η₀ indecisionLimit εcov / 32, 9 * pAP / 20,
    decSlack η₀ populations indecisionLimit εcov δ crossLimit,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    3 * cutBudget η₀ indecisionLimit εcov * flipFrac η₀ / 32,
    flipBudget η₀ populations indecisionLimit εcov δ * flipFracOf O η₀ / flipFrac η₀,
    cutBudget η₀ indecisionLimit εcov, flipFracOf O η₀, rejFrac η₀ indecisionLimit εcov,
    rejCount η₀ indecisionLimit εcov m,
    poolRej pAP (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit),
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · exact hε1
  · exact div_nonneg (mul_nonneg hs.le hεcov.le) (by norm_num)
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
  -- the gate's two sides clear their thresholds
  · intro n c hn₀ hnc hcm
    have hcR : (c : ℝ) ≤ (m : ℝ) := by rw [← hBm]; exact_mod_cast hcm
    have hnR : ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast hn₀
    have hfloor : (1 - indecisionLimit) * (m : ℝ) - 1
        ≤ ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ) := by
      rw [hBm]
      linarith [Nat.lt_floor_add_one ((1 - indecisionLimit) * (m : ℝ))]
    have hlow : (1 - indecisionLimit) * (m : ℝ) - 1 ≤ (n : ℝ) := le_trans hfloor hnR
    have hm64 : (64 : ℝ) ≤ (m : ℝ) := by
      have h1 : (64 : ℝ) / εcov ≤ (m : ℝ) := hsizeR
      rw [div_le_iff₀ hεcov] at h1
      nlinarith
    clear hBsc hBscd hBlo hBhi hBM hMceil hBdef hBk hBkappa hBm
      hb hb1 hb0 hκv
      hxexact hlowShift hhiShift hnR hfloor hsize hsizeR
      hρcap hρsfcap hρ hρsf hpAPBound
    have hhalf : (m : ℝ) / 2 ≤ (1 - indecisionLimit) * (m : ℝ) := by
      nlinarith [mul_le_mul_of_nonneg_right (by linarith : (1 : ℝ) / 2 ≤ 1 - indecisionLimit)
        hmR.le]
    have hcn : 2 * (c : ℝ) ≤ 5 * (n : ℝ) := by linarith [hcR, hlow, hm64, hhalf]
    have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg _
    have hc0 : (0 : ℝ) ≤ (c : ℝ) := Nat.cast_nonneg _
    have hss : sig η₀ ≤ 1 / 2 - O.η := by rw [sig]; linarith
    -- a wrong cut can cost the gate a whole read, so the cut budget is held under `s`
    have ha : 2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ) ≤ sig η₀ * (n : ℝ) / 2 := by
      nlinarith [mul_le_mul_of_nonneg_right (cutBudget_le η₀ indecisionLimit εcov).2.1 hc0,
        mul_le_mul_of_nonneg_left hcn (by linarith : (0 : ℝ) ≤ sig η₀ / 10)]
    have h1 : (1 - O.η) * (2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ))
        ≤ sig η₀ * (n : ℝ) / 2 := by
      have hb : (0 : ℝ) ≤ 2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ) := by positivity
      nlinarith [mul_le_mul_of_nonneg_right (by linarith : 1 - O.η ≤ 1) hb]
    have h2 : (n : ℝ) * (sig η₀ * εcov / 4 + sig η₀ * εcov / 4) ≤ (n : ℝ) * (sig η₀ / 2) := by
      refine mul_le_mul_of_nonneg_left ?_ hn0
      nlinarith [hs]
    have h3 : (n : ℝ) * (1 / 2 + sig η₀) ≤ (n : ℝ) * (1 - O.η) :=
      mul_le_mul_of_nonneg_left (by linarith) hn0
    nlinarith [h1, h2, h3]
  · rw [hBm]
    refine Nat.floor_pos.2 ?_
    have h1 : (64 : ℝ) / εcov ≤ (m : ℝ) := hsizeR
    rw [div_le_iff₀ hεcov] at h1
    nlinarith
  · rw [hBdef]
    exact solved_alpha η₀ populations hη₀ hεcov hε1 hδ hαpos hα hind1 hpAPPositive hcard
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail_seed η₀ populations hη₀ hεcov hindLim
  · rw [hBdef, hmdef]
    exact solved_roundFail η₀ populations hη₀ hη0' hεcov hε1 hδ hδ1 hpAPPositive
      (le_trans hpAPBound measureReal_le_one) hindLim hind1 hcard hρ0 hρsf0 hρcap hρsfcap

set_option maxHeartbeats 1000000 in
/-- Part 2 — the loop terminates: except with probability `δ/2`, some reachable state
passes both tests.

Each growth step draws fresh suffixes, accept-preserving with probability `≥ pAP`; once the
pool holds enough of them and the prefix count is large enough, every population's vote is
decisive on all but `indecisionLimit` of its mass.  The per-step trigger is block-local, so
`geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under `δ/2`.

Here `admitted` has to pass, not merely be sound — the gate's power, the complement of the
`α` it spends — which is why `pAP > 0` is needed rather than merely useful.
`ACCEPT_PRESERVING_GIVE_UP = 20` caps the refusals in the code; the statement carries no
cap.

Nothing here asks a population to carry both labels: the gate pools its two sides, so an
all-accepting population is admitted on its one.

`cutBudget` is held under `indecisionLimit/2` by its own definition, so the family size is
solved against a single misfire rate rather than the max of the round's two. -/
theorem loop_terminates {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0)
    (indecisionLimit εcov α : ℝ) (ρ pAP δ : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsf : collisionMass Dsf
        ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ},
      x ∉ retAll rule O.mq populations indecisionLimit α B.val} ≤ δ / 2 := by
  classical
  obtain ⟨B, hB, hpass⟩ := exists_passable O populations D Dsf indecisionLimit εcov α δ ρ
    (collisionMass Dsf) pAP hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim
    hind1
    hpAPPositive
    hpAPBound hρ hρ0 le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hρcap hρsf
  obtain ⟨τ, th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, f, c, r, rs, hmpos, hkpos, hindLim,
    hind1', hε1, hτ, hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hf0, hpAP0, hlcut, hlcl,
    hc0, hr, hpAPBound, hρsf, hheavy, hscd, hscLow, hscHigh, hcount,
    hhiUp, hloUp, hhiLo, hloLo, hga, hnloPos, hα, hEhalf, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hlpos : 0 < l := by rw [hl]; linarith
  have hl2 : 2 * l = indecisionLimit := by rw [hl]; ring
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  -- the whole failure at the one state, population by population
  have hsub : {x : Run Ω S J | ∀ B' : {B : State //
          B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ},
        x ∉ retAll rule O.mq populations indecisionLimit α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt rule O populations indecisionLimit α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_setOf_eq, not_and] at hc
    refine hx ⟨B, hB⟩ (mem_ret_of_retAt O populations ⟨j₀, hj₀⟩ indecisionLimit α B x
      (fun j hj => ?_))
    by_contra hcj
    exact hc j hj hcj
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ j ∈ populations,
      (runMeasure μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt rule O populations indecisionLimit α B j}
        ≤ roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) r rs
          ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B := by
    intro j hj
    have hstall := measureReal_stalled_le (rule := rule) hflat O populations D Dsf hsupp hsuppSf B
      (by omega) j₀ hj₀ γscr pAP tap (collisionMass Dsf) ρ rs r hγscr hpAP0 htap hpAPBound hscd
      hsig.le hscLow hcount hρsf hρsf0 (hρ j₀ hj₀) hρ0
    have hdirtyI := measureReal_dirtyMember_le (rule := rule) hflat O populations D Dsf hsupp
      hsuppSf j hj B hsig.le hmpos Δ γdirty gdirty c ρ r hΔ hγdirty hgdirty hc0 hρ0 hr (hρ j hj)
      hscd hscHigh
    have hnoclean := measureReal_noCleanRef_le D Dsf O populations B pAP tap
      (collisionMass Dsf) rs hpAP0 htap hpAPBound le_rfl hρsf0
      (by
        have hk2 : (2 : ℝ) ≤ (B.k : ℝ) := by exact_mod_cast hkpos
        linarith [hcount, hk2])
    have hcertBad := measureReal_certNotGood_le (μ := μ) D Dsf populations j B.npref r ρ hρ
      (hρ j hj) hρ0
    have hdirty : (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + c + gdirty}
        ≤ (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
            + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
          + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
            + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * collisionMass Dsf / (2 * ((rs : ℝ) + 1))) := by
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
    have hl1' : 2 * l ≤ 1 := by rw [hl]; linarith [hind1']
    have hnloB' : ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ)
        ≤ (1 - 2 * l) * (B.npref : ℝ) := by
      have hmnn : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
      have hnn : (0 : ℝ) ≤ (1 - indecisionLimit) * (B.npref : ℝ) := by nlinarith [hind1']
      have hfl := Nat.floor_le hnn
      rw [hl]
      linarith [hfl]
    have hmain := measureReal_notRetAt_le hflat O populations D Dsf hsupp hsuppSf j hj B
      hmpos hsig.le
      α τ l lcut f E ((populations.card : ℝ) * Δ + c + gdirty) th c r κ κ
      ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ (by omega)
      (Real.exp_nonneg _) (by linarith) (by rw [hl]; exact hlcl) hτ
      hl1' hnloB'
      (by positivity) hth hf0 hr hheavy _ hcertBad _ hstall _ hdirty hdec hcut hga hnloPos hα
    rw [hl2] at hmain
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  have hsum : ∑ j ∈ populations, (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt rule O populations indecisionLimit α B j}
      ≤ (populations.card : ℝ)
          * roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) r rs
            ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B := by
    refine le_trans (Finset.sum_le_sum hper) (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul]
  linarith [hsum, hbudget]

#print axioms validity_of_returned
#print axioms loop_terminates

end Loop

/-- `validity_of_returned` (whatever the loop returns is good, whenever it is returned) and
`loop_terminates` (it returns), each except w.p. `δ/2`, glued by `sound_and_terminating`. -/
theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O rule populations uni D Dsf _ _ Pre Suf η₀ indecisionLimit εcov α δ ρ
    pAP crossLimit k
    hηle hη₀ huni hflat hsupp hsuppSf hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov
    hε1 hδ _hbudget hρcap hρsf
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ ≤ 0) measureReal_nonneg
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2
      * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 16 :=
    le_trans (exp_tail_anti (Nat.cast_nonneg _) (by positivity) (by linarith))
      (le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith))
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hcoll := le_trans (poolDraws_le_of_cap η₀ populations hpAPPositive
      (le_trans hpAPBound measureReal_le_one) hδ hρsf0 hρsf)
    (show δ / (128 * ((populations.card : ℝ) + 1)) ≤ δ / 16 by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith)
  have h := sound_and_terminating (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α B.val
        ∩ FailAt rule O populations D indecisionLimit εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α B.val) δ
    (validity_of_returned hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α hindLim
      hsig hpop hηle hη₀ ρ (collisionMass Dsf) hρ le_rfl
      hρsf0 hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind hcoll)
    (le_trans (measureReal_mono (fun x hx B hB => hx B
        (retAll_subset_retBy rule O.mq huni indecisionLimit α B.val hB)) (measure_ne_top _ _))
      (loop_terminates hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α ρ pAP
        δ hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1
        hpAPPositive
        hpAPBound hρ (le_trans (tsum_nonneg (fun a => sq_nonneg _))
          (hρ hpop.choose hpop.choose_spec)) hρcap hρsf))
  refine le_trans h (measureReal_mono ?_ (measure_ne_top _ _))
  intro x hx
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not] at hx ⊢
  obtain ⟨hex, hall⟩ := hx
  refine ⟨hex, fun B hB j hj => ?_⟩
  obtain ⟨hcov, hund⟩ := hall B hB j hj
  refine ⟨le_trans hcov (measureReal_mono (fun p hp => cutCorrect_insert_one O _ _ _
    (one_not_mem_clusterAt O populations B.val x) p _ hp) (measure_ne_top _ _)), hund⟩

#print axioms clustering_correct

end OrthoDFA
