import RLStar.Proofs.LoopColl
import RLStar.Proofs.LoopGood
import RLStar.Proofs.ClusteringQuality
import RLStar.Proofs.BandKappa
import Mathlib.Data.List.GetD
import Mathlib.Data.List.ReduceOption
import RLStar.Proofs.LazyShape

/-!
# The loop's theorem

The whole run reads one noise. Each round draws off every prefix and suffix the earlier rounds
read, so no round reads a string an earlier round read but with chance zero (`loop_coll_null`).
Round `r` fails the clustering's claim, given the earlier rounds, but for chance `δc`
(`loop_cluster_round_le_persist`): off what the earlier rounds read the clustering meets its
claim at `slack - m`, and a population's average moves by at most the `m` the earlier rounds'
prefixes hold of it. It fails the lazy round's claim but for chance `lazyRoundBound`
(`loop_round_le_persist`). A harvest hands on a list of at least `nh` distinct strings, putting at
most `1/nh` on any string, within the cap. Where every one of the first `|Q| + 1` rounds meets
both claims, the loop stops in agreement (`loop_agrees_of_specs`), each harvest's list putting
more than `c/T` on a state read undecided more than `T` of the time.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The clustering's schedule -/

set_option maxHeartbeats 400000 in
/-- One schedule serves every set of the loop's populations: every state of it reads a family of
at most `k - 1` members at a band whose read, whatever its two rates, is on its rarer decided side
at most `ζ = crossLimit` times as often as undecided. It draws the one pool and
at most the top prefix count, and the prefix counts sum to at most twice the top's. Nothing is
harvested into a table, and every population is tame. -/
theorem clustering_schedule_exists {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (K : ℕ) (Pre Suf : Set (FreeMonoid α))
    (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP : ℝ)
    (hgap : 4 * (1 / 2 - η₀) ≤ O.gap) (hη₀ : η₀ < 1 / 2) (hflat : Flat Pre Suf) (hpAP : 0 < pAP)
    (hι : 0 < indecisionLimit) (hι1 : indecisionLimit ≤ 1 / 2) (hg : 0 < gateLevel)
    (hg1 : gateLevel < 1 / 2) (hcross : 0 < crossLimit) (hε : 0 < εcov) (hε1 : εcov ≤ 1 / 2)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) :
    ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ,
        0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
        RarerBounded N B.lo (B.hi + 1) p r crossLimit)
      ∧ (∀ pops, ∀ B ∈ schedule pops,
        B.nsuff = poolCount η₀ pops indecisionLimit εcov (δc / 2) pAP (crossLimit ^ 4)
        ∧ B.npref ≤ prefCount η₀ pops indecisionLimit εcov (εcov / 512) (δc / 2) gateLevel pAP
          (crossLimit ^ 4) 0
        ∧ B.k = famCount η₀ pops indecisionLimit εcov (δc / 2) (crossLimit ^ 4) + 1)
      ∧ (∀ pops, ∑ n ∈ (schedule pops).image State.npref, n
        ≤ 2 * prefCount η₀ pops indecisionLimit εcov (εcov / 512) (δc / 2) gateLevel pAP
          (crossLimit ^ 4) 0)
      ∧ ∀ pops : Finset (Option (Fin K)), none ∈ pops →
        ∀ (D : Option (Fin K) → Measure (FreeMonoid α)) (Dsf : Measure (FreeMonoid α)),
          (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
          (∀ j ∈ pops, D j Preᶜ = 0) →
          2 * εcov ≤ (D none).real {p | p ∈ O.L} → 2 * εcov ≤ (D none).real {p | p ∉ O.L} →
          Dsf Sufᶜ = 0 →
          pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 ^ 19 * (B.npref : ℝ) * (D j).real {t}
              ≤ (1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) indecisionLimit) →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 ^ 23 * ((pops.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
              ≤ min εcov indecisionLimit) →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 ^ 27 * Real.log (128 * (pops.card : ℝ) * B.nsuff
                * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
              * (D j).real {t} ≤ min εcov indecisionLimit ^ 2) →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 * Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ))
                / δc) * (D j).real {t} ≤ slack ^ 2) →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            4 * (pops.card : ℝ) * (B.npref : ℝ) * (D j).real {t} ≤ slack) →
          (∀ B ∈ schedule pops,
            ∀ t, 2 ^ 16 * (B.nsuff : ℝ) * Dsf.real {t} ≤ pAP) →
          (runMeasure μ D Dsf).real
            {x | ¬ ((∃ B ∈ schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel B)
              ∧ ∀ B ∈ schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel B →
                ∫ p, miscutProb O B.lo (B.hi + 1)
                    (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(D none)
                    ≤ 513 / 512 * εcov + slack
                  ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                    (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                    ∂(D j) ≤ 2 * indecisionLimit + slack)} ≤ δc + gateLevel := by
  classical
  obtain ⟨c, hc⟩ : ∃ c : ℝ, c = crossLimit ^ 4 := ⟨_, rfl⟩
  have hc0 : 0 < c := hc ▸ pow_pos hcross 4
  rw [← hc]
  have hH : (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))).card = 0 :=
    Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun s hs => by
      simp [allHarvest] at hs)
  have hq : ∀ pops : Finset (Option (Fin K)), none ∈ pops → _ := fun pops hpops =>
    clustering_quality_sized O pops pops none (fun _ => ∅) Pre Suf η₀ indecisionLimit εcov
      (εcov / 512) gateLevel δc pAP c slack hgap hη₀ hpops subset_rfl hflat
      (fun _ p hp => absurd hp (Finset.notMem_empty p)) hpAP hι hι1 hg hg1 hε hε1 (by positivity)
      (by linarith) hδc hδc1 hc0 hslack
  refine ⟨fun pops => if h : none ∈ pops then Classical.choose (hq pops h) else ∅, ?_, ?_, ?_, ?_⟩
  · intro pops B hB
    dsimp only at hB
    split_ifs at hB with hp
    swap
    · simp at hB
    have hs := (Classical.choose_spec (hq pops hp)).2.1 hB
    have hη0 : 0 ≤ η₀ := by have := gap_le_one O; linarith
    obtain ⟨-, -, hBk, -, -⟩ := band_of_mem_schedule hs
    have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := c)
      η₀ pops εcov (δc / 2) hη₀
    exact ⟨lohi_of_mem_schedule hs, by rw [hBk]; omega, fun N hN p r hp0 hp1 hr0 hr1 =>
      kappa_of_mem_schedule hη0 hη₀ hc0 hε hι hs N hN hcross.le hc.le hp0 hp1 hr0 hr1⟩
  · intro pops B hB
    dsimp only at hB
    split_ifs at hB with hp
    swap
    · simp at hB
    have hs := (Classical.choose_spec (hq pops hp)).2.1 hB
    rw [hH] at hs
    refine ⟨nsuff_of_mem_schedule hs, ?_, k_of_mem_schedule hs⟩
    obtain ⟨i, -, hi⟩ := npref_of_mem_schedule hs
    rw [hi]
    exact Nat.div_le_self _ _
  · intro pops
    dsimp only
    split_ifs with hp
    · have hsub : Classical.choose (hq pops hp)
          ⊆ schedule η₀ pops indecisionLimit εcov (εcov / 512) (δc / 2) gateLevel pAP c 0 :=
          fun B hB => by
        have h := (Classical.choose_spec (hq pops hp)).2.1 hB
        rwa [hH] at h
      have h := schedule_npref_sum_le (εs := εcov / 512) η₀ pops εcov (δc / 2) gateLevel pAP
        _ hsub
      have h' : ((∑ n ∈ (Classical.choose (hq pops hp)).image State.npref, n : ℕ) : ℝ)
          ≤ ((2 * prefCount η₀ pops indecisionLimit εcov (εcov / 512) (δc / 2) gateLevel pAP c 0
            : ℕ) : ℝ) := by
        rw [Nat.cast_sum]; push_cast; exact h
      exact_mod_cast h'
    · simp
  · intro pops hpops D Dsf hD hDsf hsupp hbalL hbalN hsuppSf hpAPDsf hcolT hcolC hcolH hslD htbD
      hcolSf
    simp only [dif_pos hpops] at hcolT hcolC hcolH hslD htbD hcolSf ⊢
    set states := Classical.choose (hq pops hpops)
    obtain ⟨-, hstates, hgood⟩ := Classical.choose_spec (hq pops hpops)
    have hpAP1 : pAP ≤ 1 := le_trans hpAPDsf measureReal_le_one
    have hmin : min εcov indecisionLimit / 512 ≤ min (εcov / 512) indecisionLimit :=
      le_min (by linarith [min_le_left εcov indecisionLimit])
        (by linarith [min_le_right εcov indecisionLimit, lt_min hε hι])
    have hcolE : ∀ B ∈ states, ∀ j ∈ pops, ∀ t,
        2 ^ 14 * ((pops.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
          ≤ min (εcov / 512) indecisionLimit :=
      fun B hB j hj t => by linarith [hcolC B hB j hj t]
    have hmin2 :
        (min εcov indecisionLimit) ^ 2 / 2 ^ 18 ≤ min (εcov / 512) indecisionLimit ^ 2 := by
      have := pow_le_pow_left₀ (by have := lt_min hε hι; positivity) hmin 2
      linarith
    have hkn : ∀ B ∈ states, (B.k : ℝ) ≤ B.nsuff ∧ (1 : ℝ) ≤ B.k := fun B hB => by
      have hs := hstates hB
      rw [hH] at hs
      have h1 := poolCount_ge_famCount (indecisionLimit := indecisionLimit) (crossLimit := c)
        η₀ pops εcov (δc / 2) hpAP hpAP1
      have e1 := k_of_mem_schedule hs
      have e2 := nsuff_of_mem_schedule hs
      constructor
      · have : B.k ≤ B.nsuff := by omega
        exact_mod_cast this
      · have : 1 ≤ B.k := by omega
        exact_mod_cast this
    have hpc : (1 : ℝ) ≤ pops.card := by exact_mod_cast Finset.card_pos.2 ⟨none, hpops⟩
    have hcolH' : ∀ B ∈ states, ∀ j ∈ pops, ∀ t,
        512 * Real.log (128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          * (D j).real {t} ≤ min (εcov / 512) indecisionLimit ^ 2 := fun B hB j hj t => by
      have h := hcolH B hB j hj t
      obtain ⟨hk, hk1⟩ := hkn B hB
      have hl0 : 0 ≤ (3 + 2 * (Nat.log 2 B.npref : ℝ)) := by positivity
      have hpos : 0 < 128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc := by
        positivity
      have hlog : Real.log (128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          ≤ Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc) :=
        Real.log_le_log hpos (div_le_div_of_nonneg_right (by gcongr) hδc.le)
      have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
      have h3 := mul_le_mul_of_nonneg_right hlog hD0
      linarith
    have hslD' : ∀ B ∈ states, ∀ j ∈ pops, ∀ t,
        2 * Real.log (128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          * (D j).real {t} ≤ slack ^ 2 := fun B hB j hj t => by
      have h := hslD B hB j hj t
      obtain ⟨hk, hk1⟩ := hkn B hB
      have hl0 : 0 ≤ (3 + 2 * (Nat.log 2 B.npref : ℝ)) := by positivity
      have hpos : 0 < 128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc := by
        positivity
      have hlog : Real.log (128 * (pops.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          ≤ Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc) :=
        Real.log_le_log hpos (div_le_div_of_nonneg_right (by gcongr) hδc.le)
      have hD0 : 0 ≤ (D j).real {t} := measureReal_nonneg
      have h3 := mul_le_mul_of_nonneg_right hlog hD0
      linarith
    have hg := hgood D Dsf hD hDsf hsupp hbalL hbalN hsuppSf hpAPDsf hcolT hcolE hcolH' hslD' htbD
      hcolSf
    set SpecR := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      (∃ B ∈ states, x ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel
          B)
        ∧ ∀ B ∈ states, x ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel
          B →
          ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
              ≤ 513 / 512 * εcov + slack
            ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
              ∂(D j) ≤ 2 * indecisionLimit + slack}
    have hSm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹' SpecR) :=
      measurable_streams _ none (measurableSet_specAt O pops states (2 * η₀ - 1 / 2)
        indecisionLimit εcov gateLevel (fun B F => ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂(D none)
          ≤ 513 / 512 * εcov + slack
          ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(D j)
            ≤ 2 * indecisionLimit + slack))
    have hsub : {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        (∃ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
          ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel B.val)
          ∧ ∀ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
            ∈ ret O.mq pops none (2 * η₀ - 1 / 2) indecisionLimit εcov gateLevel B.val →
            ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                ∂(D none) ≤ εcov + εcov / 512 + slack
            ∧ ∀ j ∈ pops,
              ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 1 / 2 + εcov / 512 + slack
              ∧ ∫ p, undecidedProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 2 * indecisionLimit + slack}
        ⊆ streams (fun _ => ∅) none ⁻¹' SpecR := by
      rintro x ⟨⟨B, hret⟩, hall⟩
      exact ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' =>
        ⟨by linarith [(hall ⟨B', hB'⟩ hret').1], fun j hj => ((hall ⟨B', hB'⟩ hret').2 j hj).2⟩⟩
    have hmono := measureReal_mono (μ := runMeasure μ D Dsf) hsub (measure_ne_top _ _)
    have hcompl : (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR)ᶜ
        = 1 - (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR) :=
      probReal_compl_eq_one_sub hSm
    show (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR)ᶜ ≤ δc + gateLevel
    linarith

/-! ## The theorem -/

section Lists

open Random Query

omit [DecidableEq α] in
private lemma length_unpad_le (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (p : ℕ)
    (m : FreeMonoid (Option α)) :
    (unpad (prefixOf (padIn K X x) p * m)).toList.length ≤ x.toList.length + m.toList.length := by
  have h1 : (unpad (prefixOf (padIn K X x) p * m)).toList
      = ((padIn K X x).toList.take p).reduceOption ++ m.toList.reduceOption := by
    simp [unpad, prefixOf, List.reduceOption_append]
  have h2 : ((padIn K X x).toList.take p).reduceOption.length
      ≤ (padIn K X x).toList.reduceOption.length :=
    ((List.take_sublist p _).filterMap _).length_le
  have h3 : (padIn K X x).toList.reduceOption = x.toList := by
    simp only [padIn, FreeMonoid.toList_ofList, List.reduceOption_append,
      List.reduceOption_replicate_none, List.nil_append]
    simp [List.reduceOption]
  rw [h1, List.length_append]
  have := List.reduceOption_length_le m.toList
  rw [h3] at h2
  omega

private lemma midsLe_epoch (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (xs : Fin P → FreeMonoid α) {n : ℕ}
    (h : ((epochQ C K X xs).eval read).hypothesis.tree.leaves.length ≤ n + 2) :
    MidsLe n ((epochQ C K X xs).eval read).hypothesis.tree := by
  have hinv := (runProbes_tree (padRead read) { C with walkStart := K }
    (List.ofFn fun i => padIn K X (xs i))).1.inv.cls
  have he : ((epochQ C K X xs).eval read).hypothesis.tree
      = (runProbes (padRead read) { C with walkStart := K } initialState
        (List.ofFn fun i => padIn K X (xs i))).1.tree := by
    rw [epochQ, eval_subst, level3Q, eval_map]
    rfl
  rw [he] at h ⊢
  exact (classSet_props n _ (classSet_mono' (by omega) hinv)).1

omit [DecidableEq α] in
private lemma blockQ_short (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) (X : Finset (FreeMonoid α)) {L n : ℕ} (hH : MidsLe n H.tree) :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, x.toList.length ≤ L) →
      ∀ w ∈ (blockQ K H s X xs).eval read, w.toList.length ≤ L + n
  | [], _ => by simp [blockQ]
  | x :: xs, hxs => by
    intro w hw
    simp only [blockQ, eval_bind, eval_map, List.mem_append, List.mem_map] at hw
    rcases hw with ⟨z, hz, rfl⟩ | hw
    · rw [eval_subst] at hz
      change z ∈ (siteHarvestQ K H s (padIn K X x)).eval (padRead read) at hz
      rw [eval_siteHarvestQ] at hz
      have hp := (siteHarvest_pot (padRead read) K H s (padIn K X x) z hz).1
      simp only [pot, Finset.mem_biUnion, Finset.mem_image, Finset.mem_Icc] at hp
      obtain ⟨j, -, m, hm, rfl⟩ := hp
      have h1 := length_unpad_le K X x j m
      have h2 := hH m hm
      have h3 := hxs x (by simp)
      omega
    · exact blockQ_short read K H s X hH xs (fun y hy => hxs y (by simp [hy])) w hw

omit [DecidableEq α] in
private lemma mem_blockUpdate (K : ℕ) (X : Finset (FreeMonoid α)) :
    ∀ (ℓ ws : List (FreeMonoid α)), ∀ w ∈ (blockUpdate K X ℓ ws).1, w ∈ ℓ ∨ w ∈ ws
  | ℓ, [] => by simp [blockUpdate]
  | ℓ, w' :: ws => by
    intro w hw
    by_cases h : w' ∈ ℓ
    · simp only [blockUpdate, if_pos h] at hw
      rcases mem_blockUpdate K X ℓ ws w hw with h1 | h1
      · exact Or.inl h1
      · exact Or.inr (List.mem_cons_of_mem _ h1)
    · simp only [blockUpdate, if_neg h] at hw
      rcases mem_blockUpdate K X (ℓ ++ [w']) ws w hw with h1 | h1
      · rcases List.mem_append.1 h1 with h2 | h2
        · exact Or.inl h2
        · rw [List.mem_singleton.1 h2]
          exact Or.inr List.mem_cons_self
      · exact Or.inr (List.mem_cons_of_mem _ h1)

omit [DecidableEq α] in
private lemma topUpQ_short (read : FreeMonoid α → ARU) (K nh : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) {L n : ℕ} (hH : MidsLe n H.tree) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      (∀ B ∈ Bs, ∀ t, (B t).toList.length ≤ L) → (∀ w ∈ ℓ, w.toList.length ≤ L + n) →
      (∀ ℓ' rest, (topUpQ K nh H s Bs X ℓ).eval read = some (.full ℓ', rest) →
          ∀ w ∈ ℓ', w.toList.length ≤ L + n)
        ∧ ∀ ℓ' X' rest, (topUpQ K nh H s Bs X ℓ).eval read = some (.retract ℓ' X', rest) →
          ∀ w ∈ ℓ', w.toList.length ≤ L + n
  | [], _, _, _, _ => by simp [topUpQ, eval_pure]
  | B :: Bs, X, ℓ, hB, hℓ => by
    have hws := blockQ_short read K H s X hH (List.ofFn B) fun x hx => by
      obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
      exact hB B (by simp) t
    simp only [topUpQ, eval_bind]
    generalize (blockQ K H s X (List.ofFn B)).eval read = ws at hws
    have hr : ∀ w ∈ (blockUpdate K X ℓ ws).1, w.toList.length ≤ L + n := fun w hw =>
      (mem_blockUpdate K X ℓ ws w hw).elim (hℓ w) (hws w)
    split_ifs with hfull hret
    · refine ⟨fun ℓ' rest h => ?_, fun ℓ' X' rest h => by simp [eval_pure] at h⟩
      simp only [eval_pure, Option.some.injEq, Prod.mk.injEq, TopUpEnd.full.injEq] at h
      rw [← h.1]
      exact hr
    · refine ⟨fun ℓ' rest h => by simp [eval_pure] at h, fun ℓ' X' rest h => ?_⟩
      simp only [eval_pure, Option.some.injEq, Prod.mk.injEq, TopUpEnd.retract.injEq] at h
      rw [← h.1.1]
      exact hr
    · exact topUpQ_short read K nh H s hH Bs _ _ (fun B' hB' => hB B' (by simp [hB'])) hr

private lemma le_peak_lazyEpochsQ (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {J b : ℕ} :
    ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b)) (X : Finset (FreeMonoid α))
      (ls : LazyLists α), pk ≤ ((lazyEpochsQ C K nh e pk ds X ls).eval read).peak
  | _, pk, [], X, ls => by simp [lazyEpochsQ, eval_pure]
  | e, pk, d :: ds, X, ls => by
    simp only [lazyEpochsQ, eval_bind]
    generalize (epochQ (epochSettings C e) K X d.1).eval read = res
    rcases res with ⟨hyp, _ | s | _⟩
    · exact le_max_left _ _
    · simp only [eval_bind]
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o
      rcases o with _ | ⟨_ | ⟨ℓ, X'⟩, rest⟩
      · exact le_max_left _ _
      · exact le_max_left _ _
      · exact (le_max_left _ _).trans (le_peak_lazyEpochsQ read C K nh (e + 1) _ ds X' _)
    · exact le_max_left _ _

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
private lemma mem_update_lists {ls : LazyLists α} {s : Level3Site} {ℓ : List (FreeMonoid α)}
    {Φ : FreeMonoid α → Prop} (hls : ∀ s, ∀ w ∈ ls s, Φ w) (hℓ : ∀ w ∈ ℓ, Φ w) :
    ∀ s', ∀ w ∈ Function.update ls s ℓ s', Φ w := by
  intro s' w hw
  by_cases hs : s' = s
  · subst hs
    have e : Function.update ls s' ℓ s' = ℓ := Function.update_self ..
    rw [e] at hw
    exact hℓ w hw
  · have e : Function.update ls s ℓ s' = ls s' := Function.update_of_ne hs ..
    rw [e] at hw
    exact hls s' w hw

private lemma lazyEpochsQ_short (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {J b L n : ℕ} :
    ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b)) (X : Finset (FreeMonoid α))
      (ls : LazyLists α),
      (∀ d ∈ ds, (∀ i, (d.1 i).toList.length ≤ L) ∧ ∀ j t, (d.2 j t).toList.length ≤ L) →
      (∀ s, ∀ w ∈ ls s, w.toList.length ≤ L + n) →
      ((lazyEpochsQ C K nh e pk ds X ls).eval read).peak ≤ n + 2 →
      (∀ s, ∀ w ∈ ((lazyEpochsQ C K nh e pk ds X ls).eval read).lists s,
          w.toList.length ≤ L + n)
        ∧ ∀ s ℓ, ((lazyEpochsQ C K nh e pk ds X ls).eval read).ending = .harvest s ℓ →
          ∀ w ∈ ℓ, w.toList.length ≤ L + n
  | _, pk, [], X, ls, _, hls, _ => by
    simp only [lazyEpochsQ, eval_pure]
    exact ⟨hls, fun s ℓ h => by cases h⟩
  | e, pk, d :: ds, X, ls, hd, hls, hpk => by
    obtain ⟨hd1, hd2⟩ := hd d (by simp)
    have hH := midsLe_epoch read (epochSettings C e) K X d.1 (n := n)
    simp only [lazyEpochsQ, eval_bind] at hpk ⊢
    generalize (epochQ (epochSettings C e) K X d.1).eval read = res at hH hpk ⊢
    rcases res with ⟨hyp, _ | s | _⟩
    · simp only [eval_pure]
      exact ⟨hls, fun s ℓ h => by cases h⟩
    · simp only [eval_bind] at hpk ⊢
      have hT := fun hm : MidsLe n hyp.tree => topUpQ_short read K nh hyp s hm (List.ofFn d.2) X
        (ls s) (fun B hB t => by
          obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hB
          exact hd2 j t) (hls s)
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o at hT hpk ⊢
      rcases o with _ | ⟨_ | ⟨ℓ, X'⟩, rest⟩
      · simp only [eval_pure]
        exact ⟨hls, fun s ℓ h => by cases h⟩
      · rename_i ℓ
        simp only [eval_pure] at hpk ⊢
        have hℓ := (hT (hH (le_trans (le_max_right _ _) hpk))).1 ℓ rest rfl
        refine ⟨mem_update_lists hls hℓ, fun s' ℓ' h => ?_⟩
        simp only [LazyEnd.harvest.injEq] at h
        rw [← h.2]
        exact hℓ
      · have hℓ := (hT (hH (le_trans (le_max_right _ _)
          ((le_peak_lazyEpochsQ read C K nh (e + 1) _ ds X' _).trans hpk)))).2 ℓ X' rest rfl
        exact lazyEpochsQ_short read C K nh (e + 1) _ ds X' _
          (fun d' hd' => hd d' (by simp [hd'])) (mem_update_lists hls hℓ) hpk
    · simp only [eval_pure]
      exact ⟨hls, fun s ℓ h => by cases h⟩

private lemma lazyRoundQ_short (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {E J b L n : ℕ} (X₀ : Finset (FreeMonoid α)) (d : Fin E → EpochDraws α C.probeBudget J b)
    (hd : LazyDrawsAll (fun x => x.toList.length ≤ L) d)
    (hpk : ((lazyRoundQ C K nh X₀ d).eval read).peak ≤ n + 2) {s : Level3Site}
    {ℓ : List (FreeMonoid α)} (he : ((lazyRoundQ C K nh X₀ d).eval read).ending = .harvest s ℓ) :
    ∀ w ∈ ℓ, w.toList.length ≤ L + n :=
  (lazyEpochsQ_short read C K nh 0 0 (List.ofFn d) X₀ (fun _ => [])
    (fun d' hd' => by
      obtain ⟨e, rfl⟩ := List.mem_ofFn.1 hd'
      exact hd e)
    (fun _ w hw => by simp at hw) hpk).2 s ℓ he

private lemma length_drawFrom_le (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {L : ℕ}
    (hν : ∀ᵐ x ∂ν, x.toList.length ≤ L) (u : unitInterval) :
    (drawFrom ν u).toList.length ≤ L := by
  rcases drawFrom_pos_or_one ν u with h | h
  · by_contra hlt
    have : ν {drawFrom ν u} = 0 := measure_mono_null (fun z hz => by
      rw [Set.mem_singleton_iff.1 hz]; exact hlt) (ae_iff.1 hν)
    rw [measureReal_def, this, ENNReal.toReal_zero] at h
    exact lt_irrefl _ h
  · rw [h]; simp

private lemma lazyProbes_short (P : PrefixDistribution α) (k : ℕ) (K₀ : Finset (FreeMonoid α))
    {E Pb J b : ℕ} (u : LazyUniforms E Pb J b) :
    LazyDrawsAll (fun x => x.toList.length ≤ P.L) (lazyProbes (avoidLaw P.D k K₀) u) := by
  have : IsProbabilityMeasure (avoidLaw P.D k K₀) := by unfold avoidLaw; infer_instance
  have h := avoidLaw_len P.D P.length_le k K₀
  exact fun e => ⟨fun i => length_drawFrom_le _ h _, fun j t => length_drawFrom_le _ h _⟩

omit [Fintype α] [DecidableEq α] in
private lemma harvest_of_population {o : SpecRound α} {ℓ : List (FreeMonoid α)}
    (h : o.population = some ℓ) : ∃ s, o.result.ending = .harvest s ℓ := by
  unfold SpecRound.population at h
  rcases he : o.result.ending with _ | ⟨s, ℓ'⟩ | _ <;> simp only [he, reduceCtorEq,
    Option.some.injEq] at h
  exact ⟨s, by rw [h]⟩

end Lists

section Pops

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}

/-- A loop round whose epochs' trees have at most `|Q| + 2` leaves hands on a list of strings
shorter than `Ls`. -/
lemma population_short {σ : Type*} [Fintype σ] (settings : LoopSettings K)
    (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α) {Ls : ℕ}
    (hLs : P.L + Fintype.card σ < Ls) (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : settings.Uniforms)
    (hpk : (roundOf settings O P.D h x u).result.peak ≤ Fintype.card σ + 2)
    {ℓ : List (FreeMonoid α)} (hℓ : (roundOf settings O P.D h x u).population = some ℓ) :
    listLaw ℓ {p | Ls ≤ p.toList.length} = 0 := by
  obtain ⟨s, he⟩ := harvest_of_population hℓ
  have hw := lazyRoundQ_short _ settings.level3 _ _ _ _ (lazyProbes_short P _ _ u) hpk he
  unfold listLaw
  refine popLaw_null ?_ (by simp only [Set.mem_ofPred_eq, FreeMonoid.toList_one,
    List.length_nil, not_le]; omega)
  rw [Measure.coe_finsetSum, Finset.sum_apply, Finset.sum_eq_zero_iff]
  intro z hz
  rw [Measure.dirac_apply' _ (measurableSet_of_countable _), Set.indicator_of_notMem]
  have := hw z (List.mem_toFinset.1 hz)
  simp only [Set.mem_ofPred_eq, not_le]
  omega

/-- A loop round's harvested list has no repeats and at least `listSize` strings. -/
lemma population_nodup (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : settings.Uniforms) (hnh : 0 < settings.listSize) {ℓ : List (FreeMonoid α)}
    (hℓ : (roundOf settings O D h x u).population = some ℓ) :
    ℓ.Nodup ∧ settings.listSize ≤ ℓ.length := by
  obtain ⟨s, he⟩ := harvest_of_population hℓ
  have key : ∀ {E J b : ℕ} (X₀ : Finset (FreeMonoid α))
      (d : Fin E → Random.EpochDraws α settings.level3.probeBudget J b)
      (read : FreeMonoid α → ARU), Random.TrieOK settings.level3.walkStart X₀ →
      ((Random.lazyRoundQ settings.level3 settings.level3.walkStart settings.listSize X₀ d).eval
        read).ending = .harvest s ℓ → ℓ.Nodup ∧ settings.listSize ≤ ℓ.length := by
    intro E J b X₀ d read hX he
    obtain ⟨-, -, -, hnd, -, hh, -, -⟩ := lazyRoundShape_holds read settings.level3
      settings.level3.walkStart settings.listSize E J b X₀ d hnh hX
    obtain ⟨e1, e2⟩ := hh s ℓ he
    exact ⟨e1 ▸ hnd s, e2⟩
  exact key _ _ _ (trieOK_prefixTrie _ _) he

end Pops

lemma flat_of_length (Ls : ℕ) :
    Flat {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls} := by
  intro p hp p' hp' v hv v' hv' e
  have hp : p.toList.length < Ls := hp
  have hp' : p'.toList.length < Ls := hp'
  have hv : v = 1 ∨ v.toList.length = Ls := hv
  have hv' : v' = 1 ∨ v'.toList.length = Ls := hv'
  have hl : p.toList ++ v.toList = p'.toList ++ v'.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul, e]
  have hlen : p.toList.length + v.toList.length = p'.toList.length + v'.toList.length := by
    rw [← List.length_append, ← List.length_append, hl]
  have hpp : p.toList.length = p'.toList.length := by
    rcases hv with rfl | hv <;> rcases hv' with rfl | hv' <;>
      first | omega | (simp only [FreeMonoid.toList_one, List.length_nil] at hlen; omega)
  exact FreeMonoid.toList.injective (List.append_inj_left hl hpp)

lemma real_singleton_le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (t : FreeMonoid α) : D.real {t} ≤ prefixMax D k :=
  (measureReal_mono (fun x hx => by rw [Set.mem_singleton_iff.1 hx]; rfl)
    (measure_ne_top _ _)).trans (le_prefixMax D k (prefixOf t k))

/-- A list of `n` distinct strings puts `1/n` on each. -/
lemma listLaw_atom_le {ℓ : List (FreeMonoid α)} (hℓ : ℓ.Nodup) {nh : ℕ} (hn : nh ≤ ℓ.length)
    (hnh : 0 < nh) (t : FreeMonoid α) : (listLaw ℓ).real {t} ≤ 1 / nh := by
  have hsum : ∀ A : Set (FreeMonoid α), (∑ z ∈ ℓ.toFinset, Measure.dirac z) A
      = ∑ z ∈ ℓ.toFinset, A.indicator 1 z := fun A => by
    rw [Measure.coe_finsetSum, Finset.sum_apply]
    exact Finset.sum_congr rfl fun z _ => Measure.dirac_apply' _ (measurableSet_of_countable A)
  have huniv : (∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ = ℓ.length := by
    rw [hsum]
    simp [List.toFinset_card_of_nodup hℓ]
  have hat : (∑ z ∈ ℓ.toFinset, Measure.dirac z) {t} ≤ 1 := by
    rw [hsum]
    calc ∑ z ∈ ℓ.toFinset, ({t} : Set (FreeMonoid α)).indicator 1 z
        = ∑ z ∈ ℓ.toFinset, if z = t then (1 : ENNReal) else 0 :=
          Finset.sum_congr rfl fun z _ => by simp [Set.indicator, Set.mem_singleton_iff]
      _ ≤ 1 := by
          rw [Finset.sum_ite_eq']
          split_ifs <;> simp
  have h0 : (∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ ≠ 0 := by
    rw [huniv]
    exact Nat.cast_ne_zero.2 (by omega)
  have htop : (∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ ≠ ∞ := by
    rw [huniv]
    exact ENNReal.natCast_ne_top _
  have hpos : (0 : ℝ) < ℓ.length := by exact_mod_cast hnh.trans_le hn
  have hu : (∑ z ∈ ℓ.toFinset, Measure.dirac z).real Set.univ = ℓ.length := by
    rw [measureReal_def, huniv]
    simp
  have hat' : (∑ z ∈ ℓ.toFinset, Measure.dirac z).real {t} ≤ 1 :=
    ENNReal.toReal_le_of_le_ofReal zero_le_one (by rw [ENNReal.ofReal_one]; exact hat)
  unfold listLaw
  rw [popLaw_real h0 htop, hu]
  calc _ ≤ 1 / (ℓ.length : ℝ) := div_le_div_of_nonneg_right hat' hpos.le
    _ ≤ 1 / nh := div_le_div_of_nonneg_left zero_le_one (by exact_mod_cast hnh)
        (by exact_mod_cast hn)

section Bridge

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}
  (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  [IsProbabilityMeasure D]

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure D] in
/-- Round `i` of the persistent run is the round after the history the rounds before it hand
on. -/
lemma persist_bridge (p : Ω × (Fin K → settings.RoundDraws α)) (i : ℕ)
    (hi : i < (idealRounds settings O D Dsf (persistDraws p) K).length) :
    ∃ hK : i < K, (histCodes settings O D Dsf p.1 p.2 i).length = i
      ∧ decodeHist (histCodes settings O D Dsf p.1 p.2 i)
        = specHistOf ((idealRounds settings O D Dsf (persistDraws p) K).take i)
      ∧ (idealRounds settings O D Dsf (persistDraws p) K)[i] = idealRound settings O D Dsf
        (decodeHist (histCodes settings O D Dsf p.1 p.2 i))
        (p.1, p.2 ⟨i, hK⟩) := by
  set draws := persistDraws p
  obtain ⟨hK, e⟩ := getElem_idealRounds settings O D Dsf draws K i hi
  have hiK : i ≤ K := (hi.trans_le (length_idealRounds_le settings O D Dsf draws K)).le
  have htake := take_idealRounds settings O D Dsf draws hiK
  have hall : AllHarvested (idealRounds settings O D Dsf draws i) := by
    intro o ho
    rw [← htake] at ho
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
    rw [List.getElem_take]
    simp only [List.length_take] at hj
    exact harvested_of_lt settings O D Dsf draws K j (by omega)
  have hdec := decodeHist_codesOf (idealRounds settings O D Dsf draws i) hall
  have hc : histCodes settings O D Dsf p.1 p.2 i
      = codesOf (idealRounds settings O D Dsf draws i) := rfl
  refine ⟨hK, ?_, ?_, ?_⟩
  · rw [hc, length_codesOf, length_specHistOf_pops hall,
      length_eq_of_allHarvested settings O D Dsf draws hiK hall]
  · rw [hc, hdec, htake]
  · rw [e, hc, hdec, htake]
    rfl

/-- Every earlier clustering returned, where each met its claim. -/
lemma histOK_of_specs (εcov slack : ℝ)
    (p : Ω × (Fin K → settings.RoundDraws α)) :
    ∀ i, i ≤ (idealRounds settings O D Dsf (persistDraws p) K).length →
      (∀ j (_ : j < (idealRounds settings O D Dsf (persistDraws p) K).length) (hjK : j < K),
        j < i → ClusterSpec settings O D εcov slack
          (specHistOf ((idealRounds settings O D Dsf (persistDraws p) K).take j)).pops
          (idealRunAt settings O D Dsf (persistDraws p) ⟨j, hjK⟩)) →
      HistOK settings O D Dsf p.1 p.2 i
  | 0, _, _ => trivial
  | i + 1, hi, h => by
    refine ⟨histOK_of_specs εcov slack p i (by omega) fun j hj hjK hji => h j hj hjK (by omega),
      fun hK _ => ?_⟩
    obtain ⟨hK', -, hdec, -⟩ := persist_bridge settings O D Dsf p i (by omega)
    have hs := (h i (by omega) hK' (by omega)).1
    rw [idealRunAt, ← take_idealRounds settings O D Dsf (persistDraws p) hK'.le, ← hdec] at hs
    exact hs

end Bridge

lemma card_foldr_union_le {β γ : Type*} [DecidableEq γ] (f : β → Finset γ) (l : List β) (b : ℕ)
    (h : ∀ x ∈ l, (f x).card ≤ b) : (l.foldr (fun c T => f c ∪ T) ∅).card ≤ l.length * b := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.foldr_cons, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    have h1 := h a (by simp)
    have h2 := ih fun x hx => h x (by simp [hx])
    rw [Nat.succ_mul]
    omega

omit [DecidableEq α] in
lemma measureReal_le_card_mul {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] {q : ℝ}
    (T : Finset (FreeMonoid α)) (hq : ∀ t, ν.real {t} ≤ q) : ν.real ↑T ≤ T.card * q := by
  rw [← sum_measureReal_singleton]
  calc ∑ t ∈ T, ν.real {t} ≤ ∑ _t ∈ T, q := Finset.sum_le_sum fun t _ => hq t
    _ = T.card * q := by rw [Finset.sum_const, nsmul_eq_mul]

lemma loop_reads_le {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α settings) (a b : ℕ)
    (h : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b) :
    ((idealRounds settings O D Dsf draws K).map fun o => o.prefs.card).sum ≤ K * a
      ∧ ((idealRounds settings O D Dsf draws K).map fun o => o.suffs.card).sum ≤ K * b := by
  have hlen : (idealRounds settings O D Dsf draws K).length ≤ K :=
    length_idealRounds_le settings O D Dsf draws K
  have hmem : ∀ o ∈ idealRounds settings O D Dsf draws K,
      o.prefs.card ≤ a ∧ o.suffs.card ≤ b := by
    intro o ho
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
    obtain ⟨hK, e⟩ := getElem_idealRounds settings O D Dsf draws K i hi
    rw [e]
    exact h _ _
  have hsum : ∀ (f : SpecRound α → ℕ) (c : ℕ),
      (∀ o ∈ idealRounds settings O D Dsf draws K, f o ≤ c) →
      ((idealRounds settings O D Dsf draws K).map f).sum ≤ K * c := fun f c hf => by
    refine (List.sum_le_card_nsmul _ c fun n hn => ?_).trans ?_
    · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hn
      exact hf o ho
    · rw [List.length_map, smul_eq_mul]
      exact Nat.mul_le_mul_right _ hlen
  exact ⟨hsum _ a fun o ho => (hmem o ho).1, hsum _ b fun o ho => (hmem o ho).2⟩

/-- Every round the loop runs follows rounds whose clusterings returned, returns itself, and
reads no prefix or suffix an earlier round read. -/
def RoundsFresh {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
    (Ls : ℕ) (p : Ω × (Fin K → settings.RoundDraws α)) : Prop :=
  ∀ i < (idealRounds settings O D Dsf (persistDraws p) K).length, ∃ hK : i < K,
    HistOK settings O D Dsf p.1 p.2 i
      ∧ Returns settings O D Dsf (decodeHist
          (histCodes settings O D Dsf p.1 p.2 i)) (p.1, (p.2 ⟨i, hK⟩).1)
      ∧ NoColl settings D Dsf Ls (histCodes settings O D Dsf p.1 p.2 i) p.2 i hK

/-- Off a collision, each round of the persistent run reads its family through the one noise,
its family suffixes `Ls` long and fewer than its state's `k`. -/
lemma family_of_fresh {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
    (Ls : ℕ) (p : Ω × (Fin K → settings.RoundDraws α))
    (hfr : RoundsFresh settings O D Dsf Ls p) :
    ∀ o ∈ idealRounds settings O D Dsf (persistDraws p) K,
      (∀ v ∈ o.family, v.toList.length = Ls) ∧ o.family.card + 1 ≤ o.state.k
        ∧ (∃ pops, o.state ∈ settings.schedule pops)
        ∧ o.read = fun z => familyRead O.mq o.family o.state.lo (o.state.hi + 1) z p.1 := by
  intro o ho
  obtain ⟨r, hr, rfl⟩ := List.getElem_of_mem ho
  obtain ⟨hK, -, hret, hnc⟩ := hfr r hr
  obtain ⟨hK', hlen, -, e⟩ := persist_bridge settings O D Dsf p r hr
  rw [e]
  set H := decodeHist (histCodes settings O D Dsf p.1 p.2 r)
  have hBs := (idealRound_returns settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩) hret).1
  have hpop : histAt (histCodes settings O D Dsf p.1 p.2 r) r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen)]
    rfl
  refine ⟨fun v hv => ?_, ?_, ⟨_, hBs⟩, rfl⟩
  · have hv' : v ∈ clusterAt O.mq (loopPopulations K H.pops.length)
        (idealRun D Dsf H (p.1, (p.2 ⟨r, hK'⟩).1))
        (idealRound settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩)).state := hv
    unfold clusterAt nearestRows at hv'
    obtain ⟨hv1, hv2⟩ := Finset.mem_erase.1 hv'
    rcases Finset.mem_insert.1 hv2 with h1 | h1
    · exact absurd h1 hv1
    have hpool := Finset.filter_subset _ _
      (Finset.mem_of_mem_erase (leastLossSubset_subset' _ _ _ h1))
    rcases Finset.mem_insert.1 (poolAt_sub_roundSuffs hBs _ hpool) with h2 | h2
    · exact absurd h2 hv1
    have hsub : v ∈ suffsOf settings D Dsf H (p.2 ⟨r, hK'⟩).1 := Finset.mem_erase.2 ⟨hv1, h2⟩
    rw [← hpop] at hsub
    exact hnc.1 ⟨r, hK'⟩ le_rfl v hsub
  · have : (idealRound settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩)).family.card
        ≤ (idealRound settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩)).state.k - 1 := by
      change (clusterAt O.mq _ _ _).card ≤ _
      unfold clusterAt nearestRows
      exact (Finset.card_le_card (Finset.erase_insert_subset _ _)).trans
        (leastLossSubset_card_le _ _ _)
    change (idealRound settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩)).family.card + 1 ≤ _
    rcases Nat.eq_zero_or_pos (idealRound settings O D Dsf H (p.1, p.2 ⟨r, hK'⟩)).state.k
      with h0 | h0
    · sorry
    · exact Nat.add_le_of_le_sub h0 this

/-- The loop's claim at a cap `cap`, level 3's settings `C`, lists of `nh` strings, `E` epochs of
`J` blocks of `b` probes, and `NS`, `NP` and `NF` bounding the schedule's suffix pool, prefix count
and family, the clustering run at `slack - m` and `pAP - ms`. The clustering's band reads the
rarer decided side at most `cross` times as often as undecided. A lazy round asks at most `Nr`
strings; its lists meet the slope `A = lazySlope` and slack `lazySlack` at `δH`, and the last
round has at most `C.disagreementRate + tw + mr` of the probes in `lazyDisagree`, `mr` bounding
the mass the round's probes avoid. Every round the loop runs scans within `scanCap` but for the
scan tail at `4 (K + 3) NP ≤ Tsc ≤ scanCap`. No epoch's tree has more than `|Q| + 2` leaves. -/
def LoopCore : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K Ls NS NP NF : ℕ)
    (η₀ indecisionLimit gateLevel εcov δc slack pAP cap m ms mr cross Tu : ℝ)
    (C : Random.Level3Settings) (scanCap Tsc nh E J b Nr : ℕ) (δH tw : ℝ),
  let c : ℝ := cross ^ 4
  let bsc : ℝ := (K + 1) * (2 * (scanCap.choose Tsc * (Tsc * (2 * cap)) ^ (scanCap - Tsc))
    + 2 * Real.exp (-(Tsc / 8 : ℝ)))
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < εcov → εcov ≤ 1 / 2 →
  2 * εcov + m ≤ P.D.real {p | p ∈ O.L} → 2 * εcov + m ≤ P.D.real {p | p ∉ O.L} →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ < Ls →
  0 < cap → m < slack → m ≤ 1 / 2 → ms < pAP → ms ≤ 1 / 2 →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    poolCount η₀ pops indecisionLimit εcov (δc / 2) (pAP - ms) c ≤ NS) →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    prefCount η₀ pops indecisionLimit εcov (εcov / 512) (δc / 2) gateLevel (pAP - ms) c 0
      ≤ NP) →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    famCount η₀ pops indecisionLimit εcov (δc / 2) c ≤ NF) →
  2 ^ 20 * NP * cap ≤ (1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) indecisionLimit →
  2 ^ 24 * ((K : ℝ) + 4) * NP * cap ≤ min εcov indecisionLimit →
  2 ^ 28 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1) * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc)
      * cap
    ≤ min εcov indecisionLimit ^ 2 →
  4 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1) * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc) * cap
    ≤ (slack - m) ^ 2 →
  8 * ((K : ℝ) + 1) * NP * cap ≤ slack - m →
  2 ^ 17 * NS * cap ≤ pAP - ms →
  6 * K * (K + 1) * NP * cap ≤ m →
  K * (NS + 1) * cap ≤ ms →
  1 ≤ C.walkStart → (∀ᵐ x ∂P.D, C.walkStart ≤ x.toList.length) →
  6 * (K + 1) * NP * prefixMax P.D C.walkStart ≤ mr → mr ≤ 1 / 2 → 0 < cross →
  LazyPremises σ α C nh E J (2 * prefixMax P.D C.walkStart) → 2 ≤ nh * cap →
  lazyReadCap C E J b P.L (Fintype.card σ) (Fintype.card α) ≤ Nr →
  0 < δH → 0 < tw →
  -- Each harvest puts more than `c/Tu` of its list on a state read undecided more than `Tu` of
  -- the time, `c` the gate's limit.
  0 < Tu →
  lazySlope σ Nr nh * Tu + lazySlack σ nh δH
    + Fintype.card σ * (2 * indecisionLimit + slack) / Tu < 1 →
  4 * ((K + 3) * NP) ≤ Tsc → Tsc ≤ scanCap →
  ∃ settings : LoopSettings K,
    settings.η₀ = 2 * η₀ - 1 / 2 ∧ settings.indecisionLimit = indecisionLimit
    ∧ settings.εcov = εcov ∧ settings.gateLevel = gateLevel ∧ settings.level3 = C
    ∧ settings.scanCap = scanCap ∧ settings.listSize = nh ∧ settings.epochs = E
    ∧ settings.blocks = J ∧ settings.blockSize = b ∧
    (∀ pops, ∑ n ∈ (settings.schedule pops).image State.npref, n ≤ 2 * NP) ∧
    (∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ 6 * (K + 1) * NP
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ NS + 1) ∧
    (∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ NF + 1) ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ v, Dsf.real {v} ≤ cap) →
      (settings.measure μ α).real
          {p | let rounds := idealRounds settings O P.D Dsf (persistDraws p) K
            ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Checks M O P.D C.walkStart (513 / 512 * εcov + 2 * indecisionLimit + 2 * slack)
                  (2 * indecisionLimit + slack) (C.disagreementRate + tw + mr))
              ∧ (∀ o ∈ rounds, o.result.peak ≤ Fintype.card σ + 2)
              ∧ RoundsFresh settings O P.D Dsf Ls p
              ∧ ∀ r (hr : r < K), AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r)
                  →
                RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf
                  (persistDraws p) r)) (persistDraws p ⟨r, hr⟩))}
        ≤ (Fintype.card σ + 1) * (δc + gateLevel
          + lazyRoundBound σ C E J b cross Nr P.L δH tw + bsc)

set_option maxHeartbeats 2000000 in -- the clustering's premises and the round bounds
theorem loop_core_holds : LoopCore := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K Ls NS NP NF η₀ ι gate εcov δc slack pAP cap m ms mr cross
    Tu C scanCap Tsc nh E J b Nr δH tw c bsc hL hgap hη₀ hι hι1 hg hg1 hε hε1 hbalL hbalN hδc
    hδc1 hsl hpAP hL1 hK hLs hcap hmsl hm1 hmsp hms1 hNSb hNPb hNFb hq1a hq1c hq1b hq2a hq2b hq3
    hqm hqms hk0 hKw hmr hmr1 hcr hprem hnhc hNr hδH htw hT hcond hTsc hTM
  classical
  have hc : cross ^ 4 = c := rfl
  obtain ⟨schedule₀, hband₀, hsize₀, hsum₀, hclust₀⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc (slack - m) (pAP - ms) hgap hη₀ (flat_of_length Ls) (by linarith)
    hι hι1 hg hg1 hcr hε hε1 hδc hδc1 (by linarith)
  rw [hc] at hsize₀ hsum₀
  -- the loop's populations always hold `D`'s
  set schedule : Finset (Option (Fin K)) → Finset State := fun pops =>
    if none ∈ pops then schedule₀ pops else ∅ with hschedd
  have hband : ∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
        RarerBounded N B.lo (B.hi + 1) p r cross := fun pops B hB => by
    simp only [hschedd] at hB
    split_ifs at hB
    · exact hband₀ pops B hB
    · simp at hB
  set settings : LoopSettings K :=
    ⟨schedule, 2 * η₀ - 1 / 2, ι, εcov, gate, C, scanCap, nh, E, J, b⟩ with hset
  -- the schedule's sizes
  have hsizes : ∀ pops, ∀ B ∈ settings.schedule pops, B.nsuff = poolCount η₀ pops ι εcov (δc / 2)
      (pAP - ms) c ∧ B.npref ≤ NP ∧ none ∈ pops := fun pops B hB => by
    simp only [hset, hschedd] at hB
    split_ifs at hB with hp
    · exact ⟨(hsize₀ pops B hB).1, (hsize₀ pops B hB).2.1.trans (hNPb pops hp), hp⟩
    · simp at hB
  have hks : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ NF + 1 := fun pops B hB => by
    simp only [hset, hschedd] at hB
    split_ifs at hB with hp
    · rw [(hsize₀ pops B hB).2.2]
      exact Nat.succ_le_succ (hNFb pops hp)
    · simp at hB
  have hNPsum : ∀ pops, ∑ n ∈ (settings.schedule pops).image State.npref, n ≤ 2 * NP := fun pops
      => by
    simp only [hset, hschedd]
    split_ifs with hp
    · exact (hsum₀ pops).trans (Nat.mul_le_mul_left 2 (hNPb pops hp))
    · simp
  have hreads : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ 6 * (K + 1) * NP
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ NS + 1 := by
    intro pops x
    have hpops : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    refine ⟨(card_roundPrefs_le _ _ _).trans ?_, ?_⟩
    · calc 3 * pops.card * ∑ n ∈ (settings.schedule pops).image State.npref, n
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) (hNPsum pops)
        _ = 6 * (K + 1) * NP := by ring
    · by_cases hp : none ∈ pops
      · exact (card_roundSuffs_le _ (fun B hB => (hsizes pops B hB).1) _).trans
          (Nat.succ_le_succ (hNSb pops hp))
      · have he : settings.schedule pops = ∅ := if_neg hp
        rw [he]
        exact (card_roundSuffs_le (n := 0) ∅ (by simp) x).trans (by omega)
  refine ⟨settings, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, hNPsum, hreads, hks, ?_⟩
  intro Dsf hDsf hDsfLen hpAPDsf hatom
  set k := C.walkStart
  set dec := decodeHist (α := α) with hdecd
  set μL := settings.measure μ α
  have hpm0 : 0 ≤ prefixMax P.D k := measureReal_nonneg.trans (le_prefixMax P.D k 1)
  have hnh : 0 < nh := hprem.1
  have hnhR : (1 : ℝ) ≤ nh := by exact_mod_cast hnh
  have hDat : ∀ t, P.D.real {t} ≤ cap := by
    have hy : 8 * nh * (2 * prefixMax P.D k) ≤ C.harvestRate := hprem.2.1
    have hy1 : C.harvestRate ≤ 1 := hprem.2.2.1
    have hpc : (nh : ℝ) * prefixMax P.D k ≤ nh * cap := by nlinarith
    have hpc' : prefixMax P.D k ≤ cap := le_of_mul_le_mul_left hpc (by positivity)
    exact fun t => (real_singleton_le_prefixMax P.D k t).trans hpc'
  have hinv : (1 : ℝ) / nh ≤ cap := by
    rw [div_le_iff₀ (by positivity)]
    nlinarith
  have hLs' : P.L < Ls := by omega
  have hK1 : (K : ℝ) + 1 ≤ K + 2 := by linarith
  have hpopc : ∀ pops : Finset (Option (Fin K)), ((pops.card : ℕ) : ℝ) + 1 ≤ K + 2 := by
    intro pops
    have : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    have : ((pops.card : ℕ) : ℝ) ≤ K + 1 := by exact_mod_cast this
    linarith
  have hη1 : O.η ≤ 1 := by
    have hs : 0 < 1 / 2 - η₀ := by linarith
    unfold Oracle.gap at hgap
    exact max_le (by linarith [le_max_left O.ηIn 0, le_max_right O.ηOut 0])
      (by linarith [le_max_right O.ηIn 0, le_max_left O.ηOut 0])
  have hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, ∀ t, (L j).real {t} ≤ 2 * cap) →
        2 * settings.εcov ≤ (L none).real {p | p ∈ O.L} →
        2 * settings.εcov ≤ (L none).real {p | p ∉ O.L} →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * cap) →
        (runMeasure μ L Lsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                  settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
              ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(L none)
                  ≤ 513 / 512 * εcov + (slack - m)
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - m))} ≤ δc + gate := by
    intro pops hpops L Lsf hLp hLsfp hLs hLat hLbalL hLbalN hLsfLen hpAPL hLsfat
    have hs : settings.schedule pops = schedule₀ pops := if_pos hpops
    rw [hs]
    have hLat0 : ∀ j t, 0 ≤ (L j).real {t} := fun _ _ => measureReal_nonneg
    have hNP0 : (0 : ℝ) ≤ NP := Nat.cast_nonneg _
    have hNS0 : (0 : ℝ) ≤ NS := Nat.cast_nonneg _
    refine hclust₀ pops hpops L Lsf hLp hLsfp (fun j hj => ?_) hLbalL hLbalN ?_ hpAPL
      (fun B hB j hj t => ?_)
      (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB j hj t => ?_)
      (fun B hB j hj t => ?_) (fun B hB t => ?_)
    · refine measure_mono_null (fun p hp => ?_) (hLs j hj)
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_lt] at hp ⊢
      exact hp
    · refine measure_mono_null (fun v hv => ?_) hLsfLen
      simpa using hv
    -- each of the clustering's five prefix premises is one of the cap's, at the schedule's
    -- counts and the populations' atoms, at most twice the cap
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hLj := hLat j hj t
      have hc0 := hcap.le
      calc 2 ^ 19 * (B.npref : ℝ) * (L j).real {t} ≤ 2 ^ 19 * NP * (2 * cap) := by gcongr
        _ = 2 ^ 20 * NP * cap := by ring
        _ ≤ _ := hq1a
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hpk : (pops.card : ℝ) ≤ (K : ℝ) + 1 := by linarith only [hpopc pops]
      have hLj := hLat j hj t
      have hc0 := hcap.le
      calc 2 ^ 23 * ((pops.card : ℝ) + 3) * (B.npref : ℝ) * (L j).real {t}
          ≤ 2 ^ 23 * ((K : ℝ) + 1 + 3) * NP * (2 * cap) := by gcongr
        _ = 2 ^ 24 * ((K : ℝ) + 4) * NP * cap := by ring
        _ ≤ _ := hq1c
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hnpN : B.npref ≤ NP := (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hpk : (pops.card : ℝ) ≤ (K : ℝ) + 1 := by linarith only [hpopc pops]
      have hLj := hLat j hj t
      have hc0 := hcap.le
      have hlog : (Nat.log 2 B.npref : ℝ) ≤ (Nat.log 2 NP : ℝ) := by
        exact_mod_cast Nat.log_mono_right hnpN
      have hpc : (1 : ℝ) ≤ pops.card := by exact_mod_cast Finset.card_pos.2 ⟨none, hpops⟩
      have hl0 : (0 : ℝ) ≤ (Nat.log 2 B.npref : ℝ) := Nat.cast_nonneg _
      have hBs := hsizes pops B (by rw [hs]; exact hB)
      have hns : (B.nsuff : ℝ) ≤ NS := by rw [hBs.1]; exact_mod_cast hNSb pops hpops
      have hns1 : (1 : ℝ) ≤ B.nsuff := by
        rw [hBs.1]
        have : 0 < poolCount η₀ pops ι εcov (δc / 2) (pAP - ms) c := by
          rw [poolCount]
          have : 0 < ⌈2 * ((famCount η₀ pops ι εcov (δc / 2) c : ℝ) + 1) / (pAP - ms)⌉₊ :=
            Nat.ceil_pos.2 (div_pos (by positivity) (by linarith))
          omega
        exact_mod_cast this
      have hlg : Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          ≤ Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1) * (3 + 2 * (Nat.log 2 NP : ℝ)) /
              δc) := by
        refine Real.log_le_log (div_pos (by positivity) hδc)
          (div_le_div_of_nonneg_right ?_ hδc.le)
        gcongr
        all_goals linarith
      have hpc1 : (1 : ℝ) ≤ (pops.card : ℝ) * B.nsuff := one_le_mul_of_one_le_of_one_le hpc hns1
      have hlg0 : 0 ≤ Real.log (128 * (pops.card : ℝ) * B.nsuff
          * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc) :=
        Real.log_nonneg (by rw [le_div_iff₀ hδc]; nlinarith)
      calc 2 ^ 27 * Real.log (128 * (pops.card : ℝ) * B.nsuff
              * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc) * (L j).real {t}
          ≤ 2 ^ 27 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1)
              * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc) * (2 * cap) :=
            mul_le_mul (mul_le_mul_of_nonneg_left hlg (by norm_num)) hLj measureReal_nonneg
              (mul_nonneg (by norm_num) (hlg0.trans hlg))
        _ = 2 ^ 28 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1)
              * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc) * cap := by
          ring
        _ ≤ _ := hq1b
    · have hnpN : B.npref ≤ NP := (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hpk : (pops.card : ℝ) ≤ (K : ℝ) + 1 := by linarith only [hpopc pops]
      have hLj := hLat j hj t
      have hlog : (Nat.log 2 B.npref : ℝ) ≤ (Nat.log 2 NP : ℝ) := by
        exact_mod_cast Nat.log_mono_right hnpN
      have hBs := hsizes pops B (by rw [hs]; exact hB)
      have hns : (B.nsuff : ℝ) ≤ NS := by rw [hBs.1]; exact_mod_cast hNSb pops hpops
      have hns1 : (1 : ℝ) ≤ B.nsuff := by
        rw [hBs.1]
        have : 0 < poolCount η₀ pops ι εcov (δc / 2) (pAP - ms) c := by
          rw [poolCount]
          have : 0 < ⌈2 * ((famCount η₀ pops ι εcov (δc / 2) c : ℝ) + 1) / (pAP - ms)⌉₊ :=
            Nat.ceil_pos.2 (div_pos (by positivity) (by linarith))
          omega
        exact_mod_cast this
      have hpc : (1 : ℝ) ≤ pops.card := by exact_mod_cast Finset.card_pos.2 ⟨none, hpops⟩
      have hl0 : (0 : ℝ) ≤ (Nat.log 2 B.npref : ℝ) := Nat.cast_nonneg _
      have hlg : Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc)
          ≤ Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1) * (3 + 2 * (Nat.log 2 NP : ℝ)) /
              δc) := by
        refine Real.log_le_log (div_pos (by positivity) hδc)
          (div_le_div_of_nonneg_right ?_ hδc.le)
        gcongr
        all_goals linarith
      have hpc1 : (1 : ℝ) ≤ (pops.card : ℝ) * B.nsuff := one_le_mul_of_one_le_of_one_le hpc hns1
      have hlg0 : 0 ≤ Real.log (128 * (pops.card : ℝ) * B.nsuff
          * (3 + 2 * (Nat.log 2 B.npref : ℝ)) / δc) :=
        Real.log_nonneg (by rw [le_div_iff₀ hδc]; nlinarith)
      calc 2 * Real.log (128 * (pops.card : ℝ) * B.nsuff * (3 + 2 * (Nat.log 2 B.npref : ℝ))
              / δc) * (L j).real {t}
          ≤ 2 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1)
              * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc) * (2 * cap) :=
            mul_le_mul (mul_le_mul_of_nonneg_left hlg (by norm_num)) hLj measureReal_nonneg
              (mul_nonneg (by norm_num) (hlg0.trans hlg))
        _ = 4 * Real.log (128 * ((K : ℝ) + 1) * ((NS : ℝ) + 1)
              * (3 + 2 * (Nat.log 2 NP : ℝ)) / δc) * cap := by
          ring
        _ ≤ _ := hq2a
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hnpN : B.npref ≤ NP := (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hpk : (pops.card : ℝ) ≤ (K : ℝ) + 1 := by linarith only [hpopc pops]
      have hLj := hLat j hj t
      have hc0 := hcap.le
      calc 4 * (pops.card : ℝ) * (B.npref : ℝ) * (L j).real {t}
          ≤ 4 * ((K : ℝ) + 1) * NP * (2 * cap) := by gcongr
        _ = 8 * ((K : ℝ) + 1) * NP * cap := by ring
        _ ≤ _ := hq2b
    · have hns : (B.nsuff : ℝ) ≤ NS := by
        rw [(hsizes pops B (by rw [hs]; exact hB)).1]
        exact_mod_cast hNSb pops hpops
      have hcc := mul_le_mul hns (hLsfat t) measureReal_nonneg (by positivity)
      calc 2 ^ 16 * (B.nsuff : ℝ) * Lsf.real {t}
          = 2 ^ 16 * ((B.nsuff : ℝ) * Lsf.real {t}) := by ring
        _ ≤ 2 ^ 16 * (NS * (2 * cap)) := mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 17 * NS * cap := by ring
        _ ≤ _ := hq3
  have hcodes : ∀ (ω : Ω) (g : Fin K → settings.RoundDraws α) (r : ℕ),
      ∀ code ∈ histCodes settings O P.D Dsf ω g r,
        code.2.1.card ≤ 6 * (K + 1) * NP ∧ code.2.2.card ≤ NS + 1 := by
    intro ω g r code hcode
    obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hcode
    obtain ⟨hiK, code', hget', h1, h2⟩ := histCodes_getElem settings O P.D Dsf ω g r i hi
    rw [List.getElem?_eq_getElem hi, hget, Option.some.injEq] at hget'
    subst hget'
    set pops := loopPopulations K (histAt (histCodes settings O P.D Dsf ω g r) i).pops.length
    have hpops : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    refine ⟨?_, ?_⟩
    · rw [h1, prefsOf]
      refine (card_roundPrefs_le _ _ _).trans ?_
      calc 3 * pops.card * ∑ n ∈ (settings.schedule pops).image State.npref, n
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) (hNPsum pops)
        _ = 6 * (K + 1) * NP := by ring
    · rw [h2]
      have hn : ∀ B ∈ settings.schedule pops,
          B.nsuff = poolCount η₀ pops ι εcov (δc / 2) (pAP - ms) c := fun B hB =>
        (hsizes pops B hB).1
      refine (card_roundSuffs_le _ hn _).trans (Nat.succ_le_succ ?_)
      exact hNSb pops (Finset.mem_insert_self _ _)
  -- the rounds' bad events
  set E1 : (r : ℕ) → r < K → Set (Ω × (Fin K → settings.RoundDraws α)) := fun r hr =>
    {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (dec (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
          (∀ t, (listLaw ℓ).real {t} ≤ cap) ∧ listLaw ℓ {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ Dsf.real ↑(dec (histCodes settings O P.D Dsf p.1 p.2 r)).suffs ≤ ms
        ∧ ¬ ClusterSpec settings O P.D (513 / 512 * εcov) slack (dec
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops
          (idealRun P.D Dsf (dec (histCodes settings O P.D Dsf p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
  set A := lazySlope σ Nr nh
  set sl := lazySlack σ nh δH
  set E2 : (r : ℕ) → r < K → Set (Ω × (Fin K → settings.RoundDraws α)) := fun r hr =>
    {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (dec (histCodes settings O P.D Dsf p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ RoundSpec M O P A sl (C.disagreementRate + tw + mr) k
            (idealRound settings O P.D Dsf (dec (histCodes settings O P.D Dsf p.1 p.2 r))
              (p.1, p.2 ⟨r, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
  set E3 : (r : ℕ) → r < K → Set (Ω × (Fin K → settings.RoundDraws α)) := fun r hr =>
    {p | (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (dec (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            listLaw ℓ {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ 1 / 2)
        ∧ Dsf.real ↑(dec (histCodes settings O P.D Dsf p.1 p.2 r)).suffs ≤ 1 / 2
        ∧ ¬ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
  set E4 : (r : ℕ) → r < K → Set (Ω × (Fin K → settings.RoundDraws α)) := fun r hr =>
    {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (dec (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            ∀ t, (listLaw ℓ).real {t} ≤ cap)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ ScanShort (loopPopulations K r) settings.scanCap Tsc ((K + 3) * NP)
            (idealDraws P.D Dsf
              (dec (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))}
  have h1 : ∀ r (hr : r < K), μL.real (E1 r hr) ≤ δc + gate := fun r hr =>
    loop_cluster_round_le_persist settings O P Dsf Ls (513 / 512 * εcov) (δc + gate) slack cap cap
      m ms pAP (by linarith) hm1 hms1 hclust hLs' hDat hbalL hbalN hDsfLen hpAPDsf hatom r hr
  set δ3 : ℝ := lazyRoundBound σ C E J b cross Nr P.L δH tw with hδ3d
  have h2 : ∀ r (hr : r < K), μL.real (E2 r hr) ≤ δ3 := fun r hr =>
    loop_round_le_persist M O hL P Dsf settings Ls NP mr cross δH tw Nr hNPsum hband hk0 hKw
      hη1 hcr hprem hNr hδH htw hmr hmr1 r hr
  have h3 : ∀ r (hr : r < K), μL.real (E3 r hr) = 0 := fun r hr => by
    rw [measureReal_def, loop_coll_null settings O P Dsf Ls hLs' hDsfLen r hr,
      ENNReal.toReal_zero]
  have h4 : ∀ r (hr : r < K), μL.real (E4 r hr) ≤ bsc := fun r hr =>
    loop_good_round_le_persist settings O P Dsf cap m hm1 hTsc hTM hDat r hr
  set Ev : ℕ → Set (Ω × (Fin K → settings.RoundDraws α)) := fun r =>
    if hr : r < K then ((E1 r hr ∪ E2 r hr) ∪ E3 r hr) ∪ E4 r hr else ∅ with hEvd
  have hbsc : 0 ≤ bsc := by positivity
  have hδ30 : 0 ≤ δ3 := by
    have hδ : 0 < C.δ := hprem.2.2.2.1
    have hy1 : C.harvestRate ≤ 1 := hprem.2.2.1
    have e1 : 0 ≤ (E : ℝ) * J * (1 - C.harvestRate / 4) ^ b :=
      mul_nonneg (by positivity) (pow_nonneg (by linarith) b)
    have e2 : 0 ≤ cross * Nr := mul_nonneg hcr.le (Nat.cast_nonneg _)
    have e3 : 0 ≤ ((P.L : ℝ) + 1) * (Fintype.card σ + 2) * cross / tw :=
      div_nonneg (mul_nonneg (by positivity) hcr.le) htw.le
    rw [hδ3d, lazyRoundBound]
    linarith
  have hEv : ∀ r, μL.real (Ev r) ≤ δc + gate + δ3 + bsc := by
    intro r
    simp only [hEvd]
    split_ifs with hr
    · calc μL.real (((E1 r hr ∪ E2 r hr) ∪ E3 r hr) ∪ E4 r hr)
          ≤ μL.real (E1 r hr) + μL.real (E2 r hr) + μL.real (E3 r hr) + μL.real (E4 r hr) :=
            (measureReal_union_le _ _).trans (add_le_add ((measureReal_union_le _ _).trans
              (add_le_add (measureReal_union_le _ _) le_rfl)) le_rfl)
        _ ≤ δc + gate + δ3 + 0 + bsc :=
            add_le_add (add_le_add (add_le_add (h1 r hr) (h2 r hr)) (le_of_eq (h3 r hr))) (h4 r hr)
        _ = δc + gate + δ3 + bsc := by ring
    · rw [measureReal_empty]
      linarith
  -- a failing run fails one of the first `|Q| + 1` rounds' claims
  have hsub : {p : Ω × (Fin K → settings.RoundDraws α) |
      let rounds := idealRounds settings O P.D Dsf (persistDraws p) K
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (513 / 512 * εcov + 2 * ι + 2 * slack) (2 * ι + slack)
            (C.disagreementRate + tw + mr))
        ∧ (∀ o ∈ rounds, o.result.peak ≤ Fintype.card σ + 2)
        ∧ RoundsFresh settings O P.D Dsf Ls p
        ∧ ∀ r (hr : r < K), AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
          RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf
            (persistDraws p) r)) (persistDraws p ⟨r, hr⟩))}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r := by
    intro p hp
    by_contra hno
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hno
    set Lr := idealRounds settings O P.D Dsf (persistDraws p) K with hLr
    have hspec : ∀ i, i ≤ Fintype.card σ → ∀ (hi : i < Lr.length) (hK : i < K),
        (ClusterSpec settings O P.D (513 / 512 * εcov) slack (specHistOf (Lr.take i)).pops
            (idealRunAt settings O P.D Dsf (persistDraws p) ⟨i, hK⟩)
          ∧ RoundSpec M O P A sl (C.disagreementRate + tw + mr) k Lr[i])
        ∧ (HistOK settings O P.D Dsf p.1 p.2 i
          ∧ Returns settings O P.D Dsf (dec (histCodes settings O P.D Dsf p.1 p.2 i))
            (p.1, (p.2 ⟨i, hK⟩).1)
          ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 i) p.2 i hK)
        ∧ RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf (persistDraws p)
            i)) (persistDraws p ⟨i, hK⟩) := by
      intro i
      induction i using Nat.strong_induction_on with
      | _ i ih =>
        intro hiσ hi hK'
        obtain ⟨_, hlen, hdec, e⟩ := persist_bridge settings O P.D Dsf p i hi
        have hnot : p ∉ Ev i := hno i (by omega)
        simp only [hEvd, dif_pos hK', Set.mem_union, not_or] at hnot
        obtain ⟨⟨⟨hn1, hn2⟩, hn3⟩, hn4⟩ := hnot
        set cH := histCodes settings O P.D Dsf p.1 p.2 i
        have hdec' : dec cH = specHistOf (Lr.take i) := hdec
        have hok : HistOK settings O P.D Dsf p.1 p.2 i :=
          histOK_of_specs settings O P.D Dsf (513 / 512 * εcov) slack p i hi.le
            fun j hj hjK hji => (ih j hji (by omega) hj hjK).1.1
        have hti := take_idealRounds settings O P.D Dsf (persistDraws p) hK'.le
        have hpops : ∀ ℓ ∈ (dec cH).pops,
            (∀ t, (listLaw ℓ).real {t} ≤ cap) ∧ listLaw ℓ {q | Ls ≤ q.toList.length} = 0 := by
          rw [hdec']
          intro ℓ hℓ
          obtain ⟨o, ho, hoℓ⟩ := List.mem_filterMap.1 hℓ
          obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
          simp only [List.length_take] at hj
          rw [List.getElem_take] at hoℓ
          have hji : j < i := by omega
          obtain ⟨hKj, ej⟩ := getElem_idealRounds settings O P.D Dsf (persistDraws p) K j
            (by rw [← hLr]; omega)
          have hpk := (ih j hji (by omega) (by omega) hKj).1.2.2
          rw [ej] at hoℓ hpk
          have hnd := population_nodup settings O P.D _ _ _ hnh hoℓ
          refine ⟨fun t => (listLaw_atom_le hnd.1 hnd.2 hnh t).trans hinv, ?_⟩
          exact population_short settings O P (by omega) _ _ _ hpk hoℓ
        have hpl : (dec cH).pops.length = i := by simp [hdecd, decodeHist, cH, hlen]
        have hatoms : ∀ j ∈ loopPopulations K i, ∀ t,
            (loopDistributions P.D (dec cH).pops j).real {t} ≤ cap := by
          intro j hj t
          cases j with
          | none => exact hDat t
          | some i' =>
            have hi' : i'.val < (dec cH).pops.length := by
              rw [hpl]; exact lt_of_some_mem_loopPops hj
            rw [loopDistributions, List.getElem?_eq_getElem hi']
            exact (hpops _ (List.getElem_mem _)).1 t
        have hiK : (i : ℝ) ≤ K := by exact_mod_cast hK'.le
        have hcardP : ((dec cH).prefs.card : ℝ) ≤ i * (6 * (K + 1) * NP) := by
          have := card_foldr_union_le (fun c' : HarvestCode α => c'.2.1) cH (6 * (K + 1) * NP)
            fun x hx => (hcodes p.1 p.2 i x hx).1
          rw [hlen] at this
          exact_mod_cast this
        have hcardS : ((dec cH).suffs.card : ℝ) ≤ i * (NS + 1) := by
          have := card_foldr_union_le (fun c' : HarvestCode α => c'.2.2) cH (NS + 1)
            fun x hx => (hcodes p.1 p.2 i x hx).2
          rw [hlen] at this
          exact_mod_cast this
        have hmP : ∀ j ∈ loopPopulations K i,
            (loopDistributions P.D (dec cH).pops j).real ↑(dec cH).prefs ≤ m := by
          intro j hj
          refine (measureReal_le_card_mul _ (hatoms j hj)).trans ?_
          calc ((dec cH).prefs.card : ℝ) * cap ≤ i * (6 * (K + 1) * NP) * cap :=
                mul_le_mul_of_nonneg_right hcardP hcap.le
            _ ≤ K * (6 * (K + 1) * NP) * cap := by gcongr
            _ = 6 * K * (K + 1) * NP * cap := by ring
            _ ≤ m := hqm
        have hmS : Dsf.real ↑(dec cH).suffs ≤ ms := by
          refine (measureReal_le_card_mul _ hatom).trans ?_
          calc ((dec cH).suffs.card : ℝ) * cap ≤ i * (NS + 1) * cap :=
                mul_le_mul_of_nonneg_right hcardS hcap.le
            _ ≤ K * (NS + 1) * cap := by gcongr
            _ ≤ ms := hqms
        have hnc : NoColl settings P.D Dsf Ls cH p.2 i hK' := by
          by_contra h
          exact hn3 ⟨hlen, fun ℓ hℓ => (hpops ℓ hℓ).2, fun j hj => (hmP j hj).trans hm1,
            hmS.trans hms1, h⟩
        have hcs : ClusterSpec settings O P.D (513 / 512 * εcov) slack (dec cH).pops
            (idealRun P.D Dsf (dec cH) (p.1, (p.2 ⟨i, hK'⟩).1)) := by
          by_contra h
          exact hn1 ⟨hok, hlen, hpops, hmP, hmS, h, hnc⟩
        have h4 : RoundSpec M O P A sl (C.disagreementRate + tw + mr) k
            (idealRound settings O P.D Dsf (dec cH) (p.1, p.2 ⟨i, hK'⟩)) := by
          by_contra h
          exact hn2 ⟨hok, hlen, hcs.1, h, hnc⟩
        have hcs' : ClusterSpec settings O P.D (513 / 512 * εcov) slack
            (specHistOf (Lr.take i)).pops
            (idealRunAt settings O P.D Dsf (persistDraws p) ⟨i, hK'⟩) := by
          rw [idealRunAt, ← hti, ← hdec]
          exact hcs
        have hnA : (2 + (loopPopulations K (specHistOf (Lr.take i)).pops.length).card) * NP
            ≤ (K + 3) * NP :=
          Nat.mul_le_mul_right _ (by
            have := card_loopPopulations_le K (specHistOf (Lr.take i)).pops.length
            omega)
        have hn4' : ¬ ScanShort (loopPopulations K i) settings.scanCap Tsc ((K + 3) * NP)
              (idealDraws P.D Dsf (dec cH) (p.1, (p.2 ⟨i, hK'⟩).1)) :=
          fun h => hn4 ⟨hok, hlen, fun ℓ hℓ => (hpops ℓ hℓ).1, hmP, h⟩
        have hl : (specHistOf (Lr.take i)).pops.length = i := by rw [← hdec]; exact hpl
        have hgood : RoundGood settings O P.D Dsf
            (specHistOf (idealRounds settings O P.D Dsf (persistDraws p) i))
            (persistDraws p ⟨i, hK'⟩) := by
          rw [← hti]
          refine roundGood_of settings O P.D Dsf _ _ (NP := NP) (A := (K + 3) * NP) (T := Tsc)
            (fun B hB => (hsizes _ B hB).2.1) hnA (by omega) ?_ ?_
          · rw [hl, ← hdec]
            exact hn4'
          · have h3 := hcs'.1
            unfold idealRunAt at h3
            rw [← hti] at h3
            exact h3
        rw [e]
        exact ⟨⟨hcs', h4⟩, ⟨hok, hcs.1, hnc⟩, hgood⟩
    have hsched : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi := fun pops B hB =>
      (hband pops B hB).1
    have hag := loop_agrees_of_specs M O hL P Dsf settings (persistDraws p) k (513 / 512 * εcov)
      slack A sl (C.disagreementRate + tw + mr) Tu hK hT (by linarith) hcond hsched
      fun i hi hiK hiσ => (hspec i hiσ hi hiK).1
    have hrlt : ∀ r (hr : r < K), AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
        r < Lr.length := by
      intro r hr hall
      have hlr := length_eq_of_allHarvested settings O P.D Dsf (persistDraws p) hr.le hall
      have h1 := length_idealRounds_le settings O P.D Dsf (persistDraws p) (r + 1)
      have h2 : (idealRounds settings O P.D Dsf (persistDraws p) (r + 1)).length = r + 1 := by
        rw [idealRounds_succ, dif_pos ⟨hr, hall⟩, List.length_append, hlr, List.length_singleton]
      have h3 := congrArg List.length (take_idealRounds settings O P.D Dsf (persistDraws p)
        (show r + 1 ≤ K by omega))
      have h4 := congrArg List.length hLr
      rw [List.length_take, h2] at h3
      omega
    refine hp ⟨hag, fun o ho => ?_, fun i hi => ?_, fun r hr hall => ?_⟩
    · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
      have hle := hag.1
      exact (hspec i (by omega) hi (by omega)).1.2.2
    · have hle := hag.1
      have hK : i < K := by omega
      exact ⟨hK, (hspec i (by omega) hi hK).2.1⟩
    · have hr1 := hrlt r hr hall
      have h4 := congrArg List.length hLr
      exact (hspec r (by have := hag.1; omega) hr1 hr).2.2
  calc μL.real _ ≤ μL.real (⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1), μL.real (Ev r) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (δc + gate + δ3 + bsc) :=
        Finset.sum_le_sum fun r _ => hEv r
    _ = (Fintype.card σ + 1) * (δc + gate + δ3 + bsc) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

end OrthoDFA
