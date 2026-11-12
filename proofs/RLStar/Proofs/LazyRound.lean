import RLStar.LowLevelClaims.LazyRound
import RLStar.LowLevelClaims.TrieMass
import RLStar.Proofs.LazyShape
import RLStar.Proofs.Query
import RLStar.Proofs.SeqTest

/-!
# `LazyRoundCorrect`: the proof

Epoch `e` ends badly for its level with chance at most `δ/((e + 1)(e + 2))`, and its top-up runs
out of blocks or retracts with harvest mass at least `y/2` with chance at most `J (1 − y/4)^b`.
Outside those, a retraction drops the harvest mass from at least `y` to below `y/2`, so it moves
the walk start of more than `y/2` of the probes and adds more than `y/2` to the trie's mass, which
stays at most `K`. So the epochs never run out.
-/

noncomputable section

universe u v

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type u} [Fintype α] [DecidableEq α]

/-- Σ_{u ∈ X} D{x : u ⊑ x}. -/
private def trieMass (D : Measure (FreeMonoid α)) (X : Finset (FreeMonoid α)) : ℝ :=
  ∑ u ∈ X, D.real {x | prefixOf x u.toList.length = u}

omit [DecidableEq α] in
private theorem harvestMass_sub_le (hW : WalkStartMoves.{u}) (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) {X Y : Finset (FreeMonoid α)} (hXY : X ⊆ Y) :
    harvestMass read D K X H s - harvestMass read D K Y H s ≤ trieMass D Y - trieMass D X := by
  classical
  have hsub : {x | siteHarvest (padRead read) K H s (padIn K X x) ≠ []}
      ⊆ {x | siteHarvest (padRead read) K H s (padIn K Y x) ≠ []}
        ∪ {x | walkStartIn X x ≠ walkStartIn Y x} := by
    intro x hx
    by_cases h : walkStartIn X x = walkStartIn Y x
    · left
      have hp : padIn K X x = padIn K Y x := by unfold padIn; rw [h]
      rw [Set.mem_ofPred_eq, ← hp]
      exact hx
    · exact Or.inr h
  have h1 := (measureReal_mono (μ := D) hsub).trans (measureReal_union_le _ _)
  have h2 := hW D X Y hXY
  have h3 := Finset.sum_sdiff hXY (f := fun u => D.real {x | prefixOf x u.toList.length = u})
  unfold harvestMass trieMass
  linarith

private def RoundBad (read : FreeMonoid α → ARU) (C : Level3Settings) (D : Measure (FreeMonoid α))
    (K nQ : ℕ) (r : LazyResult α) : Prop :=
  nQ + 2 < r.hypothesis.tree.leaves.length ∨
    match r.ending with
    | .consistent => C.disagreementRate < D.real (lazyDisagree read K r.expanded r.hypothesis)
    | .harvest _ _ => False
    | .failed => True

private def EpochBad (read : FreeMonoid α → ARU) (C : Level3Settings) (D : Measure (FreeMonoid α))
    (K nQ : ℕ) (X : Finset (FreeMonoid α)) (r : Level3Result (Option α)) : Prop :=
  nQ + 2 < r.hypothesis.tree.leaves.length ∨
    match r.ending with
    | .consistent => C.disagreementRate < D.real (lazyDisagree read K X r.hypothesis)
    | .harvest s => harvestMass read D K X r.hypothesis s < C.harvestRate
    | .failed => True

private def TopUpBad (read : FreeMonoid α → ARU) (C : Level3Settings)
    (D : Measure (FreeMonoid α)) (K nh : ℕ) (H : HypothesisDFA (Option α)) (s : Level3Site)
    (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)) {J b : ℕ}
    (B : Fin J → Fin b → FreeMonoid α) : Prop :=
  match (topUpQ K nh H s (List.ofFn B) X ℓ).eval read with
  | none => True
  | some (.full _, _) => False
  | some (.retract _ X', _) => C.harvestRate / 2 ≤ harvestMass read D K X' H s

private abbrev epochLaw (D : Measure (FreeMonoid α)) (P J b : ℕ) :
    Measure (EpochDraws α P J b) :=
  (Measure.pi fun _ : Fin P => D).prod (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D)

omit [DecidableEq α] in
private theorem topUpQ_full_of_zero (read : FreeMonoid α → ARU) (K : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) (X : Finset (FreeMonoid α))
    (ℓ : List (FreeMonoid α)) {J b : ℕ} (hJ : 0 < J) (B : Fin J → Fin b → FreeMonoid α) :
    ∃ ℓ' rest, (topUpQ K 0 H s (List.ofFn B) X ℓ).eval read = some (.full ℓ', rest) := by
  obtain ⟨J, rfl⟩ := Nat.exists_eq_add_of_lt hJ
  rw [List.ofFn_succ, topUpQ, Query.eval_bind]
  simp

open scoped Classical in
private theorem roundBad_le (read : FreeMonoid α → ARU) (C : Level3Settings)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (K nh nQ E J b : ℕ)
    (hW : WalkStartMoves.{u}) {q : ℝ} (hδ : 0 ≤ C.δ) (hq : 0 ≤ q) (hJ : 0 < J)
    (hmass : ∀ X, TrieOK K X → trieMass D X ≤ K)
    (hB1 : ∀ e X, e < E → (Measure.pi fun _ : Fin C.probeBudget => D).real
        {z | EpochBad read C D K nQ X ((epochQ (epochSettings C e) K X z).eval read)}
        ≤ C.δ / ((e + 1) * (e + 2)))
    (hB2 : ∀ H s X ℓ, TrieOK K X → ℓ.Nodup → ℓ.length < nh →
      (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D).real
        {B | TopUpBad read C D K nh H s X ℓ B} ≤ J * q) :
    ∀ n e (X : Finset (FreeMonoid α)) (ls : Level3Site → List (FreeMonoid α)), e + n ≤ E →
      TrieOK K X →
      (0 < nh → ∀ s, (ls s).Nodup ∧ (ls s).length < nh) →
      (K : ℝ) < trieMass D X + n * (C.harvestRate / 2) →
      (Measure.pi fun _ : Fin n => epochLaw D C.probeBudget J b)
          {d | RoundBad read C D K nQ ((lazyEpochsQ C K nh e (List.ofFn d) X ls).eval read)}
        ≤ ENNReal.ofReal (C.δ / (e + 1) + n * (J * q)) := by
  intro n
  induction n with
  | zero =>
    intro e X ls _ hX _ hK
    have := hmass X hX
    simp only [CharP.cast_eq_zero, zero_mul, add_zero] at hK
    linarith
  | succ n ih =>
    intro e X ls hn hX hls hK
    let res : EpochDraws α C.probeBudget J b → Level3Result (Option α) := fun z =>
      (epochQ (epochSettings C e) K X z.1).eval read
    let B1 : Set (EpochDraws α C.probeBudget J b) := {z | EpochBad read C D K nQ X (res z)}
    let B2 : Set (EpochDraws α C.probeBudget J b) := {z | match (res z).ending with
      | .harvest s => TopUpBad read C D K nh (res z).hypothesis s X (ls s) z.2
      | _ => False}
    have hμB1 : epochLaw D C.probeBudget J b B1 ≤ ENNReal.ofReal (C.δ / ((e + 1) * (e + 2))) := by
      have hB : B1 = {z1 | EpochBad read C D K nQ X ((epochQ (epochSettings C e) K X z1).eval read)}
          ×ˢ Set.univ := by
        ext z
        simp [B1, res]
      rw [hB, Measure.prod_prod, measure_univ, mul_one, ← ofReal_measureReal]
      exact ENNReal.ofReal_le_ofReal (hB1 e X (by omega))
    have hμB2 : epochLaw D C.probeBudget J b B2 ≤ ENNReal.ofReal (J * q) := by
      rw [Measure.prod_apply (Set.to_countable _).measurableSet]
      calc ∫⁻ z1, (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D) (Prod.mk z1 ⁻¹' B2)
            ∂(Measure.pi fun _ : Fin C.probeBudget => D)
          ≤ ∫⁻ _z1, ENNReal.ofReal (J * q) ∂(Measure.pi fun _ : Fin C.probeBudget => D) := by
            refine lintegral_mono fun z1 => ?_
            rcases hr : ((epochQ (epochSettings C e) K X z1).eval read).ending with _ | s | _
            · have : Prod.mk z1 ⁻¹' B2 = ∅ := by
                ext z2
                simp [B2, res, hr]
              simp [this]
            · have : Prod.mk z1 ⁻¹' B2 = {B | TopUpBad read C D K nh
                  ((epochQ (epochSettings C e) K X z1).eval read).hypothesis s X (ls s) B} := by
                ext z2
                simp [B2, res, hr]
              rw [this]
              rcases Nat.eq_zero_or_pos nh with h0 | h0
              · subst h0
                have : {B : Fin J → Fin b → FreeMonoid α | TopUpBad read C D K 0
                    ((epochQ (epochSettings C e) K X z1).eval read).hypothesis s X (ls s) B}
                    = ∅ := by
                  ext B
                  obtain ⟨_, _, h⟩ := topUpQ_full_of_zero read K
                    ((epochQ (epochSettings C e) K X z1).eval read).hypothesis s X (ls s) hJ B
                  simp [TopUpBad, h]
                simp [this]
              · rw [← ofReal_measureReal]
                exact ENNReal.ofReal_le_ofReal (hB2 _ s X (ls s) hX (hls h0 s).1 (hls h0 s).2)
            · have : Prod.mk z1 ⁻¹' B2 = ∅ := by
                ext z2
                simp [B2, res, hr]
              simp [this]
        _ = ENNReal.ofReal (J * q) := by rw [lintegral_const, measure_univ, mul_one]
    have hpt : ∀ z, (Measure.pi fun _ : Fin n => epochLaw D C.probeBudget J b)
        {t | Fin.cons z t ∈ {d : Fin (n + 1) → EpochDraws α C.probeBudget J b |
          RoundBad read C D K nQ ((lazyEpochsQ C K nh e (List.ofFn d) X ls).eval read)}}
        ≤ B1.indicator 1 z + B2.indicator 1 z
          + ENNReal.ofReal (C.δ / (e + 2) + n * (J * q)) := by
      intro z
      by_cases h1 : z ∈ B1
      · rw [Set.indicator_of_mem h1, add_assoc]
        exact prob_le_one.trans le_self_add
      by_cases h2 : z ∈ B2
      · rw [Set.indicator_of_mem h2, add_comm (B1.indicator 1 z), add_assoc]
        exact prob_le_one.trans le_self_add
      rw [Set.indicator_of_notMem h1, Set.indicator_of_notMem h2, zero_add, zero_add]
      simp only [Set.mem_ofPred_eq, List.ofFn_cons]
      simp only [B1, B2, res, Set.mem_ofPred_eq] at h1 h2
      rcases hr : ((epochQ (epochSettings C e) K X z.1).eval read).ending with _ | s | _
      · have : {t : Fin n → EpochDraws α C.probeBudget J b | RoundBad read C D K nQ
            ((lazyEpochsQ C K nh e (z :: List.ofFn t) X ls).eval read)} = ∅ := by
          ext t
          simp only [lazyEpochsQ, Query.eval_bind, hr, Query.eval_pure, Set.mem_ofPred_eq,
            Set.mem_empty_iff_false, iff_false]
          simpa [RoundBad, EpochBad, hr] using h1
        simp [this]
      · rcases ht : (topUpQ K nh ((epochQ (epochSettings C e) K X z.1).eval read).hypothesis s
            (List.ofFn z.2) X (ls s)).eval read with _ | ⟨_ | ⟨ℓ', X'⟩, rest⟩
        · exact absurd (by simp [hr, TopUpBad, ht]) h2
        · have : {t : Fin n → EpochDraws α C.probeBudget J b | RoundBad read C D K nQ
              ((lazyEpochsQ C K nh e (z :: List.ofFn t) X ls).eval read)} = ∅ := by
            ext t
            simp only [lazyEpochsQ, Query.eval_bind, hr, ht, Query.eval_pure, Set.mem_ofPred_eq,
              Set.mem_empty_iff_false, iff_false]
            simp only [EpochBad, hr, not_or, not_lt] at h1
            simpa [RoundBad] using h1.1
          simp [this]
        · have hnh : 0 < nh := by
            rcases Nat.eq_zero_or_pos nh with h0 | h0
            · subst h0
              obtain ⟨_, _, h⟩ := topUpQ_full_of_zero read K
                ((epochQ (epochSettings C e) K X z.1).eval read).hypothesis s X (ls s) hJ z.2
              rw [h] at ht
              cases ht
            · exact h0
          obtain ⟨hX', hXX', hnd, hlen, -, -⟩ := topUpQ_retract_inv read _ s K nh _ X (ls s) hX
            (hls hnh s).1 (hls hnh s).2 ht
          have hy1 : C.harvestRate ≤ harvestMass read D K X
              ((epochQ (epochSettings C e) K X z.1).eval read).hypothesis s := by
            simp only [EpochBad, hr, not_or, not_lt] at h1
            exact h1.2
          have hy2 : harvestMass read D K X'
              ((epochQ (epochSettings C e) K X z.1).eval read).hypothesis s
              < C.harvestRate / 2 := by
            simpa [hr, TopUpBad, ht] using h2
          have hgap := harvestMass_sub_le hW read D K
            ((epochQ (epochSettings C e) K X z.1).eval read).hypothesis s hXX'
          have hset : {t : Fin n → EpochDraws α C.probeBudget J b | RoundBad read C D K nQ
              ((lazyEpochsQ C K nh e (z :: List.ofFn t) X ls).eval read)}
              = {t | RoundBad read C D K nQ ((lazyEpochsQ C K nh (e + 1) (List.ofFn t) X'
                (Function.update ls s ℓ')).eval read)} := by
            ext t
            simp only [lazyEpochsQ, Query.eval_bind, hr, ht]
          rw [hset]
          refine (ih (e + 1) X' _ (by omega) hX' ?_ ?_).trans_eq ?_
          · intro _ s'
            by_cases hs : s' = s
            · subst hs
              simp [hnd, hlen]
            · rw [Function.update_of_ne hs]
              exact hls hnh s'
          · push_cast at hK ⊢
            linarith
          · congr 1
            push_cast
            ring
      · exact absurd (by simp [EpochBad, hr]) h1
    rw [pi_cons_apply]
    calc ∫⁻ z, _ ∂(epochLaw D C.probeBudget J b)
        ≤ ∫⁻ z, (B1.indicator 1 z + B2.indicator 1 z
          + ENNReal.ofReal (C.δ / (e + 2) + n * (J * q))) ∂(epochLaw D C.probeBudget J b) :=
          lintegral_mono hpt
      _ = epochLaw D C.probeBudget J b B1 + epochLaw D C.probeBudget J b B2
          + ENNReal.ofReal (C.δ / (e + 2) + n * (J * q)) := by
        rw [lintegral_add_right _ measurable_const, lintegral_add_left (measurable_of_countable _),
          lintegral_indicator_one (Set.to_countable _).measurableSet,
          lintegral_indicator_one (Set.to_countable _).measurableSet, lintegral_const,
          measure_univ, mul_one]
      _ ≤ ENNReal.ofReal (C.δ / ((e + 1) * (e + 2))) + ENNReal.ofReal (J * q)
          + ENNReal.ofReal (C.δ / (e + 2) + n * (J * q)) := by gcongr
      _ = ENNReal.ofReal (C.δ / (e + 1) + (n + 1 : ℕ) * (J * q)) := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        push_cast
        field_simp
        ring

end Random

open Random

theorem lazyRoundCorrect_of (hE : EpochCorrect.{u, v}) (hT : TopUpCorrect.{u})
    (hM : TrieMassBound.{u}) (hW : WalkStartMoves.{u}) : LazyRoundCorrect.{u, v} := by
  intro α _ _ σ _ M side read C D _ K nh E J b p₁ X₀ y ε nQ nα S hread hX₀ hlong hp₁ hnhp hδ hδ1
    hε hε1 hbudget hEK hJ
  have hε0 : 0 < C.disagreementRate := hε
  have hy : 0 < C.harvestRate := by unfold Level3Settings.harvestRate; linarith
  have hy1 : C.harvestRate ≤ 1 := by
    have : C.disagreementRate < 1 := hε1
    unfold Level3Settings.harvestRate
    linarith
  have hq : 0 ≤ (1 - C.harvestRate / 4) ^ b := pow_nonneg (by linarith) _
  have hJ0 : 0 < J := lt_of_lt_of_le (Nat.succ_pos _) hJ
  have hbudget' : 36 * (S : ℝ) / C.disagreementRate
      * Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) + S ≤ C.probeBudget := hbudget
  have hB1 : ∀ e X, e < E → (Measure.pi fun _ : Fin C.probeBudget => D).real
      {z | EpochBad read C D K nQ X ((epochQ (epochSettings C e) K X z).eval read)}
      ≤ C.δ / ((e + 1) * (e + 2)) := by
    intro e X he
    have hee : ((e : ℝ) + 1) * (e + 2) ≤ E * (E + 1) := by
      have : (e : ℝ) + 1 ≤ E := by exact_mod_cast he
      nlinarith
    have hpos : (0 : ℝ) < (e + 1) * (e + 2) := by positivity
    refine hE M side read (epochSettings C e) D K X hread (div_pos hδ hpos) ?_ hε hε1 ?_
    · refine lt_of_le_of_lt (div_le_self hδ.le ?_) hδ1
      nlinarith
    · change 36 * (S : ℝ) / C.disagreementRate
          * Real.log (16 * S * (S + 1) / (C.δ / ((e + 1) * (e + 2)))) + S ≤ C.probeBudget
      have hlog : Real.log (16 * S * (S + 1) / (C.δ / ((e + 1) * (e + 2))))
          ≤ Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) := by
        apply Real.log_le_log (by positivity)
        rw [div_div_eq_mul_div]
        gcongr
      have := mul_le_mul_of_nonneg_left hlog (by positivity : (0 : ℝ) ≤ 36 * S / C.disagreementRate)
      linarith
  have hB2 : ∀ H s X ℓ, TrieOK K X → ℓ.Nodup → ℓ.length < nh →
      (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D).real
        {B | TopUpBad read C D K nh H s X ℓ B} ≤ J * (1 - C.harvestRate / 4) ^ b :=
    fun H s X ℓ hX hnd hlen =>
      hT read H s D K nh J b C.harvestRate p₁ X ℓ hX hnd hlen hlong hp₁ hnhp hy1 hJ
  have hKE : (K : ℝ) < trieMass D X₀ + E * (C.harvestRate / 2) := by
    have h0 : 0 ≤ trieMass D X₀ := Finset.sum_nonneg fun _ _ => measureReal_nonneg
    have hEK' : 2 * K / C.harvestRate + 1 ≤ E := hEK
    have h1 : 2 * (K : ℝ) / C.harvestRate * C.harvestRate = 2 * K := div_mul_cancel₀ _ hy.ne'
    nlinarith [mul_le_mul_of_nonneg_right hEK' hy.le]
  have key := roundBad_le read C D K nh nQ E J b hW hδ.le hq hJ0 (fun X hX => hM D X K hX.1)
    hB1 hB2 E 0 X₀ (fun _ => []) (by omega) hX₀ (fun h _ => ⟨List.nodup_nil, h⟩) hKE
  refine (ENNReal.toReal_le_of_le_ofReal (add_nonneg (div_nonneg hδ.le (by positivity))
    (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg (Nat.cast_nonneg _) hq))) key).trans_eq ?_
  push_cast
  ring

end OrthoDFA

end
