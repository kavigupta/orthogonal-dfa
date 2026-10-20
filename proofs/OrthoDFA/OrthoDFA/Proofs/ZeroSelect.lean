import OrthoDFA.Proofs.LoopSelect
import OrthoDFA.Proofs.ZeroRuns

/-!
# The first round of a loop walked from before its avoidance depth, scored on held-out strings

Where its tree keeps at most `|Q| + 2` leaves, level 3 reads only strings beginning with some
probe's first `a` letters, or at most `a + |Q|` long, followed by the family's suffixes
(`level3_congr_zoneZ`). Held-out strings longer than that, distinct, and whose first `a` letters
are neither a clustering prefix's nor a probe's are read at none of the strings the round read
(`loop_select_zero`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Zone

open scoped Classical in
/-- The strings level 3 reads over the probes `xs` where its tree keeps at most `nQ + 2` leaves:
strings beginning with a probe's first `a` letters, or at most `a + nQ` long, followed by the
family's suffixes. -/
def level3ZoneZ (a nQ : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) (F : Finset (FreeMonoid α)) :
    Finset (FreeMonoid α) :=
  readStrings ((shortStrings (probesLen xs + nQ + 2)).filter fun z =>
    (a ≤ z.toList.length ∧ prefixOf z a ∈ probeKeys a xs) ∨ z.toList.length ≤ a + nQ) F

omit [IsProbabilityMeasure μ] in
theorem level3_congr_zoneZ (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) {a nQ : ℕ} (hka : C.walkStart ≤ a) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ w ∈ level3ZoneZ a nQ xs F, (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (hB : ((Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω)
      xs).hypothesis.tree.leaves.length ≤ nQ + 2)) :
    Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
      = Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω') xs := by
  classical
  refine Random.level3_congr_small C xs (n := nQ) (fun i j _ hjx m hm => familyRead_congr O h ?_)
    hB
  have h1 := le_probesLen xs i
  have hlp : (prefixOf (xs i) j).toList.length = j := by
    simp [prefixOf]; omega
  refine Finset.mem_filter.2 ⟨mem_shortStrings.2 ?_, ?_⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, hlp]; omega
  · by_cases hja : a ≤ j
    · left
      refine ⟨by simp only [FreeMonoid.toList_mul, List.length_append, hlp]; omega, ?_⟩
      rw [prefixOf_mul_of_le (by rw [hlp]; exact hja), prefixOf_prefixOf_of_le _ hja]
      exact Finset.mem_image.2 ⟨i, Finset.mem_univ _, rfl⟩
    · right
      simp only [FreeMonoid.toList_mul, List.length_append, hlp]; omega

end Zone

section Fail

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (N Ls L n nL nQ : ℕ) (t : ℝ) (r : ℕ) (hr : r < K)

/-- Round `r` ended consistent after `r` harvests, its tree keeping at most `nQ + 2` leaves. -/
def SelGZ (y : Option (Stage4 α × ResCode α)) : Prop :=
  ∃ t h, y = some (t, h) ∧ t.1.length = r ∧ h.2.2 = true ∧ h.1.leaves.length ≤ nQ + 2

/-- Held-out strings, distinct, of length `L`, whose first `avoidDepth` letters are neither a key of
round `r`'s prefixes nor of its probes. -/
def SelDkZ : Option (Stage4 α × ResCode α) →
    Set ((Fin K → RoundDraws α K settings.level3.probeBudget) × (Fin n → FreeMonoid α))
  | some (t, _) => {q | Function.Injective q.2 ∧ ∀ i, (q.2 i).toList.length = L
      ∧ prefixOf (q.2 i) settings.avoidDepth ∉ t.2.2.2
        ∪ probeKeys settings.avoidDepth (probesOf settings D t.2.2.2 (q.1 ⟨r, hr⟩).2)}
  | none => Set.univ

/-- Round `r` ended consistent with a small tree, and its scores strayed or a held-out string
collided. -/
def selFailZ : Set ((Ω × (Fin K → RoundDraws α K settings.level3.probeBudget))
    × (Fin n → FreeMonoid α)) :=
  {q | SelGZ nQ r (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
    ∧ q.1.2 ∈ SelOk settings D Dsf Ls r hr (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
    ∧ ((q.1.1, q.2) ∈ SelBad O L n nL t (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
      ∨ (q.1.2, q.2) ∉ SelDkZ settings D L n r hr (stageSel settings O D Dsf N r hr q.1.1 q.1.2))}

end Fail

section Bound

variable [Nonempty α]

set_option maxHeartbeats 1000000 in -- the selection's many obligations
/-- `loop_select_round_le` for a loop walked from before its avoidance depth `a`, at a round whose
tree keeps at most `nQ + 2` leaves, held-out strings longer than `a + nQ`. -/
theorem loop_select_zero (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (Ls L n nL nQ NP : ℕ) (t : ℝ) (ht : 0 ≤ t) (hkL : settings.avoidDepth + nQ < L)
    (hka : settings.level3.walkStart ≤ settings.avoidDepth)
    (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP) (r : ℕ) (hr : r < K) :
    ((loopMeasure μ α K settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      (selFailZ settings O P.D Dsf (P.L + settings.level3.probeBudget + 2) Ls L n nL nQ t r hr)
      ≤ nL * (4 * Real.exp (-(n * t ^ 2 / 2)))
        + (n * (6 * (K + 1) * NP + settings.level3.probeBudget)
            / (Fintype.card α : ℝ) ^ settings.avoidDepth
          + n ^ 2 / (Fintype.card α : ℝ) ^ L) := by
  classical
  set T := settings.level3.probeBudget
  set k := settings.avoidDepth
  set kw := settings.level3.walkStart
  set N := P.L + settings.level3.probeBudget + 2
  have hN : P.L + settings.level3.probeBudget + 2 ≤ N := le_rfl
  set rK : Fin K := ⟨r, hr⟩
  set dec := decodeHist P.D kw with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  have hA0 : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  set Y := Option (Stage4 α × ResCode α)
  set stage := stageSel settings O P.D Dsf N r hr
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some (t, _) => (histReads settings P.D Dsf N t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf P.D Dsf (dec t.1) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K))))
        ∪ level3ZoneZ k nQ (probesOf settings P.D t.2.2.2 (x rK).2) t.2.2.1
    | none => ∅ with hWd
  set Z : Y → (Fin n → FreeMonoid α) → Finset (FreeMonoid α) := fun y b => match y with
    | some (t, _) => readStrings (Finset.univ.image b) t.2.2.1
    | none => ∅ with hZd
  set δ₁ : ℝ := nL * (4 * Real.exp (-(n * t ^ 2 / 2)))
  set δ₂ : ℝ := n * (6 * (K + 1) * NP + T) / (Fintype.card α : ℝ) ^ k
    + n ^ 2 / (Fintype.card α : ℝ) ^ L
  have hδ₁ : 0 ≤ δ₁ := by positivity
  have hδ₂ : 0 ≤ δ₂ := by positivity
  have hmain := select_le O (Measure.pi fun _ : Fin K => roundLaw α K T)
    (Measure.pi fun _ : Fin n => uniformStrings (α := α) L) stage (SelGZ nQ r)
    (fun y hy => by
      obtain ⟨t', h', rfl, -, -, -⟩ := hy
      have hset : {p : Ω × (Fin K → RoundDraws α K T) | stage p.1 p.2 = some (t', h')}
          = {p | stage4 settings O P.D Dsf N r hr p.1 p.2 = some t'}
            ∩ (fun p : Ω × (Fin K → RoundDraws α K T) =>
                (p.1, probesOf settings P.D t'.2.2.2 (p.2 rK).2)) ⁻¹'
              {e : Ω × (Fin T → FreeMonoid α) | resCode (Random.indecisiveLStarLevel3 settings.level3
                (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z e.1) e.2) = h'} := by
        ext p
        simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_preimage, stage, stageSel,
          Option.map_eq_some_iff]
        constructor
        · rintro ⟨t'', ht'', he⟩
          simp only [Prod.mk.injEq] at he
          obtain ⟨rfl, he⟩ := he
          exact ⟨ht'', he⟩
        · rintro ⟨ht, he⟩
          exact ⟨t', ht, by simp only [Prod.mk.injEq]; exact ⟨trivial, he⟩⟩
      rw [hset]
      refine (measurableSet_stage4 settings O P.D Dsf N P.length_le hN r hr _).inter ?_
      refine (measurable_fst.prodMk ((measurable_probesOf settings P.D _).comp
        (measurable_snd.comp ((measurable_pi_apply rK).comp measurable_snd)))) ?_
      refine measurableSet_prod_countable_right _ fun xs => ?_
      refine measurableSet_of_reads O (readStrings (shortStrings (probesLen xs + 1
        + settings.level3.probeBudget + 2)) t'.2.2.1) _ fun ω ω' h hω => ?_
      have e : Random.indecisiveLStarLevel3 settings.level3
            (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω) xs
          = Random.indecisiveLStarLevel3 settings.level3
            (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω') xs :=
        level3_result_congr O settings.level3 t'.2.2.1 _ _ xs h
      change resCode (Random.indecisiveLStarLevel3 settings.level3
        (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω) xs) = h' at hω
      show resCode (Random.indecisiveLStarLevel3 settings.level3
        (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω') xs) = h'
      rw [← e]
      exact hω)
    W
    (fun y x ω ω' hy hs h => by
      obtain ⟨t', h', rfl, -, -, hsmall⟩ := hy
      obtain ⟨t'', ht4, he⟩ := Option.map_eq_some_iff.1 hs
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, he⟩ := he
      obtain ⟨c, B, F, K₀⟩ := t''
      have h4 := stage4_stop settings O P.D Dsf N P.length_le hN r hr c B F K₀ x ω ω' ω₀ ht4
        (fun w hw => h w (Finset.mem_union_left _ hw))
      have hres : stageRes settings O P.D (c, B, F, K₀) ω (x rK).2
          = stageRes settings O P.D (c, B, F, K₀) ω' (x rK).2 := by
        refine level3_congr_zoneZ O settings.level3 F B.lo (B.hi + 1) hka _ (fun w hw =>
          h w (Finset.mem_union_right _ hw)) ?_
        have := congrArg (fun c : ResCode α => c.1) he
        simp only [resCode] at this
        show (stageRes settings O P.D (c, B, F, K₀) ω (x rK).2).hypothesis.tree.leaves.length ≤ _
        rw [this]
        exact hsmall
      show (stage4 settings O P.D Dsf N r hr ω' x).map _ = _
      rw [h4, Option.map_some, ← hres, he])
    (SelOk settings P.D Dsf Ls r hr)
    (measurableSet_selOk settings P.D Dsf Ls r hr)
    (SelBad O L n nL t)
    (fun y => by
      rcases y with _ | ⟨t', h'⟩
      · exact MeasurableSet.empty
      · show MeasurableSet (if _ then _ else _)
        split_ifs
        · exact measurableSet_scoreBad O _ _ _ _ L n t
        · exact MeasurableSet.empty)
    Z
    (fun y b ω ω' h hb => by
      rcases y with _ | ⟨t', h'⟩
      · exact hb
      · change (ω, b) ∈ (if SuffixFree t'.2.2.1
            ∧ h'.hyp.tree.leaves.length ≤ nL
          then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅) at hb
        show (ω', b) ∈ (if SuffixFree t'.2.2.1
            ∧ h'.hyp.tree.leaves.length ≤ nL
          then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅)
        split_ifs at hb ⊢
        · exact scoreBad_congr O _ _ _ _ L n t b h hb
        · exact hb)
    (SelDkZ settings P.D L n r hr)
    (fun y => by
      rcases y with _ | ⟨⟨c, B, F, K₀⟩, h'⟩
      · exact MeasurableSet.univ
      · refine measurableSet_prod_countable_right _ fun b => ?_
        have hset : {x : Fin K → RoundDraws α K T |
            (x, b) ∈ SelDkZ settings P.D L n r hr (some ((c, B, F, K₀), h'))}
            = (fun x : Fin K → RoundDraws α K T => probesOf settings P.D K₀ (x rK).2) ⁻¹'
              {xs | Function.Injective b ∧ ∀ i, (b i).toList.length = L
                ∧ prefixOf (b i) k ∉ K₀ ∪ probeKeys k xs} := rfl
        rw [hset]
        exact ((measurable_probesOf settings P.D K₀).comp
          (measurable_snd.comp (measurable_pi_apply rK))) (Set.to_countable _).measurableSet)
    (fun y x b hy hx hb => by
      obtain ⟨t', h', rfl, hlen, -, -⟩ := hy
      obtain ⟨c, B, F, K₀⟩ := t'
      obtain ⟨hnc, hF, hK⟩ := hx
      obtain ⟨hinj, hbi⟩ := hb
      change c.length = r at hlen
      have hpop : histAt settings P.D c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hlen)]
        rfl
      have hlenF : ∀ v ∈ F, v.toList.length = Ls := fun v hv => hnc.1 rK le_rfl v (hF hv)
      have hd1 := disjoint_hist_family settings P.D Dsf N Ls L c x r hr ω₀ hnc F hF
      rw [hpop] at hd1 hK
      show Disjoint ((histReads settings P.D Dsf N c x r ω₀
          ∪ schedReads (settings.schedule (loopPopulations K (dec c).pops.length))
            (loopPopulations K (dec c).pops.length)
            ((ω₀, runOf P.D Dsf (dec c) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K))))
          ∪ level3ZoneZ k nQ (probesOf settings P.D K₀ (x rK).2) F)
        (readStrings (Finset.univ.image b) F)
      refine Finset.disjoint_union_left.2 ⟨hd1.mono_right ?_, ?_⟩
      · intro w hw
        obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hz, hv⟩ := Finset.mem_product.1 hzv
        obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hz
        refine Finset.mem_sdiff.2 ⟨Finset.mem_image.2 ⟨(b i, v), Finset.mem_product.2
          ⟨mem_shortStrings.2 (hbi i).1.le, hv⟩, rfl⟩, fun hw' => ?_⟩
        obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
        obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
        dsimp only at he
        obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he ((hlenF v' hv').trans (hlenF v hv).symm)
        apply (hbi i).2
        refine Finset.mem_union_left _ ?_
        rw [hK]
        exact Finset.mem_image.2 ⟨p, hp, by rw [hzp]⟩
      · refine Finset.disjoint_left.2 fun w hw hw' => ?_
        obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hz, hv⟩ := Finset.mem_product.1 hzv
        obtain ⟨⟨z', v'⟩, hzv', he⟩ := Finset.mem_image.1 hw'
        obtain ⟨hz', hv'⟩ := Finset.mem_product.1 hzv'
        obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hz'
        dsimp only at he
        obtain ⟨-, hzz⟩ := suffix_eq_of_mul_eq he ((hlenF v' hv').trans (hlenF v hv).symm)
        rcases (Finset.mem_filter.1 hz).2 with hzk | hzs
        · apply (hbi i).2
          refine Finset.mem_union_right _ ?_
          dsimp only
          rw [hzz]
          exact hzk.2
        · have hl := (hbi i).1
          have hzz' : z = b i := hzz.symm
          rw [hzz', hl] at hzs
          omega)
    (ENNReal.ofReal δ₁) (ENNReal.ofReal δ₂)
    (fun y hy => by
      obtain ⟨t', h', rfl, -, -, -⟩ := hy
      show (μ.prod _) (if SuffixFree t'.2.2.1
          ∧ h'.hyp.tree.leaves.length ≤ nL
        then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅) ≤ _
      split_ifs with hc
      · refine (scoreBad_le O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) hc.1 L n t ht).trans
          (ENNReal.ofReal_le_ofReal ?_)
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast hc.2) (by positivity)
      · simp)
    (fun y hy x hx => by
      obtain ⟨t', h', rfl, -, -, -⟩ := hy
      obtain ⟨c, B, F, K₀⟩ := t'
      obtain ⟨-, -, hK⟩ := hx
      set PK := probeKeys k (probesOf settings P.D K₀ (x rK).2)
      refine (measure_mono ?_).trans ((batch_collide_le L k n (by omega) (K₀ ∪ PK)).trans
        (ENNReal.ofReal_le_ofReal ?_))
      · intro b hb
        simp only [Set.mem_ofPred_eq] at hb ⊢
        by_contra hno
        push Not at hno
        exact hb ⟨hno.1, fun i => ⟨(hno.2 i).1, (hno.2 i).2⟩⟩
      · have hPK : PK.card ≤ T := by
          refine Finset.card_image_le.trans (le_of_eq ?_)
          simp [T]
        change K₀ = _ at hK
        have hK₀ : K₀.card ≤ 6 * (K + 1) * NP := by
          rw [hK]
          refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
          have hpops : (loopPopulations K (histAt settings P.D c r).pops.length).card ≤ K + 1 := by
            have := Finset.card_le_univ (loopPopulations K (histAt settings P.D c r).pops.length)
            simpa using this
          have := hNP (loopPopulations K (histAt settings P.D c r).pops.length)
          calc 3 * (loopPopulations K (histAt settings P.D c r).pops.length).card
                * ∑ B ∈ settings.schedule (loopPopulations K (histAt settings P.D c r).pops.length),
                  B.npref
              ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
            _ = 6 * (K + 1) * NP := by ring
        have hcard : ((K₀ ∪ PK).card : ℝ) ≤ 6 * (K + 1) * NP + T := by
          have h1 := Finset.card_union_le K₀ PK
          have h2 : ((K₀ ∪ PK).card : ℕ) ≤ 6 * (K + 1) * NP + T := by omega
          exact_mod_cast h2
        have hAk : (0 : ℝ) < (Fintype.card α : ℝ) ^ k := by positivity
        have h3 : (n : ℝ) * ((K₀ ∪ PK).card : ℝ) ≤ n * (6 * (K + 1) * NP + T) :=
          mul_le_mul_of_nonneg_left hcard (Nat.cast_nonneg _)
        exact add_le_add (div_le_div_of_nonneg_right h3 hAk.le) le_rfl)
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) (hmain.trans (le_of_eq ?_))
  rw [ENNReal.ofReal_add hδ₁ hδ₂]

end Bound

end OrthoDFA

end
