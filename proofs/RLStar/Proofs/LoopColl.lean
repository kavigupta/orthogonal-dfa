import RLStar.Proofs.LoopE2Round
import RLStar.Proofs.RawWindow

/-!
# Rounds read new strings

Each round draws its prefixes and suffixes off every one the earlier rounds read, by uniforms, and
a uniform below `1` draws a string of positive mass off them (`drawFrom_offLaw`). So where every
uniform is below `1`, which has chance one, no round reads a prefix or a suffix an earlier round
read, and every string read has a length its distribution gives mass (`loop_coll_null`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Sets

lemma mem_foldr_union {β γ : Type*} [DecidableEq γ] (f : β → Finset γ) (l : List β) (x : γ) :
    x ∈ l.foldr (fun c T => f c ∪ T) ∅ ↔ ∃ c ∈ l, x ∈ f c := by
  induction l with
  | nil => simp
  | cons a l ih => simp [ih]

variable (settings : LoopSettings K) (D : Measure (FreeMonoid α))

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma histAt_pops (c : List (HarvestCode α)) (i : ℕ) :
    (histAt c i).pops = (decodeHist c).pops.take i := by
  simp [histAt, decodeHist, List.map_take]

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma histAt_prefs_sub (c : List (HarvestCode α)) (i : ℕ) :
    (histAt c i).prefs ⊆ (decodeHist c).prefs := by
  intro x hx
  simp only [histAt, decodeHist] at hx ⊢
  obtain ⟨a, ha, hx⟩ := (mem_foldr_union _ _ x).1 hx
  exact (mem_foldr_union _ _ x).2 ⟨a, List.mem_of_mem_take ha, hx⟩

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma histAt_suffs_sub (c : List (HarvestCode α)) (i : ℕ) :
    (histAt c i).suffs ⊆ (decodeHist c).suffs := by
  intro x hx
  simp only [histAt, decodeHist] at hx ⊢
  obtain ⟨a, ha, hx⟩ := (mem_foldr_union _ _ x).1 hx
  exact (mem_foldr_union _ _ x).2 ⟨a, List.mem_of_mem_take ha, hx⟩

lemma loopDistributions_take (k : ℕ) (pops : List (SpecSampler α)) {i : ℕ}
    {j : Option (Fin K)} (hj : j ∈ loopPopulations K i) :
    loopDistributions D k (pops.take i) j = loopDistributions D k pops j := by
  cases j with
  | none => rfl
  | some i' =>
    have hi' := lt_of_some_mem_loopPops hj
    simp only [loopDistributions, List.getElem?_take_of_lt hi']

lemma loopPopulations_mono {i r : ℕ} (h : i ≤ r) : loopPopulations K i ⊆ loopPopulations K r := by
  intro j hj
  cases j with
  | none => exact Finset.mem_insert_self _ _
  | some i' =>
    have := lt_of_some_mem_loopPops hj
    simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or, Finset.mem_image,
      Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq]
    exact ⟨i', by omega, rfl⟩

end Sets

section Streams

variable {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

lemma nthDrawIn_draw (A : Set S) (d : S) (w : ℕ → S) (k : ℕ) :
    nthDrawIn A d w k = d ∨ ∃ i, nthDrawIn A d w k = w i := by
  rcases nthDrawIn_spec A d w k with h | ⟨i, -, h⟩
  · exact .inl h
  · exact .inr ⟨i, h.symm⟩

lemma nthDrawIn_nthDrawIn_draw (A : Set S) (w z : ℕ → S) (a k : ℕ) :
    ∃ i, nthDrawIn A (nthDrawIn A (z a) w 0) w k = w i ∨ nthDrawIn A (nthDrawIn A (z a) w 0) w k
      = z i := by
  rcases nthDrawIn_draw A (nthDrawIn A (z a) w 0) w k with h | ⟨i, h⟩
  · rw [h]
    rcases nthDrawIn_draw A (z a) w 0 with h' | ⟨i, h'⟩
    · exact ⟨a, .inr h'⟩
    · exact ⟨i, .inl h'⟩
  · exact ⟨i, .inl h⟩

open scoped Classical in
/-- With nothing harvested, every entry of a population's two sequences is a draw of one of
its two streams. -/
lemma streams_empty_draw (uni : J) (x : Draws Ω S J) (j : J) (n : ℕ) :
    (∃ i, prefixDraw j n (streams (fun _ => ∅) uni x) = x.1.2.1.2 j i
      ∨ prefixDraw j n (streams (fun _ => ∅) uni x) = x.1.2.2 j i)
    ∧ ∃ i, certPrefix j n (streams (fun _ => ∅) uni x) = x.1.2.1.2 j i
      ∨ certPrefix j n (streams (fun _ => ∅) uni x) = x.1.2.2 j i := by
  by_cases hj : j = uni
  · subst hj
    simp only [prefixDraw, certPrefix, streams, if_true]
    exact ⟨⟨n, .inl rfl⟩, ⟨n, .inr rfl⟩⟩
  · simp only [prefixDraw, certPrefix, streams, if_neg hj, tableSeqOf, reserveSeqOf,
      prependStream, Finset.toList_empty, List.headD_nil, List.length_nil, Nat.not_lt_zero,
      dite_false, Nat.sub_zero]
    exact ⟨nthDrawIn_nthDrawIn_draw _ _ _ 0 n, nthDrawIn_nthDrawIn_draw _ _ _ 1 n⟩

end Streams

section Draws

variable (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
/-- A prefix a round's clustering reads is a draw of one of its populations. -/
lemma mem_prefsOf_draw {h : SpecHist α} {c : ClusterRest α K settings.level3.harvestReadCap} {p : FreeMonoid α}
    (hp : p ∈ prefsOf settings D Dsf h c) :
    ∃ j ∈ loopPopulations K h.pops.length, ∃ n,
      p = fillDraw D settings.level3.walkStart settings.level3.harvestReadCap h.pops h.prefs j (c.2.1 j n)
        ∨ p = drawFrom (offLaw (loopDistributions D settings.level3.walkStart h.pops j) h.prefs) (c.2.2.1 j n) := by
  obtain ⟨B, -, hp⟩ := Finset.mem_biUnion.1 hp
  rcases Finset.mem_union.1 hp with hp | hp
  · obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨n, hn⟩ := mem_freshDraws hp
    obtain ⟨i, hi⟩ := (streams_empty_draw none (idealDraws (Ω := Unit) D Dsf settings.level3.walkStart settings.level3.harvestReadCap h ((), c)) j n).1
    exact ⟨j, hj, i, hn ▸ hi⟩
  · obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨e, -, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨n, hn⟩ := mem_freshDraws hp
    obtain ⟨i, hi⟩ := (streams_empty_draw none (idealDraws (Ω := Unit) D Dsf settings.level3.walkStart settings.level3.harvestReadCap h ((), c)) j n).2
    exact ⟨j, hj, i, hn ▸ hi⟩

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
/-- A suffix a round's clustering reads, but `ε`, is a draw of the suffix distribution. -/
lemma mem_suffsOf_draw {h : SpecHist α} {c : ClusterRest α K settings.level3.harvestReadCap} {v : FreeMonoid α}
    (hv : v ∈ suffsOf settings D Dsf h c) : ∃ n, v = drawFrom (offLaw Dsf h.suffs) (c.1 n) := by
  obtain ⟨hv1, hv⟩ := Finset.mem_erase.1 hv
  obtain ⟨B, -, hv⟩ := Finset.mem_biUnion.1 hv
  rcases Finset.mem_insert.1 hv with h1 | hv
  · exact absurd h1 hv1
  · obtain ⟨n, hn⟩ := mem_freshDraws hv
    exact ⟨n, hn.symm⟩

/-- Through round `r`, every uniform the clusterings draw by is below `1`. -/
def AllLt {T : ℕ} (x : Fin K → RoundDraws α K settings.level3.harvestReadCap T) (r : ℕ) : Prop :=
  ∀ i : Fin K, i.1 ≤ r → (∀ n, ((x i).1.1 n : ℝ) < 1)
    ∧ (∀ j n, (((x i).1.2.1 j n).1 : ℝ) < 1 ∧ ∀ t, (((x i).1.2.1 j n).2 t : ℝ) < 1)
    ∧ ∀ j n, ((x i).1.2.2.1 j n : ℝ) < 1

lemma volume_eq_one_unit : (volume : Measure unitInterval) {u | ¬ (u : ℝ) < 1} = 0 := by
  have h := volume_Ico_unit (a := 0) (b := 1) le_rfl zero_le_one le_rfl
  have hs : {u : unitInterval | ¬ (u : ℝ) < 1} = {u : unitInterval | 0 ≤ (u : ℝ) ∧ (u : ℝ) < 1}ᶜ := by
    ext u
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_and, not_lt]
    exact ⟨fun h _ => h, fun h => h u.2.1⟩
  have hm : MeasurableSet {u : unitInterval | 0 ≤ (u : ℝ) ∧ (u : ℝ) < 1} :=
    (measurableSet_le measurable_const continuous_subtype_val.measurable).inter
      (measurableSet_lt continuous_subtype_val.measurable measurable_const)
  rw [hs, measure_compl hm (measure_ne_top _ _), h, measure_univ]
  simp

omit [DecidableEq α] in
/-- Every uniform of a probe's draw is below `1` almost surely. -/
lemma drawLaw_lt_ae (N : ℕ) :
    drawLaw N {u | ¬ ((u.1 : ℝ) < 1 ∧ ∀ t, (u.2 t : ℝ) < 1)} = 0 := by
  have h1 : MeasurableSet {u : unitInterval | ¬ (u : ℝ) < 1} :=
    (measurableSet_lt continuous_subtype_val.measurable measurable_const).compl
  have hsub : {u : unitInterval × (Fin N → unitInterval) | ¬ ((u.1 : ℝ) < 1 ∧ ∀ t, (u.2 t : ℝ) < 1)}
      ⊆ Prod.fst ⁻¹' {u | ¬ (u : ℝ) < 1}
        ∪ ⋃ t, (fun u : unitInterval × (Fin N → unitInterval) => u.2 t) ⁻¹' {u | ¬ (u : ℝ) < 1} := by
    intro u hu
    simp only [not_and_or, not_forall] at hu
    simp only [Set.mem_union, Set.mem_preimage, Set.mem_setOf_eq, Set.mem_iUnion]
    rcases hu with hu | ⟨t, hu⟩
    · exact Or.inl hu
    · exact Or.inr ⟨t, hu⟩
  have hfst : MeasurePreserving (Prod.fst : unitInterval × (Fin N → unitInterval) → unitInterval)
      (drawLaw N) volume := measurePreserving_fst
  have hco : ∀ t : Fin N, MeasurePreserving
      (fun u : unitInterval × (Fin N → unitInterval) => u.2 t) (drawLaw N) volume := fun t =>
    (measurePreserving_eval (fun _ : Fin N => (volume : Measure unitInterval)) t).comp
      measurePreserving_snd
  refine measure_mono_null hsub (measure_union_null ?_ (measure_iUnion_null fun t => ?_))
  · rw [hfst.measure_preimage h1.nullMeasurableSet, volume_eq_one_unit]
  · rw [(hco t).measure_preimage h1.nullMeasurableSet, volume_eq_one_unit]

omit [DecidableEq α] in
lemma allLt_ae {T : ℕ} (r : ℕ) :
    (loopMeasure μ α K settings.level3.harvestReadCap T) {p | ¬ AllLt settings p.2 r} = 0 := by
  have hrest : ∀ i : Fin K, MeasurePreserving
      (fun p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap T) => (p.2 i).1)
      (loopMeasure μ α K settings.level3.harvestReadCap T)
      ((Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
        ((Measure.pi fun _ : Option (Fin K) =>
            Measure.infinitePi fun _ : ℕ => drawLaw settings.level3.harvestReadCap).prod
          ((Measure.pi fun _ : Option (Fin K) =>
            Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
            (Measure.infinitePi fun _ : FreeMonoid α => fairCoin)))) := fun i =>
    measurePreserving_fst.comp ((measurePreserving_eval _ i).comp measurePreserving_snd)
  have hev : ∀ n, MeasurePreserving (fun f : ℕ → unitInterval => f n)
      (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun n =>
    ⟨measurable_pi_apply n, Measure.infinitePi_map_eval _ n⟩
  have hev2 : ∀ j n, MeasurePreserving (fun f : Option (Fin K) → ℕ → unitInterval => f j n)
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun j n =>
    (hev n).comp (measurePreserving_eval _ j)
  have hevD : ∀ j n, MeasurePreserving
      (fun f : Option (Fin K) → ℕ → unitInterval × (Fin settings.level3.harvestReadCap → unitInterval)
        => f j n)
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => drawLaw settings.level3.harvestReadCap)
      (drawLaw settings.level3.harvestReadCap) := fun j n =>
    (⟨measurable_pi_apply n, Measure.infinitePi_map_eval _ n⟩ : MeasurePreserving
      (fun f : ℕ → unitInterval × (Fin settings.level3.harvestReadCap → unitInterval) => f n)
      (Measure.infinitePi fun _ : ℕ => drawLaw settings.level3.harvestReadCap)
      (drawLaw settings.level3.harvestReadCap)).comp (measurePreserving_eval _ j)
  have h1 : MeasurableSet {u : unitInterval | ¬ (u : ℝ) < 1} :=
    (measurableSet_lt continuous_subtype_val.measurable measurable_const).compl
  have hD1 : MeasurableSet {u : unitInterval × (Fin settings.level3.harvestReadCap → unitInterval) |
      ¬ ((u.1 : ℝ) < 1 ∧ ∀ t, (u.2 t : ℝ) < 1)} := by
    have e : {u : unitInterval × (Fin settings.level3.harvestReadCap → unitInterval) |
        ¬ ((u.1 : ℝ) < 1 ∧ ∀ t, (u.2 t : ℝ) < 1)}
        = ({u | (u.1 : ℝ) < 1} ∩ ⋂ t, {u | (u.2 t : ℝ) < 1})ᶜ := by
      ext u; simp [Set.mem_iInter]
    rw [e]
    exact ((measurableSet_lt (measurable_subtype_coe.comp measurable_fst)
      measurable_const).inter (MeasurableSet.iInter fun t => measurableSet_lt
        (measurable_subtype_coe.comp ((measurable_pi_apply t).comp measurable_snd))
        measurable_const)).compl
  have hnull : ∀ {f : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap T) → unitInterval},
      MeasurePreserving f (loopMeasure μ α K settings.level3.harvestReadCap T) volume →
        (loopMeasure μ α K settings.level3.harvestReadCap T) {x | ¬ (f x : ℝ) < 1} = 0 :=
    fun hf => by
    rw [show {x | ¬ (_ : ℝ) < 1} = _ ⁻¹' {u : unitInterval | ¬ (u : ℝ) < 1} from rfl,
      hf.measure_preimage h1.nullMeasurableSet, volume_eq_one_unit]
  have hset : {p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap T) | ¬ AllLt settings p.2 r}
      ⊆ ⋃ i : Fin K, ⋃ n : ℕ, ({p | ¬ (((p.2 i).1.1 n : unitInterval) : ℝ) < 1}
        ∪ ⋃ j : Option (Fin K), ({p | ¬ ((((p.2 i).1.2.1 j n).1 : ℝ) < 1
            ∧ ∀ t, (((p.2 i).1.2.1 j n).2 t : ℝ) < 1)}
          ∪ {p | ¬ (((p.2 i).1.2.2.1 j n : unitInterval) : ℝ) < 1})) := by
    intro p hp
    simp only [AllLt, not_forall, not_and_or] at hp
    obtain ⟨i, -, hp⟩ := hp
    simp only [Set.mem_iUnion, Set.mem_union, Set.mem_setOf_eq]
    rcases hp with ⟨n, hn⟩ | ⟨j, n, hn⟩ | ⟨j, n, hn⟩
    · exact ⟨i, n, Or.inl hn⟩
    · exact ⟨i, n, Or.inr ⟨j, Or.inl (by rw [not_and_or, not_forall]; exact hn)⟩⟩
    · exact ⟨i, n, Or.inr ⟨j, Or.inr hn⟩⟩
  refine measure_mono_null hset (measure_iUnion_null fun i => measure_iUnion_null fun n =>
    measure_union_null ?_ (measure_iUnion_null fun j => measure_union_null ?_ ?_))
  · exact hnull ((hev n).comp (measurePreserving_fst.comp (hrest i)))
  · have hm := (hevD j n).comp (measurePreserving_fst.comp (measurePreserving_snd.comp
      (hrest i)))
    have := hm.measure_preimage hD1.nullMeasurableSet
    rw [drawLaw_lt_ae] at this
    exact this
  · exact hnull ((hev2 j n).comp (measurePreserving_fst.comp (measurePreserving_snd.comp
      (measurePreserving_snd.comp (hrest i)))))

end Draws

section Codes

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))

omit [IsProbabilityMeasure μ] in
/-- Each earlier round's data hold the prefixes and suffixes its clustering read. -/
lemma histCodes_getElem [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    ∀ r i, i < (histCodes settings O D Dsf ω g r).length → ∃ hiK : i < K, ∃ code,
      (histCodes settings O D Dsf ω g r)[i]? = some code
      ∧ code.2.2.2.1 = prefsOf settings D Dsf
            (histAt (histCodes settings O D Dsf ω g r) i)
          (g ⟨i, hiK⟩).1
      ∧ code.2.2.2.2 = roundSuffs (settings.schedule (loopPopulations K
            (histAt (histCodes settings O D Dsf ω g r) i).pops.length))
          (((), runOf settings D Dsf (histAt (histCodes settings O D Dsf ω g r) i)
            (g ⟨i, hiK⟩).1) : Run Unit (FreeMonoid α) (Option (Fin K)))
  | 0, i, hi => by simp [histCodes, idealRounds, codesOf] at hi
  | r + 1, i, hi => by
    by_cases hrK : r < K
    · have hy := codesOf_idealRounds_succ settings O D Dsf (ae_lenBound hlen) (N := codeLen settings D) le_rfl
        (fun i => (ω, g i)) r hrK
      set c := histCodes settings O D Dsf ω g r with hcd
      have hy' : histCodes settings O D Dsf ω g (r + 1) = if c.length = r then
          c ++ (popCode (codeLen settings D)
            (idealRound settings O D Dsf (decodeHist c)
              (ω, g ⟨r, hrK⟩))).toList else c := hy
      have ih := histCodes_getElem hlen ω g r
      by_cases hcl : c.length = r
      · rw [if_pos hcl] at hy'
        rw [hy'] at hi ⊢
        by_cases hic : i < c.length
        · obtain ⟨hiK, code, hget, h1, h2⟩ := ih i hic
          have ht : ∀ l : List (HarvestCode α), histAt (c ++ l) i
              = histAt c i := fun l => by
            simp only [histAt, List.take_append_of_le_length hic.le]
          refine ⟨hiK, code, ?_, ?_, ?_⟩
          · rw [List.getElem?_append_left hic]; exact hget
          · rw [ht]; exact h1
          · rw [ht]; exact h2
        · revert hi
          rcases ho : popCode (codeLen settings D) (idealRound settings O D Dsf
              (decodeHist c) (ω, g ⟨r, hrK⟩)) with _ | code
          · intro hi
            simp only [Option.toList_none, List.append_nil] at hi
            exact absurd hi hic
          · intro hi
            simp only [Option.toList_some, List.length_append, List.length_cons,
              List.length_nil] at hi
            have hir : i = r := by omega
            subst hir
            have ht : histAt (c ++ [code]) i
                = decodeHist c := by
              simp only [histAt, List.take_append_of_le_length (le_of_eq hcl.symm),
                List.take_of_length_le (le_of_eq hcl)]
            simp only [Option.toList_some, ht]
            unfold popCode at ho
            obtain ⟨hv, -, hcode⟩ := Option.map_eq_some_iff.1 ho
            refine ⟨hrK, code, ?_, ?_, ?_⟩
            · rw [List.getElem?_append_right (le_of_eq hcl)]
              simp [hcl]
            · rw [← hcode]; rfl
            · rw [← hcode]; rfl
      · rw [if_neg hcl] at hy'
        rw [hy'] at hi ⊢
        exact ih i hi
    · have hst : histCodes settings O D Dsf ω g (r + 1) = histCodes settings O D Dsf ω g r := by
        simp only [histCodes]
        rw [idealRounds_succ, dif_neg (fun h' => hrK h'.1)]
      rw [hst] at hi ⊢
      exact histCodes_getElem hlen ω g r i hi

end Codes

section Coll

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

omit [DecidableEq α] in
lemma mem_of_pos {ν : Measure (FreeMonoid α)} {A : Set (FreeMonoid α)} (hA : ν Aᶜ = 0)
    {t : FreeMonoid α} (ht : 0 < ν.real {t}) : t ∈ A := by
  by_contra h
  have : ν {t} = 0 := measure_mono_null (fun z hz => by
    rw [Set.mem_singleton_iff.1 hz]; exact h) hA
  rw [measureReal_def, this, ENNReal.toReal_zero] at ht
  exact lt_irrefl _ ht

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] [DecidableEq α] in
/-- Where its uniforms are below `1`, a population's filled draw is off `T`, at a string the
population gives mass. -/
lemma fillDraw_pos (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    {u : unitInterval × (Fin N → unitInterval)} (h0 : loopDistributions D k pops j (↑T)ᶜ ≠ 0)
    (hu : (u.1 : ℝ) < 1) (hu2 : ∀ t, (u.2 t : ℝ) < 1) :
    fillDraw D k N pops T j u ∉ T
      ∧ 0 < (loopDistributions D k pops j).real {fillDraw D k N pops T j u} := by
  classical
  rcases j with _ | i
  · exact drawFrom_offLaw h0 hu
  rcases hs : pops[i.val]? with _ | s
  · have hl : loopDistributions D k pops (some i) = Measure.dirac 1 := by
      simp only [loopDistributions, hs]
    have hf : fillDraw D k N pops T (some i) u = 1 := by simp only [fillDraw, hs]
    rw [hf, hl]
    rw [hl] at h0
    refine ⟨fun h1 => h0 ?_, ?_⟩
    · simp [h1]
    · rw [measureReal_def, Measure.dirac_apply_of_mem (Set.mem_singleton 1)]
      exact zero_lt_one
  have hl : loopDistributions D k pops (some i)
      = popLaw (Random.harvestLaw s.read D k s.hyp s.site) := by
    simp only [loopDistributions, hs]
  rw [hl] at h0 ⊢
  simp only [fillDraw, hs]
  rcases hh : Random.harvestFirst s.read k s.hyp s.site ↑T
      (List.ofFn fun t => drawFrom D (u.2 t)) with _ | z
  · exact drawFrom_offLaw h0 hu
  simp only [Option.getD_some]
  unfold Random.harvestFirst at hh
  obtain ⟨x, hx, hz⟩ := List.exists_of_findSome?_eq_some hh
  obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
  rw [Option.filter_eq_some_iff] at hz
  have hzT : z ∉ T := by simpa using hz.2
  have hzmem := List.mem_of_mem_head? hz.1
  have hx0 : 0 < D {drawFrom D (u.2 t)} := (ENNReal.toReal_pos_iff.1 (drawFrom_pos D (hu2 t))).1
  have hpos : 0 < Random.harvestLaw s.read D k s.hyp s.site {z} := by
    rw [Random.harvestLaw_apply]
    have hf1 : (1 : ENNReal) ≤ (((Random.siteHarvest s.read k s.hyp s.site
        (drawFrom D (u.2 t))).filter (· ∈ ({z} : Set (FreeMonoid α)))).length : ENNReal) := by
      exact_mod_cast List.length_pos_of_mem (List.mem_filter.2 ⟨hzmem, by simp⟩)
    calc (0 : ENNReal) < 1 * D {drawFrom D (u.2 t)} := by simpa using hx0
      _ ≤ _ := mul_le_mul_left hf1 _
      _ = ∫⁻ y in {drawFrom D (u.2 t)}, (((Random.siteHarvest s.read k s.hyp s.site y).filter
            (· ∈ ({z} : Set (FreeMonoid α)))).length : ENNReal) ∂D := by rw [lintegral_singleton]
      _ ≤ _ := setLIntegral_le_lintegral _ _
  have h0' : Random.harvestLaw s.read D k s.hyp s.site Set.univ ≠ 0 :=
    (lt_of_lt_of_le hpos (measure_mono (Set.subset_univ _))).ne'
  have htop := Random.harvestLaw_univ_ne_top s.read D k s.hyp s.site
  refine ⟨hzT, ?_⟩
  rw [popLaw_real h0' htop]
  exact div_pos (ENNReal.toReal_pos hpos.ne' (measure_ne_top_of_subset (Set.subset_univ _) htop))
    (ENNReal.toReal_pos h0' htop)

omit [IsProbabilityMeasure μ] in
/-- Where every uniform is below `1`, no round through `r` reads a string an earlier round read,
and every string read has a length its distribution gives mass. -/
theorem noColl_of_allLt (Ls : ℕ)
    (hLs : P.L < Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) (r : ℕ) (hr : r < K)
    (hlen : (histCodes settings O P.D Dsf ω g r).length = r)
    (hsupp : ∀ ν ∈ samplerLaws P.D settings.level3.walkStart (decodeHist
      (histCodes settings O P.D Dsf ω g r)).pops, ν {q | Ls ≤ q.toList.length} = 0)
    (hpm : ∀ j ∈ loopPopulations K r, (loopDistributions P.D settings.level3.walkStart (decodeHist (histCodes settings O P.D Dsf ω g r)).pops j).real
        ↑(decodeHist (histCodes settings O P.D Dsf ω g r)).prefs
        ≤ 1 / 2)
    (hsm : Dsf.real ↑(decodeHist
      (histCodes settings O P.D Dsf ω g r)).suffs ≤ 1 / 2)
    (hlt : AllLt settings g r) :
    NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf ω g r) g r hr := by
  classical
  set c := histCodes settings O P.D Dsf ω g r
  set H := decodeHist c
  have hpl : H.pops.length = r := by simp [H, decodeHist, hlen]
  have hHr : histAt c r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen), H]
  have hpsH : ∀ i, (histAt c i).pops.length ≤ r := fun i => by
    rw [histAt_pops, List.length_take, hpl]; exact min_le_right _ _
  -- each population a round through `r` reads is one of round `r`'s, and puts mass on short strings
  have hdist : ∀ i, ∀ j ∈ loopPopulations K (histAt c i).pops.length,
      loopDistributions P.D settings.level3.walkStart (histAt c i).pops j = loopDistributions P.D settings.level3.walkStart H.pops j
        ∧ j ∈ loopPopulations K r := by
    intro i j hj
    have hj' : j ∈ loopPopulations K i := by
      refine loopPopulations_mono ?_ hj
      rw [histAt_pops]; exact List.length_take_le _ _
    rw [histAt_pops]
    exact ⟨loopDistributions_take P.D _ H.pops hj', loopPopulations_mono (hpsH i) hj⟩
  have hshort : ∀ j ∈ loopPopulations K r, loopDistributions P.D settings.level3.walkStart H.pops j
      {q | Ls ≤ q.toList.length} = 0 := by
    intro j hj
    cases j with
    | none =>
      refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
      simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
      omega
    | some i' =>
      have hi' : i'.val < H.pops.length := by rw [hpl]; exact lt_of_some_mem_loopPops hj
      rw [loopDistributions_eq_popDists]
      simp only [popDists]
      rw [List.getD_eq_getElem (hn := by rw [samplerLaws_length]; exact hi')]
      exact hsupp _ (List.getElem_mem _)
  -- the draws of round `i ≤ r`
  have hpref : ∀ i : Fin K, i.1 ≤ r → ∀ p ∈ prefsOf settings P.D Dsf (histAt c i)
      (g i).1, p ∉ (histAt c i).prefs ∧ p.toList.length < Ls := by
    intro i hir p hp
    obtain ⟨j, hj, n, hpn⟩ := mem_prefsOf_draw settings P.D Dsf hp
    obtain ⟨hd, hjr⟩ := hdist i j hj
    have hm : (loopDistributions P.D settings.level3.walkStart H.pops j).real ↑(histAt c i).prefs ≤ 1 / 2 :=
      (measureReal_mono (Finset.coe_subset.2 (histAt_prefs_sub c i))).trans
        (hpm j hjr)
    have h0 : loopDistributions P.D settings.level3.walkStart (histAt c i).pops j (↑(histAt c i).prefs)ᶜ ≠ 0 := by
      rw [hd]; exact compl_ne_zero hm
    obtain ⟨hT, hpos⟩ : p ∉ (histAt c i).prefs
        ∧ 0 < (loopDistributions P.D settings.level3.walkStart (histAt c i).pops j).real {p} := by
      rcases hpn with h | h
      · rw [h]
        exact fillDraw_pos P.D _ _ _ _ j h0 ((hlt i hir).2.1 j n).1 ((hlt i hir).2.1 j n).2
      · rw [h]; exact drawFrom_offLaw h0 ((hlt i hir).2.2 j n)
    rw [hd] at hpos
    refine ⟨hT, ?_⟩
    have := mem_of_pos (A := {q : FreeMonoid α | q.toList.length < Ls})
      (by
        refine measure_mono_null (fun x hx => ?_) (hshort j hjr)
        simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_lt] at hx ⊢
        exact hx) hpos
    exact this
  have hsuf : ∀ i : Fin K, i.1 ≤ r → ∀ v ∈ suffsOf settings P.D Dsf (histAt c i)
      (g i).1, v ∉ (histAt c i).suffs ∧ v.toList.length = Ls := by
    intro i hir v hv
    obtain ⟨n, rfl⟩ := mem_suffsOf_draw settings P.D Dsf hv
    have hm : Dsf.real ↑(histAt c i).suffs ≤ 1 / 2 :=
      (measureReal_mono (Finset.coe_subset.2 (histAt_suffs_sub c i))).trans hsm
    obtain ⟨hT, hpos⟩ := drawFrom_offLaw (compl_ne_zero hm) ((hlt i hir).1 n)
    refine ⟨hT, ?_⟩
    have := mem_of_pos (A := {q : FreeMonoid α | q.toList.length = Ls})
      (by
        refine measure_mono_null (fun x hx => ?_) hDsf
        simpa using hx) hpos
    exact this
  -- an earlier round's reads are in the history round `r` draws off
  have hcons : ∀ i : Fin K, i.1 < r →
      prefsOf settings P.D Dsf (histAt c i) (g i).1 ⊆ H.prefs
      ∧ suffsOf settings P.D Dsf (histAt c i) (g i).1 ⊆ H.suffs := by
    intro i hir
    have hic : i.1 < c.length := by omega
    obtain ⟨hiK, code, hget, h1, h2⟩ :=
      histCodes_getElem settings O P.D Dsf P.length_le ω g r i hic
    have hmem : code ∈ c := List.mem_of_getElem? hget
    refine ⟨fun p hp => ?_, fun v hv => ?_⟩
    · simp only [H, decodeHist]
      refine (mem_foldr_union _ _ p).2 ⟨code, hmem, ?_⟩
      rw [h1]
      exact hp
    · simp only [H, decodeHist]
      refine (mem_foldr_union _ _ v).2 ⟨code, hmem, ?_⟩
      rw [h2]
      exact Finset.mem_of_mem_erase hv
  refine ⟨fun i hir v hv => (hsuf i hir v hv).2, fun i hir p hp => (hpref i hir p hp).2,
    fun i hir => Finset.disjoint_left.2 fun v hv hv' => ?_,
    fun i hir => Finset.disjoint_left.2 fun p hp hp' => ?_⟩
  · have := (hsuf ⟨r, hr⟩ le_rfl v hv').1
    rw [hHr] at this
    exact this ((hcons i hir).2 hv)
  · have := (hpref ⟨r, hr⟩ le_rfl p hp').1
    rw [hHr] at this
    exact this ((hcons i hir).1 hp)

/-- Round `r` reads a prefix or a suffix an earlier round read, or a string of a length its
distribution gives no mass, with chance zero. -/
theorem loop_coll_null (Ls : ℕ)
    (hLs : P.L < Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0) (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.harvestReadCap settings.level3.probeBudget)
      {p | (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ν ∈ samplerLaws P.D settings.level3.walkStart (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops, ν {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D settings.level3.walkStart (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(decodeHist
                (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ 1 / 2)
        ∧ Dsf.real ↑(decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).suffs ≤ 1 / 2
        ∧ ¬ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
      = 0 :=
  measure_mono_null (by
    intro p hp hlt
    obtain ⟨hlen, hsupp, hpm, hsm, hnc⟩ := hp
    exact hnc (noColl_of_allLt settings O P Dsf Ls hLs hDsf p.1 p.2 r hr hlen hsupp hpm
      hsm hlt))
    (allLt_ae settings r)

end Coll

end OrthoDFA
