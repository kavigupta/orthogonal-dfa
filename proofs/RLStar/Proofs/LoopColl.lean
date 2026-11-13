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

lemma loopDistributions_take (pops : List (List (FreeMonoid α))) {i : ℕ}
    {j : Option (Fin K)} (hj : j ∈ loopPopulations K i) :
    loopDistributions D (pops.take i) j = loopDistributions D pops j := by
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
lemma mem_prefsOf_draw {h : SpecHist α} {c : ClusterRest α K} {p : FreeMonoid α}
    (hp : p ∈ prefsOf settings D Dsf h c) :
    ∃ j ∈ loopPopulations K h.pops.length, ∃ n,
      p = drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.1 j n)
        ∨ p = drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.1 j n) := by
  obtain ⟨B, -, hp⟩ := Finset.mem_biUnion.1 hp
  rcases Finset.mem_union.1 hp with hp | hp
  · obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨n, hn⟩ := mem_freshDraws hp
    obtain ⟨i, hi⟩ := (streams_empty_draw none (idealDraws (Ω := Unit) D Dsf h ((), c)) j n).1
    exact ⟨j, hj, i, hn ▸ hi⟩
  · obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨e, -, hp⟩ := Finset.mem_biUnion.1 hp
    obtain ⟨n, hn⟩ := mem_freshDraws hp
    obtain ⟨i, hi⟩ := (streams_empty_draw none (idealDraws (Ω := Unit) D Dsf h ((), c)) j n).2
    exact ⟨j, hj, i, hn ▸ hi⟩

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
/-- A suffix a round's clustering reads, but `ε`, is a draw of the suffix distribution. -/
lemma mem_suffsOf_draw {h : SpecHist α} {c : ClusterRest α K} {v : FreeMonoid α}
    (hv : v ∈ suffsOf settings D Dsf h c) : ∃ n, v = drawFrom (offLaw Dsf h.suffs) (c.1 n) := by
  obtain ⟨hv1, hv⟩ := Finset.mem_erase.1 hv
  obtain ⟨B, -, hv⟩ := Finset.mem_biUnion.1 hv
  rcases Finset.mem_insert.1 hv with h1 | hv
  · exact absurd h1 hv1
  · obtain ⟨n, hn⟩ := mem_freshDraws hv
    exact ⟨n, hn.symm⟩

/-- Through round `r`, every uniform the clusterings draw by is below `1`. -/
def AllLt (x : Fin K → settings.RoundDraws α) (r : ℕ) : Prop :=
  ∀ i : Fin K, i.1 ≤ r → (∀ n, ((x i).1.1 n : ℝ) < 1)
    ∧ (∀ j n, ((x i).1.2.1 j n : ℝ) < 1) ∧ ∀ j n, ((x i).1.2.2.1 j n : ℝ) < 1

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
lemma allLt_ae (r : ℕ) : (settings.measure μ α) {p | ¬ AllLt settings p.2 r} = 0 := by
  have hrest : ∀ i : Fin K, MeasurePreserving (fun p : settings.Draws Ω α => (p.2 i).1)
      (settings.measure μ α) (restLaw α K) := fun i =>
    measurePreserving_fst.comp ((measurePreserving_eval _ i).comp measurePreserving_snd)
  have hev : ∀ n, MeasurePreserving (fun f : ℕ → unitInterval => f n)
      (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun n =>
    ⟨measurable_pi_apply n, Measure.infinitePi_map_eval _ n⟩
  have hev2 : ∀ j n, MeasurePreserving (fun f : Option (Fin K) → ℕ → unitInterval => f j n)
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)) volume := fun j n =>
    (hev n).comp (measurePreserving_eval _ j)
  have h1 : MeasurableSet {u : unitInterval | ¬ (u : ℝ) < 1} :=
    (measurableSet_lt continuous_subtype_val.measurable measurable_const).compl
  have hnull : ∀ {f : settings.Draws Ω α → unitInterval},
      MeasurePreserving f (settings.measure μ α) volume →
        (settings.measure μ α) {x | ¬ (f x : ℝ) < 1} = 0 := fun hf => by
    rw [show {x | ¬ (_ : ℝ) < 1} = _ ⁻¹' {u : unitInterval | ¬ (u : ℝ) < 1} from rfl,
      hf.measure_preimage h1.nullMeasurableSet, volume_eq_one_unit]
  have hset : {p : settings.Draws Ω α | ¬ AllLt settings p.2 r}
      ⊆ ⋃ i : Fin K, ⋃ n : ℕ, ({p | ¬ (((p.2 i).1.1 n : unitInterval) : ℝ) < 1}
        ∪ ⋃ j : Option (Fin K), ({p | ¬ (((p.2 i).1.2.1 j n : unitInterval) : ℝ) < 1}
          ∪ {p | ¬ (((p.2 i).1.2.2.1 j n : unitInterval) : ℝ) < 1})) := by
    intro p hp
    simp only [AllLt, not_forall, not_and_or] at hp
    obtain ⟨i, -, hp⟩ := hp
    simp only [Set.mem_iUnion, Set.mem_union, Set.mem_ofPred_eq]
    rcases hp with ⟨n, hn⟩ | ⟨j, n, hn⟩ | ⟨j, n, hn⟩
    · exact ⟨i, n, Or.inl hn⟩
    · exact ⟨i, n, Or.inr ⟨j, Or.inl hn⟩⟩
    · exact ⟨i, n, Or.inr ⟨j, Or.inr hn⟩⟩
  refine measure_mono_null hset (measure_iUnion_null fun i => measure_iUnion_null fun n =>
    measure_union_null ?_ (measure_iUnion_null fun j => measure_union_null ?_ ?_))
  · exact hnull ((hev n).comp (measurePreserving_fst.comp (hrest i)))
  · exact hnull ((hev2 j n).comp (measurePreserving_fst.comp (measurePreserving_snd.comp
      (hrest i))))
  · exact hnull ((hev2 j n).comp (measurePreserving_fst.comp (measurePreserving_snd.comp
      (measurePreserving_snd.comp (hrest i)))))

end Draws

section Codes

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))

omit [IsProbabilityMeasure μ] in
/-- Each earlier round's data hold the prefixes and suffixes its clustering read. -/
lemma histCodes_getElem (ω : Ω) (g : Fin K → settings.RoundDraws α) :
    ∀ r i, i < (histCodes settings O D Dsf ω g r).length → ∃ hiK : i < K, ∃ code,
      (histCodes settings O D Dsf ω g r)[i]? = some code
      ∧ code.2.1 = prefsOf settings D Dsf
            (histAt (histCodes settings O D Dsf ω g r) i)
          (g ⟨i, hiK⟩).1
      ∧ code.2.2 = roundSuffs (settings.schedule (loopPopulations K
            (histAt (histCodes settings O D Dsf ω g r) i).pops.length))
          (((), runOf settings D Dsf (histAt (histCodes settings O D Dsf ω g r) i)
            (g ⟨i, hiK⟩).1) : Run Unit (FreeMonoid α) (Option (Fin K)))
  | 0, i, hi => by simp [histCodes, idealRounds, codesOf] at hi
  | r + 1, i, hi => by
    by_cases hrK : r < K
    · have hy := codesOf_idealRounds_succ settings O D Dsf (fun i => (ω, g i)) r hrK
      set c := histCodes settings O D Dsf ω g r
      have hy' : histCodes settings O D Dsf ω g (r + 1) = if c.length = r then
          c ++ (popCode (idealRound settings O D Dsf (decodeHist c) (ω, g ⟨r, hrK⟩))).toList
          else c := hy
      have ih := histCodes_getElem ω g r
      by_cases hcl : c.length = r
      · rw [if_pos hcl] at hy'
        rw [hy'] at hi ⊢
        by_cases hic : i < c.length
        · obtain ⟨hiK, code, hget, h1, h2⟩ := ih i hic
          have ht : ∀ l : List (HarvestCode α), histAt (c ++ l) i = histAt c i := fun l => by
            simp only [histAt, List.take_append_of_le_length hic.le]
          refine ⟨hiK, code, ?_, ?_, ?_⟩
          · rw [List.getElem?_append_left hic]; exact hget
          · rw [ht]; exact h1
          · rw [ht]; exact h2
        · revert hi
          rcases ho : popCode (idealRound settings O D Dsf (decodeHist c) (ω, g ⟨r, hrK⟩))
            with _ | code
          · intro hi
            simp only [Option.toList_none, List.append_nil] at hi
            exact absurd hi hic
          · intro hi
            simp only [Option.toList_some, List.length_append, List.length_cons,
              List.length_nil] at hi
            have hir : i = r := by omega
            subst hir
            have ht : histAt (c ++ [code]) i = decodeHist c := by
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
      exact histCodes_getElem ω g r i hi

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

omit [IsProbabilityMeasure μ] in
/-- Where every uniform is below `1`, no round through `r` reads a string an earlier round read,
and every string read has a length its distribution gives mass. -/
theorem noColl_of_allLt (Ls : ℕ)
    (hLs : P.L < Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0) (ω : Ω)
    (g : Fin K → settings.RoundDraws α) (r : ℕ) (hr : r < K)
    (hlen : (histCodes settings O P.D Dsf ω g r).length = r)
    (hsupp : ∀ ℓ ∈ (decodeHist (histCodes settings O P.D Dsf ω g r)).pops,
      listLaw ℓ {q | Ls ≤ q.toList.length} = 0)
    (hpm : ∀ j ∈ loopPopulations K r,
      (loopDistributions P.D (decodeHist (histCodes settings O P.D Dsf ω g r)).pops j).real
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
  -- each population a round through `r` reads is one of round `r`'s
  have hdist : ∀ i, ∀ j ∈ loopPopulations K (histAt c i).pops.length,
      loopDistributions P.D (histAt c i).pops j = loopDistributions P.D H.pops j
        ∧ j ∈ loopPopulations K r := by
    intro i j hj
    have hj' : j ∈ loopPopulations K i := by
      refine loopPopulations_mono ?_ hj
      rw [histAt_pops]; exact List.length_take_le _ _
    rw [histAt_pops]
    exact ⟨loopDistributions_take P.D H.pops hj', loopPopulations_mono (hpsH i) hj⟩
  have hshort : ∀ j ∈ loopPopulations K r, loopDistributions P.D H.pops j
      {q | Ls ≤ q.toList.length} = 0 := by
    intro j hj
    cases j with
    | none =>
      refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
      simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
      omega
    | some i' =>
      have hi' : i'.val < H.pops.length := by rw [hpl]; exact lt_of_some_mem_loopPops hj
      simp only [loopDistributions, List.getElem?_eq_getElem hi']
      exact hsupp _ (List.getElem_mem _)
  have hpref : ∀ i : Fin K, i.1 ≤ r → ∀ p ∈ prefsOf settings P.D Dsf (histAt c i) (g i).1,
      p ∉ (histAt c i).prefs ∧ p.toList.length < Ls := by
    intro i hir p hp
    obtain ⟨j, hj, n, hpn⟩ := mem_prefsOf_draw settings P.D Dsf hp
    obtain ⟨hd, hjr⟩ := hdist i j hj
    have hm : (loopDistributions P.D H.pops j).real ↑(histAt c i).prefs ≤ 1 / 2 :=
      (measureReal_mono (Finset.coe_subset.2 (histAt_prefs_sub c i))).trans (hpm j hjr)
    have h0 : loopDistributions P.D (histAt c i).pops j (↑(histAt c i).prefs)ᶜ ≠ 0 := by
      rw [hd]; exact compl_ne_zero hm
    obtain ⟨hT, hpos⟩ : p ∉ (histAt c i).prefs
        ∧ 0 < (loopDistributions P.D (histAt c i).pops j).real {p} := by
      rcases hpn with h | h
      · rw [h]; exact drawFrom_offLaw h0 ((hlt i hir).2.1 j n)
      · rw [h]; exact drawFrom_offLaw h0 ((hlt i hir).2.2 j n)
    rw [hd] at hpos
    refine ⟨hT, mem_of_pos (A := {q : FreeMonoid α | q.toList.length < Ls}) ?_ hpos⟩
    refine measure_mono_null (fun x hx => ?_) (hshort j hjr)
    simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hx ⊢
    exact hx
  have hsuf : ∀ i : Fin K, i.1 ≤ r → ∀ v ∈ suffsOf settings P.D Dsf (histAt c i) (g i).1,
      v ∉ (histAt c i).suffs ∧ v.toList.length = Ls := by
    intro i hir v hv
    obtain ⟨n, rfl⟩ := mem_suffsOf_draw settings P.D Dsf hv
    have hm : Dsf.real ↑(histAt c i).suffs ≤ 1 / 2 :=
      (measureReal_mono (Finset.coe_subset.2 (histAt_suffs_sub c i))).trans hsm
    obtain ⟨hT, hpos⟩ := drawFrom_offLaw (compl_ne_zero hm) ((hlt i hir).1 n)
    refine ⟨hT, mem_of_pos (A := {q : FreeMonoid α | q.toList.length = Ls}) ?_ hpos⟩
    exact measure_mono_null (fun x hx => by simpa using hx) hDsf
  -- an earlier round's reads are in the history round `r` draws off
  have hcons : ∀ i : Fin K, i.1 < r →
      prefsOf settings P.D Dsf (histAt c i) (g i).1 ⊆ H.prefs
      ∧ suffsOf settings P.D Dsf (histAt c i) (g i).1 ⊆ H.suffs := by
    intro i hir
    have hic : i.1 < c.length := by omega
    obtain ⟨hiK, code, hget, h1, h2⟩ := histCodes_getElem settings O P.D Dsf ω g r i hic
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
    (settings.measure μ α)
      {p | (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            listLaw ℓ {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r,
            (loopDistributions P.D
              (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
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
