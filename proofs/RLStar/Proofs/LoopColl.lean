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
  sorry -- closes by: main's proof with volume_eq_one_unit for every stream's uniform

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
            (g ⟨i, hiK⟩).1) : Run Unit (FreeMonoid α) (Option (Fin K))) := by
  sorry -- closes by: main's induction with codesOf_idealRounds_succ, popCode

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
  sorry -- closes by: main's proof with drawFrom_offLaw for list populations, histCodes_getElem

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
