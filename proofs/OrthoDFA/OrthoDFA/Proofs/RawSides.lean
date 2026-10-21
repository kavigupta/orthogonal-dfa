import OrthoDFA.Proofs.RawMeas
import OrthoDFA.Proofs.RawTerm

/-!
# A tame population through one side of the hash

For a fixed hash, a named population's table and certification sequences are i.i.d. of its law
on the table side and on the reserve side.  When the hash splits its mass evenly these laws keep
its atoms within `3ρ`, and its mass on any set the family picks is the reserve side's up to the
hash's imbalance on that set.  The family reads the hash only on a window of each population's
draws, so the imbalance on the set it picks is a deviation over the hash outside the window.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]

section Laws

lemma sideLawT_named (H : J → Finset S) (uni : J) (D : J → Measure S)
    [∀ j, IsProbabilityMeasure (D j)] (g : S → Bool) {j : J} (hju : j ≠ uni)
    (hA : D j (onSide (allHarvest H) true g) ≠ 0) :
    sideLawT H uni D g j
      = prependMeasure (H j).toList ((D j)[|onSide (allHarvest H) true g]) := by
  classical
  obtain ⟨μ₁, μ₂, h1, h2, hlaw, hT, -⟩ := map_sides (D j) (H j).toList
    (onSide (allHarvest H) true g) (onSide (allHarvest H) false g)
    (onSide_disjoint (allHarvest H) g)
  have hm := (measurable_tableSeqOf_at (H j).toList (allHarvest H) g).prodMk
    (measurable_reserveSeqOf_at (allHarvest H) g)
  have hfst : sideLawT H uni D g j = μ₁ := by
    simp only [sideLawT, hju, if_false]
    haveI := h2
    have := congrArg (Measure.map Prod.fst) hlaw
    rw [Measure.map_map measurable_fst hm, Measure.map_fst_prod, measure_univ, one_smul] at this
    exact this
  rw [hfst, hT hA]
  rfl

lemma sideLawC_named (H : J → Finset S) (uni : J) (D : J → Measure S)
    [∀ j, IsProbabilityMeasure (D j)] (g : S → Bool) {j : J} (hju : j ≠ uni)
    (hR : D j (onSide (allHarvest H) false g) ≠ 0) :
    sideLawC H uni D g j = iidStream ((D j)[|onSide (allHarvest H) false g]) := by
  classical
  obtain ⟨μ₁, μ₂, h1, h2, hlaw, -, hC⟩ := map_sides (D j) (H j).toList
    (onSide (allHarvest H) true g) (onSide (allHarvest H) false g)
    (onSide_disjoint (allHarvest H) g)
  have hm := (measurable_tableSeqOf_at (H j).toList (allHarvest H) g).prodMk
    (measurable_reserveSeqOf_at (allHarvest H) g)
  have hsnd : sideLawC H uni D g j = μ₂ := by
    simp only [sideLawC, hju, if_false]
    haveI := h1
    have := congrArg (Measure.map Prod.snd) hlaw
    rw [Measure.map_map measurable_snd hm, Measure.map_snd_prod, measure_univ, one_smul] at this
    exact this
  rw [hsnd, hC hR]

/-- A side holding a third of the mass has atoms at most `3ρ`. -/
lemma cond_atom_le (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ} (hρ : ∀ t, D.real {t} ≤ ρ)
    (A : Set S) (hA : 1 / 3 ≤ D.real A) (t : S) : (D[|A]).real {t} ≤ 3 * ρ := by
  rw [cond_real_apply]
  have hA0 : 0 < D.real A := by linarith
  have h1 : D.real (A ∩ {t}) ≤ ρ := le_trans (measureReal_mono Set.inter_subset_right
    (measure_ne_top _ _)) (hρ t)
  have h2 : (D.real A)⁻¹ ≤ 3 := by
    rw [inv_le_comm₀ hA0 (by norm_num)]; linarith
  have h3 : 0 ≤ D.real (A ∩ {t}) := measureReal_nonneg
  have h4 : 0 ≤ (D.real A)⁻¹ := inv_nonneg.2 hA0.le
  nlinarith

/-- An even split leaves each side a third of the mass. -/
lemma side_third (D : Measure S) [IsProbabilityMeasure D] (Hs : Finset S) (g : S → Bool)
    {e : ℝ} (he : e ≤ 1 / 8) (hH : D.real ↑Hs ≤ 1 / 8)
    (hS : |D.real (onSide Hs true g) - D.real (onSide Hs false g)| ≤ e) :
    1 / 3 ≤ D.real (onSide Hs true g) ∧ 1 / 3 ≤ D.real (onSide Hs false g) := by
  have hp := sideOf_partition D Hs g Set.univ
  simp only [Set.univ_inter, probReal_univ] at hp
  have hH0 : 0 ≤ D.real (Set.univ ∩ (↑Hs : Set S)) := measureReal_nonneg
  have hHle : D.real (Set.univ ∩ (↑Hs : Set S)) ≤ 1 / 8 := by rw [Set.univ_inter]; exact hH
  rw [abs_le] at hS
  constructor <;> linarith

lemma pos_of_atoms (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ}
    (hρ : ∀ t, D.real {t} ≤ ρ) : 0 < ρ := by
  by_contra hc
  push Not at hc
  have h0 : ∀ t, D.real {t} = 0 := fun t => le_antisymm (le_trans (hρ t) hc) measureReal_nonneg
  have := measureReal_eq_tsum D Set.univ
  simp only [probReal_univ, Set.indicator_univ, h0, tsum_zero] at this
  exact one_ne_zero this

end Laws

section Window

variable (H : J → Finset S) (uni : J)

/-- Every population's window, side by side. -/
noncomputable def winAll (populations : Finset J) (K : ℕ) (r : Run Ω S J) : Finset S :=
  populations.biUnion (fun j' => winAt H K j' r)

open scoped Classical in
/-- Every named population's full window holds `N` table-side strings. -/
def fillOK (populations : Finset J) (K N : ℕ) (x : Draws Ω S J) : Prop :=
  ∀ j' ∈ populations, j' ≠ uni → (winAt H K j' x.1).card = K →
    N ≤ ((winAt H K j' x.1).filter (fun s => x.2 s = true)).card

omit [MeasurableSpace Ω] in
lemma card_winAll_le (populations : Finset J) (K : ℕ) (r : Run Ω S J) :
    (winAll H populations K r).card ≤ populations.card * K := by
  refine le_trans Finset.card_biUnion_le ?_
  calc ∑ j' ∈ populations, (winAt H K j' r).card ≤ ∑ _j' ∈ populations, K :=
        Finset.sum_le_sum (fun j' _ => window_card_le _ K _)
    _ = populations.card * K := by rw [Finset.sum_const, smul_eq_mul]

open scoped Classical in
/-- A table read off a window that is full with `N` on the table side, or that holds every
draw, is read off the hash on the window. -/
lemma prefixesOf_streams_congr (K N : ℕ) (r : Run Ω S J) (g g' : S → Bool) (j' : J) {m : ℕ}
    (hm : m ≤ N) (hagree : ∀ s ∈ winAt H K j' r, g s = g' s)
    (hfill : j' ≠ uni → (winAt H K j' r).card = K →
      N ≤ ((winAt H K j' r).filter (fun s => g s = true)).card) :
    prefixesOf j' m (streams H uni (r, g)) = prefixesOf j' m (streams H uni (r, g')) := by
  classical
  by_cases hju : j' = uni
  · subst hju
    show freshDraws ∅ (fun i => prefixDraw j' i (streams H j' (r, g))) m
      = freshDraws ∅ (fun i => prefixDraw j' i (streams H j' (r, g'))) m
    rw [prefixDraw_streams_uni, prefixDraw_streams_uni]
  show freshDraws ∅ (fun i => prefixDraw j' i (streams H uni (r, g))) m
    = freshDraws ∅ (fun i => prefixDraw j' i (streams H uni (r, g'))) m
  rw [prefixDraw_streams_named H uni hju, prefixDraw_streams_named H uni hju]
  simp only
  set Hs := allHarvest H with hHs
  set w := r.2.1.2 j' with hw
  set z := r.2.2 j' with hz
  set L := (H j').toList with hL
  set A := onSide Hs true g with hA
  set A' := onSide Hs true g' with hA'
  set n₀ := winIdx (↑Hs : Set S)ᶜ K w with hn₀
  have hZ : winAt H K j' r = rawIn (↑Hs : Set S)ᶜ w n₀ := rfl
  have hmemA : ∀ i, w i ∈ (↑Hs : Set S)ᶜ → w i ∈ winAt H K j' r → (w i ∈ A ↔ w i ∈ A') := by
    intro i hQ hZi
    have hnot : w i ∉ Hs := by simpa using hQ
    simp only [hA, hA', onSide, Set.mem_ofPred_eq, hnot, not_false_eq_true, and_true,
      hagree _ hZi]
  have hmemH : ∀ i, w i ∉ (↑Hs : Set S)ᶜ → (w i ∈ A ↔ w i ∈ A') := by
    intro i hQ
    have hin : w i ∈ Hs := by simpa using hQ
    simp only [hA, hA', onSide, Set.mem_ofPred_eq, hin, not_true_eq_false, and_false]
  rcases window_full_or_all (↑Hs : Set S)ᶜ K w with hfull | hall
  · -- the table fills inside the window
    have hlist : drawsIn A w n₀ = drawsIn A' w n₀ := by
      refine entries_congr_set (fun i hi => ?_)
      by_cases hQ : w i ∈ (↑Hs : Set S)ᶜ
      · exact hmemA i hQ (by rw [hZ]; exact mem_rawIn hi hQ)
      · exact hmemH i hQ
    have hAQ : A ⊆ (↑Hs : Set S)ᶜ := fun s hs => hs.2
    have hrawA : rawIn A w n₀ = (winAt H K j' r).filter (fun s => g s = true) := by
      rw [hZ, rawIn_filter hAQ w n₀]
      refine Finset.filter_congr (fun s hs => ?_)
      have hsQ : s ∉ Hs := by
        have := mem_of_mem_rawIn hs
        simpa using this
      simp only [hA, onSide, Set.mem_ofPred_eq, hsQ, not_false_eq_true, and_true]
    have hdisj : Disjoint L.toFinset (rawIn A w n₀) := by
      rw [Finset.disjoint_left]
      intro s hs hs'
      exact (mem_of_mem_rawIn hs').2 (harvest_subset H j' (Finset.mem_toList.1
        (List.mem_toFinset.1 hs)))
    set pad := L.headD (nthDrawIn A (z 0) w 0) with hpad
    set pad' := L.headD (nthDrawIn A' (z 0) w 0) with hpad'
    have hseq : tableSeqOf L A w z = prependStream L (nthDrawIn A pad w) := rfl
    have hseq' : tableSeqOf L A' w z = prependStream L (nthDrawIn A' pad' w) := rfl
    have hacc : acceptedAt ∅ (tableSeqOf L A w z) (L.length + (drawsIn A w n₀).length)
        = L.toFinset ∪ rawIn A w n₀ := by
      rw [hseq, acceptedAt_prepend, acceptedAt_sub_eq_rawIn,
        Finset.sdiff_eq_self_of_disjoint hdisj.symm]
    refine freshDraws_congr ∅ (n := L.length + (drawsIn A w n₀).length) (fun k hk => ?_) ?_
    · rw [hseq, hseq']
      unfold prependStream
      split_ifs with hkL
      · rfl
      · have hk' : k - L.length < (drawsIn A w n₀).length := by omega
        have hk'' : k - L.length < (drawsIn A' w n₀).length := by rw [← hlist]; exact hk'
        rw [sub_eq_getD A pad w hk', sub_eq_getD A' pad' w hk'', hlist,
          List.getD_eq_getElem _ _ hk'', List.getD_eq_getElem _ _ hk'']
    · rw [hacc, Finset.card_union_of_disjoint hdisj, hrawA]
      have := hfill hju hfull
      omega
  · -- the window holds every draw
    have hmem : ∀ i, (w i ∈ A ↔ w i ∈ A') := by
      intro i
      by_cases hQ : w i ∈ (↑Hs : Set S)ᶜ
      · exact hmemA i hQ (hall (i + 1) (mem_rawIn (Nat.lt_succ_self i) hQ))
      · exact hmemH i hQ
    have hlist : ∀ n, drawsIn A w n = drawsIn A' w n :=
      fun n => entries_congr_set (fun i _ => hmem i)
    have hsub : ∀ d, nthDrawIn A d w = nthDrawIn A' d w := by
      intro d
      funext k
      unfold nthDrawIn
      simp only [hlist]
    have hT : tableSeqOf L A w z = tableSeqOf L A' w z := by
      unfold tableSeqOf
      simp only [hsub]
    rw [hT]

end Window

section Events

variable (O : Oracle μ S) (populations : Finset J) (H : J → Finset S) (uni : J)

open scoped Classical in
/-- Where the family at `B` cuts wrongly (`true`) or is undecided (`false`). -/
def famSet (b : Bool) (y : Run Ω S J) (B : State) : Set S :=
  if b then {p | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations y B) p (oracleNoise y)}
  else {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations y B) p (oracleNoise y)}

lemma clusterBy_congr (B : State) {y y' : Run Ω S J}
    (hP : prefixesAt populations B.npref y = prefixesAt populations B.npref y')
    (hC : poolAt B.nsuff y = poolAt B.nsuff y') (hω : oracleNoise y = oracleNoise y') :
    clusterBy rule O.mq populations y B = clusterBy rule O.mq populations y' B := by
  unfold clusterBy screenedAt
  rw [hP, hC, hω]

open scoped Classical in
lemma fillOK_congr (K N : ℕ) (r : Run Ω S J) {g g' : S → Bool}
    (hagree : ∀ s ∈ winAll H populations K r, g s = g' s) :
    fillOK H uni populations K N (r, g) ↔ fillOK H uni populations K N (r, g') := by
  have hf : ∀ j' ∈ populations, (winAt H K j' r).filter (fun s => g s = true)
      = (winAt H K j' r).filter (fun s => g' s = true) := fun j' hj =>
    Finset.filter_congr (fun s hs => by
      rw [hagree s (Finset.mem_biUnion.2 ⟨j', hj, hs⟩)])
  unfold fillOK
  refine forall₂_congr (fun j' hj => ?_)
  simp only [hf j' hj]

open scoped Classical in
/-- Under a full fill, the set the family picks reads the hash only on the windows. -/
lemma famSet_congr (K N : ℕ) (b : Bool) (B : State) (hB : B.npref ≤ N) (r : Run Ω S J)
    {g g' : S → Bool} (hagree : ∀ s ∈ winAll H populations K r, g s = g' s)
    (hfill : fillOK H uni populations K N (r, g)) :
    famSet (rule := rule) O populations b (streams H uni (r, g)) B
      = famSet (rule := rule) O populations b (streams H uni (r, g')) B := by
  have hP : prefixesAt populations B.npref (streams H uni (r, g))
      = prefixesAt populations B.npref (streams H uni (r, g')) :=
    Finset.biUnion_congr rfl (fun j' hj => prefixesOf_streams_congr H uni K N r g g' j' hB
      (fun s hs => hagree s (Finset.mem_biUnion.2 ⟨j', hj, hs⟩))
      (fun hju hfull => hfill j' hj hju hfull))
  have hcl := clusterBy_congr (rule := rule) O populations B hP rfl rfl
  unfold famSet familyBy
  rw [hcl]
  rfl

open scoped Classical in
/-- The family's set where every window is full, and nothing elsewhere. -/
noncomputable def famSetF (K N : ℕ) (b : Bool) (B : State) (x : Draws Ω S J) : Set S :=
  if fillOK H uni populations K N x then
    famSet (rule := rule) O populations b (streams H uni x) B
  else ∅

lemma famSetF_congr (K N : ℕ) (b : Bool) (B : State) (hB : B.npref ≤ N) (r : Run Ω S J)
    {g g' : S → Bool} (hagree : ∀ s ∈ winAll H populations K r, g s = g' s) :
    famSetF (rule := rule) O populations H uni K N b B (r, g)
      = famSetF (rule := rule) O populations H uni K N b B (r, g') := by
  classical
  unfold famSetF
  by_cases hf : fillOK H uni populations K N (r, g)
  · rw [if_pos hf, if_pos ((fillOK_congr populations H uni K N r hagree).1 hf)]
    exact famSet_congr O populations H uni K N b B hB r hagree hf
  · rw [if_neg hf, if_neg (fun h => hf ((fillOK_congr populations H uni K N r hagree).2 h))]

open scoped Classical in
lemma measurableSet_fillOK (K N : ℕ) :
    MeasurableSet {x : Draws Ω S J | fillOK H uni populations K N x} := by
  classical
  refine measurableSet_coinEvent (fun r : Run Ω S J => fun j' => winAt H K j' r)
    (fun a => ?_) (fun a => populations.biUnion a)
    (fun a g => ∀ j' ∈ populations, j' ≠ uni → (a j').card = K →
      N ≤ ((a j').filter (fun s => g s = true)).card) (fun a g g' hgg => ?_)
  · have : {r : Run Ω S J | (fun j' => winAt H K j' r) = a}
        = ⋂ j', {r : Run Ω S J | winAt H K j' r = a j'} := by
      ext r; simp [funext_iff]
    rw [this]
    exact MeasurableSet.iInter (fun j' => measurableSet_winAt_eq H K j' (a j'))
  · have hf : ∀ j' ∈ populations, (a j').filter (fun s => g s = true)
        = (a j').filter (fun s => g' s = true) :=
      fun j' hj => Finset.filter_congr (fun s hs => by
        rw [hgg s (Finset.mem_biUnion.2 ⟨j', hj, hs⟩)])
    refine forall₂_congr (fun j' hj => ?_)
    simp only [hf j' hj]

open scoped Classical in
lemma measurableSet_mem_famSet (b : Bool) (B : State) (s : S) :
    MeasurableSet {y : Run Ω S J | s ∈ famSet (rule := rule) O populations b y B} := by
  classical
  cases b
  · exact measurableSet_family_noise O populations B
      (fun A ω => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A) s ω)
      (fun A => noiseAlg_le O Set.univ _ (measurableSet_decided' O B.lo (B.hi + 1) _ s))
  · exact measurableSet_family_noise O populations B
      (fun A ω => ¬ cutCorrect O B.lo B.hi A s ω)
      (fun A => measurableSet_cutCorrect O B.lo B.hi A s)

open scoped Classical in
/-- A population's mass on the family's set, on one side of the hash, is measurable in the
draws and the hash. -/
lemma measurable_famMass (D : Measure S) (K N : ℕ) (b side : Bool) (B : State) :
    Measurable (fun x : Draws Ω S J =>
      D.real (famSetF (rule := rule) O populations H uni K N b B x
        ∩ onSide (allHarvest H) side x.2)) := by
  classical
  have hBad : ∀ s, MeasurableSet {x : Draws Ω S J | x ∈ {x' : Draws Ω S J |
      s ∈ famSetF (rule := rule) O populations H uni K N b B x'
        ∩ onSide (allHarvest H) side x'.2}} := by
    intro s
    have hrw : {x : Draws Ω S J | x ∈ {x' : Draws Ω S J |
        s ∈ famSetF (rule := rule) O populations H uni K N b B x'
          ∩ onSide (allHarvest H) side x'.2}}
        = {x | fillOK H uni populations K N x}
          ∩ (streams H uni ⁻¹' {y | s ∈ famSet (rule := rule) O populations b y B})
          ∩ ((fun x : Draws Ω S J => x.2 s) ⁻¹' {side} ∩ {_x | s ∉ allHarvest H}) := by
      ext x
      simp only [famSetF, onSide, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_preimage,
        Set.mem_singleton_iff]
      by_cases hf : fillOK H uni populations K N x
      · simp [hf]
      · simp [hf]
    rw [hrw]
    refine ((measurableSet_fillOK populations H uni K N).inter (measurable_streams H uni
      (measurableSet_mem_famSet O populations b B s))).inter
      ((((measurable_pi_apply s).comp measurable_snd) (measurableSet_singleton side)).inter
        (MeasurableSet.const _))
  have h := measurable_badMass (Ω := Draws Ω S J) D
    (fun s => {x' : Draws Ω S J | s ∈ famSetF (rule := rule) O populations H uni K N b B x'
      ∩ onSide (allHarvest H) side x'.2}) hBad
  exact ENNReal.measurable_toReal.comp h

open scoped Classical in
/-- The hash splits a population's mass on the family's set unevenly. -/
noncomputable def imbSet (D : Measure S) (K N : ℕ) (e : ℝ) (b : Bool) (B : State) :
    Set (Draws Ω S J) :=
  {x | e < |D.real (famSetF (rule := rule) O populations H uni K N b B x
      ∩ onSide (allHarvest H) true x.2)
    - D.real (famSetF (rule := rule) O populations H uni K N b B x
      ∩ onSide (allHarvest H) false x.2)|}

lemma measurableSet_imbSet (D : Measure S) (K N : ℕ) (e : ℝ) (b : Bool) (B : State) :
    MeasurableSet (imbSet (rule := rule) O populations H uni D K N e b B) :=
  measurableSet_lt measurable_const (continuous_abs.measurable.comp
    ((measurable_famMass O populations H uni D K N b true B).sub
      (measurable_famMass O populations H uni D K N b false B)))

/-- At a fixed draw the hash splits the family's set unevenly with chance `2 exp(-e²/(8ρ))`,
when the windows hold at most a quarter of `e` of the population's mass. -/
lemma coins_imbSet_le (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ}
    (hρ : ∀ t, D.real {t} ≤ ρ) (K N : ℕ) {e : ℝ} (he : 0 < e) (b : Bool) (B : State)
    (hB : B.npref ≤ N) (r : Run Ω S J) (hU : D.real ↑(winAll H populations K r) ≤ e / 4) :
    (coins S).real {g | (r, g) ∈ imbSet (rule := rule) O populations H uni D K N e b B}
      ≤ 2 * Real.exp (-(e / 2) ^ 2 / (2 * ρ)) := by
  refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _))
    (side_balance D hρ (allHarvest H) (winAll H populations K r)
      (fun g => famSetF (rule := rule) O populations H uni K N b B (r, g))
      (fun g g' hgg => famSetF_congr O populations H uni K N b B hB r hgg)
      (t := e / 2) (ε := e / 4) (by positivity) (by positivity))
  simp only [imbSet, Set.mem_ofPred_eq] at hg ⊢
  linarith

/-- The hash leaves a population's sides uneven. -/
def sideBad (D : Measure S) (Hs : Finset S) (e : ℝ) : Set (S → Bool) :=
  {g | e < |D.real (onSide Hs true g) - D.real (onSide Hs false g)|}

lemma measurableSet_sideBad (D : Measure S) (Hs : Finset S) (e : ℝ) :
    MeasurableSet (sideBad D Hs e) := by
  classical
  have hm : ∀ side : Bool, Measurable (fun g : S → Bool => D.real (onSide Hs side g)) := by
    intro side
    have hBad : ∀ s,
        MeasurableSet {g : S → Bool | g ∈ {g' : S → Bool | s ∈ onSide Hs side g'}} := by
      intro s
      have : {g : S → Bool | g ∈ {g' : S → Bool | s ∈ onSide Hs side g'}}
          = (fun g : S → Bool => g s) ⁻¹' {side} ∩ {_g | s ∉ Hs} := by
        ext g; simp [onSide]
      rw [this]
      exact ((measurable_pi_apply s) (measurableSet_singleton side)).inter
        (MeasurableSet.const _)
    exact ENNReal.measurable_toReal.comp (measurable_badMass (Ω := S → Bool) D
      (fun s => {g' : S → Bool | s ∈ onSide Hs side g'}) hBad)
  exact measurableSet_lt measurable_const (continuous_abs.measurable.comp ((hm true).sub
    (hm false)))

lemma coins_sideBad_le (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ}
    (hρ : ∀ t, D.real {t} ≤ ρ) (Hs : Finset S) {e : ℝ} (he : 0 < e) :
    (coins S).real (sideBad D Hs e) ≤ 2 * Real.exp (-(e / 2) ^ 2 / (2 * ρ)) := by
  refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _))
    (side_balance D hρ Hs ∅ (fun _ => Set.univ) (fun _ _ _ => rfl)
      (t := e / 2) (ε := e / 2) (by positivity) (by positivity))
  simp only [sideBad, Set.mem_ofPred_eq, Finset.coe_empty, measureReal_empty, add_zero,
    Set.univ_inter] at hg ⊢
  linarith

omit [MeasurableSpace Ω] in
open scoped Classical in
/-- At a fixed draw, some window of `4N` strings holds fewer than `N` on the table side with
chance at most `|J| exp(-N/2)`. -/
lemma coins_notFill_le (N : ℕ) (hN : 0 < N) (r : Run Ω S J) :
    (coins S).real {g | ¬ fillOK H uni populations (4 * N) N (r, g)}
      ≤ populations.card * Real.exp (-((N : ℝ) / 2)) := by
  classical
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  set T : J → Set (S → Bool) := fun j' => if (winAt H (4 * N) j' r).card = 4 * N then
    {g | ((winAt H (4 * N) j' r).card : ℝ) / 2 + N
      ≤ (((winAt H (4 * N) j' r).filter (fun s => g s = false)).card : ℝ)} else ∅ with hT
  have hsub : {g | ¬ fillOK H uni populations (4 * N) N (r, g)} ⊆ ⋃ j' ∈ populations, T j' := by
    intro g hg
    simp only [fillOK, Set.mem_ofPred_eq, not_forall, not_le, exists_prop] at hg
    obtain ⟨j', hj, -, hfull, hlt⟩ := hg
    refine Set.mem_biUnion hj ?_
    simp only [hT, if_pos hfull, Set.mem_ofPred_eq]
    have hsplit := Finset.card_filter_add_card_filter_not (s := winAt H (4 * N) j' r)
      (fun s => g s = true)
    have e : (winAt H (4 * N) j' r).filter (fun s => ¬ g s = true)
        = (winAt H (4 * N) j' r).filter (fun s => g s = false) :=
      Finset.filter_congr (fun s _ => by simp)
    rw [e, hfull] at hsplit
    rw [hfull]
    have : ((winAt H (4 * N) j' r).filter (fun s => g s = false)).card ≥ 3 * N := by omega
    have h3 : (((winAt H (4 * N) j' r).filter (fun s => g s = false)).card : ℝ) ≥ 3 * N := by
      exact_mod_cast this
    push_cast
    linarith
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hone : ∀ j', (coins S).real (T j') ≤ Real.exp (-((N : ℝ) / 2)) := by
    intro j'
    simp only [hT]
    split_ifs with hfull
    · refine le_trans (coins_window_side _ false (Nat.cast_nonneg N) (by omega)) ?_
      rw [hfull]
      refine Real.exp_le_exp.2 (le_of_eq ?_)
      push_cast
      field_simp
      ring
    · rw [measureReal_empty]; positivity
  calc ∑ j' ∈ populations, (coins S).real (T j')
      ≤ ∑ _j' ∈ populations, Real.exp (-((N : ℝ) / 2)) :=
        Finset.sum_le_sum (fun j' _ => hone j')
    _ = populations.card * Real.exp (-((N : ℝ) / 2)) := by
        rw [Finset.sum_const, nsmul_eq_mul]

end Events

end OrthoDFA
