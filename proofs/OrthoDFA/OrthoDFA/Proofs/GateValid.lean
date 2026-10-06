import OrthoDFA.Proofs.Adaptive
import Mathlib.NumberTheory.ZetaValues

/-!
# What the gate certifies

The certification draws are the run's last factor, so with the noise and the table held fixed
they are i.i.d. draws of the population, which the family's cut and the seed's reads split into
fixed sets.  Each of `misAdmits`' four intervals then misses at most at its level
(`pi_rate_miss_le`, `pi_mass_miss_le`), and a family admitted with all four holding the
population's own mass and rates has `misShareOf` under the limit.  Each state is its own look,
and the looks' levels sum to `α`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-- The four intervals `misAdmits` reads hold `m`, `1 − m`, `a` and `r`. -/
def gateCovers (level : ℝ) (c : (ℕ × ℕ) × (ℕ × ℕ)) (m a r : ℝ) : Prop :=
  cpCovers (c.1.2 + c.2.2) c.1.2 level m ∧ cpCovers (c.1.2 + c.2.2) c.2.2 level (1 - m)
    ∧ cpCovers c.1.2 c.1.1 level a ∧ cpCovers c.2.2 c.2.1 level r

lemma misShare_le_of_covers {η₀ level limit : ℝ} {c : (ℕ × ℕ) × (ℕ × ℕ)} {m a r : ℝ}
    (hadm : misAdmits η₀ level limit c) (hm : m ∈ Set.Icc (0 : ℝ) 1) (ha : a ∈ Set.Icc (0 : ℝ) 1)
    (hr : r ∈ Set.Icc (0 : ℝ) 1) (hc : gateCovers level c m a r) : misShare η₀ m a r ≤ limit :=
  hadm m a r hm ha hr hc.1 hc.2.1 hc.2.2.1 hc.2.2.2

/-- A share of a side, as `misShareOf` reads a rate. -/
lemma ratio_mem (ν : Measure S) [IsProbabilityMeasure ν] (A B : Set S) :
    ν.real (A ∩ B) / ν.real A ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨div_nonneg measureReal_nonneg measureReal_nonneg, ?_⟩
  rcases eq_or_lt_of_le (measureReal_nonneg : (0 : ℝ) ≤ ν.real A) with h0 | hpos
  · rw [← h0, div_zero]; norm_num
  · rw [div_le_one hpos]; exact measureReal_mono Set.inter_subset_left

open scoped Classical in
/-- The counts `splitCounts` reads, over draws given as a function. -/
noncomputable def countsFin {n : ℕ} (A B : Set S) (x : Fin n → S) : (ℕ × ℕ) × (ℕ × ℕ) :=
  (((univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card, (univ.filter (fun i => x i ∈ A)).card),
    ((univ.filter (fun i => x i ∈ Aᶜ ∧ x i ∈ B)).card, (univ.filter (fun i => x i ∈ Aᶜ)).card))

open scoped Classical in
/-- Over i.i.d. draws, the four intervals all hold, except at four times their level. -/
theorem pi_gate_miss_le (ν : Measure S) [IsProbabilityMeasure ν] (A B : Set S) (n : ℕ)
    (level : ℝ) (hl : 0 ≤ level) :
    (Measure.pi fun _ : Fin n => ν).real
      {x | ¬ gateCovers level (countsFin A B x) (ν.real A) (ν.real (A ∩ B) / ν.real A)
          (ν.real (Aᶜ ∩ B) / ν.real Aᶜ)} ≤ 4 * level := by
  classical
  have hmeasA : MeasurableSet A := (Set.to_countable _).measurableSet
  have hcompl : ν.real Aᶜ = 1 - ν.real A := by rw [measureReal_compl hmeasA, probReal_univ]
  have hn : ∀ x : Fin n → S, (univ.filter (fun i => x i ∈ A)).card
      + (univ.filter (fun i => x i ∈ Aᶜ)).card = n := by
    intro x
    rw [show (univ.filter (fun i => x i ∈ Aᶜ)) = univ.filter (fun i => ¬ x i ∈ A) from rfl,
      Finset.card_filter_add_card_filter_not, Finset.card_univ, Fintype.card_fin]
  have hsub : {x : Fin n → S | ¬ gateCovers level (countsFin A B x) (ν.real A)
        (ν.real (A ∩ B) / ν.real A) (ν.real (Aᶜ ∩ B) / ν.real Aᶜ)}
      ⊆ (({x | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)}
          ∪ {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ Aᶜ)).card level (ν.real Aᶜ)})
        ∪ {x | ¬ cpCovers (univ.filter (fun i => x i ∈ A)).card
            (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card level (ν.real (A ∩ B) / ν.real A)})
        ∪ {x | ¬ cpCovers (univ.filter (fun i => x i ∈ Aᶜ)).card
            (univ.filter (fun i => x i ∈ Aᶜ ∧ x i ∈ B)).card level
            (ν.real (Aᶜ ∩ B) / ν.real Aᶜ)} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := hc
    apply hx
    refine ⟨?_, ?_, h3, h4⟩
    · show cpCovers ((univ.filter (fun i => x i ∈ A)).card
          + (univ.filter (fun i => x i ∈ Aᶜ)).card) _ level _
      rw [hn x]; exact h1
    · show cpCovers ((univ.filter (fun i => x i ∈ A)).card
          + (univ.filter (fun i => x i ∈ Aᶜ)).card) _ level (1 - ν.real A)
      rw [hn x, ← hcompl]; exact h2
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have u1 := measureReal_union_le (μ := Measure.pi fun _ : Fin n => ν)
    ({x : Fin n → S | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)}
      ∪ {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ Aᶜ)).card level (ν.real Aᶜ)}
      ∪ {x | ¬ cpCovers (univ.filter (fun i => x i ∈ A)).card
          (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card level (ν.real (A ∩ B) / ν.real A)})
    {x | ¬ cpCovers (univ.filter (fun i => x i ∈ Aᶜ)).card
        (univ.filter (fun i => x i ∈ Aᶜ ∧ x i ∈ B)).card level (ν.real (Aᶜ ∩ B) / ν.real Aᶜ)}
  have u2 := measureReal_union_le (μ := Measure.pi fun _ : Fin n => ν)
    ({x : Fin n → S | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)}
      ∪ {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ Aᶜ)).card level (ν.real Aᶜ)})
    {x | ¬ cpCovers (univ.filter (fun i => x i ∈ A)).card
          (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card level (ν.real (A ∩ B) / ν.real A)}
  have u3 := measureReal_union_le (μ := Measure.pi fun _ : Fin n => ν)
    {x : Fin n → S | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)}
    {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ Aᶜ)).card level (ν.real Aᶜ)}
  have e1 : (Measure.pi fun _ : Fin n => ν).real
      {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)} ≤ level := by
    convert pi_mass_miss_le ν A n level hl
  have e2 : (Measure.pi fun _ : Fin n => ν).real
      {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ Aᶜ)).card level (ν.real Aᶜ)} ≤ level := by
    convert pi_mass_miss_le ν Aᶜ n level hl
  have e3 : (Measure.pi fun _ : Fin n => ν).real
      {x | ¬ cpCovers (univ.filter (fun i => x i ∈ A)).card
          (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card level (ν.real (A ∩ B) / ν.real A)}
      ≤ level := by
    convert pi_rate_miss_le ν A B n level hl
  have e4 : (Measure.pi fun _ : Fin n => ν).real
      {x | ¬ cpCovers (univ.filter (fun i => x i ∈ Aᶜ)).card
          (univ.filter (fun i => x i ∈ Aᶜ ∧ x i ∈ B)).card level
          (ν.real (Aᶜ ∩ B) / ν.real Aᶜ)} ≤ level := by
    convert pi_rate_miss_le ν Aᶜ B n level hl
  linarith

/-- The certification stream's first `n` draws of one population are i.i.d. -/
lemma map_streamBlock (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)] (j : J) (n : ℕ) :
    Measure.map (fun c : J → ℕ → S => fun i : Fin n => c j i)
        (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
      = Measure.pi (fun _ : Fin n => D j) := by
  have hstep : (fun c : J → ℕ → S => fun i : Fin n => c j i)
      = (fun s : ℕ → S => (fun i : Fin n => s i.val)) ∘ (fun c : J → ℕ → S => c j) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop),
    (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j).map_eq]
  refine (Measure.pi_eq (μ := fun _ : Fin n => D j) fun t ht => ?_).symm
  have hpre : (fun s : ℕ → S => (fun i : Fin n => s i.val)) ⁻¹' Set.univ.pi t
      = Set.pi ↑(Finset.range n) (fun i => if h : i < n then t ⟨i, h⟩ else Set.univ) := by
    ext s
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_range,
      Set.mem_Iio]
    constructor
    · intro h i hi
      rw [dif_pos hi]
      exact h ⟨i, hi⟩
    · intro h i
      have := h i.val i.isLt
      simpa [dif_pos i.isLt] using this
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre, Measure.infinitePi_pi]
  · rw [← Fin.prod_univ_eq_prod_range]
    exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
  · intro i _
    split_ifs with h
    exacts [ht ⟨i, h⟩, .univ]

lemma card_range_filter_fin (n : ℕ) (P : ℕ → Prop) [DecidablePred P]
    [DecidablePred (fun i : Fin n => P i)] :
    ((Finset.range n).filter P).card = (univ.filter (fun i : Fin n => P i)).card := by
  rw [Finset.card_filter, Finset.card_filter, ← Fin.sum_univ_eq_sum_range]
  exact Finset.sum_congr rfl (fun i _ => by split_ifs <;> rfl)

open scoped Classical in
/-- The intervals the gate reads at population `j` miss the population's own mass and rates. -/
noncomputable def gateMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (j : J) (B : State) (level : ℝ) : Set (Run Ω S J) :=
  {x | ¬ gateCovers level
      (countsFin {p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p (oracleNoise x)}
        {p | O.mq p (oracleNoise x) = 1} (fun i : Fin B.npref => certPrefix j i.val x))
      ((D j).real {p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p (oracleNoise x)})
      ((D j).real ({p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p (oracleNoise x)}
          ∩ {p | O.mq p (oracleNoise x) = 1})
        / (D j).real {p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p (oracleNoise x)})
      ((D j).real ({p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p (oracleNoise x)}ᶜ
          ∩ {p | O.mq p (oracleNoise x) = 1})
        / (D j).real {p | cutAccepts O.mq (clusterBy rule O.mq populations x B) p
            (oracleNoise x)}ᶜ)}

open scoped Classical in
/-- At a fixed family and fixed draws, the miss is an event about the noise. -/
lemma measurableSet_gateMissFixed (O : Oracle μ S) (Dj : Measure S) {n : ℕ} (A₀ : Finset S)
    (tt : Fin n → S) (level : ℝ) :
    MeasurableSet {ω : Ω | ¬ gateCovers level
      (countsFin {p | cutAccepts O.mq A₀ p ω} {p | O.mq p ω = 1} tt)
      (Dj.real {p | cutAccepts O.mq A₀ p ω})
      (Dj.real ({p | cutAccepts O.mq A₀ p ω} ∩ {p | O.mq p ω = 1})
        / Dj.real {p | cutAccepts O.mq A₀ p ω})
      (Dj.real ({p | cutAccepts O.mq A₀ p ω}ᶜ ∩ {p | O.mq p ω = 1})
        / Dj.real {p | cutAccepts O.mq A₀ p ω}ᶜ)} := by
  classical
  have hacc : ∀ p, MeasurableSet {ω : Ω | cutAccepts O.mq A₀ p ω} := fun p =>
    noiseAlg_le O Set.univ _ (measurableSet_cutAccepts O A₀ p)
  have hread : ∀ p, MeasurableSet {ω : Ω | O.mq p ω = 1} := fun p =>
    noiseAlg_le O Set.univ _ (measurableSet_mq_eq_one O (Set.mem_univ p))
  have hmass : ∀ (Bad : S → Set Ω), (∀ p, MeasurableSet (Bad p)) →
      Measurable (fun ω => Dj.real {p | ω ∈ Bad p}) := fun Bad h =>
    ENNReal.measurable_toReal.comp (measurable_badMass Dj Bad h)
  have hm : Measurable (fun ω => Dj.real {p | cutAccepts O.mq A₀ p ω}) :=
    hmass (fun p => {ω | cutAccepts O.mq A₀ p ω}) hacc
  have hm1 : Measurable (fun ω => Dj.real ({p | cutAccepts O.mq A₀ p ω} ∩ {p | O.mq p ω = 1})) :=
    hmass (fun p => {ω | cutAccepts O.mq A₀ p ω} ∩ {ω | O.mq p ω = 1})
      (fun p => (hacc p).inter (hread p))
  have hm0 : Measurable (fun ω => Dj.real {p | cutAccepts O.mq A₀ p ω}ᶜ) :=
    hmass (fun p => {ω | cutAccepts O.mq A₀ p ω}ᶜ) (fun p => (hacc p).compl)
  have hm01 : Measurable
      (fun ω => Dj.real ({p | cutAccepts O.mq A₀ p ω}ᶜ ∩ {p | O.mq p ω = 1})) :=
    hmass (fun p => {ω | cutAccepts O.mq A₀ p ω}ᶜ ∩ {ω | O.mq p ω = 1})
      (fun p => (hacc p).compl.inter (hread p))
  -- the counts take finitely many values, each on a measurable set
  have hfin : ∀ (P : Fin n → Set Ω), (∀ i, MeasurableSet (P i)) →
      Measurable (fun ω => (univ.filter (fun i => ω ∈ P i)).card) := by
    intro P hP
    refine measurable_to_countable' (fun k => ?_)
    have : (fun ω => (univ.filter (fun i => ω ∈ P i)).card) ⁻¹' {k}
        = ⋃ T ∈ (univ : Finset (Fin n)).powerset.filter (fun T => T.card = k),
          ⋂ i, {ω | ω ∈ P i ↔ i ∈ T} := by
      ext ω
      simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_iUnion, Set.mem_iInter,
        Finset.mem_filter, Finset.mem_powerset, exists_prop, Set.mem_setOf_eq]
      constructor
      · intro h
        exact ⟨univ.filter (fun i => ω ∈ P i), ⟨Finset.filter_subset _ _, h⟩,
          fun i => by simp⟩
      · rintro ⟨T, ⟨-, hk⟩, hT⟩
        rw [← hk]
        congr 1
        ext i
        simp [hT i]
    rw [this]
    refine Finset.measurableSet_biUnion _ (fun T _ => MeasurableSet.iInter (fun i => ?_))
    by_cases hi : i ∈ T
    · have e : {ω | ω ∈ P i ↔ i ∈ T} = P i := by ext ω; simp [hi]
      rw [e]; exact hP i
    · have e : {ω | ω ∈ P i ↔ i ∈ T} = (P i)ᶜ := by ext ω; simp [hi]
      rw [e]; exact (hP i).compl
  have hc11 := hfin (fun i => {ω | tt i ∈ {p | cutAccepts O.mq A₀ p ω}
      ∧ tt i ∈ {p | O.mq p ω = 1}}) (fun i => (hacc (tt i)).inter (hread (tt i)))
  have hc12 := hfin (fun i => {ω | tt i ∈ {p | cutAccepts O.mq A₀ p ω}}) (fun i => hacc (tt i))
  have hc01 := hfin (fun i => {ω | tt i ∈ {p | cutAccepts O.mq A₀ p ω}ᶜ
      ∧ tt i ∈ {p | O.mq p ω = 1}}) (fun i => (hacc (tt i)).compl.inter (hread (tt i)))
  have hc02 := hfin (fun i => {ω | tt i ∈ {p | cutAccepts O.mq A₀ p ω}ᶜ})
    (fun i => (hacc (tt i)).compl)
  have hcnt : Measurable (fun ω => countsFin (n := n) {p | cutAccepts O.mq A₀ p ω}
      {p | O.mq p ω = 1} tt) := by
    convert (hc11.prodMk hc12).prodMk (hc01.prodMk hc02) using 1
    funext ω
    simp only [countsFin]
    congr 1 <;> congr 1 <;> congr 1 <;> ext i <;> simp
  have hrates : Measurable (fun ω => (Dj.real {p | cutAccepts O.mq A₀ p ω},
      Dj.real ({p | cutAccepts O.mq A₀ p ω} ∩ {p | O.mq p ω = 1})
        / Dj.real {p | cutAccepts O.mq A₀ p ω},
      Dj.real ({p | cutAccepts O.mq A₀ p ω}ᶜ ∩ {p | O.mq p ω = 1})
        / Dj.real {p | cutAccepts O.mq A₀ p ω}ᶜ)) :=
    hm.prodMk ((hm1.div hm).prodMk (hm01.div hm0))
  -- and the intervals are closed in each rate, so the event is measurable in the pair
  have hsf : ∀ N k, Measurable (fun p : ℝ => binomSfGe N p k) := by
    intro N k; unfold binomSfGe; fun_prop
  have hcdf : ∀ N k, Measurable (fun p : ℝ => binomCdfLe N p k) := by
    intro N k; unfold binomCdfLe; fun_prop
  have hcov : ∀ N k (g : ℝ × ℝ × ℝ → ℝ), Measurable g →
      MeasurableSet {w : ℝ × ℝ × ℝ | cpCovers N k level (g w)} := fun N k g hg =>
    (measurableSet_le measurable_const ((hsf N k).comp hg)).inter
      (measurableSet_le measurable_const ((hcdf N k).comp hg))
  have hset : MeasurableSet {z : ((ℕ × ℕ) × (ℕ × ℕ)) × (ℝ × ℝ × ℝ) |
      ¬ gateCovers level z.1 z.2.1 z.2.2.1 z.2.2.2} := by
    have : {z : ((ℕ × ℕ) × (ℕ × ℕ)) × (ℝ × ℝ × ℝ) | ¬ gateCovers level z.1 z.2.1 z.2.2.1 z.2.2.2}
        = ⋃ c : (ℕ × ℕ) × (ℕ × ℕ), {c} ×ˢ {w : ℝ × ℝ × ℝ | ¬ gateCovers level c w.1 w.2.1 w.2.2} := by
      ext ⟨c, w⟩
      simp
    rw [this]
    refine MeasurableSet.iUnion (fun c => (measurableSet_singleton c).prod ?_)
    have e : {w : ℝ × ℝ × ℝ | gateCovers level c w.1 w.2.1 w.2.2}
        = {w : ℝ × ℝ × ℝ | cpCovers (c.1.2 + c.2.2) c.1.2 level w.1}
          ∩ {w | cpCovers (c.1.2 + c.2.2) c.2.2 level (1 - w.1)}
          ∩ {w | cpCovers c.1.2 c.1.1 level w.2.1}
          ∩ {w | cpCovers c.2.2 c.2.1 level w.2.2} := by
      ext w; simp [gateCovers, and_assoc]
    have hmeas : MeasurableSet {w : ℝ × ℝ × ℝ | gateCovers level c w.1 w.2.1 w.2.2} := by
      rw [e]
      exact (((hcov (c.1.2 + c.2.2) c.1.2 Prod.fst measurable_fst).inter
        (hcov (c.1.2 + c.2.2) c.2.2 (fun w => 1 - w.1)
          (measurable_const.sub measurable_fst))).inter
        (hcov c.1.2 c.1.1 (fun w => w.2.1) (measurable_fst.comp measurable_snd))).inter
        (hcov c.2.2 c.2.1 (fun w => w.2.2) (measurable_snd.comp measurable_snd))
    exact hmeas.compl
  exact (hcnt.prodMk hrates) hset

open scoped Classical in
lemma measurableSet_gateMiss (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (j : J) (B : State) (level : ℝ) :
    MeasurableSet (gateMiss rule O populations D j B level) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω : Ω | ¬ gateCovers level
        (countsFin {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}
          {p | O.mq p ω = 1} tt)
        ((D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω})
        ((D j).real ({p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}
            ∩ {p | O.mq p ω = 1})
          / (D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω})
        ((D j).real ({p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}ᶜ
            ∩ {p | O.mq p ω = 1})
          / (D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}ᶜ)}
      else (∅ : Set (Run Ω S J))) := by
    intro P C tt
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | ¬ gateCovers level
          (countsFin {p | cutAccepts O.mq A₀ p ω} {p | O.mq p ω = 1} tt)
          ((D j).real {p | cutAccepts O.mq A₀ p ω})
          ((D j).real ({p | cutAccepts O.mq A₀ p ω} ∩ {p | O.mq p ω = 1})
            / (D j).real {p | cutAccepts O.mq A₀ p ω})
          ((D j).real ({p | cutAccepts O.mq A₀ p ω}ᶜ ∩ {p | O.mq p ω = 1})
            / (D j).real {p | cutAccepts O.mq A₀ p ω}ᶜ)})
        (fun A₀ => measurableSet_gateMissFixed O (D j) A₀ tt level))
    · exact MeasurableSet.empty
  have hrw : gateMiss rule O populations D j B level
      = {x : Run Ω S J | x ∈ (fun P C (tt : Fin B.npref → S) => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω : Ω | ¬ gateCovers level
            (countsFin {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}
              {p | O.mq p ω = 1} tt)
            ((D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω})
            ((D j).real ({p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}
                ∩ {p | O.mq p ω = 1})
              / (D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω})
            ((D j).real ({p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}ᶜ
                ∩ {p | O.mq p ω = 1})
              / (D j).real {p | cutAccepts O.mq (clusterOf rule O B.sc B.scd P C B.k ω) p ω}ᶜ)}
          else (∅ : Set (Run Ω S J)))
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

/-- At a fixed noise and table, the certification draws miss at most at four times the
level. -/
theorem gateMiss_le (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (B : State) (level : ℝ) (hl : 0 ≤ level) :
    (runMeasure μ D Dsf).real (gateMiss rule O populations D j B level) ≤ 4 * level := by
  classical
  have hE : runMeasure μ D Dsf (gateMiss rule O populations D j B level)
      ≤ ENNReal.ofReal (4 * level) := by
    refine runMeasure_slice_cert_le D Dsf _ (measurableSet_gateMiss rule O populations D j B
      level) _ (fun y => ?_)
    set F : Finset S := clusterBy rule O.mq populations
      ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hFdef
    set A : Set S := {p | cutAccepts O.mq F p y.1} with hA
    set Bs : Set S := {p | O.mq p y.1 = 1} with hBs
    have hpre : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ gateMiss rule O populations D j B level}
        = (fun c : J → ℕ → S => fun i : Fin B.npref => c j i) ⁻¹'
          {x | ¬ gateCovers level (countsFin A Bs x) ((D j).real A)
            ((D j).real (A ∩ Bs) / (D j).real A) ((D j).real (Aᶜ ∩ Bs) / (D j).real Aᶜ)} := by
      ext c
      rfl
    rw [hpre, ← Measure.map_apply (by fun_prop) ((Set.to_countable _).measurableSet),
      map_streamBlock D j B.npref, ← ENNReal.ofReal_toReal (measure_ne_top _ _)]
    exact ENNReal.ofReal_le_ofReal (pi_gate_miss_le (D j) A Bs B.npref level hl)
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) hE

lemma lookLevel_nonneg {α : ℝ} (hα : 0 ≤ α) (k : ℕ) : 0 ≤ lookLevel α k := by
  unfold lookLevel; positivity

/-- The looks' levels sum to `α`: `∑ 1/(k+1)² = π²/6`. -/
lemma hasSum_lookLevel (α : ℝ) : HasSum (fun k : ℕ => lookLevel α k) α := by
  have h0 : HasSum (fun n : ℕ => (1 : ℝ) / ((n : ℝ) + 1) ^ 2) (Real.pi ^ 2 / 6) := by
    have := (hasSum_nat_add_iff' 1).2 hasSum_zeta_two
    simpa using this
  have h1 := h0.mul_left (α * 6 / Real.pi ^ 2)
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  have hf : (fun k : ℕ => lookLevel α k)
      = fun i : ℕ => α * 6 / Real.pi ^ 2 * (1 / ((i : ℝ) + 1) ^ 2) := by
    funext k; unfold lookLevel; field_simp
  have hv : α * 6 / Real.pi ^ 2 * (Real.pi ^ 2 / 6) = α := by field_simp
  rw [hv] at h1
  rw [hf]
  exact h1

open scoped Classical in
/-- What an admitted family certifies: a run reaching any of `s`, its looks distinct, with a
family whose cut reads as misclassifying more than a population's limit, has chance at most
`α`. -/
theorem gate_valid (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (uni : J)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (η₀ indecisionLimit εcov α : ℝ) (hα : 0 ≤ α) (s : Finset State)
    (hlook : ∀ B ∈ s, ∀ B' ∈ s, B.look = B'.look → B = B') :
    (runMeasure μ D Dsf).real {x | ∃ B ∈ s,
        x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∧ ∃ j ∈ populations, missLimit uni j εcov
          < misShareOf η₀ O.mq (D j) (clusterBy rule O.mq populations x B) (oracleNoise x)}
      ≤ α := by
  classical
  set lev : State → ℝ := fun B => lookLevel α B.look / (4 * populations.card) with hlev
  have hlev0 : ∀ B, 0 ≤ lev B := fun B => by
    rw [hlev]; exact div_nonneg (lookLevel_nonneg hα _) (by positivity)
  have hsub : {x | ∃ B ∈ s,
        x ∈ retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
        ∧ ∃ j ∈ populations, missLimit uni j εcov
          < misShareOf η₀ O.mq (D j) (clusterBy rule O.mq populations x B) (oracleNoise x)}
      ⊆ ⋃ B ∈ s, ⋃ j ∈ populations, gateMiss rule O populations D j B (lev B) := by
    rintro x ⟨B, hB, hret, j, hj, hgt⟩
    simp only [Set.mem_iUnion]
    refine ⟨B, hB, j, hj, ?_⟩
    intro hcov
    set F := clusterBy rule O.mq populations x B with hF
    have hadm := hret.2.2 j hj
    have hcnt : splitCounts O.mq F j B.npref x
        = countsFin {p | cutAccepts O.mq F p (oracleNoise x)} {p | O.mq p (oracleNoise x) = 1}
          (fun i : Fin B.npref => certPrefix j i.val x) := by
      simp only [splitCounts, countsFin]
      congr 1 <;> congr 1 <;> rw [card_range_filter_fin] <;> congr 1 <;> ext i <;> simp
    rw [hcnt] at hadm
    have hle := misShare_le_of_covers hadm ⟨measureReal_nonneg, measureReal_le_one⟩
      (ratio_mem (D j) _ _) (ratio_mem (D j) _ _) hcov
    exact absurd hle (not_le.2 hgt)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ B ∈ s, (runMeasure μ D Dsf).real
      (⋃ j ∈ populations, gateMiss rule O populations D j B (lev B)) ≤ lookLevel α B.look := by
    intro B _
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    refine le_trans (Finset.sum_le_sum (fun j _ =>
      gateMiss_le rule O populations D Dsf j B (lev B) (hlev0 B))) ?_
    rw [Finset.sum_const, nsmul_eq_mul, hlev]
    rcases Nat.eq_zero_or_pos populations.card with h0 | hpos
    · rw [h0]; simp only [Nat.cast_zero, zero_mul]; exact lookLevel_nonneg hα _
    · have hc : (0 : ℝ) < populations.card := by exact_mod_cast hpos
      rw [show (populations.card : ℝ) * (4 * (lookLevel α B.look / (4 * populations.card)))
          = lookLevel α B.look by field_simp]
  refine le_trans (Finset.sum_le_sum hper) ?_
  rw [← Finset.sum_image (f := fun k => lookLevel α k) (g := State.look) hlook]
  exact sum_le_hasSum _ (fun k _ => lookLevel_nonneg hα k) (hasSum_lookLevel α)

end OrthoDFA
