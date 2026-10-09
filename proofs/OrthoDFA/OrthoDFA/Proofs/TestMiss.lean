import OrthoDFA.Proofs.SplitPower
import OrthoDFA.Proofs.FreshSelect

/-!
# A split test on fresh strings misses with chance at most `e^{−τ²}`

The test's counted strings, sided by their members, with sides whose mean bits differ by enough.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Real

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

open scoped Classical in
/-- The counted strings on the side `b`. -/
noncomputable def sideOf (ts : List (FreeMonoid α × Bool)) (b : Bool) : Finset (FreeMonoid α) :=
  ((ts.filter fun p => p.2 = b).map Prod.fst).toFinset

/-- The mean of the oracle's answers over `S`. -/
noncomputable def meanBit (O : Oracle μ (FreeMonoid α)) (S : Finset (FreeMonoid α)) : ℝ :=
  (∑ w ∈ S, μ[O.mq w]) / S.card

/-- `2ab/(a + b)`. -/
noncomputable def harm (a b : ℕ) : ℝ := 2 * a * b / (a + b)

/-- The two-sample statistic of the answers at `ω` on the sides of `ts`. -/
noncomputable def sideStat (O : Oracle μ (FreeMonoid α)) (ts : List (FreeMonoid α × Bool))
    (ω : Ω) : ℝ :=
  harm (sideOf ts true).card (sideOf ts false).card
    * ((∑ w ∈ sideOf ts true, O.mq w ω) / (sideOf ts true).card
      - (∑ w ∈ sideOf ts false, O.mq w ω) / (sideOf ts false).card) ^ 2

/-- The test of the sides of `ts` at threshold `θ`, with both sides read, misses by a margin `τ`
of signal: `√H` times the gap of the sides' mean answers is at least `√θ + τ`. -/
def PowerCase (O : Oracle μ (FreeMonoid α)) (θ τ : ℝ) (ts : List (FreeMonoid α × Bool)) :
    Prop :=
  (sideOf ts true).Nonempty ∧ (sideOf ts false).Nonempty
    ∧ √θ + τ ≤ √(harm (sideOf ts true).card (sideOf ts false).card)
      * (meanBit O (sideOf ts true) - meanBit O (sideOf ts false))

/-- The answers on the strings of `ts` fall short of `θ`. -/
def Misses (O : Oracle μ (FreeMonoid α)) (θ : ℝ) (ts : List (FreeMonoid α × Bool)) : Set Ω :=
  {ω | sideStat O ts ω < θ}

/-- Every string of `ts` appears once. -/
def SidesApart (ts : List (FreeMonoid α × Bool)) : Prop := (ts.map Prod.fst).Nodup

theorem sides_disjoint {ts : List (FreeMonoid α × Bool)} (h : SidesApart ts) :
    Disjoint (sideOf ts true) (sideOf ts false) := by
  classical
  rw [Finset.disjoint_left]
  intro w hw hw'
  simp only [sideOf, List.mem_toFinset, List.mem_map, List.mem_filter, decide_eq_true_eq] at hw hw'
  obtain ⟨p, ⟨hp, hp2⟩, rfl⟩ := hw
  obtain ⟨q, ⟨hq, hq2⟩, he⟩ := hw'
  have := List.inj_on_of_nodup_map h hq hp he
  rw [this, hp2] at hq2
  exact Bool.noConfusion hq2

theorem mq_measurable_noiseAlg [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {S : Set (FreeMonoid α)}
    {w : FreeMonoid α} (hw : w ∈ S) : Measurable[noiseAlg O S] (O.mq w) := by
  have hn : Measurable[noiseAlg O S] (O.noise w) := fun s hs =>
    measurableSet_noise_preimage O hw hs
  exact measurable_const.add (measurable_const.mul hn)

theorem misses_measurable [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (θ : ℝ)
    (ts : List (FreeMonoid α × Bool)) :
    MeasurableSet[noiseAlg O ↑((ts.map Prod.fst).toFinset)] (Misses O θ ts) := by
  classical
  have hm : ∀ b, ∀ w ∈ sideOf ts b,
      Measurable[noiseAlg O ↑((ts.map Prod.fst).toFinset)] (O.mq w) := by
    intro b w hw
    refine mq_measurable_noiseAlg O ?_
    simp only [sideOf, List.mem_toFinset, List.mem_map, List.mem_filter] at hw
    obtain ⟨p, ⟨hp, -⟩, rfl⟩ := hw
    exact List.mem_toFinset.2 (List.mem_map.2 ⟨p, hp, rfl⟩)
  have hs : ∀ b, Measurable[noiseAlg O ↑((ts.map Prod.fst).toFinset)]
      fun ω => ∑ w ∈ sideOf ts b, O.mq w ω := fun b =>
    Finset.measurable_sum _ fun w hw => hm b w hw
  have : Measurable[noiseAlg O ↑((ts.map Prod.fst).toFinset)] (sideStat O ts) :=
    measurable_const.mul ((((hs true).div_const _).sub ((hs false).div_const _)).pow_const 2)
  exact measurableSet_lt this measurable_const

theorem misses_le [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) {θ τ : ℝ}
    {ts : List (FreeMonoid α × Bool)} (hts : SidesApart ts) (hτ : 0 ≤ τ)
    (hc : PowerCase O θ τ ts) : μ.real (Misses O θ ts) ≤ exp (-τ ^ 2) := by
  obtain ⟨ha, hb, hgap⟩ := hc
  set a := sideOf ts true
  set b := sideOf ts false
  set H := harm a.card b.card
  have hna : (0 : ℝ) < a.card := by exact_mod_cast ha.card_pos
  have hnb : (0 : ℝ) < b.card := by exact_mod_cast hb.card_pos
  have hθ : √θ ≤ √H * (meanBit O a - meanBit O b) := by linarith
  have hpow := two_sample_power (μ := μ) (fun w => O.mq w) a b (meanBit O a) (meanBit O b) θ
    (fun w => (mq_meas O w).aemeasurable) (mq_indep O) (fun w => mq_icc O w)
    (sides_disjoint hts) ha hb
    (by unfold meanBit; field_simp; exact le_rfl)
    (by unfold meanBit; field_simp; exact le_rfl) hθ
  refine (le_of_eq ?_).trans (hpow.trans ?_)
  · rfl
  · apply Real.exp_le_exp.2
    have h1 : τ ≤ √(2 * (a.card : ℝ) * b.card / (a.card + b.card))
        * (meanBit O a - meanBit O b) - √θ := by
      have : H = 2 * (a.card : ℝ) * b.card / (a.card + b.card) := rfl
      rw [← this]; linarith
    exact neg_le_neg (pow_le_pow_left₀ hτ h1 2)

end OrthoDFA
