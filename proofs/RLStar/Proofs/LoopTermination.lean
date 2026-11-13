import RLStar.Proofs.LoopTransfer
import RLStar.Proofs.FamilyRead
import RLStar.LowLevelClaims.FamilyOracle
import RLStar.LowLevelClaims.ClusteringQuality
import RLStar.LowLevelClaims.LazyRound

/-!
# The loop stops in agreement once its rounds meet their claims

A round's claims are the clustering's (`ClusterSpec`) and the lazy round's (`RoundSpec`), the
latter putting at most `A·T + sl` of a harvest's list on the states the round's family leaves
undecided at most `T` of the time. Every later family is held by the FNR gate to
`c = 2·indecisionLimit + slack` undecided on each earlier harvest's list, so a state with more
than `c/T` of that list is read undecided at most `T` of the time by every later family
(`good_of_heavy`). Where

    A·T + sl + |Q|·c/T < 1,

the list puts more than `|Q|·c/T` on the states read undecided more than `T` of the time, so more
than `c/T` on one of them: each harvest has such a state, never again read undecided more than `T`
of the time, and one of the first `|Q| + 1` rounds does not harvest.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {σ : Type*}

open scoped Classical in
/-- Each state's read distribution under `F` cut at `kl` and `kh`, as `SuffixFamily.readDist`
has it. -/
def familyReadDist (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (q : σ) : ARU → ℝ :=
  if h : ∃ w : FreeMonoid α, M.eval w.toList = q then readProb O F kl kh h.choose else fun _ => 0

lemma readProb_eq_familyReadDist [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    (M : _root_.DFA α σ) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (w : FreeMonoid α) :
    readProb O F kl kh w = familyReadDist O M F kl kh (M.eval w.toList) := by
  classical
  have h : ∃ w' : FreeMonoid α, M.eval w'.toList = M.eval w.toList := ⟨w, rfl⟩
  simp only [familyReadDist, dif_pos h]
  funext rd
  have key : ∀ x v : FreeMonoid α,
      x * v ∈ O.L ↔ M.evalFrom (M.eval x.toList) v.toList ∈ M.accept := by
    intro x v
    rw [hL, FreeMonoid.toList_mul]
    simp only [DFA.eval, DFA.evalFrom_of_append]
  have e : acceptCount O.L F w = acceptCount O.L F h.choose := by
    unfold acceptCount
    congr 1
    refine Finset.filter_congr fun v _ => ?_
    rw [key, key, h.choose_spec]
  rw [readProb_eq_readLaw, readProb_eq_readLaw, e]

/-! ## The read against the clustering's cut -/

lemma readProb_undecided_le [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) (lo hi : ℕ) (z : FreeMonoid α) :
    readProb O F lo (hi + 1) z .undecided ≤ undecidedProb O lo (hi + 1) (insert 1 F) z := by
  refine measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)
  simp only [Set.mem_ofPred_eq, familyRead_eq, readOf] at hω ⊢
  rw [decided, voteCount_insert O.mq hF]
  split_ifs at hω ⊢ <;> omega

lemma misreadProb_le [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) {lo hi : ℕ} (hlh : lo ≤ hi) (p : FreeMonoid α) :
    misreadProb O F lo (hi + 1) p
      ≤ miscutProb O lo (hi + 1) (insert 1 F) p + undecidedProb O lo (hi + 1) (insert 1 F) p := by
  classical
  refine (measureReal_mono (s₂ := {ω | ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω}
    ∪ {ω | ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω}) (fun ω hω => ?_)
      (measure_ne_top _ _)).trans (measureReal_union_le _ _)
  simp only [Set.mem_ofPred_eq, familyRead_eq] at hω
  simp only [Set.mem_union, Set.mem_ofPred_eq, cutCorrect, decided]
  have hb := voteCount_insert O.mq hF p ω
  have hb01 : (if O.mq (p * 1) ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  set v := voteCount O.mq F p ω
  set vf := voteCount O.mq (insert 1 F) p ω
  by_cases hp : p ∈ O.L
  · have hl : O.label p = 1 := by simp [Oracle.label, hp]
    rw [if_pos hp] at hω
    have hv : v ≤ lo := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h2
    by_cases hlo : vf ≤ lo
    · left
      intro hcc
      have := hcc.2 hlo
      rw [hl] at this
      norm_num at this
    · right
      omega
  · have hl : O.label p = 0 := by simp [Oracle.label, hp]
    rw [if_neg hp] at hω
    have hv : hi + 1 ≤ v := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h1
    by_cases hhi : hi + 1 < vf
    · left
      intro hcc
      have := hcc.1 hhi
      rw [hl] at this
      norm_num at this
    · right
      omega

/-! ## Integrals -/

lemma integrable_of_unit (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν]
    (f : FreeMonoid α → ℝ) (h0 : ∀ p, 0 ≤ f p) (h1 : ∀ p, f p ≤ 1) : Integrable f ν :=
  Integrable.of_bound (measurable_of_countable f).aestronglyMeasurable 1
    (ae_of_all _ fun p => by rw [Real.norm_eq_abs, abs_of_nonneg (h0 p)]; exact h1 p)

/-! ## A population's heaviest bad state -/

open scoped Classical in
/-- More than `1/a` of a population at bad states puts more than `1/(a|Q|)` of it on one of
them. -/
lemma exists_heavy_state (M : _root_.DFA α σ) [Fintype σ] (bad : σ → Prop)
    (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] {a : ℝ} (ha : 0 ≤ a)
    (h : 1 < a * ν.real {z | bad (M.eval z.toList)}) :
    ∃ q, bad q ∧ 1 < a * Fintype.card σ * ν.real {z | M.eval z.toList = q} := by
  by_contra hno
  push Not at hno
  have hset : {z : FreeMonoid α | bad (M.eval z.toList)}
      = ⋃ q ∈ Finset.univ.filter bad, {z | M.eval z.toList = q} := by
    ext z; simp
  have hsum : ν.real {z : FreeMonoid α | bad (M.eval z.toList)}
      = ∑ q ∈ Finset.univ.filter bad, ν.real {z : FreeMonoid α | M.eval z.toList = q} := by
    rw [hset]
    refine measureReal_biUnion_finset (s := Finset.univ.filter bad)
      (f := fun q => {z : FreeMonoid α | M.eval z.toList = q}) ?_
      (fun q _ => measurableSet_of_countable _) (fun q _ => measure_ne_top _ _)
    intro q _ q' _ hqq'
    refine Set.disjoint_left.2 fun z h1 h2 => hqq' ?_
    simp only [Set.mem_setOf_eq] at h1 h2
    rw [← h1, ← h2]
  have hle : ∑ q ∈ Finset.univ.filter bad,
      a * Fintype.card σ * ν.real {z : FreeMonoid α | M.eval z.toList = q}
        ≤ ∑ _q ∈ Finset.univ.filter bad, (1 : ℝ) :=
    Finset.sum_le_sum fun q hq => hno q (Finset.mem_filter.1 hq).2
  rw [← Finset.mul_sum, ← hsum, Finset.sum_const, nsmul_eq_mul, mul_one] at hle
  have hcard : ((Finset.univ.filter bad).card : ℝ) ≤ Fintype.card σ := by
    exact_mod_cast Finset.card_filter_le _ _
  have hpos : (0 : ℝ) < Fintype.card σ := by
    rcases Nat.eq_zero_or_pos (Fintype.card σ) with h0 | h0
    · have : (Finset.univ.filter bad).card = 0 := by
        have := Finset.card_filter_le (Finset.univ : Finset σ) bad
        rw [Finset.card_univ] at this; omega
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff] at this
      have hz : ν.real {z : FreeMonoid α | bad (M.eval z.toList)} = 0 := by
        rw [hsum, Finset.sum_eq_zero]
        intro q hq
        exact absurd (Finset.mem_filter.1 hq).2 (this (Finset.mem_univ q))
      rw [hz, mul_zero] at h
      linarith
    · exact_mod_cast h0
  nlinarith

open scoped Classical in
/-- A state holding more than `c/T` of a population is read undecided at most `T` of the time by a
family the FNR gate holds to `c` undecided on that population. -/
lemma good_of_heavy [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) (lo hi : ℕ) (ν : Measure (FreeMonoid α))
    [IsProbabilityMeasure ν] (q : σ) {T c : ℝ}
    (hheavy : c < T * ν.real {z | M.eval z.toList = q})
    (hgate : ∫ p, undecidedProb O lo (hi + 1) (insert 1 F) p ∂ν ≤ c) :
    familyReadDist O M F lo (hi + 1) q .undecided ≤ T := by
  set u := familyReadDist O M F lo (hi + 1) q .undecided
  set Sq := {z : FreeMonoid α | M.eval z.toList = q}
  set g := fun p => undecidedProb O lo (hi + 1) (insert 1 F) p
  have hg0 : ∀ p, 0 ≤ g p := fun p => measureReal_nonneg
  have hlow : ∀ p, Sq.indicator (fun _ => u) p ≤ g p := by
    intro p
    by_cases hp : p ∈ Sq
    · rw [Set.indicator_of_mem hp]
      have := readProb_undecided_le O hF lo hi p
      rw [readProb_eq_familyReadDist O M hL, Set.mem_setOf_eq.mp hp] at this
      exact this
    · rw [Set.indicator_of_notMem hp]
      exact hg0 p
  have hint : u * ν.real Sq ≤ ∫ p, g p ∂ν := by
    calc u * ν.real Sq = ∫ p, Sq.indicator (fun _ => u) p ∂ν := by
          rw [integral_indicator_const _ (measurableSet_of_countable _), smul_eq_mul, mul_comm]
      _ ≤ ∫ p, g p ∂ν := integral_mono
          ((integrable_const u).indicator (measurableSet_of_countable _))
          (integrable_of_unit ν g hg0 fun _ => measureReal_le_one) hlow
  have hc0 : 0 ≤ c := le_trans (integral_nonneg hg0) hgate
  have hS0 : 0 ≤ ν.real Sq := measureReal_nonneg
  by_contra hbad
  push Not at hbad
  nlinarith

/-! ## The rounds' claims -/

variable {K : ℕ}

/-- What the clustering claims of a round run after the lists `pops` at its run `x`: it returns at
some state of the schedule, and at every state it could return at, its family miscuts `D` at most
`εcov + slack` of the time and leaves each population undecided at most
`2·indecisionLimit + slack`. -/
def ClusterSpec (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (εcov slack : ℝ) (pops : List (List (FreeMonoid α)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  (∃ B ∈ settings.schedule (loopPopulations K pops.length),
      x ∈ ret O.mq (loopPopulations K pops.length) none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B)
    ∧ ∀ B ∈ settings.schedule (loopPopulations K pops.length),
      x ∈ ret O.mq (loopPopulations K pops.length) none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B →
      ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq (loopPopulations K pops.length) x B) p
          ∂D ≤ εcov + slack
        ∧ ∀ j ∈ loopPopulations K pops.length,
          ∫ p, undecidedProb O B.lo (B.hi + 1)
              (familyAt O.mq (loopPopulations K pops.length) x B) p
            ∂(loopDistributions D pops j) ≤ 2 * settings.indecisionLimit + slack

/-- What the loop needs of a lazy round: it ends well, a harvest's list putting at most `A·T + sl`
on the states the round's family leaves undecided at most `T` of the time, and a consistent round
with at most `ε` of the probes in `lazyDisagree`; and no epoch's tree passes `|Q| + 2` leaves. -/
def RoundSpec [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (A sl ε : ℝ) (k : ℕ) (o : SpecRound α) : Prop :=
  LooseEndsWell M (fun q => familyReadDist O M o.family o.state.lo (o.state.hi + 1) q .undecided)
      A sl P.D ε (Random.lazyDisagree o.read k o.result.expanded o.result.hypothesis)
      o.result.ending
    ∧ o.result.peak ≤ Fintype.card σ + 2

/-- The round agrees, at most `εd` of `D` is in `lazyDisagree`, and no epoch's tree passes
`|Q| + 2` leaves. -/
def SpecRound.Checks [Fintype σ] (_M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (k : ℕ) (εr εu εd : ℝ) (o : SpecRound α) : Prop :=
  o.Agrees O D εr εu
    ∧ D.real (Random.lazyDisagree o.read k o.result.expanded o.result.hypothesis) ≤ εd
    ∧ o.result.peak ≤ Fintype.card σ + 2

/-! ## A round the clustering's claim holds of -/

section RoundFacts

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (h : SpecHist α)
  (d : LoopRoundDraws Ω α settings)

lemma idealRound_returns
    (hB : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf h (d.1, d.2.1) ∈ ret O.mq (loopPopulations K h.pops.length) none
        settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B) :
    (idealRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
      ∧ idealRun D Dsf h (d.1, d.2.1) ∈ ret O.mq (loopPopulations K h.pops.length) none
        settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel (idealRound settings O D Dsf h d).state := by
  obtain ⟨B, hB, hr⟩ := hB
  exact Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
    (loopPopulations K h.pops.length) ∧ idealRun D Dsf h (d.1, d.2.1) ∈ ret O.mq
      (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    ⟨B, hB, hr⟩

lemma one_notMem_idealRound_family : 1 ∉ (idealRound settings O D Dsf h d).family :=
  Finset.notMem_erase _ _

/-- The clustering's claim, at the state the round returned at. -/
lemma idealRound_quality {D' : Measure (FreeMonoid α)} {εcov slack : ℝ}
    (hs : ClusterSpec settings O D' εcov slack h.pops
      (idealRun D Dsf h (d.1, d.2.1))) :
    (idealRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
    ∧ ∫ p, miscutProb O (idealRound settings O D Dsf h d).state.lo
          ((idealRound settings O D Dsf h d).state.hi + 1)
          (insert 1 (idealRound settings O D Dsf h d).family) p ∂D' ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K h.pops.length,
        ∫ p, undecidedProb O (idealRound settings O D Dsf h d).state.lo
            ((idealRound settings O D Dsf h d).state.hi + 1)
            (insert 1 (idealRound settings O D Dsf h d).family) p
          ∂(loopDistributions D' h.pops j)
          ≤ 2 * settings.indecisionLimit + slack := by
  have hr := idealRound_returns settings O D Dsf h d hs.1
  exact ⟨hr.1, hs.2 _ hr.1 hr.2⟩

end RoundFacts

omit [MeasurableSpace Ω] in
lemma length_specHistOf_pops {rounds : List (SpecRound α)} (h : AllHarvested rounds) :
    (specHistOf rounds).pops.length = rounds.length := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, o.population = some ℓ := by
      have ho := h o (by simp)
      unfold SpecRound.population
      revert ho
      cases o.result.ending <;> simp [Random.LazyEnd.IsHarvest]
    have ih' := ih fun o' ho' => h o' (by simp [ho'])
    simp only [specHistOf, List.filterMap_cons, hℓ, List.length_cons] at ih' ⊢
    rw [ih']

/-! ## The loop stops in agreement -/

theorem loop_agrees_of_specs [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) (settings : LoopSettings K)
    (draws : Fin K → LoopRoundDraws Ω α settings) (k : ℕ) (εcov slack A sl εd T : ℝ)
    (hK : Fintype.card σ + 1 ≤ K) (hT : 0 < T)
    (hc : 0 < 2 * settings.indecisionLimit + slack)
    (hcond : A * T + sl + Fintype.card σ * (2 * settings.indecisionLimit + slack) / T < 1)
    (hsched : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi)
    (hspec : ∀ i (hi : i < (idealRounds settings O P.D Dsf draws K).length) (hiK : i < K),
      i ≤ Fintype.card σ →
      ClusterSpec settings O P.D εcov slack
          (specHistOf ((idealRounds settings O P.D Dsf draws K).take i)).pops
          (idealRunAt settings O P.D Dsf draws ⟨i, hiK⟩)
        ∧ RoundSpec M O P A sl εd k (idealRounds settings O P.D Dsf draws K)[i]) :
    (idealRounds settings O P.D Dsf draws K).length ≤ Fintype.card σ + 1
      ∧ ∃ o ∈ (idealRounds settings O P.D Dsf draws K).getLast?,
        o.Checks M O P.D k (εcov + 2 * settings.indecisionLimit + 2 * slack)
          (2 * settings.indecisionLimit + slack) εd := by
  sorry -- closes by: main's proof on listLaw, exists_heavy_state, good_of_heavy

end OrthoDFA

end
