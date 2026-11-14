import RLStar.Proofs.LoopTransfer
import RLStar.Proofs.FamilyRead
import RLStar.LowLevelClaims.FamilyOracle
import RLStar.LowLevelClaims.ClusteringQuality
import RLStar.LowLevelClaims.LazyRound

/-!
# The loop stops in agreement once its rounds meet their claims

A round's claims are the clustering's (`ClusterSpec`) and the lazy round's (`RoundSpec`), the
latter putting at most `A·T + sl` of a harvest's law, past `m`, on the states the round's family
leaves undecided at most `T` of the time. Every later family is held by the FNR gate to
`c = 2·indecisionLimit + slack` undecided on each earlier harvest's population, so a state with
more than `c/T` of that population is read undecided at most `T` of the time by every later
family (`good_of_heavy`). Where

    A·T + sl + m/y + |Q|·c/T < 1,

the population puts more than `|Q|·c/T` on the states read undecided more than `T` of the time,
so more than `c/T` on one of them: each harvest has such a state, never again read undecided more
than `T` of the time, and one of the first `|Q| + 1` rounds does not harvest.
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

/-- What the clustering claims of a round run after the samplers `pops` at its run `x`: it returns
at some state of the schedule, and at every state it could return at, its family miscuts `D` at
most `εcov + slack` of the time and leaves each population undecided at most
`2·indecisionLimit + slack`. -/
def ClusterSpec (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (εcov slack : ℝ) (pops : List (SpecSampler α))
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
            ∂(loopDistributions D settings.level3.walkStart pops j)
            ≤ 2 * settings.indecisionLimit + slack

lemma clusterSpec_alike (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) {L : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (εcov slack : ℝ) {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    ClusterSpec settings O D εcov slack pops x ↔ ClusterSpec settings O D εcov slack pops' x := by
  unfold ClusterSpec
  rw [hp.length_eq]
  simp only [loopDistributions_alike hp D hD settings.level3.walkStart]

/-- What the loop needs of a lazy round: it ends well, a harvest's law yielding `y`, putting at
most `pc` on a string and at most `A·T + sl` of itself, past `m`, on the states the round's family
leaves undecided at most `T` of the time, and a consistent round with at most `ε` of the probes in
`lazyDisagree`; and no epoch's tree passes `|Q| + 2` leaves. -/
def RoundSpec [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (A sl m pc y ε : ℝ) (k : ℕ) (o : SpecRound α) : Prop :=
  LooseEndsWell M (fun q => familyReadDist O M o.family o.state.lo (o.state.hi + 1) q .undecided)
      A sl m pc y P.D ε (Random.lazyDisagree o.read k o.result.expanded o.result.hypothesis)
      (Random.lazyHarvestLaw o.read P.D k o.result.expanded o.result.hypothesis) o.result.ending
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
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    (idealRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
      ∧ idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel (idealRound settings O D Dsf h d).state := by
  obtain ⟨B, hB, hr⟩ := hB
  exact Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
    (loopPopulations K h.pops.length) ∧ idealRun D Dsf settings.level3.walkStart
      settings.level3.harvestReadCap h (d.1, d.2.1) ∈ ret O.mq (loopPopulations K h.pops.length)
        none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    ⟨B, hB, hr⟩

lemma one_notMem_idealRound_family : 1 ∉ (idealRound settings O D Dsf h d).family :=
  Finset.notMem_erase _ _

/-- The clustering's claim, at the state the round returned at. -/
lemma idealRound_quality {D' : Measure (FreeMonoid α)} {εcov slack : ℝ}
    (hs : ClusterSpec settings O D' εcov slack h.pops
      (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h
        (d.1, d.2.1))) :
    (idealRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
    ∧ ∫ p, miscutProb O (idealRound settings O D Dsf h d).state.lo
          ((idealRound settings O D Dsf h d).state.hi + 1)
          (insert 1 (idealRound settings O D Dsf h d).family) p ∂D' ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K h.pops.length,
        ∫ p, undecidedProb O (idealRound settings O D Dsf h d).state.lo
            ((idealRound settings O D Dsf h d).state.hi + 1)
            (insert 1 (idealRound settings O D Dsf h d).family) p
          ∂(loopDistributions D' settings.level3.walkStart h.pops j)
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
    obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, o.sampler = some ℓ := by
      have ho := h o (by simp)
      unfold SpecRound.sampler
      revert ho
      cases o.result.ending <;> simp [Random.LazyEnd.IsHarvest]
    have ih' := ih fun o' ho' => h o' (by simp [ho'])
    simp only [specHistOf, List.filterMap_cons, hℓ, List.length_cons] at ih' ⊢
    rw [ih']

omit [MeasurableSpace Ω] in
lemma length_harvestsOf {D : Measure (FreeMonoid α)} {Kw : ℕ} {rounds : List (SpecRound α)}
    (h : AllHarvested rounds) : (harvestsOf D Kw rounds).length = rounds.length := by
  rw [← histOf_pops, samplerLaws_length, length_specHistOf_pops h]

/-! ## The loop stops in agreement -/

theorem loop_agrees_of_specs [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) (settings : LoopSettings K)
    (draws : Fin K → LoopRoundDraws Ω α settings) (k : ℕ) (hk : settings.level3.walkStart = k)
    (εcov slack A sl εd T : ℝ) (m pc y : ℝ) (hK : Fintype.card σ + 1 ≤ K) (hT : 0 < T)
    (hy : 0 < y) (hm0 : 0 ≤ m) (hc : 0 < 2 * settings.indecisionLimit + slack)
    (hcond : A * T + sl + m / y
      + Fintype.card σ * (2 * settings.indecisionLimit + slack) / T < 1)
    (hsched : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi)
    (hspec : ∀ i (hi : i < (idealRounds settings O P.D Dsf draws K).length) (hiK : i < K),
      i ≤ Fintype.card σ →
      ClusterSpec settings O P.D εcov slack
          (specHistOf ((idealRounds settings O P.D Dsf draws K).take i)).pops
          (idealRunAt settings O P.D Dsf draws ⟨i, hiK⟩)
        ∧ RoundSpec M O P A sl m pc y εd k (idealRounds settings O P.D Dsf draws K)[i]) :
    (idealRounds settings O P.D Dsf draws K).length ≤ Fintype.card σ + 1
      ∧ ∃ o ∈ (idealRounds settings O P.D Dsf draws K).getLast?,
        o.Checks M O P.D k (εcov + 2 * settings.indecisionLimit + 2 * slack)
          (2 * settings.indecisionLimit + slack) εd := by
  classical
  set L := idealRounds settings O P.D Dsf draws K with hLdef
  set c := 2 * settings.indecisionLimit + slack with hcd
  -- each round, as the clustering's claim has it
  have hround : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      L[i].state.lo ≤ L[i].state.hi ∧ 1 ∉ L[i].family
      ∧ ∫ p, miscutProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p ∂P.D
          ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K (harvestsOf P.D k (L.take i)).length,
        ∫ p, undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
          ∂(popDists P.D (harvestsOf P.D k (L.take i)) j) ≤ c := by
    intro i hi hiσ
    obtain ⟨hiK, e⟩ := getElem_idealRounds settings O P.D Dsf draws K i hi
    have hs := (hspec i hi hiK hiσ).1
    rw [idealRunAt, ← take_idealRounds settings O P.D Dsf draws hiK.le] at hs
    rw [e]
    obtain ⟨hmem, hq⟩ := idealRound_quality settings O P.D Dsf (specHistOf (L.take i))
      (draws ⟨i, hiK⟩) (D' := P.D) hs
    rw [hk] at hq
    rw [← samplerLaws_length P.D k (specHistOf (L.take i)).pops, histOf_pops] at hq
    simp only [loopDistributions_eq_popDists, histOf_pops] at hq
    exact ⟨hsched _ _ hmem, one_notMem_idealRound_family settings O P.D Dsf
      (specHistOf (L.take i)) (draws ⟨i, hiK⟩), hq⟩
  -- a round that does not harvest agrees
  have hagree : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      ¬ L[i].result.ending.IsHarvest →
      L[i].Checks M O P.D k (εcov + 2 * settings.indecisionLimit + 2 * slack) c εd := by
    intro i hi hiσ hn
    obtain ⟨hlh, h1, hmis, hund⟩ := hround i hi hiσ
    obtain ⟨h4, hpeak⟩ := (hspec i hi (by omega) hiσ).2
    have hcons : L[i].result.ending = .consistent ∧ P.D.real
        (Random.lazyDisagree L[i].read k L[i].result.expanded L[i].result.hypothesis) ≤ εd := by
      rcases hE : L[i].result.ending with _ | ⟨s, ℓ⟩ | _
      · rw [hE] at h4; exact ⟨rfl, h4⟩
      · rw [hE] at hn; exact absurd trivial hn
      · rw [hE] at h4; exact h4.elim
    have hnone : none ∈ loopPopulations K (harvestsOf P.D k (L.take i)).length := by
      simp [loopPopulations]
    have hund' := hund none hnone
    simp only [popDists] at hund'
    have hb := (integrable_of_unit P.D (fun p => miscutProb O L[i].state.lo
      (L[i].state.hi + 1) (insert 1 L[i].family) p) (fun _ => measureReal_nonneg)
      (fun _ => measureReal_le_one)).add (integrable_of_unit P.D (fun p => undecidedProb O
        L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p)
        (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
    refine ⟨⟨hcons.1, ?_, ?_⟩, hcons.2, hpeak⟩
    · calc ∫ p, misreadProb O L[i].family L[i].state.lo (L[i].state.hi + 1) p ∂P.D
          ≤ ∫ p, (miscutProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
              + undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p)
            ∂P.D :=
            integral_mono (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
              (fun _ => measureReal_le_one)) hb (fun p => misreadProb_le O h1 hlh p)
        _ = ∫ p, miscutProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p ∂P.D
            + ∫ p, undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
              ∂P.D :=
            integral_add (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
              (fun _ => measureReal_le_one)) (integrable_of_unit P.D _
                (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
        _ ≤ εcov + 2 * settings.indecisionLimit + 2 * slack := by linarith
    · calc ∫ p, readProb O L[i].family L[i].state.lo (L[i].state.hi + 1) p .undecided ∂P.D
          ≤ ∫ p, undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
            ∂P.D :=
            integral_mono (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
              (fun _ => measureReal_le_one)) (integrable_of_unit P.D _
                (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
              (fun p => readProb_undecided_le O h1 _ _ p)
        _ ≤ c := hund'
  -- some round among the first `|Q| + 1` does not harvest
  have hstop : ∃ i, ∃ hi : i < L.length, i ≤ Fintype.card σ
      ∧ ¬ L[i].result.ending.IsHarvest := by
    by_contra hno
    push Not at hno
    by_cases hlen : Fintype.card σ + 1 ≤ L.length
    · -- `|Q| + 1` harvests, each forcing a new state good
      have hall : ∀ j ≤ Fintype.card σ, AllHarvested (L.take j) := by
        intro j hj o ho
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
        rw [List.getElem_take]
        exact hno i (by simp at hi; omega) (by simp at hi; omega)
      have hzs : ∀ i (hi : i ≤ Fintype.card σ), ∃ s ℓ,
          (L[i]'(by omega)).result.ending = .harvest s ℓ := by
        intro i hi
        have h := hno i (by omega) hi
        revert h
        cases (L[i]'(by omega)).result.ending <;> simp [Random.LazyEnd.IsHarvest]
      choose s ℓ he using hzs
      set ν : ∀ i, i ≤ Fintype.card σ → Measure (FreeMonoid α) := fun i hi =>
        popLaw (Random.lazyHarvestLaw (L[i]'(by omega)).read P.D k
          (L[i]'(by omega)).result.expanded (L[i]'(by omega)).result.hypothesis (s i hi))
      have hpop : ∀ i (hi : i ≤ Fintype.card σ), (L[i]'(by omega)).population P.D k
          = some (ν i hi) :=
        fun i hi => by simp [SpecRound.population, he i hi, ν]
      have hQ : (0 : ℝ) < Fintype.card σ := by
        have : Nonempty σ := ⟨M.start⟩
        exact_mod_cast Fintype.card_pos
      have hbad : ∀ i (hi : i ≤ Fintype.card σ), ∃ q,
          ¬ familyReadDist O M (L[i]'(by omega)).family (L[i]'(by omega)).state.lo
            ((L[i]'(by omega)).state.hi + 1) q .undecided ≤ T
          ∧ c < T * (ν i hi).real {z | M.eval z.toList = q} := by
        intro i hi
        have h4 := (hspec i (by omega) (by omega) hi).2.1
        rw [he i hi] at h4
        set law := Random.lazyHarvestLaw (L[i]'(by omega)).read P.D k
          (L[i]'(by omega)).result.expanded (L[i]'(by omega)).result.hypothesis (s i hi)
        set U := fun q => familyReadDist O M (L[i]'(by omega)).family
          (L[i]'(by omega)).state.lo ((L[i]'(by omega)).state.hi + 1) q .undecided
        set bad := fun q => ¬ U q ≤ T
        have hlow : law.real {z | U (M.eval z.toList) ≤ T}
            ≤ (A * T + sl) * law.real Set.univ + m := h4.1 T hT.le
        have hyl : y ≤ law.real Set.univ := h4.2.2
        have htop : law Set.univ ≠ ⊤ := lazyHarvestLaw_univ_ne_top _ P.D k _ _ (s i hi)
        have hpos : 0 < law.real Set.univ := lt_of_lt_of_le hy hyl
        have h0 : law Set.univ ≠ 0 := fun h0 => by
          rw [measureReal_def, h0, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
        have hsplit : law.real {z | bad (M.eval z.toList)}
            = law.real Set.univ - law.real {z | U (M.eval z.toList) ≤ T} := by
          have hc' : {z : FreeMonoid α | bad (M.eval z.toList)}
              = {z | U (M.eval z.toList) ≤ T}ᶜ := by ext z; simp [bad]
          have hu := measureReal_union (μ := law) (disjoint_compl_right (a := {z : FreeMonoid α |
            U (M.eval z.toList) ≤ T})) (Set.to_countable _).measurableSet
            (measure_ne_top_of_subset (Set.subset_univ _) htop)
            (measure_ne_top_of_subset (Set.subset_univ _) htop)
          rw [Set.union_compl_self] at hu
          rw [hc']
          linarith
        have hmy : m / law.real Set.univ ≤ m / y := div_le_div_of_nonneg_left hm0 hy hyl
        have hpb : Fintype.card σ * c / T < (popLaw law).real {z | bad (M.eval z.toList)} := by
          rw [popLaw_real h0 htop, hsplit, sub_div, div_self hpos.ne']
          have : law.real {z | U (M.eval z.toList) ≤ T} / law.real Set.univ
              ≤ A * T + sl + m / law.real Set.univ := by
            rw [div_le_iff₀ hpos, add_mul, div_mul_cancel₀ _ hpos.ne']
            exact hlow
          linarith
        have hQc : 0 < Fintype.card σ * c := mul_pos hQ hc
        have hpb' : 1 < T / (Fintype.card σ * c) * (ν i hi).real {z | bad (M.eval z.toList)} := by
          change 1 < T / (Fintype.card σ * c) * (popLaw law).real {z | bad (M.eval z.toList)}
          calc (1 : ℝ) = T / (Fintype.card σ * c) * (Fintype.card σ * c / T) := by
                field_simp
            _ < _ := mul_lt_mul_of_pos_left hpb (by positivity)
        obtain ⟨q, hqb, hq⟩ := exists_heavy_state M bad (ν i hi) (by positivity) hpb'
        refine ⟨q, hqb, ?_⟩
        have e : T / (Fintype.card σ * c) * Fintype.card σ
              * (ν i hi).real {z | M.eval z.toList = q}
            = T * (ν i hi).real {z | M.eval z.toList = q} / c := by
          field_simp
        rw [e, lt_div_iff₀ hc] at hq
        linarith
      choose q hq using hbad
      have hgood : ∀ i j (hi : i ≤ Fintype.card σ) (hj : j ≤ Fintype.card σ), i < j →
          familyReadDist O M (L[j]'(by omega)).family (L[j]'(by omega)).state.lo
            ((L[j]'(by omega)).state.hi + 1) (q i hi) .undecided ≤ T := by
        intro i j hi hj hij
        obtain ⟨-, h1, -, hund⟩ := hround j (by omega) hj
        have hlenj : (harvestsOf P.D k (L.take j)).length = j := by
          rw [length_harvestsOf (hall j hj), List.length_take]; omega
        have hiK : i < K := by omega
        have hmem : some ⟨i, hiK⟩ ∈ loopPopulations K (harvestsOf P.D k (L.take j)).length := by
          simp [loopPopulations, hlenj, hij]
        have hg := hund _ hmem
        simp only [popDists] at hg
        rw [harvestsOf_take_getD P.D hij (by omega) (hall j hj) (hpop i hi)] at hg
        exact good_of_heavy M O hL h1 _ _ (ν i hi) (q i hi) (hq i hi).2 hg
      have hinj : Function.Injective fun i : Fin (Fintype.card σ + 1) =>
          q i.val (Nat.lt_succ_iff.1 i.isLt) := by
        intro a b hab
        simp only at hab
        by_contra hne
        rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hne) with h | h
        · have := hgood a b (Nat.lt_succ_iff.1 a.isLt) (Nat.lt_succ_iff.1 b.isLt) h
          rw [hab] at this
          exact (hq b _).1 this
        · have := hgood b a (Nat.lt_succ_iff.1 b.isLt) (Nat.lt_succ_iff.1 a.isLt) h
          rw [← hab] at this
          exact (hq a _).1 this
      have := Fintype.card_le_of_injective _ hinj
      simp at this
    · -- fewer than `|Q| + 1` rounds, so the last did not harvest
      have hst := idealRounds_stops settings O P.D Dsf draws
      rw [← hLdef] at hst
      rcases hst with h | ⟨o, ho, hn⟩
      · omega
      · rw [List.getLast?_eq_getElem?] at ho
        obtain ⟨hlt, rfl⟩ := List.getElem?_eq_some_iff.1 ho
        exact hn (hno _ hlt (by omega))
  obtain ⟨i, hi, hiσ, hn⟩ := hstop
  have hlast : i + 1 = L.length := by
    by_contra hne
    exact hn (harvested_of_lt settings O P.D Dsf draws K i (by rw [← hLdef]; omega))
  refine ⟨by omega, L[i], ?_, hagree i hi hiσ hn⟩
  rw [List.getLast?_eq_getElem?, ← hlast]
  simp

end OrthoDFA

end
