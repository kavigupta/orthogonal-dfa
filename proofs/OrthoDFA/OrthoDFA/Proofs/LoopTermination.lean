import OrthoDFA.Proofs.LoopTransfer
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.Level5Law
import OrthoDFA.Proofs.SiftSet

/-!
# The loop stops in agreement once its rounds meet their claims

A round's claims are the clustering's (`ClusterSpec`) and level 5's (`Level5Spec`), the latter's
bad states those the round's family leaves undecided more than `1.5θ` of the time. A harvest
whose sampler finds more than a quarter of what it finds at bad states hands on a population more
than `1/(4|Q|)` of which is at one of them. Every later family is held by the FNR gate to
`2·indecisionLimit + slack` undecided on that population, which at

    1.5θ = 4|Q|·(2·indecisionLimit + slack)

that state would exceed were it still bad. So each harvest makes a new state good for every later
family, and one of the first `|Q| + 1` rounds does not harvest.
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
  simp only [Set.mem_ofPred_eq, familyRead, readOf] at hω ⊢
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
  simp only [Set.mem_ofPred_eq, familyRead] at hω
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
/-- A state holding more than `1/(4|Q|)` of a population is read undecided at most `1.5θ` of the
time by a family the FNR gate holds to `c` undecided on that population, where
`1.5θ = 4|Q|·c`. -/
lemma good_of_heavy [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) (lo hi : ℕ) (ν : Measure (FreeMonoid α))
    [IsProbabilityMeasure ν] (q : σ) {θ c : ℝ} (hθ : 3 / 2 * θ = 4 * Fintype.card σ * c)
    (hheavy : 1 < 4 * Fintype.card σ * ν.real {z | M.eval z.toList = q})
    (hgate : ∫ p, undecidedProb O lo (hi + 1) (insert 1 F) p ∂ν ≤ c) :
    familyReadDist O M F lo (hi + 1) q .undecided ≤ 3 / 2 * θ := by
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
  by_contra hbad
  push Not at hbad
  rw [hθ] at hbad
  have hc0 : 0 ≤ c := le_trans (integral_nonneg hg0) hgate
  have hQ0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
  have hu : 0 < u := lt_of_le_of_lt (by positivity) hbad
  have h1 := mul_lt_mul_of_pos_left hheavy hu
  nlinarith

/-! ## The rounds' claims -/

variable {K : ℕ}

/-- What the clustering claims of a round run after `harvests`, at its run `x`: it returns at some
state of the schedule, and at every state it could return at, its family miscuts `D` at most
`εcov + slack` of the time and leaves each population undecided at most
`2·indecisionLimit + slack`. -/
def ClusterSpec (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (εcov slack : ℝ) (harvests : List (Measure (FreeMonoid α)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  (∃ B ∈ settings.schedule (loopPopulations K harvests.length),
      x ∈ ret O.mq (loopPopulations K harvests.length) none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    ∧ ∀ B ∈ settings.schedule (loopPopulations K harvests.length),
      x ∈ ret O.mq (loopPopulations K harvests.length) none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B →
      ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq (loopPopulations K harvests.length) x B) p
          ∂D ≤ εcov + slack
        ∧ ∀ j ∈ loopPopulations K harvests.length,
          ∫ p, undecidedProb O B.lo (B.hi + 1)
              (familyAt O.mq (loopPopulations K harvests.length) x B) p
            ∂(loopDistributions D harvests j) ≤ 2 * settings.indecisionLimit + slack

/-- What the loop needs of level 5 at a round: it ends well, a quarter of its harvest's sampler at
bad states and the sampler yielding `y`, a state being bad where the round's family leaves it
undecided more than `1.5θ` of the time, and a consistent round checking at most `εd` of the
probes, those whose start or end sift stops undecided among them where `sift`. -/
def Level5Spec [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (θ εd y : ℝ) (k J : ℕ) (ū : ℝ) (sift : Bool) (o : LoopRound Ω α K) : Prop :=
  let dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1)
  QuarterEndsWell M (fun q => ¬ dist q .undecided ≤ 3 / 2 * θ) P.D εd
    (checkedProbes M dist P.D o.read k (Fintype.card σ + 2) J ū o.result.hypothesis
      ∪ siftedProbes sift o.read k o.result.hypothesis) k
    (harvestLaw P.D (fun z => (o.read z).cut) k o.result.hypothesis) y o.result.ending

/-- The round agrees, and its probes checked against the family's read distributions, with those
whose start or end sift stops undecided where `sift`, are at most `εd` of `D`. -/
def LoopRound.Checks [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (k J : ℕ) (ū : ℝ) (sift : Bool) (εr εu εd : ℝ)
    (o : LoopRound Ω α K) : Prop :=
  o.Agrees O D k εr εu εd
    ∧ D.real (checkedProbes M (familyReadDist O M o.family o.state.lo (o.state.hi + 1)) D o.read
      k (Fintype.card σ + 2) J ū o.result.hypothesis ∪ siftedProbes sift o.read k
        o.result.hypothesis) ≤ εd

/-- The goodness level the loop runs level 5's claim at. -/
def loopTheta (σ : Type*) [Fintype σ] (indecisionLimit slack : ℝ) : ℝ :=
  8 * Fintype.card σ * (2 * indecisionLimit + slack) / 3

/-! ## A round the clustering's claim holds of -/

section RoundFacts

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (h : LoopHist α)
  (d : LoopRoundDraws Ω α K settings.level5.probeBudget)

lemma loopRound_returns
    (hB : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      clusterRun D Dsf h (d.1, d.2.1) ∈ ret O.mq (loopPopulations K h.pops.length) none
        settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B) :
    (loopRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
      ∧ (loopRound settings O D Dsf h d).run ∈ ret O.mq (loopPopulations K h.pops.length) none
        settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel (loopRound settings O D Dsf h d).state := by
  obtain ⟨B, hB, hr⟩ := hB
  exact Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
    (loopPopulations K h.pops.length) ∧ clusterRun D Dsf h (d.1, d.2.1) ∈ ret O.mq
      (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    ⟨B, hB, hr⟩

lemma one_notMem_loopRound_family : 1 ∉ (loopRound settings O D Dsf h d).family :=
  Finset.notMem_erase _ _

/-- The clustering's claim, at the state the round returned at. -/
lemma loopRound_quality {D' : Measure (FreeMonoid α)} {εcov slack : ℝ}
    (hs : ClusterSpec settings O D' εcov slack h.pops (loopRound settings O D Dsf h d).run) :
    (loopRound settings O D Dsf h d).state ∈ settings.schedule (loopPopulations K h.pops.length)
    ∧ ∫ p, miscutProb O (loopRound settings O D Dsf h d).state.lo
          ((loopRound settings O D Dsf h d).state.hi + 1)
          (insert 1 (loopRound settings O D Dsf h d).family) p ∂D' ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K h.pops.length,
        ∫ p, undecidedProb O (loopRound settings O D Dsf h d).state.lo
            ((loopRound settings O D Dsf h d).state.hi + 1)
            (insert 1 (loopRound settings O D Dsf h d).family) p
          ∂(loopDistributions D' h.pops j) ≤ 2 * settings.indecisionLimit + slack := by
  have hr := loopRound_returns settings O D Dsf h d hs.1
  exact ⟨hr.1, hs.2 _ hr.1 hr.2⟩

end RoundFacts

omit [MeasurableSpace Ω] in
lemma length_harvestsOf {D : Measure (FreeMonoid α)} {k : ℕ} {rounds : List (LoopRound Ω α K)}
    (h : AllHarvested rounds) : (harvestsOf D k rounds).length = rounds.length := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    obtain ⟨zo, hzo⟩ := Option.ne_none_iff_exists'.1 (h o (by simp))
    have hpo : o.population D k
        = some (popLaw (harvestLaw D (fun z => (o.read z).cut) k o.result.hypothesis zo)) := by
      simp [LoopRound.population, hzo]
    simp only [harvestsOf, List.filterMap_cons, hpo, List.length_cons] at ih ⊢
    rw [ih fun o' ho' => h o' (by simp [ho'])]

/-! ## The loop stops in agreement -/

theorem loop_agrees_of_specs [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) (settings : LoopSettings K)
    (draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget) (k J : ℕ) (ū : ℝ)
    (sift : Bool) (hk : settings.level5.walkStart = k) (εcov slack εd y : ℝ) (hK : Fintype.card σ + 1 ≤ K)
    (hsched : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi)
    (hspec : ∀ i (hi : i < (loopRounds settings O P.D Dsf draws K).length), i ≤ Fintype.card σ →
      ClusterSpec settings O P.D εcov slack
          (harvestsOf P.D k ((loopRounds settings O P.D Dsf draws K).take i))
          (loopRounds settings O P.D Dsf draws K)[i].run
        ∧ Level5Spec M O P (loopTheta σ settings.indecisionLimit slack) εd y k J ū sift
          (loopRounds settings O P.D Dsf draws K)[i]) :
    (indecisiveLStarLoop settings O P.D Dsf draws).length ≤ Fintype.card σ + 1
      ∧ ∃ o ∈ (indecisiveLStarLoop settings O P.D Dsf draws).getLast?,
        o.Checks M O P.D k J ū sift (εcov + 2 * settings.indecisionLimit + 2 * slack)
          (2 * settings.indecisionLimit + slack) εd := by
  classical
  unfold indecisiveLStarLoop
  set L := loopRounds settings O P.D Dsf draws K with hLdef
  set c := 2 * settings.indecisionLimit + slack with hc
  have hθ : 3 / 2 * loopTheta σ settings.indecisionLimit slack = 4 * Fintype.card σ * c := by
    rw [loopTheta, hc]; ring
  -- each round, as the clustering's claim has it
  have hround : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      L[i].state.lo ≤ L[i].state.hi ∧ 1 ∉ L[i].family
      ∧ ∫ p, miscutProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p ∂P.D
          ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K (harvestsOf P.D k (L.take i)).length,
        ∫ p, undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
          ∂(loopDistributions P.D (harvestsOf P.D k (L.take i)) j) ≤ c := by
    intro i hi hiσ
    obtain ⟨hiK, e⟩ := getElem_loopRounds settings O P.D Dsf draws K i hi
    have hs := (hspec i hi hiσ).1
    rw [hk] at e
    rw [e] at hs ⊢
    obtain ⟨hmem, hq⟩ := loopRound_quality settings O P.D Dsf (histOf P.D k (L.take i))
      (draws ⟨i, hiK⟩) (D' := P.D) (εcov := εcov) (slack := slack) hs
    exact ⟨hsched _ _ hmem, one_notMem_loopRound_family settings O P.D Dsf
      (histOf P.D k (L.take i)) (draws ⟨i, hiK⟩), hq⟩
  -- a round that does not harvest agrees
  have hagree : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      L[i].result.ending.harvested = none →
      L[i].Checks M O P.D k J ū sift (εcov + 2 * settings.indecisionLimit + 2 * slack) c εd := by
    intro i hi hiσ hn
    obtain ⟨hlh, h1, hmis, hund⟩ := hround i hi hiσ
    have h4 := (hspec i hi hiσ).2
    simp only [Level5Spec] at h4
    have hcons : L[i].result.ending = .consistent ∧
        P.D.real (checkedProbes M (familyReadDist O M L[i].family L[i].state.lo
          (L[i].state.hi + 1)) P.D L[i].read k (Fintype.card σ + 2) J ū
            L[i].result.hypothesis ∪ siftedProbes sift L[i].read k L[i].result.hypothesis) ≤ εd := by
      rcases hE : L[i].result.ending with _ | zs | _
      · rw [hE] at h4; exact ⟨rfl, h4⟩
      · rw [hE] at hn; simp [RoundEnd.harvested] at hn
      · rw [hE] at h4; exact absurd h4 (by simp [QuarterEndsWell])
    have hnone : none ∈ loopPopulations K (harvestsOf P.D k (L.take i)).length := by
      simp [loopPopulations]
    have hund' := hund none hnone
    simp only [loopDistributions] at hund'
    have hb := (integrable_of_unit P.D (fun p => miscutProb O L[i].state.lo
      (L[i].state.hi + 1) (insert 1 L[i].family) p) (fun _ => measureReal_nonneg)
      (fun _ => measureReal_le_one)).add (integrable_of_unit P.D (fun p => undecidedProb O
        L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p)
        (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
    have hsrch : P.D.real (searchingProbes L[i].read k L[i].result.hypothesis) ≤ εd :=
      (measureReal_mono (fun x hx => .inl (.inl (.inl hx))) (measure_ne_top _ _)).trans hcons.2
    refine ⟨⟨hcons.1, ?_, ?_, hsrch⟩, hcons.2⟩
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
      ∧ L[i].result.ending.harvested = none := by
    by_contra hno
    push Not at hno
    by_cases hlen : Fintype.card σ + 1 ≤ L.length
    · -- `|Q| + 1` harvests, each forcing a new state good
      have hall : ∀ j ≤ Fintype.card σ, AllHarvested (L.take j) := by
        intro j hj o ho
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
        rw [List.getElem_take]
        exact hno i (by simp at hi; omega) (by simp at hi; omega)
      have hzs : ∀ i (hi : i ≤ Fintype.card σ), ∃ h,
          (L[i]'(by omega)).result.ending.harvested = some h :=
        fun i hi => Option.ne_none_iff_exists'.1 (hno i (by omega) hi)
      choose hv hzs using hzs
      -- the population each harvest hands on
      set ν : ∀ i, i ≤ Fintype.card σ → Measure (FreeMonoid α) := fun i hi =>
        popLaw (harvestLaw P.D (fun z => ((L[i]'(by omega)).read z).cut) k
          (L[i]'(by omega)).result.hypothesis (hv i hi))
      have hpop : ∀ i (hi : i ≤ Fintype.card σ), (L[i]'(by omega)).population P.D k
          = some (ν i hi) := by
        intro i hi
        unfold LoopRound.population
        rw [hzs i hi]
        rfl
      have hbad : ∀ i (hi : i ≤ Fintype.card σ), ∃ q,
          ¬ familyReadDist O M (L[i]'(by omega)).family (L[i]'(by omega)).state.lo
            ((L[i]'(by omega)).state.hi + 1) q .undecided
              ≤ 3 / 2 * loopTheta σ settings.indecisionLimit slack
          ∧ 1 < 4 * Fintype.card σ * (ν i hi).real {z | M.eval z.toList = q} := by
        intro i hi
        have h4 := (hspec i (by omega) hi).2
        simp only [Level5Spec] at h4
        have he : (L[i]'(by omega)).result.ending = .harvest (hv i hi) := by
          have := hzs i hi
          revert this
          cases (L[i]'(by omega)).result.ending <;> simp [RoundEnd.harvested]
        rw [he] at h4
        set law := harvestLaw P.D (fun z => ((L[i]'(by omega)).read z).cut) k
          (L[i]'(by omega)).result.hypothesis (hv i hi)
        set bad := fun q => ¬ familyReadDist O M (L[i]'(by omega)).family
          (L[i]'(by omega)).state.lo ((L[i]'(by omega)).state.hi + 1) q .undecided
            ≤ 3 / 2 * loopTheta σ settings.indecisionLimit slack
        have hmost : law.real Set.univ < 4 * law.real {z | bad (M.eval z.toList)} := h4.1
        have htop : law Set.univ ≠ ⊤ :=
          harvestLaw_univ_ne_top P.D _ k P.length_le _ (hv i hi)
        have hle : law.real {z | bad (M.eval z.toList)} ≤ law.real Set.univ :=
          measureReal_mono (Set.subset_univ _) htop
        have hpos : 0 < law.real Set.univ := by
          have : 0 ≤ law.real {z | bad (M.eval z.toList)} := measureReal_nonneg
          linarith
        have h0 : law Set.univ ≠ 0 := fun h0 => by
          rw [measureReal_def, h0, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
        have hpb : 1 < 4 * (ν i hi).real {z | bad (M.eval z.toList)} := by
          change 1 < 4 * (popLaw law).real {z | bad (M.eval z.toList)}
          rw [popLaw_real h0 htop, mul_div_assoc', lt_div_iff₀ hpos]
          linarith
        exact exists_heavy_state M bad (ν i hi) (by norm_num) hpb
      choose q hq using hbad
      have hgood : ∀ i j (hi : i ≤ Fintype.card σ) (hj : j ≤ Fintype.card σ), i < j →
          familyReadDist O M (L[j]'(by omega)).family (L[j]'(by omega)).state.lo
            ((L[j]'(by omega)).state.hi + 1) (q i hi) .undecided
              ≤ 3 / 2 * loopTheta σ settings.indecisionLimit slack := by
        intro i j hi hj hij
        obtain ⟨-, h1, -, hund⟩ := hround j (by omega) hj
        have hlenj : (harvestsOf P.D k (L.take j)).length = j := by
          rw [length_harvestsOf (hall j hj), List.length_take]; omega
        have hiK : i < K := by omega
        have hmem : some ⟨i, hiK⟩ ∈ loopPopulations K (harvestsOf P.D k (L.take j)).length := by
          simp [loopPopulations, hlenj, hij]
        have hg := hund _ hmem
        simp only [loopDistributions] at hg
        rw [harvestsOf_take_getD P.D hij (by omega) (hall j hj) (hpop i hi)] at hg
        exact good_of_heavy M O hL h1 _ _ (ν i hi) (q i hi) hθ (hq i hi).2 hg
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
      have hst := loopRounds_stops settings O P.D Dsf draws
      rw [← hLdef] at hst
      rcases hst with h | ⟨o, ho, hn⟩
      · omega
      · rw [List.getLast?_eq_getElem?] at ho
        obtain ⟨hlt, rfl⟩ := List.getElem?_eq_some_iff.1 ho
        exact hno _ hlt (by omega) hn
  obtain ⟨i, hi, hiσ, hn⟩ := hstop
  have hlast : i + 1 = L.length := by
    by_contra hne
    exact harvested_of_lt settings O P.D Dsf draws K i (by rw [← hLdef]; omega) hn
  refine ⟨by omega, L[i], ?_, hagree i hi hiσ hn⟩
  rw [List.getLast?_eq_getElem?, ← hlast]
  simp

end OrthoDFA

end
