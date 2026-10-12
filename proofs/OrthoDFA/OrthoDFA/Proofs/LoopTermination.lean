import OrthoDFA.Proofs.LoopRounds
import OrthoDFA.Proofs.FamilyRead

/-!
# The loop stops in agreement once its rounds meet their claims

A round's claims are the clustering's (`ClusterSpec`) and level 4's (`Level4Spec`), the latter's
bad states those the round's family leaves undecided more than `1.5θ` of the time. A harvest more
than half of whose strings are at bad states puts more than `1/(2|Q|)` of itself on one of them.
Every later family is held by the FNR gate to `2·indecisionLimit + slack` undecided on that
harvest, which at

    1.5θ = 2|Q|·(2·indecisionLimit + slack)

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

lemma integral_list_dirac (f : FreeMonoid α → ℝ) (zs : List (FreeMonoid α)) :
    Integrable f (zs.map Measure.dirac).sum
      ∧ ∫ p, f p ∂(zs.map Measure.dirac).sum = (zs.map f).sum := by
  induction zs with
  | nil => simp
  | cons z zs ih =>
    have hz : Integrable f (Measure.dirac z) := integrable_dirac enorm_lt_top
    simp only [List.map_cons, List.sum_cons]
    exact ⟨hz.add_measure ih.1, by rw [integral_add_measure hz ih.1, integral_dirac, ih.2]⟩

lemma integral_harvestMeasure (f : FreeMonoid α → ℝ) {zs : List (FreeMonoid α)} (hzs : zs ≠ []) :
    ∫ p, f p ∂(harvestMeasure zs) = (zs.map f).sum / zs.length := by
  rw [harvestMeasure, if_neg hzs, integral_smul_measure, (integral_list_dirac f zs).2]
  simp [ENNReal.toReal_inv, div_eq_inv_mul]

/-! ## A harvest's heaviest bad state -/

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
lemma sum_count_eq (M : _root_.DFA α σ) [Fintype σ] (bad : σ → Prop) (zs : List (FreeMonoid α)) :
    ∑ q ∈ Finset.univ.filter bad, (zs.filter fun z => M.eval z.toList = q).length
      = (zs.filter fun z => bad (M.eval z.toList)).length := by
  induction zs with
  | nil => simp
  | cons z zs ih =>
    have e : ∀ q, (List.filter (fun x => decide (M.eval x.toList = q)) (z :: zs)).length
        = (List.filter (fun x => decide (M.eval x.toList = q)) zs).length
          + if M.eval z.toList = q then 1 else 0 := by
      intro q
      by_cases h : M.eval z.toList = q <;> simp [h]
    simp_rw [e, Finset.sum_add_distrib, ih]
    rw [Finset.sum_ite_eq]
    by_cases h : bad (M.eval z.toList) <;> simp [h]

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
/-- More than half of `zs` at bad states puts more than `1/(2|Q|)` of it on one of them. -/
lemma exists_heavy_state (M : _root_.DFA α σ) [Fintype σ] (bad : σ → Prop)
    (zs : List (FreeMonoid α))
    (h : zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length) :
    ∃ q, bad q
      ∧ zs.length < 2 * Fintype.card σ * (zs.filter fun z => M.eval z.toList = q).length := by
  by_contra hno
  push Not at hno
  have hsum := sum_count_eq M bad zs
  have hle : ∑ q ∈ Finset.univ.filter bad,
      2 * Fintype.card σ * (zs.filter fun z => M.eval z.toList = q).length
        ≤ ∑ _q ∈ Finset.univ.filter bad, zs.length :=
    Finset.sum_le_sum fun q hq => hno q (Finset.mem_filter.1 hq).2
  rw [← Finset.mul_sum, hsum, Finset.sum_const, smul_eq_mul] at hle
  have hcard : (Finset.univ.filter bad).card ≤ Fintype.card σ := Finset.card_filter_le _ _
  have hpos : 0 < Fintype.card σ := by
    obtain ⟨z, hz⟩ : ∃ z, z ∈ zs.filter fun z => bad (M.eval z.toList) :=
      List.exists_mem_of_length_pos (by omega)
    exact Fintype.card_pos_iff.2 ⟨M.eval z.toList⟩
  have := Nat.mul_le_mul_right zs.length hcard
  nlinarith

open scoped Classical in
/-- A state holding more than `1/(2|Q|)` of a harvest is read undecided at most `1.5θ` of the time
by a family the FNR gate holds to `c` undecided on that harvest, where `1.5θ = 2|Q|·c`. -/
lemma good_of_heavy [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    {F : Finset (FreeMonoid α)} (hF : 1 ∉ F) (lo hi : ℕ) (zs : List (FreeMonoid α)) (q : σ)
    {θ c : ℝ} (hθ : 3 / 2 * θ = 2 * Fintype.card σ * c)
    (hheavy : zs.length < 2 * Fintype.card σ * (zs.filter fun z => M.eval z.toList = q).length)
    (hgate : ∫ p, undecidedProb O lo (hi + 1) (insert 1 F) p ∂(harvestMeasure zs) ≤ c) :
    familyReadDist O M F lo (hi + 1) q .undecided ≤ 3 / 2 * θ := by
  have hzs : zs ≠ [] := by rintro rfl; simp at hheavy
  set zq := zs.filter fun z => M.eval z.toList = q with hzq
  set u := familyReadDist O M F lo (hi + 1) q .undecided with hu
  set g := fun p => undecidedProb O lo (hi + 1) (insert 1 F) p with hg
  have hlen : (0 : ℝ) < zs.length := by
    have := List.length_pos_of_ne_nil hzs
    exact_mod_cast this
  have hg0 : ∀ p, 0 ≤ g p := fun p => measureReal_nonneg
  have hint := integral_harvestMeasure g hzs
  rw [hint] at hgate
  have hsub : (zq.map g).sum ≤ (zs.map g).sum :=
    ((List.filter_sublist).map g).sum_le_sum (fun a ha => by
      obtain ⟨b, -, rfl⟩ := List.mem_map.1 ha; exact hg0 b)
  have hzqg : ∀ z ∈ zq, u ≤ g z := by
    intro z hz
    have hzq' : M.eval z.toList = q := by simpa [hzq] using (List.mem_filter.1 hz).2
    have := readProb_undecided_le O hF lo hi z
    rw [readProb_eq_familyReadDist O M hL, hzq'] at this
    exact this
  have hlow : (zq.length : ℝ) * u ≤ (zq.map g).sum := by
    have := List.card_nsmul_le_sum (zq.map g) u (fun x hx => by
      obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hx; exact hzqg b hb)
    simpa [nsmul_eq_mul] using this
  by_contra hbad
  push Not at hbad
  have hheavy' : (zs.length : ℝ) < 2 * Fintype.card σ * zq.length := by exact_mod_cast hheavy
  have hzq0 : (0 : ℝ) < zq.length := by
    have : 0 < zq.length := by
      rcases Nat.eq_zero_or_pos zq.length with h0 | h0
      · rw [h0] at hheavy; simp at hheavy
      · exact h0
    exact_mod_cast this
  rw [div_le_iff₀ hlen] at hgate
  by_cases hc : 0 ≤ c
  · have h1 : (zq.length : ℝ) * (3 / 2 * θ) < zq.length * u := mul_lt_mul_of_pos_left hbad hzq0
    rw [hθ] at h1
    have h2 : c * zs.length ≤ c * (2 * Fintype.card σ * zq.length) :=
      mul_le_mul_of_nonneg_left hheavy'.le hc
    nlinarith
  · have : 0 ≤ (zs.map g).sum := List.sum_nonneg (fun x hx => by
      obtain ⟨b, -, rfl⟩ := List.mem_map.1 hx; exact hg0 b)
    nlinarith

/-! ## The rounds' claims -/

variable {K : ℕ}

/-- What the clustering claims of a round run after `harvests`, at its run `x`: it returns at some
state of the schedule, and at every state it could return at, its family miscuts `D` at most
`εcov + slack` of the time and leaves each population undecided at most
`2·indecisionLimit + slack`. -/
def ClusterSpec (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (εcov slack : ℝ) (harvests : List (List (FreeMonoid α)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  (∃ B ∈ settings.schedule (loopPopulations K harvests.length),
      x ∈ ret O.mq (loopPopulations K harvests.length) none settings.indecisionLimit
        settings.gateLevel B)
    ∧ ∀ B ∈ settings.schedule (loopPopulations K harvests.length),
      x ∈ ret O.mq (loopPopulations K harvests.length) none settings.indecisionLimit
        settings.gateLevel B →
      ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq (loopPopulations K harvests.length) x B) p
          ∂D ≤ εcov + slack
        ∧ ∀ j ∈ loopPopulations K harvests.length,
          ∫ p, undecidedProb O B.lo (B.hi + 1)
              (familyAt O.mq (loopPopulations K harvests.length) x B) p
            ∂(loopDistributions D harvests j) ≤ 2 * settings.indecisionLimit + slack

/-- What level 4 claims of a round: it ends well, a state being bad where the round's family
leaves it undecided more than `1.5θ` of the time. -/
def Level4Spec (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (θ δ εd : ℝ) (k : ℕ) (o : LoopRound Ω α K) : Prop :=
  let dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1)
  let V := undecidedCount P M dist
  EndsWell M (fun q => ¬ dist q .undecided ≤ 3 / 2 * θ) P.D εd
    (searchingProbes o.read k o.result.hypothesis) P.firstAnchor P.L
    (level4Replay o.read o.result.hypothesis)
    ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ))) / (P.L + 1 - P.firstAnchor : ℕ))
    o.result.ending

/-- The goodness level the loop runs level 4's claim at. -/
def loopTheta (σ : Type*) [Fintype σ] (indecisionLimit slack : ℝ) : ℝ :=
  4 * Fintype.card σ * (2 * indecisionLimit + slack) / 3

/-! ## A round the clustering's claim holds of -/

section RoundFacts

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (harvests : List (List (FreeMonoid α))) (d : LoopRoundDraws Ω α K settings.level4.probeBudget)

lemma loopRound_returns
    (h : ∃ B ∈ settings.schedule (loopPopulations K harvests.length),
      clusterRun harvests d.1 ∈ ret O.mq (loopPopulations K harvests.length) none
        settings.indecisionLimit settings.gateLevel B) :
    (loopRound settings O harvests d).state ∈ settings.schedule (loopPopulations K harvests.length)
      ∧ (loopRound settings O harvests d).run ∈ ret O.mq (loopPopulations K harvests.length) none
        settings.indecisionLimit settings.gateLevel (loopRound settings O harvests d).state := by
  obtain ⟨B, hB, hr⟩ := h
  exact Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
    (loopPopulations K harvests.length) ∧ clusterRun harvests d.1 ∈ ret O.mq
      (loopPopulations K harvests.length) none settings.indecisionLimit settings.gateLevel B)
    ⟨B, hB, hr⟩

lemma one_notMem_loopRound_family : 1 ∉ (loopRound settings O harvests d).family :=
  Finset.notMem_erase _ _

/-- The clustering's claim, at the state the round returned at. -/
lemma loopRound_quality {D : Measure (FreeMonoid α)} {εcov slack : ℝ}
    (h : ClusterSpec settings O D εcov slack harvests (loopRound settings O harvests d).run) :
    (loopRound settings O harvests d).state ∈ settings.schedule (loopPopulations K harvests.length)
    ∧ ∫ p, miscutProb O (loopRound settings O harvests d).state.lo
          ((loopRound settings O harvests d).state.hi + 1)
          (insert 1 (loopRound settings O harvests d).family) p ∂D ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K harvests.length,
        ∫ p, undecidedProb O (loopRound settings O harvests d).state.lo
            ((loopRound settings O harvests d).state.hi + 1)
            (insert 1 (loopRound settings O harvests d).family) p
          ∂(loopDistributions D harvests j) ≤ 2 * settings.indecisionLimit + slack := by
  have hr := loopRound_returns settings O harvests d h.1
  exact ⟨hr.1, h.2 _ hr.1 hr.2⟩

end RoundFacts

omit [Fintype α] [DecidableEq α] [MeasurableSpace Ω] in
lemma length_harvestsOf {rounds : List (LoopRound Ω α K)} (h : AllHarvested rounds) :
    (harvestsOf rounds).length = rounds.length := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    obtain ⟨zo, hzo⟩ := Option.ne_none_iff_exists'.1 (h o (by simp))
    simp only [harvestsOf, List.filterMap_cons, hzo, List.length_cons] at ih ⊢
    rw [ih fun o' ho' => h o' (by simp [ho'])]

/-! ## The loop stops in agreement -/

theorem loop_agrees_of_specs [IsProbabilityMeasure μ] [Fintype σ] (M : _root_.DFA α σ)
    (O : Oracle μ (FreeMonoid α)) (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (settings : LoopSettings K)
    (draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget) (k : ℕ)
    (εcov slack εd δ : ℝ) (hK : Fintype.card σ + 1 ≤ K)
    (hsched : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi)
    (hspec : ∀ i (hi : i < (loopRounds settings O draws K).length), i ≤ Fintype.card σ →
      ClusterSpec settings O P.D εcov slack (harvestsOf ((loopRounds settings O draws K).take i))
          (loopRounds settings O draws K)[i].run
        ∧ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) δ εd k
          (loopRounds settings O draws K)[i]) :
    (indecisiveLStarLoop settings O draws).length ≤ Fintype.card σ + 1
      ∧ ∃ o ∈ (indecisiveLStarLoop settings O draws).getLast?,
        o.Agrees O P.D k (εcov + 2 * settings.indecisionLimit + 2 * slack)
          (2 * settings.indecisionLimit + slack) εd := by
  classical
  unfold indecisiveLStarLoop
  set L := loopRounds settings O draws K with hLdef
  set c := 2 * settings.indecisionLimit + slack with hc
  have hθ : 3 / 2 * loopTheta σ settings.indecisionLimit slack = 2 * Fintype.card σ * c := by
    rw [loopTheta, hc]; ring
  -- each round, as the clustering's claim has it
  have hround : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      L[i].state.lo ≤ L[i].state.hi ∧ 1 ∉ L[i].family
      ∧ ∫ p, miscutProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p ∂P.D
          ≤ εcov + slack
      ∧ ∀ j ∈ loopPopulations K (harvestsOf (L.take i)).length,
        ∫ p, undecidedProb O L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p
          ∂(loopDistributions P.D (harvestsOf (L.take i)) j) ≤ c := by
    intro i hi hiσ
    obtain ⟨hiK, e⟩ := getElem_loopRounds settings O draws K i hi
    have hs := (hspec i hi hiσ).1
    rw [e] at hs ⊢
    obtain ⟨hmem, hq⟩ := loopRound_quality settings O _ _ hs
    exact ⟨hsched _ _ hmem, one_notMem_loopRound_family settings O _ _, hq⟩
  -- a round that does not harvest agrees
  have hagree : ∀ i (hi : i < L.length), i ≤ Fintype.card σ →
      L[i].result.ending.harvested = none →
      L[i].Agrees O P.D k (εcov + 2 * settings.indecisionLimit + 2 * slack) c εd := by
    intro i hi hiσ hn
    obtain ⟨hlh, h1, hmis, hund⟩ := hround i hi hiσ
    have h4 := (hspec i hi hiσ).2
    simp only [Level4Spec] at h4
    have hcons : L[i].result.ending = .consistent ∧
        P.D.real (searchingProbes L[i].read k L[i].result.hypothesis) ≤ εd := by
      rcases hE : L[i].result.ending with _ | zs | _
      · rw [hE] at h4; exact ⟨rfl, h4⟩
      · rw [hE] at hn; simp [RoundEnd.harvested] at hn
      · rw [hE] at h4; exact absurd h4 (by simp [EndsWell])
    have hnone : none ∈ loopPopulations K (harvestsOf (L.take i)).length := by
      simp [loopPopulations]
    have hund' := hund none hnone
    simp only [loopDistributions] at hund'
    have hb := (integrable_of_unit P.D (fun p => miscutProb O L[i].state.lo
      (L[i].state.hi + 1) (insert 1 L[i].family) p) (fun _ => measureReal_nonneg)
      (fun _ => measureReal_le_one)).add (integrable_of_unit P.D (fun p => undecidedProb O
        L[i].state.lo (L[i].state.hi + 1) (insert 1 L[i].family) p)
        (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
    refine ⟨hcons.1, ?_, ?_, hcons.2⟩
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
      have hzs : ∀ i (hi : i ≤ Fintype.card σ), ∃ zs,
          (L[i]'(by omega)).result.ending.harvested = some zs :=
        fun i hi => Option.ne_none_iff_exists'.1 (hno i (by omega) hi)
      choose zs hzs using hzs
      have hbad : ∀ i (hi : i ≤ Fintype.card σ), ∃ q,
          ¬ familyReadDist O M (L[i]'(by omega)).family (L[i]'(by omega)).state.lo
            ((L[i]'(by omega)).state.hi + 1) q .undecided
              ≤ 3 / 2 * loopTheta σ settings.indecisionLimit slack
          ∧ (zs i hi).length < 2 * Fintype.card σ
            * ((zs i hi).filter fun z => M.eval z.toList = q).length := by
        intro i hi
        have h4 := (hspec i (by omega) hi).2
        simp only [Level4Spec] at h4
        have he : (L[i]'(by omega)).result.ending = .harvest (zs i hi) := by
          have := hzs i hi
          revert this
          cases (L[i]'(by omega)).result.ending <;> simp [RoundEnd.harvested]
        rw [he] at h4
        exact exists_heavy_state M _ _ h4.1
      choose q hq using hbad
      have hgood : ∀ i j (hi : i ≤ Fintype.card σ) (hj : j ≤ Fintype.card σ), i < j →
          familyReadDist O M (L[j]'(by omega)).family (L[j]'(by omega)).state.lo
            ((L[j]'(by omega)).state.hi + 1) (q i hi) .undecided
              ≤ 3 / 2 * loopTheta σ settings.indecisionLimit slack := by
        intro i j hi hj hij
        obtain ⟨-, h1, -, hund⟩ := hround j (by omega) hj
        have hlenj : (harvestsOf (L.take j)).length = j := by
          rw [length_harvestsOf (hall j hj), List.length_take]; omega
        have hiK : i < K := by omega
        have hmem : some ⟨i, hiK⟩ ∈ loopPopulations K (harvestsOf (L.take j)).length := by
          simp [loopPopulations, hlenj, hij]
        have hg := hund _ hmem
        simp only [loopDistributions] at hg
        rw [harvestsOf_take_getD hij (by omega) (hall j hj) (hzs i hi)] at hg
        exact good_of_heavy M O hL h1 _ _ (zs i hi) (q i hi) hθ (hq i hi).2 hg
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
      have hst := loopRounds_stops settings O draws
      rw [← hLdef] at hst
      rcases hst with h | ⟨o, ho, hn⟩
      · omega
      · rw [List.getLast?_eq_getElem?] at ho
        obtain ⟨hlt, rfl⟩ := List.getElem?_eq_some_iff.1 ho
        exact hno _ hlt (by omega) hn
  obtain ⟨i, hi, hiσ, hn⟩ := hstop
  have hlast : i + 1 = L.length := by
    by_contra hne
    exact harvested_of_lt settings O draws K i (by rw [← hLdef]; omega) hn
  refine ⟨by omega, L[i], ?_, hagree i hi hiσ hn⟩
  rw [List.getLast?_eq_getElem?, ← hlast]
  simp

end OrthoDFA

end
