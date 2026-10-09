import OrthoDFA.FamilyRead
import OrthoDFA.Proofs.Adaptive

/-!
# The family's read: the proofs

A string's vote reads the oracle's bits at `w·F` and no others.  Suffix-freeness keeps those
blocks disjoint across strings, which is the independence.  Within one string the bits are
independent at two rates, so the vote is Poisson-binomial and its law is fixed by how many of
`w·F` lie in the language.  The shipped parameters' laws are computed exactly in integers.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

open scoped Classical in
noncomputable def Oracle.acceptRate (O : Oracle μ S) (u : S) : ℝ :=
  if u ∈ O.L then 1 - O.ηIn else O.ηOut

lemma measureReal_mq_one (O : Oracle μ S) (u : S) :
    μ.real {ω | O.mq u ω = 1} = O.acceptRate u := by
  rw [measureReal_mq_eq_one, mq_mean]
  unfold Oracle.rate Oracle.label Oracle.acceptRate
  by_cases h : u ∈ O.L
  · rw [if_pos h, if_pos h, Set.indicator_of_mem h, Pi.one_apply]; ring
  · rw [if_neg h, if_neg h, Set.indicator_of_notMem h]; ring

open scoped Classical in
/-- How many of `w·F` lie in `L`. -/
noncomputable def acceptCount (L : Set S) (F : Finset S) (w : S) : ℕ :=
  (F.filter (fun v => w * v ∈ L)).card

lemma acceptCount_le (L : Set S) (F : Finset S) (w : S) : acceptCount L F w ≤ F.card := by
  classical
  unfold acceptCount
  exact Finset.card_filter_le _ _

lemma measurable_voteCount_block (O : Oracle μ S) (F : Finset S) (w : S) {T : Set S}
    (hT : ∀ v ∈ F, w * v ∈ T) :
    Measurable[noiseAlg O T] (fun ω => voteCount O.mq F w ω) := by
  classical
  have hfun : (fun ω => voteCount O.mq F w ω)
      = fun ω => ∑ v ∈ F, (if O.mq (w * v) ω = 1 then 1 else 0) := by
    funext ω; unfold voteCount; rw [Finset.card_filter]
  rw [hfun]
  exact Finset.measurable_sum _ (fun v hv =>
    Measurable.ite (measurableSet_mq_eq_one O (hT v hv)) measurable_const measurable_const)

omit [MeasurableSpace Ω] in
lemma voteCount_le_card (mq : S → Ω → ℝ) (F : Finset S) (w : S) (ω : Ω) :
    voteCount mq F w ω ≤ F.card :=
  Finset.card_filter_le _ _

/-! ## Independence across strings -/

lemma disjoint_block_of_suffixFree {α : Type*} [Countable α] {F : Finset (FreeMonoid α)}
    (hF : SuffixFree F) {w w' : FreeMonoid α} (h : w ≠ w') :
    Disjoint (F.image (w * ·)) (F.image (w' * ·)) := by
  rw [Finset.disjoint_left]
  intro u hu hu'
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hu
  obtain ⟨v', hv', he⟩ := Finset.mem_image.1 hu'
  have hl : w'.toList ++ v'.toList = w.toList ++ v.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul]
    exact congrArg FreeMonoid.toList he
  have h1 : v.toList <:+ w.toList ++ v.toList := List.suffix_append _ _
  have h2 : v'.toList <:+ w.toList ++ v.toList := hl ▸ List.suffix_append _ _
  have hvv : v = v' := by
    rcases List.suffix_or_suffix_of_suffix h1 h2 with hs | hs
    · exact hF v hv v' hv' hs
    · exact (hF v' hv' v hv hs).symm
  subst hvv
  exact h (mul_right_cancel he).symm

theorem vote_iIndep {α : Type*} [Countable α] (O : Oracle μ (FreeMonoid α))
    {F : Finset (FreeMonoid α)} (hF : SuffixFree F) :
    iIndepFun (fun w ω => voteCount O.mq F w ω) μ := by
  have hle : ∀ u, MeasurableSpace.comap (O.noise u) (inferInstance : MeasurableSpace ℝ)
      ≤ ‹MeasurableSpace Ω› := fun u => (O.noise_meas u).comap_le
  have hgroup := iIndep_biSup_of_disjoint hle
    ((iIndepFun_iff_iIndep (fun _ => (inferInstance : MeasurableSpace ℝ)) O.noise μ).1
      O.noise_indep)
    (fun w => F.image (w * ·)) (fun w w' h => disjoint_block_of_suffixFree hF h)
  refine (iIndepFun_iff_iIndep (fun _ => (inferInstance : MeasurableSpace ℕ)) _ μ).2 ?_
  refine iIndep_of_iIndep_of_le hgroup (fun w => ?_)
  have hsup : (⨆ u ∈ F.image (w * ·), MeasurableSpace.comap (O.noise u) inferInstance)
      = noiseAlg O ↑(F.image (w * ·)) := by
    unfold noiseAlg
    exact iSup_congr (fun u => by simp)
  rw [hsup]
  exact (measurable_voteCount_block O F w
    (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv))).comap_le

theorem read_iIndep {α : Type*} [Countable α] (O : Oracle μ (FreeMonoid α))
    {F : Finset (FreeMonoid α)} (hF : SuffixFree F) (kl kh : ℕ) :
    iIndepFun (fun w => familyRead O.mq F kl kh w) μ :=
  (vote_iIndep O hF).comp (fun _ => readOf kl kh) (fun _ => measurable_from_nat)

/-! ## The vote's law -/

omit [MeasurableSpace Ω] in
lemma voteCount_insert (mq : S → Ω → ℝ) {F : Finset S} {x : S} (hx : x ∉ F) (w : S)
    (ω : Ω) :
    voteCount mq (insert x F) w ω = voteCount mq F w ω + if mq (w * x) ω = 1 then 1 else 0 := by
  unfold voteCount
  rw [Finset.filter_insert]
  split_ifs with h
  · rw [Finset.card_insert_of_notMem (fun h' => hx (Finset.mem_filter.1 h').1)]
  · rfl

omit [IsProbabilityMeasure μ] in
lemma measureReal_inter_of_indep {m₁ m₂ : MeasurableSpace Ω} (h : Indep m₁ m₂ μ)
    {s t : Set Ω} (hs : MeasurableSet[m₁] s) (ht : MeasurableSet[m₂] t) :
    μ.real (s ∩ t) = μ.real s * μ.real t := by
  rw [measureReal_def, (Indep_iff _ _ _).1 h s t hs ht, ENNReal.toReal_mul]
  rfl

lemma vote_prob_foldr (O : Oracle μ S) (w : S) (F : Finset S) (k : ℕ) :
    μ.real {ω | voteCount O.mq F w ω = k}
      = (F.val.map (fun v => O.acceptRate (w * v))).foldr pbStep pbBase k := by
  classical
  induction F using Finset.induction_on generalizing k with
  | empty =>
    simp only [Finset.empty_val, Multiset.map_zero, Multiset.foldr_zero, pbBase]
    have h0 : ∀ ω, voteCount O.mq (∅ : Finset S) w ω = 0 := fun ω => by simp [voteCount]
    simp only [h0]
    split_ifs with hk
    · subst hk; simp
    · have : {ω : Ω | 0 = k} = ∅ := by ext ω; simp [Ne.symm hk]
      rw [this, measureReal_empty]
  | insert x F hx ih =>
    rw [Finset.insert_val_of_notMem hx, Multiset.map_cons, Multiset.foldr_cons]
    set B : Set Ω := {ω | O.mq (w * x) ω = 1} with hBdef
    have hT : Disjoint (↑(F.image (w * ·)) : Set S) {w * x} := by
      rw [Set.disjoint_singleton_right, Finset.mem_coe, Finset.mem_image]
      rintro ⟨v, hv, he⟩
      exact hx (mul_left_cancel he ▸ hv)
    have hind := indep_noiseAlg O hT
    have hE : ∀ j, MeasurableSet[noiseAlg O ↑(F.image (w * ·))]
        {ω | voteCount O.mq F w ω = j} := fun j =>
      measurable_voteCount_block O F w
        (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv)) (measurableSet_singleton j)
    have hB : MeasurableSet[noiseAlg O {w * x}] B :=
      measurableSet_mq_eq_one O (Set.mem_singleton _)
    have hBm : MeasurableSet B := noiseAlg_le O _ _ hB
    have hpB : μ.real B = O.acceptRate (w * x) := measureReal_mq_one O (w * x)
    have hpBc : μ.real Bᶜ = 1 - O.acceptRate (w * x) := by
      rw [measureReal_compl hBm, probReal_univ, hpB]
    have hEB : ∀ j, μ.real ({ω | voteCount O.mq F w ω = j} ∩ B)
        = (F.val.map (fun v => O.acceptRate (w * v))).foldr pbStep pbBase j
          * O.acceptRate (w * x) := by
      intro j; rw [measureReal_inter_of_indep hind (hE j) hB, ih j, hpB]
    have hEBc : ∀ j, μ.real ({ω | voteCount O.mq F w ω = j} ∩ Bᶜ)
        = (F.val.map (fun v => O.acceptRate (w * v))).foldr pbStep pbBase j
          * (1 - O.acceptRate (w * x)) := by
      intro j; rw [measureReal_inter_of_indep hind (hE j) hB.compl, ih j, hpBc]
    cases k with
    | zero =>
      have hset : {ω | voteCount O.mq (insert x F) w ω = 0}
          = {ω | voteCount O.mq F w ω = 0} ∩ Bᶜ := by
        ext ω
        simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_compl_iff, hBdef,
          voteCount_insert O.mq hx]
        by_cases hb : O.mq (w * x) ω = 1 <;> simp [hb]
      rw [hset, hEBc 0]
      simp only [pbStep]
      ring
    | succ k =>
      have hset : {ω | voteCount O.mq (insert x F) w ω = k + 1}
          = ({ω | voteCount O.mq F w ω = k + 1} ∩ Bᶜ) ∪ ({ω | voteCount O.mq F w ω = k} ∩ B) := by
        ext ω
        simp only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_inter_iff, Set.mem_compl_iff, hBdef,
          voteCount_insert O.mq hx]
        by_cases hb : O.mq (w * x) ω = 1 <;> simp [hb]
      have hdisj : Disjoint ({ω | voteCount O.mq F w ω = k + 1} ∩ Bᶜ)
          ({ω | voteCount O.mq F w ω = k} ∩ B) :=
        Set.disjoint_left.2 (fun ω h1 h2 => h1.2 h2.2)
      rw [hset, measureReal_union hdisj ((noiseAlg_le O _ _ (hE k)).inter hBm), hEBc, hEB]
      simp only [pbStep]
      ring

omit [IsProbabilityMeasure μ] in
open scoped Classical in
lemma map_acceptRate (O : Oracle μ S) (F : Finset S) (w : S) :
    F.val.map (fun v => O.acceptRate (w * v))
      = Multiset.replicate (acceptCount O.L F w) (1 - O.ηIn)
        + Multiset.replicate (F.card - acceptCount O.L F w) O.ηOut := by
  have hsplit := Multiset.filter_add_not (fun v => w * v ∈ O.L) F.val
  have hc : F.card - acceptCount O.L F w = (F.filter (fun v => ¬ w * v ∈ O.L)).card := by
    have := Finset.card_filter_add_card_filter_not (s := F) (fun v => w * v ∈ O.L)
    unfold acceptCount
    omega
  conv_lhs => rw [← hsplit]
  rw [Multiset.map_add, hc]
  congr 1
  · rw [Multiset.map_congr rfl (fun v hv => by
      rw [Oracle.acceptRate, if_pos (Multiset.of_mem_filter hv)]), Multiset.map_const']
    rfl
  · rw [Multiset.map_congr rfl (fun v hv => by
      rw [Oracle.acceptRate, if_neg (Multiset.of_mem_filter (p := fun a => w * a ∉ O.L) hv)]),
      Multiset.map_const']
    rfl

theorem vote_prob (O : Oracle μ S) (F : Finset S) (w : S) (k : ℕ) :
    μ.real {ω | voteCount O.mq F w ω = k}
      = voteLaw F.card (acceptCount O.L F w) (1 - O.ηIn) O.ηOut k := by
  rw [vote_prob_foldr, map_acceptRate]
  rfl

theorem readProb_eq_readLaw (O : Oracle μ S) (F : Finset S) (kl kh : ℕ) (w : S) (rd : Read) :
    readProb O F kl kh w rd
      = readLaw F.card (acceptCount O.L F w) (1 - O.ηIn) O.ηOut kl kh rd := by
  classical
  have hset : {ω | familyRead O.mq F kl kh w ω = rd}
      = ⋃ j ∈ (Finset.range (F.card + 1)).filter (fun j => readOf kl kh j = rd),
          {ω | voteCount O.mq F w ω = j} := by
    ext ω
    simp only [familyRead, Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_filter,
      Finset.mem_range, exists_prop]
    constructor
    · intro h
      exact ⟨_, ⟨Nat.lt_succ_of_le (voteCount_le_card _ _ _ _), h⟩, rfl⟩
    · rintro ⟨j, ⟨_, hj⟩, hω⟩
      rw [hω]; exact hj
  rw [readProb, hset, measureReal_biUnion_finset (f := fun j => {ω | voteCount O.mq F w ω = j})
    (fun i _ j _ hij => Set.disjoint_left.2 (fun ω h1 h2 => hij (h1.symm.trans h2)))
    (fun j _ => measurable_voteCount O F w (measurableSet_singleton j)),
    readLaw, Finset.sum_filter]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  split_ifs
  · rw [vote_prob]
  · rfl

/-! ## Same law within a DFA state -/

omit [IsProbabilityMeasure μ] in
open scoped Classical in
lemma acceptCount_eq_state {α σ : Type*} [Countable α] (O : Oracle μ (FreeMonoid α))
    (M : DFA α σ) (hL : ∀ u, u ∈ O.L ↔ M.eval u.toList ∈ M.accept)
    (F : Finset (FreeMonoid α)) (w : FreeMonoid α) :
    acceptCount O.L F w
      = (F.filter (fun v => M.evalFrom (M.eval w.toList) v.toList ∈ M.accept)).card := by
  unfold acceptCount
  congr 1
  refine Finset.filter_congr (fun v _ => ?_)
  rw [hL, FreeMonoid.toList_mul]
  simp only [DFA.eval, DFA.evalFrom_of_append]

theorem family_read_by_state_holds : FamilyReadByState := by
  intro Ω _ μ _ α σ _ O M F kl kh hL w w' hq rd
  rw [readProb_eq_readLaw, readProb_eq_readLaw, acceptCount_eq_state O M hL,
    acceptCount_eq_state O M hL, hq]

theorem family_read_independent_holds : FamilyReadIndependent := by
  intro Ω _ μ _ α _ O F kl kh hF
  exact read_iIndep O hF kl kh

/-! ## Tails -/

lemma meanVote_eq (O : Oracle μ S) (F : Finset S) (w : S) :
    meanVote O F w = acceptCount O.L F w * (1 - O.ηIn)
      + ((F.card - acceptCount O.L F w : ℕ) : ℝ) * O.ηOut := by
  unfold meanVote
  rw [Finset.sum_congr rfl (fun v _ => measureReal_mq_one O (w * v)),
    Finset.sum_eq_multiset_sum, map_acceptRate, Multiset.sum_add, Multiset.sum_replicate,
    Multiset.sum_replicate, nsmul_eq_mul, nsmul_eq_mul]

lemma exp_tail_le {N : ℕ} {d e : ℝ} (hd : 0 ≤ d) (hde : d ≤ e) :
    Real.exp (-2 * (N : ℝ) * (e / N) ^ 2) ≤ Real.exp (-2 * d ^ 2 / N) := by
  rcases Nat.eq_zero_or_pos N with h0 | hpos
  · simp [h0]
  have hc : (0 : ℝ) < N := by exact_mod_cast hpos
  refine Real.exp_le_exp.2 ?_
  have e1 : -2 * (N : ℝ) * (e / N) ^ 2 = -2 * e ^ 2 / N := by field_simp
  rw [e1]
  exact div_le_div_of_nonneg_right (by nlinarith [pow_le_pow_left₀ hd hde 2]) hc.le

lemma accept_tail (O : Oracle μ S) (F : Finset S) (w : S) {kl kh : ℕ} (hk : kl ≤ kh)
    (hm : meanVote O F w ≤ kl) :
    μ.real {ω | kh ≤ voteCount O.mq F w ω} ≤ Real.exp (-2 * ((kh : ℝ) - kl) ^ 2 / F.card) := by
  classical
  have hmean : meanVote O F w = ∑ v ∈ F, μ[O.mq (w * v)] :=
    Finset.sum_congr rfl (fun v _ => measureReal_mq_eq_one O (w * v))
  have hkl : (kl : ℝ) ≤ kh := by exact_mod_cast hk
  rcases Nat.eq_zero_or_pos F.card with h0 | hpos
  · rw [h0, Nat.cast_zero, div_zero, Real.exp_zero]; exact measureReal_le_one
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hpos
  set γ : ℝ := ((kh : ℝ) - meanVote O F w) / F.card with hγ
  have hγ0 : 0 ≤ γ := div_nonneg (by linarith) hc.le
  have hT := sumUpper_le_total (fun v : S => O.mq (w * v)) F (meanVote O F w) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O w) (fun v => mq_icc O _)
    hmean.ge hγ0
  have hcg : meanVote O F w + (F.card : ℝ) * γ = kh := by rw [hγ]; field_simp; ring
  rw [hcg] at hT
  refine le_trans (measureReal_le_of_ae_imp ?_) (hT.trans (exp_tail_le (by linarith) ?_))
  · filter_upwards [voteCount_eq_voteSum O F w] with ω heq h
    have h' : (kh : ℝ) ≤ (voteCount O.mq F w ω : ℝ) := by exact_mod_cast h
    rw [heq] at h'
    exact h'
  · linarith

lemma reject_tail (O : Oracle μ S) (F : Finset S) (w : S) {kl kh : ℕ} (hk : kl ≤ kh)
    (hm : (kh : ℝ) ≤ meanVote O F w) :
    μ.real {ω | voteCount O.mq F w ω ≤ kl} ≤ Real.exp (-2 * ((kh : ℝ) - kl) ^ 2 / F.card) := by
  classical
  have hmean : meanVote O F w = ∑ v ∈ F, μ[O.mq (w * v)] :=
    Finset.sum_congr rfl (fun v _ => measureReal_mq_eq_one O (w * v))
  have hkl : (kl : ℝ) ≤ kh := by exact_mod_cast hk
  rcases Nat.eq_zero_or_pos F.card with h0 | hpos
  · rw [h0, Nat.cast_zero, div_zero, Real.exp_zero]; exact measureReal_le_one
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hpos
  set γ : ℝ := (meanVote O F w - kl) / F.card with hγ
  have hγ0 : 0 ≤ γ := div_nonneg (by linarith) hc.le
  have hT := sumLower_le_total (fun v : S => O.mq (w * v)) F (meanVote O F w) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O w) (fun v => mq_icc O _)
    hmean.le hγ0
  have hcg : meanVote O F w - (F.card : ℝ) * γ = kl := by rw [hγ]; field_simp; ring
  rw [hcg] at hT
  refine le_trans (measureReal_le_of_ae_imp ?_) (hT.trans (exp_tail_le (by linarith) ?_))
  · filter_upwards [voteCount_eq_voteSum O F w] with ω heq h
    have h' : (voteCount O.mq F w ω : ℝ) ≤ kl := by exact_mod_cast h
    rw [heq] at h'
    exact h'
  · linarith

/-! ## The trichotomy -/

omit [IsProbabilityMeasure μ] in
lemma readProb_accept (O : Oracle μ S) (F : Finset S) (kl kh : ℕ) (w : S) :
    readProb O F kl kh w .accept = μ.real {ω | kh ≤ voteCount O.mq F w ω} := by
  unfold readProb familyRead readOf
  congr 1
  ext ω
  simp only [Set.mem_ofPred_eq]
  split_ifs <;> simp_all

omit [IsProbabilityMeasure μ] in
lemma readProb_reject (O : Oracle μ S) (F : Finset S) {kl kh : ℕ} (hk : kl < kh) (w : S) :
    readProb O F kl kh w .reject = μ.real {ω | voteCount O.mq F w ω ≤ kl} := by
  unfold readProb familyRead readOf
  congr 1
  ext ω
  simp only [Set.mem_ofPred_eq]
  split_ifs <;> simp_all
  omega

theorem family_read_trichotomy_holds : FamilyReadTrichotomy := by
  intro Ω _ μ _ S _ O F kl kh hk hband w
  have hmean := meanVote_eq O F w
  by_cases h1 : meanVote O F w ≤ kl
  · left
    rw [readProb_accept]
    exact accept_tail O F w hk.le h1
  by_cases h2 : (kh : ℝ) ≤ meanVote O F w
  · right; left
    rw [readProb_reject O F hk]
    exact reject_tail O F w hk.le h2
  right; right
  rw [readProb_eq_readLaw]
  replace h1 := not_le.1 h1
  replace h2 := not_le.1 h2
  rw [hmean] at h1 h2
  exact hband _ (acceptCount_le _ _ _) h1 h2

/-! ## The shipped parameters, computed exactly

Rates `4/5` and `1/5` are weights `(1, 4)` and `(4, 1)` over `5`, so `5⁶²·P[X = k]` is an
integer and the whole law is one integer recurrence. -/

def stepAux (u v : ℕ) : ℕ → List ℕ → List ℕ
  | prev, [] => [v * prev]
  | prev, x :: xs => (u * x + v * prev) :: stepAux u v x xs

def dpList : List (ℕ × ℕ) → List ℕ
  | [] => [1]
  | (u, v) :: l => stepAux u v 0 (dpList l)

lemma stepAux_getD (u v : ℕ) (L : List ℕ) :
    ∀ (prev k : ℕ), (stepAux u v prev L).getD k 0 = u * L.getD k 0 + v * (prev :: L).getD k 0 := by
  induction L with
  | nil =>
    intro prev k
    cases k <;> simp [stepAux]
  | cons x xs ih =>
    intro prev k
    cases k with
    | zero => simp [stepAux]
    | succ k =>
      simp only [stepAux, List.getD_cons_succ]
      exact ih x k

lemma dpList_getD (l : List (ℕ × ℕ)) (hl : ∀ x ∈ l, x.1 + x.2 = 5) (k : ℕ) :
    ((dpList l).getD k 0 : ℝ)
      = 5 ^ l.length * (l.map (fun x => (x.2 : ℝ) / 5)).foldr pbStep pbBase k := by
  induction l generalizing k with
  | nil =>
    cases k <;> simp [dpList, pbBase]
  | cons x l ih =>
    obtain ⟨u, v⟩ := x
    have huv : (u : ℝ) = 5 - v := by
      have := hl (u, v) List.mem_cons_self
      have h' : ((u + v : ℕ) : ℝ) = 5 := by exact_mod_cast this
      push_cast at h'
      linarith
    have ih' := fun k => ih (fun y hy => hl y (List.mem_cons_of_mem _ hy)) k
    simp only [dpList, List.map_cons, List.foldr_cons, List.length_cons]
    rw [stepAux_getD]
    cases k with
    | zero =>
      simp only [List.getD_cons_zero, pbStep]
      push_cast
      rw [ih' 0, huv]
      ring
    | succ k =>
      simp only [List.getD_cons_succ, pbStep]
      push_cast
      rw [ih' (k + 1), ih' k, huv]
      ring

def shippedVotes (a : ℕ) : List ℕ :=
  dpList (List.replicate a (1, 4) ++ List.replicate (62 - a) (4, 1))

def shippedMass (a : ℕ) (rd : Read) : ℕ :=
  ∑ j ∈ Finset.range 63, if readOf 20 42 j = rd then (shippedVotes a).getD j 0 else 0

set_option maxRecDepth 100000 in
lemma shippedMass_check : ∀ a < 63,
    shippedMass a .accept * 10 ^ 10 ≤ 5 ^ 62 ∨ shippedMass a .reject * 10 ^ 10 ≤ 5 ^ 62
      ∨ 5 ^ 62 ≤ 3 * shippedMass a .undecided := by
  decide +kernel

set_option maxRecDepth 100000 in
lemma shippedMass_band_check : ∀ a < 63, 13 ≤ a → a ≤ 49 →
    5 ^ 62 ≤ 3 * shippedMass a .undecided := by
  decide +kernel

lemma voteLaw_shipped {a : ℕ} (ha : a ≤ 62) (j : ℕ) :
    voteLaw 62 a (1 - 1 / 5) (1 / 5) j = ((shippedVotes a).getD j 0 : ℝ) / 5 ^ 62 := by
  have hl : ∀ x ∈ List.replicate a (1, 4) ++ List.replicate (62 - a) (4, 1), x.1 + x.2 = 5 := by
    intro x hx
    rcases List.mem_append.1 hx with h | h <;> rw [(List.mem_replicate.1 h).2]
    · rfl
    · rfl
  have hlen : (List.replicate a (1, 4) ++ List.replicate (62 - a) ((4 : ℕ), (1 : ℕ))).length
      = 62 := by simp; omega
  rw [shippedVotes, dpList_getD _ hl j, hlen, voteLaw, ← Multiset.coe_replicate,
    ← Multiset.coe_replicate, Multiset.coe_add, Multiset.coe_foldr]
  simp only [List.map_append, List.map_replicate]
  norm_num

lemma readLaw_shipped {a : ℕ} (ha : a ≤ 62) (rd : Read) :
    readLaw 62 a (1 - 1 / 5) (1 / 5) 20 42 rd = (shippedMass a rd : ℝ) / 5 ^ 62 := by
  rw [readLaw, shippedMass, Nat.cast_sum, Finset.sum_div]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  split_ifs
  · rw [voteLaw_shipped ha]
  · simp

lemma shipped_trichotomy {a : ℕ} (ha : a ≤ 62) :
    readLaw 62 a (1 - 1 / 5) (1 / 5) 20 42 .accept ≤ 1 / 10 ^ 10
    ∨ readLaw 62 a (1 - 1 / 5) (1 / 5) 20 42 .reject ≤ 1 / 10 ^ 10
    ∨ 1 / 3 ≤ readLaw 62 a (1 - 1 / 5) (1 / 5) 20 42 .undecided := by
  simp only [readLaw_shipped ha]
  have h5 : (0 : ℝ) < 5 ^ 62 := by positivity
  rcases shippedMass_check a (by omega) with h | h | h
  · left
    rw [div_le_div_iff₀ h5 (by positivity)]
    exact_mod_cast (by omega : shippedMass a .accept * 10 ^ 10 ≤ 1 * 5 ^ 62)
  · right; left
    rw [div_le_div_iff₀ h5 (by positivity)]
    exact_mod_cast (by omega : shippedMass a .reject * 10 ^ 10 ≤ 1 * 5 ^ 62)
  · right; right
    rw [div_le_div_iff₀ (by norm_num) h5]
    exact_mod_cast (by omega : 1 * 5 ^ 62 ≤ shippedMass a .undecided * 3)

/-- The parametric trichotomy's band holds at the shipped parameters. -/
theorem shipped_band_holds : BandHolds 62 20 42 (1 / 5) (1 / 5) := by
  intro a ha h1 h2
  rw [Nat.cast_sub ha] at h1 h2
  push_cast at h1 h2
  have ha13 : 13 ≤ a := by
    by_contra hc
    have : (a : ℝ) ≤ 12 := by exact_mod_cast (by omega : a ≤ 12)
    linarith
  have ha49 : a ≤ 49 := by
    by_contra hc
    have : (50 : ℝ) ≤ a := by exact_mod_cast (by omega : 50 ≤ a)
    linarith
  rw [readLaw_shipped ha, le_div_iff₀ (by positivity)]
  have := shippedMass_band_check a (by omega) ha13 ha49
  have h' : ((5 ^ 62 : ℕ) : ℝ) ≤ 3 * (shippedMass a .undecided : ℝ) := by exact_mod_cast this
  push_cast at h'
  linarith

theorem family_read_guarantee_holds : FamilyReadGuarantee := by
  classical
  intro Ω _ μ _ α σ _ O M F hL hF hN hIn hOut
  refine ⟨read_iIndep O hF 20 42, fun q => ?_⟩
  set a := (F.filter (fun v => M.evalFrom q v.toList ∈ M.accept)).card with ha
  have ha62 : a ≤ 62 := hN ▸ Finset.card_filter_le _ _
  refine ⟨readLaw 62 a (1 - 1 / 5) (1 / 5) 20 42, fun w hw rd => ?_, shipped_trichotomy ha62⟩
  rw [readProb_eq_readLaw, acceptCount_eq_state O M hL, hw, hN, hIn, hOut]

end OrthoDFA
