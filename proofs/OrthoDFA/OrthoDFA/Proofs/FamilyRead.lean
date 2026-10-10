import OrthoDFA.FamilyRead
import OrthoDFA.Proofs.Adaptive

/-!
# The family's read: the proofs

A string's vote reads the oracle's bits at `w·F` and no others.  Suffix-freeness keeps those
blocks disjoint across strings, which is the independence.  Within one string the bits are
independent at two rates, so the vote is a sum of two binomials and its law is fixed by how
many of `w·F` lie in the language.  The shipped parameters' laws are computed exactly in
integers.
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

/-! ## The vote's law

A sum of independent bits at rates `p₁, …, p_N` has the Poisson-binomial law, built one bit at a
time by `pbStep`. -/

def pbStep (p : ℝ) (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => (1 - p) * f 0
  | k + 1 => (1 - p) * f (k + 1) + p * f k

def pbBase (k : ℕ) : ℝ := if k = 0 then 1 else 0

instance : LeftCommutative pbStep where
  left_comm p q f := by
    funext k
    rcases k with _ | _ | k <;> simp only [pbStep] <;> ring

/-- `P[X = k]` for `X` the sum of `Bin(a, p)` and an independent `Bin(N − a, r)`. -/
noncomputable def voteLaw (N a : ℕ) (p r : ℝ) (k : ℕ) : ℝ :=
  ∑ x ∈ Finset.antidiagonal k,
    (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
      * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))

noncomputable def readLaw (N a : ℕ) (p r : ℝ) (kl kh : ℕ) (rd : ARU) : ℝ :=
  ∑ j ∈ Finset.range (N + 1), if readOf kl kh j = rd then voteLaw N a p r j else 0

noncomputable def binTerm (n : ℕ) (p : ℝ) (i : ℕ) : ℝ := n.choose i * p ^ i * (1 - p) ^ (n - i)

lemma binTerm_zero_succ (n : ℕ) (p : ℝ) : binTerm (n + 1) p 0 = (1 - p) * binTerm n p 0 := by
  simp [binTerm, pow_succ]
  ring

lemma binTerm_succ (n i : ℕ) (p : ℝ) :
    binTerm (n + 1) p (i + 1) = (1 - p) * binTerm n p (i + 1) + p * binTerm n p i := by
  rcases Nat.lt_or_ge i n with h | h
  · obtain ⟨m, rfl⟩ : ∃ m, n = i + 1 + m := ⟨n - (i + 1), by omega⟩
    simp only [binTerm, Nat.choose_succ_succ, Nat.cast_add]
    rw [show i + 1 + m + 1 - (i + 1) = m + 1 by omega, show i + 1 + m - (i + 1) = m by omega,
      show i + 1 + m - i = m + 1 by omega]
    ring
  · simp only [binTerm]
    rw [Nat.choose_succ_succ, Nat.choose_eq_zero_of_lt (by omega : n < i + 1),
      show n + 1 - (i + 1) = n - i by omega]
    push_cast
    ring

lemma foldr_replicate (n : ℕ) (p : ℝ) (h : ℕ → ℝ) (k : ℕ) :
    (List.replicate n p).foldr pbStep h k
      = ∑ x ∈ Finset.antidiagonal k, binTerm n p x.1 * h x.2 := by
  induction n generalizing k with
  | zero =>
    cases k with
    | zero => simp [binTerm]
    | succ k =>
      rw [Finset.Nat.sum_antidiagonal_succ]
      simp [binTerm, Nat.choose_zero_succ]
  | succ n ih =>
    rw [List.replicate_succ, List.foldr_cons]
    cases k with
    | zero =>
      simp only [pbStep, ih 0, Finset.Nat.antidiagonal_zero, Finset.sum_singleton,
        binTerm_zero_succ]
      ring
    | succ k =>
      simp only [pbStep]
      rw [ih (k + 1), ih k, Finset.Nat.sum_antidiagonal_succ, Finset.Nat.sum_antidiagonal_succ,
        binTerm_zero_succ]
      have hs : ∑ x ∈ Finset.antidiagonal k, binTerm (n + 1) p (x.1 + 1) * h x.2
          = (1 - p) * ∑ x ∈ Finset.antidiagonal k, binTerm n p (x.1 + 1) * h x.2
            + p * ∑ x ∈ Finset.antidiagonal k, binTerm n p x.1 * h x.2 := by
        rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl (fun x _ => by rw [binTerm_succ]; ring)
      rw [hs]
      ring

lemma voteLaw_eq_foldr (N a : ℕ) (p r : ℝ) (k : ℕ) :
    voteLaw N a p r k
      = (List.replicate a p ++ List.replicate (N - a) r).foldr pbStep pbBase k := by
  rw [List.foldr_append, foldr_replicate, voteLaw]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [foldr_replicate, Finset.sum_eq_single (x.2, 0)]
  · simp [binTerm, pbBase]
  · intro y hy hne
    have h0 : y.2 ≠ 0 := by
      intro h0
      apply hne
      have := Finset.mem_antidiagonal.1 hy
      ext <;> simp_all
    simp [pbBase, h0]
  · intro h
    exact absurd (Finset.mem_antidiagonal.2 (by simp)) h


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
  rw [vote_prob_foldr, map_acceptRate, voteLaw_eq_foldr, ← Multiset.coe_replicate,
    ← Multiset.coe_replicate, Multiset.coe_add, Multiset.coe_foldr]

theorem readProb_eq_readLaw (O : Oracle μ S) (F : Finset S) (kl kh : ℕ) (w : S) (rd : ARU) :
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


/-! ## The reduction to the parameters -/

theorem family_read_trichotomy_holds : FamilyReadTrichotomy := by
  classical
  intro Ω _ μ _ α σ _ O M F kl kh ε ε₂ κ hL hF hband
  refine ⟨read_iIndep O hF kl kh, fun q => ?_⟩
  refine ⟨readLaw F.card (F.filter (fun v => M.evalFrom q v.toList ∈ M.accept)).card
    (1 - O.ηIn) O.ηOut kl kh, fun w hw => funext fun rd => ?_,
    hband _ (Finset.card_filter_le _ _)⟩
  rw [readProb_eq_readLaw, acceptCount_eq_state O M hL, hw]

/-! ## Why the rule checks `BandPasses`

Its cross and FNR criteria alone admit a band some state reads badly: `(2, 15)` over 15 at
`center = 17/30`, where the state with no accepting member reads as `Bin(15, 67/500)`. -/

/-- `binom_cdf(k, N, q)`. -/
noncomputable def binomCdf (N k : ℕ) (q : ℝ) : ℝ :=
  ∑ j ∈ Finset.range (k + 1), (N.choose j : ℝ) * q ^ j * (1 - q) ^ (N - j)

lemma voteLaw_zero (N : ℕ) (p r : ℝ) (k : ℕ) :
    voteLaw N 0 p r k = (N.choose k : ℝ) * r ^ k * (1 - r) ^ (N - k) := by
  rw [voteLaw, Finset.sum_eq_single (0, k)]
  · simp
  · intro x hx hne
    have h1 : x.1 ≠ 0 := by
      intro h0
      apply hne
      have := Finset.mem_antidiagonal.1 hx
      ext <;> simp_all
    simp [Nat.choose_eq_zero_of_lt (Nat.pos_of_ne_zero h1)]
  · intro h
    exact absurd (Finset.mem_antidiagonal.2 (by simp)) h

theorem selection_not_trichotomy :
    max (binomCdf 15 2 ((15 - 1) / 15)) (1 - binomCdf 15 (15 - 1) (2 / 15)) ≤ (2 / 15) ^ 15
    ∧ max (binomCdf 15 (15 - 1) (17 / 30 + (17 / 30 - 67 / 500))
          - binomCdf 15 2 (17 / 30 + (17 / 30 - 67 / 500)))
        (binomCdf 15 (15 - 1) (17 / 30 - (17 / 30 - 67 / 500))
          - binomCdf 15 2 (17 / 30 - (17 / 30 - 67 / 500))) ≤ 33 / 100
    ∧ ∀ ε₂ κ, ¬ BandPasses 15 2 15 (17 / 30 + (17 / 30 - 67 / 500))
        (17 / 30 - (17 / 30 - 67 / 500)) ((2 / 15) ^ 15) ε₂ κ := by
  refine ⟨?_, ?_, fun ε₂ κ h => ?_⟩
  · simp only [binomCdf, Finset.sum_range_succ, Finset.sum_range_zero]
    norm_num [Nat.choose]
  · simp only [binomCdf, Finset.sum_range_succ, Finset.sum_range_zero]
    norm_num [Nat.choose]
  · have h0 : readLaw 15 0 (17 / 30 + (17 / 30 - 67 / 500)) (17 / 30 - (17 / 30 - 67 / 500)) 2
          15 .accept ≤ (2 / 15) ^ 15
        ∨ readLaw 15 0 (17 / 30 + (17 / 30 - 67 / 500)) (17 / 30 - (17 / 30 - 67 / 500)) 2 15
          .reject ≤ (2 / 15) ^ 15
        ∨ 1 / 3 ≤ readLaw 15 0 (17 / 30 + (17 / 30 - 67 / 500))
          (17 / 30 - (17 / 30 - 67 / 500)) 2 15 .undecided :=
      (h 0 (by norm_num)).imp_right (Or.imp_right fun h => h.2.1)
    rcases h0 with h | h | h <;>
      simp only [readLaw, voteLaw_zero, Finset.sum_range_succ, Finset.sum_range_zero,
        readOf] at h <;>
      norm_num [Nat.choose] at h <;>
      simp only [reduceCtorEq, ↓reduceIte] at h <;>
      norm_num at h

end OrthoDFA
