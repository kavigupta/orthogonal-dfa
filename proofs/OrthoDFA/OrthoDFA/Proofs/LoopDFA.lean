import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.UniformStrings
import OrthoDFA.Proofs.DFABridge
import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.LoopSelect

/-!
# The learner's DFA: the proof

`loop_simple_checks` at the window's probes leaves a last round with at most `ε + ε/96` of the
probes disagreeing, searching or stopping undecided at their start or end sift, and at most half
of each state's strings shorter than `L + 1`, plus `ε + ε/96`, misled: reading, or extended by a
letter reading, a string on its state's rarer side. From there the bridge is deterministic
(`dfa_of_checks`): the tree's root reads the string itself, so a covered state's edges are true
(`edge_of_occupied`), it is read on its true side (`sideOf_eq_of_mass`), and the covered start's
true leaf misjudges at most what re-rooting does (`classify_start_le`).

The family reads a string of length `L` wrongly at most `(1025/1024) N εcov` of the time and
leaves it undecided at most `(3/8) N εcov`, with `N εcov ≤ ξ/8`, so off a stray score the picked
leaf misjudges at most `ξ` more (`pickLeaf_le`). Each
round that can be last strays or collides with chance at most `δ/(2(|Q|+1))`
(`loop_select_round_le`), so the selection costs half of `δ` and the loop the other half.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The tree's root reads the string itself. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

theorem rootOne_splitAt {T : DTree α} (h : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := h
  rcases p with _ | ⟨_ | _, p⟩ <;> exact ⟨_, _, rfl⟩

theorem rootOne_of_classSet : ∀ (n : ℕ) (T : DTree α), T ∈ Random.classSet n → RootOne T
  | 0, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_singleton] at h
    exact ⟨_, _, h⟩
  | n + 1, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T₀, hT₀, _, _, rfl⟩
    · exact rootOne_of_classSet n T h
    · exact rootOne_splitAt (rootOne_of_classSet n T₀ hT₀) _ _

section Reads

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem readProb_sum (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (w : FreeMonoid α) :
    readProb O F kl kh w .accept + readProb O F kl kh w .reject
      + readProb O F kl kh w .undecided = 1 := by
  have hm : ∀ v : ARU, MeasurableSet {ω | familyRead O.mq F kl kh w ω = v} := fun v =>
    (measurable_voteCount O F w) ((Set.to_countable {j | readOf kl kh j = v}).measurableSet)
  unfold readProb
  rw [← measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_setOf_eq] at h1 h2; rw [h1] at h2; cases h2) (hm _),
    ← measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_union, Set.mem_setOf_eq] at h1 h2
      rcases h1 with h1 | h1 <;> rw [h1] at h2 <;> cases h2) (hm _)]
  have : {ω | familyRead O.mq F kl kh w ω = .accept} ∪ {ω | familyRead O.mq F kl kh w ω = .reject}
      ∪ {ω | familyRead O.mq F kl kh w ω = .undecided} = Set.univ := by
    ext ω
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    rcases familyRead O.mq F kl kh w ω <;> simp
  rw [this, probReal_univ]

variable {σ : Type*} (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ)
  (F : Finset (FreeMonoid α)) (kl kh : ℕ)

theorem familyReadDist_nonneg (q : σ) (v : ARU) : 0 ≤ familyReadDist O M F kl kh q v := by
  unfold familyReadDist
  split_ifs
  · exact measureReal_nonneg
  · exact le_rfl

theorem familyReadDist_le_one (q : σ) (v : ARU) : familyReadDist O M F kl kh q v ≤ 1 := by
  unfold familyReadDist
  split_ifs
  · exact measureReal_le_one
  · norm_num

theorem familyReadDist_sum (q : σ) (hq : ∃ w : FreeMonoid α, M.eval w.toList = q) :
    familyReadDist O M F kl kh q .accept + familyReadDist O M F kl kh q .reject
      + familyReadDist O M F kl kh q .undecided = 1 := by
  unfold familyReadDist
  rw [dif_pos hq]
  exact readProb_sum O F kl kh _

end Reads

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A last round whose disagreeing probes over the window are at most `εB`, and whose misled ones
are at most half of each state's strings shorter than `L + 1`, plus `εm`, both small enough
against the coverage `c`, has a leaf from which its hypothesis misjudges at most what re-rooting
at a covered start does. -/
theorem dfa_of_checks (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) {c εr εu εB εm γ : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M (L + 1)) εm) (hεm : 0 ≤ εm)
    (hεB : 2 * (((Fintype.card α : ℝ) + 1)
      * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εB
      + ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εm) < c)
    (hcov : (2 * εr + εu) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) < c)
    (hceil : ∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ γ := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := L + Fintype.card σ + 3 with hL₁d
  have hkL₁ : k ≤ L₁ := by omega
  set D : Measure (FreeMonoid α) := windowStrings k L₁ with hDd
  haveI : IsProbabilityMeasure D := windowStrings_isProbability hkL₁
  set H := o.result.hypothesis with hHd
  set dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1) with hdistd
  set side : σ → Bool := fun q => readSide (dist q)
  have hd0 := familyReadDist_nonneg O M o.family o.state.lo (o.state.hi + 1)
  have hd1 := familyReadDist_le_one O M o.family o.state.lo (o.state.hi + 1)
  obtain ⟨⟨hcons, hmis, hund⟩, hchk, hcls, hfew⟩ := hch
  have hroot : ∃ r a, H.tree = .node 1 r a := rootOne_of_classSet _ _ hcls
  -- the family's reads over the window
  have hund' : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εu := by
    refine le_trans (le_of_eq ?_) hund
    congr 1
    funext p
    rw [readProb_eq_familyReadDist O M hL]
  have hmis' : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εr := by
    refine le_trans (le_of_eq ?_) hmis
    congr 1
    funext p
    have e := congrFun (readProb_eq_familyReadDist O M hL o.family o.state.lo (o.state.hi + 1) p)
    unfold misreadProb
    by_cases hp : M.eval p.toList ∈ M.accept
    · simp only [if_pos hp, if_pos ((hL p).2 hp)]
      exact (e .reject).symm
    · simp only [if_neg hp, if_neg (fun h => hp ((hL p).1 h))]
      exact (e .accept).symm
  have hN0 : (0 : ℝ) < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    linarith
  have hεB0 : 0 ≤ εB := measureReal_nonneg.trans hchk
  -- the covered states
  set Sc := coveredStates M L c with hSd
  have hreach : ∀ q ∈ Sc, ∃ w : FreeMonoid α, M.eval w.toList = q := by
    intro q hq
    by_contra hno
    push Not at hno
    have hempty : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
      exact fun j _ _ => hno _
    have h := hq
    simp only [hSd, coveredStates, Set.mem_setOf_eq, hempty, measureReal_empty] at h
    have hc0 : 0 < c := by
      have : 0 ≤ ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εB := by positivity
      have : 0 ≤ ((L₁ : ℝ) + 1 - k) * εm := by positivity
      have e : 2 * (((Fintype.card α : ℝ) + 1)
          * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εB
          + ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εm)
          = 2 * (((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εB
            + ((L₁ : ℝ) + 1 - k) * εm) := rfl
      linarith
    linarith
  have hocc : ∀ q ∈ Sc, c ≤ ∑ i ∈ Finset.Ico k (L + 1),
      (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} := by
    intro q hq
    have := occupancy_of_covered M hq
    rwa [Finset.Ico_add_one_right_eq_Icc]
  have hE : ∀ q ∈ Sc, ∀ a, ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w) := fun q hq a =>
    edge_of_occupied M dist o.read H (by omega : k < L + 1) (by omega : L + 1 ≤ L₁) q a hchk
      (hfew q a) (hocc q hq) hεB
  have hside : ∀ q ∈ Sc, side q = decide (q ∈ M.accept) := by
    intro q hq
    refine sideOf_eq_of_mass M D dist hd0 hd1 q
      (familyReadDist_sum O M _ _ _ q (hreach q hq)) hmis' hund' (b := c / ((L₁ : ℝ) + 1 - k))
      ?_ ?_
    · rw [windowStrings_real hkL₁, div_le_div_iff_of_pos_right hN0]
      refine (hocc q hq).trans ?_
      rw [Finset.Ico_add_one_right_eq_Icc]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.Icc_subset_Icc le_rfl (by omega))
        fun _ _ _ => measureReal_nonneg
    · rw [lt_div_iff₀ hN0]
      have e : (2 * εr + εu) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
          = (2 * εr + εu) * ((L₁ : ℝ) + 1 - k) := rfl
      linarith
  obtain ⟨q₀, -, hacc⟩ := hceil
  exact ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _,
    classify_start_le M side H hroot Sc hE hside L q₀ hacc⟩

end Bridge

section Select

variable [Nonempty α]

theorem integral_uniform_le_window {k L L₁ : ℕ} (hk : k ≤ L) (hL : L ≤ L₁)
    (g : FreeMonoid α → ℝ) (hg : ∀ x, 0 ≤ g x) :
    ∫ x, g x ∂(uniformStrings L) ≤ ((L₁ : ℝ) + 1 - k) * ∫ x, g x ∂(windowStrings k L₁) := by
  have hN : (0 : ℝ) < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hk.trans hL
    linarith
  rw [integral_windowStrings (hk.trans hL), integral_uniformStrings, mul_div_cancel₀ _ hN.ne']
  have hu : ∀ n, 0 ≤ uavg n g := fun n =>
    div_nonneg (Finset.sum_nonneg fun v _ => hg _) (by positivity)
  exact Finset.single_le_sum (f := fun n => uavg n g) (fun n _ => hu n)
    (Finset.mem_Icc.2 ⟨hk, hL⟩)

lemma rb_nonneg {Q s pAP δ' τ cL : ℝ} (hQ : 0 ≤ Q) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hcL : 0 ≤ cL) :
    0 ≤ 161000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2)
        * ((50 * max (Real.log (20 / τ)) cL / s ^ 2 + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2)
        / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / (s ^ 4 * τ ^ 2) := by
  have hm0 : 0 ≤ max (Real.log (20 / τ)) cL := le_max_of_le_right hcL
  have hN0 : 0 ≤ 50 * max (Real.log (20 / τ)) cL / s ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) hm0) (sq_nonneg s)
  have hN1 : 0 ≤ 20 / s := by positivity
  have h1 : 1 ≤ (50 * max (Real.log (20 / τ)) cL / s ^ 2 + 20 / s + 28) / pAP := by
    rw [le_div_iff₀ hp]
    linarith
  have h2 : 0 ≤ 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := by
    refine div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg ?_)) (by positivity)
    rw [le_div_iff₀ (by positivity)]
    linarith
  refine div_nonneg (mul_nonneg (by positivity) (Real.log_nonneg ?_)) (by positivity)
  have hD1 : δ' / 2 * (δ' / 4) * (pAP / 2) * τ ≤ 1 := by
    have : δ' / 2 * (δ' / 4) * (pAP / 2) ≤ 1 := by nlinarith
    nlinarith
  rw [le_div_iff₀ (by positivity), one_mul]
  nlinarith

lemma dfa_select_arith {Q A δ ξ RB KpG T NPf : ℝ} {n k L : ℕ} (hQ : 0 ≤ Q) (hA : 2 ≤ A)
    (hδ : 0 < δ) (hξ : 0 < ξ) (hkL : k ≤ L)
    (hn : 32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2 ≤ n) (hT : T ≤ KpG)
    (hNP : NPf ≤ RB)
    (h3 : (Q + 1) * n * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ ≤ δ / 4) :
    (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
      + (n * (6 * (Q + 2) * NPf + T) / A ^ k + n ^ 2 / A ^ L) ≤ δ / (2 * (Q + 1)) := by
  set X : ℝ := 16 * (Q + 1) * (Q + 2) / δ with hXd
  have hX0 : 0 < X := by positivity
  have hnX : Real.log X ≤ n * (ξ / 4) ^ 2 / 2 := by
    rw [div_le_iff₀ (by positivity)] at hn
    have e : (n : ℝ) * (ξ / 4) ^ 2 / 2 = (n : ℝ) * ξ ^ 2 / 32 := by ring
    rw [e]
    linarith
  have hexp : Real.exp (-(n * (ξ / 4) ^ 2 / 2)) ≤ δ / (16 * (Q + 1) * (Q + 2)) := by
    calc Real.exp (-(n * (ξ / 4) ^ 2 / 2)) ≤ Real.exp (-Real.log X) :=
          Real.exp_le_exp.2 (by linarith)
      _ = δ / (16 * (Q + 1) * (Q + 2)) := by rw [Real.exp_neg, Real.exp_log hX0, hXd, inv_div]
  have h1 : (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2))) ≤ δ / (4 * (Q + 1)) := by
    calc (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
        ≤ (Q + 2) * (4 * (δ / (16 * (Q + 1) * (Q + 2)))) := by gcongr
      _ = δ / (4 * (Q + 1)) := by field_simp; ring
  have hAk : (0 : ℝ) < A ^ k := by positivity
  have hAkL : A ^ k ≤ A ^ L := pow_le_pow_right₀ (by linarith) hkL
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hb1 : (n : ℝ) ^ 2 / A ^ L ≤ (n : ℝ) ^ 2 / A ^ k :=
    div_le_div_of_nonneg_left (by positivity) hAk hAkL
  have hb2 : (n : ℝ) * (6 * (Q + 2) * NPf + T) ≤ n * (6 * (Q + 2) * RB + KpG) := by
    gcongr
  have hQ1 : (0 : ℝ) < Q + 1 := by linarith
  have h3' : (n : ℝ) * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ ≤ δ / (4 * (Q + 1)) := by
    rw [le_div_iff₀ (by positivity)]
    nlinarith
  have h2 : (n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L
      ≤ δ / (4 * (Q + 1)) := by
    calc (n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L
        ≤ n * (6 * (Q + 2) * RB + KpG) / A ^ k + (n : ℝ) ^ 2 / A ^ k :=
          add_le_add (div_le_div_of_nonneg_right hb2 hAk.le) hb1
      _ = (n : ℝ) * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ := by field_simp
      _ ≤ δ / (4 * (Q + 1)) := h3'
  calc (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
        + ((n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L)
      ≤ δ / (4 * (Q + 1)) + δ / (4 * (Q + 1)) := add_le_add h1 h2
    _ = δ / (2 * (Q + 1)) := by field_simp; ring

end Select

/-- The claim's rates, from the coverage `c`. -/
lemma dfa_rates {Q A N ℓ c ξ ε εcov ι : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hN : 1 ≤ N)
    (hℓ : 0 ≤ ℓ) (hNℓ : ℓ ≤ 2 * N) (hc : 0 < c) (hc1 : c ≤ 1) (hξ : 0 < ξ)
    (he : ε = c / (4 * (A + 1) * N)) (hεc : εcov = min ε (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) :
    0 < ε ∧ ε ≤ 1 / 2 ∧ 0 < εcov ∧ εcov ≤ ε ∧ 0 < ι ∧ ι ≤ εcov / 8
      ∧ 2 * ((A + 1) * N * (ε + ε / 96) + N * (ε + ε / 96)) < c ∧ 3 * εcov * N < c
      ∧ ℓ * ε < 3 / 8 := by
  have hX : 0 < 4 * (A + 1) * N := by positivity
  have hq : ε * (4 * (A + 1) * N) = c := by rw [he]; field_simp
  have he0 : 0 < ε := by rw [he]; positivity
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min he0 (by positivity)
  have hεcε : εcov ≤ ε := by rw [hεc]; exact min_le_left _ _
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := by rw [hι]; exact min_le_left _ _
  have h12 : 12 ≤ 4 * (A + 1) * N := by nlinarith
  refine ⟨he0, ?_, hεc0, hεcε, hι0, hι8, ?_, ?_, ?_⟩
  · have : ε * 12 ≤ 1 := by nlinarith
    linarith
  · have hεN : 0 < ε * N := by positivity
    have e : 2 * ((A + 1) * N * (ε + ε / 96) + N * (ε + ε / 96)) = 97 / 48 * (A + 2) * (ε * N) := by
      ring
    have e2 : c = 4 * (A + 1) * (ε * N) := by rw [← hq]; ring
    rw [e, e2]
    nlinarith
  · have h3 : 3 * εcov * N ≤ 3 * ε * N := by gcongr
    have e : 3 * ε * N * (4 * (A + 1)) = 3 * c := by rw [← hq]; ring
    have hA1 : (12 : ℝ) ≤ 4 * (A + 1) := by linarith
    have : 3 * ε * N * 12 ≤ 3 * c := by
      rw [← e]
      exact mul_le_mul_of_nonneg_left hA1 (by positivity)
    nlinarith
  · have h2 : ℓ * ε ≤ 2 * N * ε := mul_le_mul_of_nonneg_right hNℓ he0.le
    have h3 : 2 * N * ε * (4 * (A + 1)) = 2 * c := by rw [← hq]; ring
    have h4 : 2 * N * ε * 12 ≤ 2 * c := by
      rw [← h3]
      exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    nlinarith

set_option maxHeartbeats 8000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_simple_counts {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c γ δ pAP ξ : ℝ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := c / (4 * (A + 1) * N)
    let εcov : ℝ := min ε (ξ / (8 * N))
    let δ' : ℝ := δ / (8 * (Q + 1))
    let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
          (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * J * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
    pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) →
    (A ^ k)⁻¹ ≤ cap * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * (A ^ k)⁻¹ ≤ ε →
    (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4 →
    16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (∀ draws : Fin (Fintype.card σ + 1)
          → LoopRoundDraws Ω α (Fintype.card σ + 1) settings.level3.probeBudget,
        let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
          (uniformStrings (L₁ + Fintype.card σ + 3)) draws
        ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1)) ∧
      ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
          (Measure.pi fun _ : Fin n => uniformStrings L)).real
        {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
              (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
          ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
              o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
                o.result.hypothesis.rerootedDFA o.read q.2 x
                  (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
        ≤ δ := by
  intro Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB cap Kc n
    hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hk1 hk2 hk3 hk4
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  have hℓN : ((L₁ : ℝ) + 1) ≤ 2 * N := by
    rw [hNe]
    have : (2 * k : ℕ) ≤ L₁ + 1 := by show 2 * ((L + 1) / 2) ≤ L + Fintype.card σ + 3 + 1; omega
    have : (2 : ℝ) * k ≤ L₁ + 1 := by exact_mod_cast this
    linarith
  obtain ⟨q₀, hq₀, hq₀acc⟩ := hceil
  have hc1 : c ≤ 1 := hq₀.trans measureReal_le_one
  have hwN : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 2 * N := by
    have h1 : 1 ≤ L₁ := by change 1 ≤ L + Fintype.card σ + 3; omega
    have h2 : Nat.clog 2 L₁ ≤ L₁ - 1 :=
      Nat.clog_le_of_le_pow (by have := (L₁ - 1).lt_two_pow_self; omega)
    have : 2 + Nat.clog 2 L₁ ≤ L₁ + 1 := by omega
    have : 2 + (Nat.clog 2 L₁ : ℝ) ≤ L₁ + 1 := by exact_mod_cast this
    linarith
  obtain ⟨hε0, hε12, hεc0, hεcε, hι0, hι8, hbad, hcN, hιL⟩ :=
    dfa_rates (ℓ := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hA2' hN1 (by positivity) hwN hc hc1 hξ rfl rfl
      rfl
  have hεcov1 : εcov ≤ 1 := by linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1' : δ' < 1 := by
    show δ / (8 * (Q + 1)) < 1
    rw [div_lt_one (by positivity)]
    linarith
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hKc1 : (1 : ℝ) ≤ Kc := by
    show (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    have h1 : (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ :=
      one_le_pow₀ (by nlinarith [pow_nonneg (show (0 : ℝ) ≤ Q + 1 by linarith) 3])
    have h2 : (1 : ℝ) ≤ (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) :=
      one_le_pow₀ (by linarith)
    nlinarith
  have hlogK : 2 ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 2 ≤ 16 := by
      calc Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
        _ ≤ 3 * 3 := by gcongr <;> linarith
        _ ≤ 16 := by norm_num
    refine this.trans ?_
    rw [le_div_iff₀ hδ'0]
    nlinarith
  -- the walk starts after at least one letter
  have hk0 : 1 ≤ k := by
    by_contra h
    have hk : k = 0 := by omega
    rw [hk, pow_zero, inv_one, mul_one] at hk2
    linarith
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  have : IsProbabilityMeasure P.D := windowStrings_isProbability hkL₁
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hreads, hmain⟩ :=
    loop_simple_checks M O P (shortAt M (L + 1)) η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1' hpAP (by show 1 ≤ L + Fintype.card σ + 3; omega)
    hk0 hιL (hpm.trans hk1)
    ((mul_le_mul_of_nonneg_left hpm (mul_nonneg (by norm_num) (by linarith [hlogK]))).trans hk2)
    ((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hk4)
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by
    show 1 / 2 - η₀ ≤ 1 / 2
    linarith
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
    (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
  have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
  have hW1 : 1 ≤ W := le_max_right _ _
  have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hτ0 hτ1
    (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
  have hPB0 : 0 ≤ PB := by
    have hm : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) :=
      le_max_of_le_left (Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith))
    have hl : 0 ≤ Real.log ((J + 2) / (δ' / 2)) :=
      Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; show _ ≤ Q + 2 + 2; linarith)
    have h1 : 0 ≤ 50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 := by positivity
    show 0 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    positivity
  refine ⟨settings, hwalk, hdis, hbud, fun draws => ?_, ?_⟩
  · obtain ⟨hpr, hsu⟩ := loop_reads_le settings O (windowStrings k L₁)
      (uniformStrings (L₁ + Fintype.card σ + 3)) draws _ _ hreads
    have hQ : Q = (Fintype.card σ : ℝ) := rfl
    have e : ∀ (f : LoopRound Ω α (Fintype.card σ + 1) → ℕ)
        (l : List (LoopRound Ω α (Fintype.card σ + 1))),
        (l.map fun o => (f o : ℝ)).sum = ((l.map f).sum : ℝ) := fun f l => by
      rw [Nat.cast_list_sum, List.map_map]; rfl
    refine ⟨(e _ _).trans_le ((Nat.cast_le.2 hpr).trans ?_),
      (e _ _).trans_le ((Nat.cast_le.2 hsu).trans ?_)⟩
    · calc (((Fintype.card σ + 1) * (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊) : ℕ) : ℝ)
          = (Q + 1) * (6 * (Q + 2) * (⌊RB⌋₊ : ℝ)) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (6 * (Q + 2) * RB) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (Nat.floor_le hRB0)
            (by linarith)) (by linarith)
    · calc (((Fintype.card σ + 1) * (⌊PB⌋₊ + 1) : ℕ) : ℝ)
          = (Q + 1) * ((⌊PB⌋₊ : ℝ) + 1) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (PB + 1) :=
          mul_le_mul_of_nonneg_left (by linarith [Nat.floor_le hPB0]) (by linarith)
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    by_contra hc0
    push Not at hc0
    have : cap * ε / 1536 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 hε0.le
      linarith
    linarith
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    refine h1.trans (hk1.trans ?_)
    have : cap * ε ≤ cap * 1 := mul_le_mul_of_nonneg_left (by linarith) hcap0.le
    linarith
  set Dsf : Measure (FreeMonoid α) := uniformStrings (L₁ + Fintype.card σ + 3) with hDsfd
  have hD := hmain Dsf inferInstance (uniformStrings_length _) hpAPu hatom
  set T := settings.level3.probeBudget
  set μL := loopMeasure μ α (Fintype.card σ + 1) T
  set β := Measure.pi fun _ : Fin n => uniformStrings (α := α) L
  set Nn := P.L + T + 2
  set Ls := P.L + Fintype.card σ + 3
  set t : ℝ := ξ / 4 with htd
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T)) :=
    {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96) (shortAt M (L + 1))
            (ε + ε / 96))
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then
      selFail settings O P.D Dsf Nn Ls L n (Fintype.card σ + 2) t r hr else ∅
    with hSeld
  -- the loop's last round, consistent and fresh, picks well unless its scores stray or a
  -- held-out string collides
  have hsub : {q : (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
        × (Fin n → FreeMonoid α) |
      let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
            o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
              (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ⊆ (Prod.fst ⁻¹' Bloop) ∪ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Sel r := by
    intro q hq
    by_cases hlp : q.1 ∈ Bloop
    · exact Or.inl hlp
    right
    have hlp' := not_not.1 hlp
    obtain ⟨⟨hlen, o, ho, hch⟩, hfr⟩ := hlp'
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨r, hr, -, tt, hst, hG, hOk, hB, hF, hread, hhyp⟩ :=
      stageSel_last settings O P Dsf Ls q.1 hfr o ho hcons
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad O L n (Fintype.card σ + 2) t (some (tt, resCode o.result))
        ∧ (q.1.2, q.2) ∈ SelDk settings P.D L n r hr (some (tt, resCode o.result)) := by
      constructor
      · intro h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inl h⟩
      · by_contra h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inr h⟩
    obtain ⟨hnb, hdk⟩ := hnot'
    obtain ⟨hinj, -⟩ := hdk
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
      (Random.classSet_props _ _ hch.2.2.1).2
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ Fintype.card σ + 2
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    -- the best leaf, and the family's reads on the strings of length `L`
    have hs := dfa_of_checks M O hL o L hch (by linarith) ?_ ?_ ⟨q₀, hq₀, hq₀acc⟩
    · have hn : 0 < n := by
        refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
        rw [one_lt_div hδ]
        nlinarith only [hQ0, hδ1, mul_nonneg hQ0 hQ0]
      have hεcov_ξ : εcov ≤ ξ / (8 * N) := min_le_right _ _
      have hNε : N * εcov ≤ ξ / 8 := by
        have h := mul_le_mul_of_nonneg_left hεcov_ξ (by linarith : (0 : ℝ) ≤ N)
        rw [show N * (ξ / (8 * N)) = ξ / 8 by field_simp] at h
        exact h
      have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
          ∂(uniformStrings L) ≤ 3 / 8 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.2 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have hm : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
          ≤ 1025 / 1024 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
        t (3 / 8 * (N * εcov)) (1025 / 1024 * (N * εcov)) γ q.1.1 q.2 hinj hgood hu hm hs
      rw [hread]
      have : γ + 2 * t + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) ≤ γ + ξ := by
        have h0 : 0 ≤ N * εcov := mul_nonneg (zero_le_one.trans hN1) hεc0.le
        rw [htd]; linarith
      exact h.trans this
    · show 2 * ((A + 1) * N * (ε + ε / 96) + N * (ε + ε / 96)) < c
      exact hbad
    · show (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov) * N < c
      have h0 : 0 ≤ εcov * N := mul_nonneg hεc0.le (zero_le_one.trans hN1)
      linarith only [hcN, h0]
  -- the loop's failure, and each round's selection
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans ?_
    show (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) ≤ δ / 2
    have e : (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) = 5 * δ / 16 := by field_simp; ring
    rw [e]; linarith
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    refine (loop_select_round_le settings O P Dsf Ls L n (Fintype.card σ + 2) ⌊RB⌋₊ t
      (by positivity) (hwalk ▸ hkL) hnp r hr').trans ?_
    rw [hwalk]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    exact dfa_select_arith hQ0 hA2' hδ hξ hkL (Nat.le_ceil _) hbud (Nat.floor_le hRB0) hk3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  refine (measureReal_union_le _ _).trans ?_
  refine (add_le_add hBl ((measureReal_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum hsel))).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  show δ / 2 + ((Fintype.card σ + 1 : ℕ) : ℝ) * (δ / (2 * (Q + 1))) = δ
  push_cast
  field_simp
  ring

theorem learner_loop_dfa_simple_holds : LearnerLoopDFASimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hk1 hk2 hk3 hk4
  obtain ⟨settings, h1, h2, h3, -, h4⟩ := learner_loop_dfa_simple_counts M O L η₀ c γ δ pAP ξ
    hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hk1 hk2 hk3 hk4
  exact ⟨settings, h1, h2, h3, h4⟩

/-- `LearnerLoopDFA`'s `Θ`. -/
def loopΘ (Q A c ξ η₀ : ℝ) : ℝ :=
  (4 * (A + 1) / c + 8 / ξ) / (2 * (1 / 2 - η₀) ^ 2) + 8 * (4 * (A + 1) / c + 8 / ξ)
    + 4608 * Q * (Q + 1) * (4 * (A + 1) / c)

/-- `LearnerLoopDFA`'s `W₀`. -/
def loopW₀ (Q A δ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let S := Q * ((Q + 2) * A + 1) + 2 * A + 1
  Real.exp 1 * Real.sqrt δ'
    * (16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)) / 2 + 1

/-- Bounds `LearnerLoopDFASimple`'s suffix pool `PB` at `ℓ = L₁ + 1`. -/
def loopPool (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let s := 1 / 2 - η₀
  (50 * (max (Real.log (20 * loopΘ Q A c ξ η₀)) (Real.log (8 * loopW₀ Q A δ / δ'))
      + 2 * Real.log ℓ) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2

/-- Bounds `LearnerLoopDFASimple`'s prefix count `RB` at `ℓ = L₁ + 1`. -/
def loopCount (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let Θ := loopΘ Q A c ξ η₀
  161000 * (Q + 2) ^ 2 * Θ ^ 2 * ℓ ^ 2
    * Real.log (16 * (Q + 4) * loopPool ℓ Q A δ c ξ pAP η₀ * Θ * ℓ / (δ' ^ 2 * pAP))
    / (1 / 2 - η₀) ^ 4

/-- Bounds the probe budget `B₃` at `ℓ = L₁ + 1`. -/
def loopBudget (ℓ Q A δ c : ℝ) : ℝ :=
  let S := Q * ((Q + 2) * A + 1) + 2 * A + 1
  (36 * S * (4 * (A + 1) / c) * Real.log (16 * S * (S + 1) / (δ / (8 * (Q + 1)))) + S + 1) * ℓ

set_option maxRecDepth 4000 in
set_option maxHeartbeats 8000000 in -- unfolding the simple claim's settings
theorem learner_loop_dfa_counts {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]
    {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c γ δ pAP ξ : ℝ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := c / (4 * (A + 1) * N)
    let δ' : ℝ := δ / (8 * (Q + 1))
    let s : ℝ := 1 / 2 - η₀
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
    let aε : ℝ := 4 * (A + 1) / c
    let Θ : ℝ := loopΘ Q A c ξ η₀
    let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let W₀ : ℝ := loopW₀ Q A δ
    let P₀ : ℝ :=
      (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
    let R₀ : ℝ := 161000 * (Q + 2) ^ 2 * Θ ^ 2
      * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
    let E₁ : ℝ := (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    let E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε
    let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
    let E₃ : ℝ := 16 * K₁ ^ 2 / δ'
    let T : ℝ := 2 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4
    let L₀ : ℝ := 2 * (Real.log (E + E₂ + E₃) + 6 * Real.log T) / Real.log A
      + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12))
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
    pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) →
    L₀ ≤ L →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (∀ draws : Fin (Fintype.card σ + 1)
          → LoopRoundDraws Ω α (Fintype.card σ + 1) settings.level3.probeBudget,
        let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
          (uniformStrings (L₁ + Fintype.card σ + 3)) draws
        ((rounds.map fun o => o.prefs.card).sum : ℝ)
            ≤ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A δ c ξ pAP η₀)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ)
            ≤ (Q + 1) * (loopPool (L + Q + 4) Q A δ c ξ pAP η₀ + 1)
          ∧ (rounds.length * settings.level3.probeBudget : ℝ)
            ≤ (Q + 1) * loopBudget (L + Q + 4) Q A δ c) ∧
      ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
          (Measure.pi fun _ : Fin n => uniformStrings L)).real
        {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
              (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
          ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
              o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
                o.result.hypothesis.rerootedDFA o.read q.2 x
                  (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
        ≤ δ := by
  intro Q A k L₁ N ε δ' s S B₃ n aε Θ K₁ W₀ P₀ R₀ E₁ Kc E E₂ E₃ T L₀
    hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hkr : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hN1 : 1 ≤ N := by show (1 : ℝ) ≤ (L₁ : ℝ) + 1 - k; linarith
  have hNℓ : N ≤ (L₁ : ℝ) + 1 := by show (L₁ : ℝ) + 1 - k ≤ _; linarith
  have hℓ1 : (1 : ℝ) ≤ (L₁ : ℝ) + 1 := by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith
  have hc1 : c ≤ 1 := by
    obtain ⟨q₀, hq₀, -⟩ := hceil
    exact hq₀.trans measureReal_le_one
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by show 1 / 2 - η₀ ≤ 1 / 2; linarith
  set ℓ : ℝ := (L₁ : ℝ) + 1 with hℓd
  set Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  set εcov : ℝ := min ε (ξ / (8 * N)) with hεcd
  set ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1))) with hιd
  set τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with hτd
  set J : ℝ := Q + 2 with hJd
  set W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1 with hWd
  set PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 with hPBd
  set RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2) with hRBd
  set cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1))))) with hcapd
  obtain ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hΘ1, haε0⟩ :=
    dfa_rate_bounds (ℓ := ℓ) (Ce := aε + 8 / ξ) (Θ := Θ) hQ1 hA hℓ1 hN1 hNℓ hc hc1 hξ hη0 hη₀
      rfl rfl rfl rfl rfl rfl rfl rfl
  have hτ1 : τ ≤ 1 := by
    have h1 : ι ≤ εcov / 8 := min_le_left _ _
    have h2 : εcov ≤ ε := min_le_left _ _
    have h3 : ε ≤ 1 := rate_le_one hc1 hA hN1
    exact (min_le_right _ _).trans (by linarith)
  have hS1 : 1 ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith
  have hlS : 0 ≤ Real.log (16 * S * (S + 1) / δ') := log_budget_nonneg hS1 hδ'0 hδ'1
  have hB₃ : B₃ ≤ K₁ * ℓ := budget_le (by linarith) hε0 hinvε hℓ1 hlS
  have hB₃0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hK₁0 : 0 ≤ K₁ := by
    show 0 ≤ 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hWW : W ≤ W₀ * ℓ ^ 2 := by
    have hWm0 : 0 ≤ Wm := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
      positivity
    exact w_le hℓ1 hWm0
  have hW1 : 1 ≤ W := le_max_right _ _
  have hW₀1 : 1 ≤ W₀ := by
    have : 0 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      have : 0 ≤ Wm := by
        show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
        positivity
      positivity
    show 1 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
    linarith
  obtain ⟨hPB1, hPB⟩ := pb_le (Q := Q) (W := W) (W₀ := W₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1 hW1 hWW
    hδ'0 hδ'1 hs0 hs1 hpAP hp1
  have hP₀1 : 1 ≤ P₀ := pool_ge_one hQ0 hs0 hs1 hpAP hp1 hΘ1 hW₀1 hδ'0 hδ'1
  have hPBJ : PB = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := rfl
  obtain ⟨hRB0, hRB⟩ := rb_le (Q := Q) (PB := PB) (P₀ := P₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1
    (by rw [hPBJ]; exact hPB1) (by rw [hPBJ]; exact hPB) hP₀1 hδ'0 hδ'1 hs0 hpAP hp1
  have hR₀ : 0 ≤ R₀ := by
    have := log_count_nonneg hQ0 hP₀1 hΘ1 hδ'0 hδ'1 hpAP hp1
    show 0 ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2
      * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
    positivity
  have hτm : τ ≤ min (εcov / 2) ι := by
    refine le_min ?_ (min_le_right _ _)
    have ha := gateA_le_one hη0 hη₀
    have ha0 := (gateA_pos hη0 hη₀).le
    have hsq : ((1 - 2 * η₀) / (1 - η₀)) ^ 2 ≤ 1 := by
      have : (1 - 2 * η₀) / (1 - η₀) = gateA η₀ := rfl
      rw [this]; nlinarith only [ha, ha0]
    have hεc0 : 0 ≤ εcov / 2 := by
      have : 0 < εcov := lt_min hε0 (by positivity)
      positivity
    exact (min_le_left _ _).trans (mul_le_of_le_one_right hεc0 hsq)
  have hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hcap := cap_ge (Q := Q) (RB := RB) (R₀ := R₀) (PB := PB) (P₀ := P₀) (E₁ := E₁) hQ0 hτ0
    hinvτ hι0 hinvι hΘ1 hℓ1 hRB0 (by rw [hRBd]; exact hRB) (by linarith) (by rw [hPBJ]; exact hPB)
    (by linarith) hR₀ hδ'0 hδ'1 hpAP hp1 hs0 hs1 hτm rfl
  have hE₁0 : 0 < E₁ := by
    show 0 < (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
    have : 0 < 2 ^ 20 * Θ * (R₀ + 1) / s := by positivity
    have : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' := by positivity
    have : 0 ≤ 2 ^ 16 * (P₀ + 2) / (pAP * δ') := by positivity
    positivity
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlg : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
  have hEE : 1 ≤ E + E₂ := by
    have hE₁ : 1 ≤ E₁ := by
      show 1 ≤ (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
        + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
      have h1 : 1 ≤ 2 ^ 20 * Θ * (R₀ + 1) / s := by
        rw [le_div_iff₀ hs0, one_mul]
        have : 1 ≤ Θ * (R₀ + 1) := one_le_mul_of_one_le_of_one_le hΘ1 (by linarith only [hR₀])
        linarith only [this, hs1]
      have h2 : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' := by positivity
      have h3 : 0 ≤ 2 ^ 16 * (P₀ + 2) / (pAP * δ') := by positivity
      have h4 : (1 : ℝ) ≤ (Q + 3) ^ 2 := one_le_pow₀ (by linarith only [hQ0])
      exact one_le_mul_of_one_le_of_one_le h4 (by linarith only [h1, h2, h3])
    have haε : 1 ≤ aε := by
      show 1 ≤ 4 * (A + 1) / c
      rw [le_div_iff₀ hc, one_mul]
      linarith only [hA, hc1]
    have hE₂ : 0 ≤ E₂ := by
      show 0 ≤ 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
      positivity
    have h1 : 0 ≤ 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε := by positivity
    have h2 : 1 ≤ 1536 * aε * E₁ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by norm_num) haε) hE₁
    show 1 ≤ 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε + E₂
    exact le_add_of_le_of_nonneg (le_add_of_le_of_nonneg h2 h1) hE₂
  have hE₃ : 0 ≤ E₃ := by show 0 ≤ 16 * K₁ ^ 2 / δ'; positivity
  have hpow3 : ℓ ^ 6 * (E + E₂ + E₃) ≤ A ^ k :=
    dfa_pow (L := (L : ℝ)) (Q := Q) hA
      (by show ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 = _; push_cast; ring) hQ0
      (Nat.cast_nonneg L) (by linarith) (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega))
      rfl hL₀
  have hℓ6 : 0 ≤ ℓ ^ 6 := by positivity
  have hpow : ℓ ^ 6 * (E + E₂) ≤ A ^ k :=
    (mul_le_mul_of_nonneg_left (by linarith only [hE₃]) hℓ6).trans hpow3
  obtain ⟨h1, h2, h3⟩ := dfa_final (Ak := A ^ k) (J := J) (RB := RB) (B₃ := B₃) (by positivity)
    hℓ1 hε0 hinvε haε0 hcap hE₁0 hlg rfl rfl hQ0 (Nat.cast_nonneg n) hδ rfl hRB0
    (by rw [hRBd]; exact hRB) hB₃0 hB₃ hpow
  have h4 : 16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' := by
    have hAk : 0 < A ^ k := by positivity
    have hB2 : B₃ ^ 2 ≤ K₁ ^ 2 * ℓ ^ 6 := by
      have h1 : B₃ ^ 2 ≤ (K₁ * ℓ) ^ 2 := pow_le_pow_left₀ hB₃0 hB₃ 2
      have h2 : ℓ ^ 2 ≤ ℓ ^ 6 := pow_le_pow_right₀ hℓ1 (by norm_num)
      have h3 : (K₁ * ℓ) ^ 2 = K₁ ^ 2 * ℓ ^ 2 := by ring
      have h4 := mul_le_mul_of_nonneg_left h2 (sq_nonneg K₁)
      linarith only [h1, h3, h4]
    have hE3 : ℓ ^ 6 * E₃ ≤ A ^ k := by
      have : ℓ ^ 6 * E₃ ≤ ℓ ^ 6 * (E + E₂ + E₃) :=
        mul_le_mul_of_nonneg_left (by linarith only [hEE]) hℓ6
      linarith only [this, hpow3]
    rw [← div_eq_mul_inv, div_le_iff₀ hAk]
    have e : ℓ ^ 6 * E₃ * δ' = 16 * K₁ ^ 2 * ℓ ^ 6 := by
      show ℓ ^ 6 * (16 * K₁ ^ 2 / δ') * δ' = _; field_simp
    have h5 := mul_le_mul_of_nonneg_right hE3 hδ'0.le
    linarith only [h5, e, hB2]
  obtain ⟨settings, hw, hd, hb, hreads, hprob⟩ := learner_loop_dfa_simple_counts M O L η₀ c γ δ pAP
    ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil h1 h2 h3 h4
  refine ⟨settings, hw, hd, hb, fun draws => ?_, hprob⟩
  have hℓe : (L : ℝ) + Q + 4 = ℓ := by
    show (L : ℝ) + (Fintype.card σ : ℝ) + 4 = ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1
    push_cast; ring
  rw [hℓe]
  have hPBf : PB ≤ loopPool ℓ Q A δ c ξ pAP η₀ :=
    pool_le hτ0 hinvτ hΘ1 hℓ1 hW1 hWW hδ'0 hs0 hpAP
  have hQ1' : (0 : ℝ) ≤ Q + 1 := by linarith
  obtain ⟨hpr, hsu⟩ := hreads draws
  refine ⟨hpr.trans ?_, hsu.trans ?_, ?_⟩
  · have := count_le hQ0 hτ0 hτ1 hinvτ hPB1 hPBf hδ'0 hδ'1 hs0 hpAP hp1
    gcongr
    exact this
  · gcongr
  · have hlen := length_loopRounds_le settings O (windowStrings k L₁)
      (uniformStrings (L₁ + Fintype.card σ + 3)) draws (Fintype.card σ + 1)
    have hlen' : ((indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card σ + 3)) draws).length : ℝ) ≤ Q + 1 := by
      show _ ≤ (Fintype.card σ : ℝ) + 1
      exact_mod_cast hlen
    have hbud := hb.trans hB₃
    calc ((indecisiveLStarLoop settings O (windowStrings k L₁)
          (uniformStrings (L₁ + Fintype.card σ + 3)) draws).length
          * settings.level3.probeBudget : ℝ)
        ≤ (Q + 1) * (K₁ * ℓ) := mul_le_mul hlen' hbud (Nat.cast_nonneg _) hQ1'
      _ = (Q + 1) * loopBudget ℓ Q A δ c := rfl

theorem learner_loop_dfa_holds : LearnerLoopDFA := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ Q A k L₁ N ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀
  obtain ⟨settings, h1, h2, h3, -, h4⟩ := learner_loop_dfa_counts M O L η₀ c γ δ pAP ξ
    hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀
  exact ⟨settings, h1, h2, h3, h4⟩

end OrthoDFA

end
