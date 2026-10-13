import OrthoDFA.Proofs.EpsTree
import OrthoDFA.Proofs.RandomRound

/-!
# The random round with rare wrong reads: the proofs

A record at a target that is not real comes from a probe that can read a string wrong: its two
sifts end at leaves, and with every read on their way right they would end at the true leaves of
a state and of its successor, making the target real. Records are kept until a split, so a fake
fix needs `recordsToMove` such records since the last split, possibly over many stretches. Their
count is a race against the probes that can read a string wrong, at most `Gw` of the probes under
one tree, so `fakeQ`, one binomial tail for each split left and the current race's, is a
supermartingale over the draws (`fake_step`), and the fake fixes cost at most
`maxLeaves · P(Bin(P, Gw) ≥ recordsToMove)` over the round. The stretches are the random round's,
cut short at a fake fix (`anyBZ_le`), with `GInv` in place of the reads being right.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory
open scoped ENNReal
open OrthoDFA.Ideal (searchOutcome siftAgrees probe probeOrAgree Outcome
  agrees_inl EdgesOK OutcomeOK)

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level4Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

/-! ## Where a record is made -/

theorem searchOutcome_edge (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)
    (un : Option (α × List Bool)) (lo hi : ℕ) (hle : lo ≤ hi) {p : List Bool} {c : α}
    {u : FreeMonoid α} {t : List Bool}
    (h : searchOutcome read T x st un hi (tripartiteSearch false (siftAgrees read T x st) lo hi).1
      = .wrongEdge p c u t) :
    ∃ q, lo < q ∧ q ≤ hi ∧ u = prefixOf x (q - 1) ∧ x.toList[q - 1]? = some c := by
  have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch false (siftAgrees read T x st) lo hi).1 = r at hs h
  cases r with
  | edge q =>
    obtain ⟨h1, h2, -, -⟩ := hs
    simp only [searchOutcome] at h
    split at h
    · rename_i c' t' hc ht
      simp only [Outcome.wrongEdge.injEq] at h
      obtain ⟨-, rfl, rfl, -⟩ := h
      exact ⟨q, h1, h2, rfl, hc⟩
    · simp at h
  | allAgree => simp only [searchOutcome] at h; split at h <;> simp at h
  | _ => simp [searchOutcome] at h

/-- A record's prefix and letter are the probe's, past its first `k` letters. -/
theorem probe_edge (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) {p : List Bool} {c : α}
    {u : FreeMonoid α} {t : List Bool} (h : probe false read T E k x = .wrongEdge p c u t) :
    ∃ i, k ≤ i ∧ i < x.toList.length ∧ u = prefixOf x i ∧ x.toList[i]? = some c := by
  unfold probe at h
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · rw [ha] at h
    simp only at h
    have hpos := length_walk_pos E a (x.toList.drop k)
    have hlt : ∀ q, x.toList[q - 1]? = some c → q - 1 < x.toList.length := fun q hc => by
      by_contra hcon
      rw [List.getElem?_eq_none (by omega)] at hc
      simp at hc
    split at h
    · split at h
      · simp at h
      · obtain ⟨q, h1, -, h3, h4⟩ := searchOutcome_edge read T x _ _ k _ (by omega) h
        exact ⟨q - 1, by omega, hlt q h4, h3, h4⟩
    · obtain ⟨q, h1, -, h3, h4⟩ := searchOutcome_edge read T x _ _ k _ (by omega) h
      exact ⟨q - 1, by omega, hlt q h4, h3, h4⟩
  · rw [ha] at h
    simp at h

/-- The probes that can read a string wrong. -/
def PotWrong (k : ℕ) (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, read z = if side (M.eval z.toList) then .reject else .accept}

/-- A record at a target that is not real comes from a probe that can read a string wrong. -/
theorem untrue_wrong {T : DTree α} {E : Edges α} (hE : EdgesOK read T E)
    {τ : List Bool × α × List Bool} {x : FreeMonoid α} (hx : fR read C T E τ x = true)
    (hr : ¬ RealT M side T τ.1 τ.2.1 τ.2.2) : x ∈ PotWrong read M side C.walkStart T := by
  classical
  obtain ⟨p, c, t⟩ := τ
  simp only [fR, decide_eq_true_eq] at hx
  have hOK := Ideal.probeOrAgree_ok false read T E C.walkStart hE x
  generalize ho : probeOrAgree false read T E C.walkStart x = o at hx hOK
  cases o with
  | wrongEdge p' c' u t' =>
    simp only [recOf, Option.some.injEq, Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    obtain ⟨-, hu, huc, -⟩ := hOK
    have hk : C.walkStart ≤ x.toList.length := by
      by_contra hc
      simp [probeOrAgree, show x.toList.length < C.walkStart by omega] at ho
    have ho' : probe false read T E C.walkStart x = .wrongEdge p' c' u t' := by
      simpa [probeOrAgree, show ¬ x.toList.length < C.walkStart by omega] using ho
    obtain ⟨i, hi1, hi2, rfl, hc⟩ := probe_edge read T E C.walkStart x ho'
    by_contra hw
    simp only [PotWrong, Set.mem_ofPred_eq, not_exists, not_and] at hw
    have hp := sift_true M side read T _ p' hu fun m hm =>
      hw _ (mem_pot hi1 (by omega) hm)
    have hsucc := prefixOf_succ hc
    rw [← hsucc] at huc
    have ht := sift_true M side read T _ t' huc fun m hm =>
      hw _ (mem_pot (by omega) (by omega) hm)
    apply hr
    refine ⟨M.eval (prefixOf x i).toList, hp.symm, ?_⟩
    rw [ht, hsucc, eval_mul]
    rfl
  | agree => simp [recOf] at hx
  | undecidedStart _ | undecided _ _ => simp [recOf] at hx
  | unlearnedEdge p' c' u t' => simp [recOf] at hx

/-! ## The round -/

variable (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α)) (ε : ℝ) (Ns : ℕ) (y : ℝ)

open Finset in
theorem stretchRisk_nonneg (C : Level4Settings) (nα L Ns N₁ : ℕ) (p θr εd' θpt' : ℝ)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (ha : 0 ≤ C.testLevel) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) :
    0 ≤ stretchRisk C nα L Ns N₁ p θr εd' θpt' := by
  unfold stretchRisk
  have h01 : ∀ (j : ℕ) (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 1 else 0 := fun _ P _ => by
    split_ifs <;> norm_num
  have := binomSfGe_le_one hθr0 hθr1 Ns C.recordsToMove
  have := binomSfGe_le_one hεd'0 hεd'1 Ns (max C.firstLook ⌈C.searchShare * Ns⌉₊)
  have := binE_nonneg hεd'0 hεd'1 (n := N₁)
    (f := fun s => if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
        ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1
      else 0) fun j => h01 j _
  have : 0 ≤ ∑ s ∈ Icc (max C.firstLook ⌈C.searchShare * Ns⌉₊) Ns,
      binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) :=
    sum_nonneg fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 j _
  have : 0 ≤ ∑ n ∈ Icc C.firstLook Ns, (binomSfGe n p (hFire C.startRate C.testLevel n)
      + C.testLevel) :=
    sum_nonneg fun n _ => add_nonneg (binomSfGe_nonneg hp0 hp1 _ _) ha
  have : 0 ≤ ∑ n ∈ Icc 1 Ns,
      ((C.maxLeaves * nα : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1)))
        + binomSfGe n p (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊))) :=
    sum_nonneg fun n _ => add_nonneg (by positivity) (binomSfGe_nonneg hp0 hp1 _ _)
  linarith

/-- A run with no bad step ends well, given enough probes. -/
theorem ends_of_not_anyB (X : Level4State α → FreeMonoid α → Prop) (hm : 1 ≤ C.recordsToMove) :
    ∀ (xs : List (FreeMonoid α)) (s : Level4State α), Inv read C s → s.probeCount < Ns →
      Ψr C s * Ns + (Ns - s.probeCount) ≤ xs.length →
      ¬ AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) s xs →
      EndsWell M (BadAt U θ) D ε
        {w | Disagrees read ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
            C.walkStart w}
        C.walkStart (harvestLaw read D C.walkStart
          ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩) y
        (runProbes read C s xs).2
  | [], s, _, hn, hlen, _ => by simp at hlen; omega
  | x :: xs, s, hs, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨hb1, hb2⟩ := hb
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes, hst]
      by_cases hsm : Same s s'
      · obtain ⟨hs', hΨ, hn'⟩ := step_same read C hs hst hsm.1 hsm.2
        have hlt : s'.probeCount < Ns := by
          have : s'.probeCount ≠ Ns := fun h => hb1 (Or.inr ⟨s', hst, hsm, h⟩)
          omega
        refine ends_of_not_anyB X hm xs s' hs' hlt ?_ (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · obtain ⟨hs', hf, hΨ⟩ := step_change read C hm hs hst hsm
        have h0 : s'.probeCount = 0 := by rw [hf]; rfl
        refine ends_of_not_anyB X hm xs s' hs' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr C s' + 1) * Ns ≤ Ψr C s * Ns := Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [runProbes, hst]
      by_contra hne
      exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)


/-! ## Fake fixes since the last split -/

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B Z : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D] (I F : S → Prop) (Ψ : S → ℕ) (b : ℝ≥0∞)
  (Q : ℕ → S → ℝ≥0∞)

theorem anyB_mono {B' : S → X → Prop} (h : ∀ s x, B s x → B' s x) :
    ∀ s xs, AnyB step B s xs → AnyB step B' s xs
  | _, [], hb => hb
  | s, x :: xs, hb => by
    rcases hb with hb | ⟨s', hs', hb⟩
    · exact Or.inl (h s x hb)
    · exact Or.inr ⟨s', hs', anyB_mono h s' xs hb⟩

/-- The chance of a bad step or a step `Z` is at most the current segment's chance of a bad step
before any step `Z`, plus `Ψ` times a fresh segment's, plus a potential `Q` that pays for the
steps `Z`, given the draws left. -/
theorem anyBZ_le [∀ s x, Decidable (Z s x)]
    (hsame : ∀ s x s', I s → step s x = .inl s' → same s s' → ¬ B s x → ¬ Z s x →
      I s' ∧ Ψ s' = Ψ s)
    (hchg : ∀ s x s', I s → step s x = .inl s' → ¬ same s s' → ¬ B s x → ¬ Z s x →
      I s' ∧ F s' ∧ Ψ s' < Ψ s)
    (hseg : ∀ s, I s → F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegBZ step B Z same s (List.ofFn xs)} ≤ b)
    (hQ : ∀ N s, I s → ∫⁻ x, ((if Z s x then 1 else 0)
      + if Z s x then 0 else Sum.elim (Q N) (fun _ => 0) (step s x)) ∂D ≤ Q (N + 1) s) :
    ∀ N s, I s → (Measure.pi fun _ : Fin N => D)
        {xs | AnyB step (fun s x => B s x ∨ Z s x) s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegBZ step B Z same s (List.ofFn xs)}
        + Ψ s * b + Q N s := by
  intro N
  induction N with
  | zero =>
    intro s _
    have : {xs : Fin 0 → X | AnyB step (fun s x => B s x ∨ Z s x) s (List.ofFn xs)} = ∅ := by
      ext xs; simp [AnyB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s hs
    rw [Ideal.pi_succ_apply, Ideal.pi_succ_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin N => D) {xs | Fin.cons x xs ∈
          {xs : Fin (N + 1) → X | AnyB step (fun s x => B s x ∨ Z s x) s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ x, ((Measure.pi fun _ : Fin N => D) {xs | Fin.cons x xs ∈
            {xs : Fin (N + 1) → X | SegBZ step B Z same s (List.ofFn xs)}}
          + Ψ s * b + ((if Z s x then 1 else 0)
            + if Z s x then 0 else Sum.elim (Q N) (fun _ => 0) (step s x))) ∂D :=
          lintegral_mono fun x => ?_
      _ = ∫⁻ x, (Measure.pi fun _ : Fin N => D) {xs | Fin.cons x xs ∈
            {xs : Fin (N + 1) → X | SegBZ step B Z same s (List.ofFn xs)}} ∂D
          + Ψ s * b + ∫⁻ x, ((if Z s x then 1 else 0)
            + if Z s x then 0 else Sum.elim (Q N) (fun _ => 0) (step s x)) ∂D := by
          rw [lintegral_add_right _ (measurable_of_countable _),
            lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ _ := add_le_add le_rfl (hQ N s hs)
    simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, AnyB, SegBZ]
    by_cases hz : Z s x
    · simp only [hz, or_true, true_or, Set.ofPred_true, measure_univ, if_true]
      exact le_trans (le_add_right le_rfl) le_add_self
    · simp only [hz, or_false, not_false_eq_true, true_and, if_false, zero_add]
      by_cases hB : B s x
      · simp only [hB, true_or, Set.ofPred_true, measure_univ]
        exact le_trans (le_add_right le_rfl) (le_add_right le_rfl)
      · simp only [hB, false_or]
        rcases hst : step s x with s' | e
        · simp only [Sum.inl.injEq, exists_eq_left', Sum.elim_inl]
          by_cases hsm : same s s'
          · obtain ⟨hs', hΨ⟩ := hsame s x s' hs hst hsm hB hz
            simp only [hsm, true_and]
            rw [← hΨ]
            exact ih s' hs'
          · obtain ⟨hs', hF, hΨ⟩ := hchg s x s' hs hst hsm hB hz
            simp only [hsm, false_and, Set.ofPred_false, measure_empty, zero_add]
            calc _ ≤ _ := ih s' hs'
              _ ≤ b + Ψ s' * b + Q N s' := add_le_add (add_le_add (hseg s' hs' hF N) le_rfl) le_rfl
              _ = ((Ψ s' + 1 : ℕ) : ℝ≥0∞) * b + Q N s' := by push_cast; ring
              _ ≤ Ψ s * b + Q N s' := by gcongr; exact_mod_cast hΨ
        · simp

end Compose

open scoped Classical in
/-- The records since the last split at targets that are not real. -/
noncomputable def wrongCount (s : Level4State α) : ℕ :=
  s.records.countP fun τ => decide (¬ RealT M side s.tree τ.1 τ.2.1 τ.2.2)

/-- What the fake fixes still risk with `N` draws left: one race to `recordsToMove` records at a
target that is not real for each split left, and the current one. -/
noncomputable def fakeQ (Gw : ℝ) (N : ℕ) (s : Level4State α) : ℝ≥0∞ :=
  ENNReal.ofReal ((C.maxLeaves - s.tree.leaves.length : ℕ) * binomSfGe N Gw C.recordsToMove
    + binomSfGe N Gw (C.recordsToMove - wrongCount M side s))

omit [Fintype α] in
open scoped Classical in
theorem wrongCount_append (s : Level4State α) (τ : List Bool × α × List Bool) :
    wrongCount M side { s with records := s.records ++ [τ] }
      = wrongCount M side s + if RealT M side s.tree τ.1 τ.2.1 τ.2.2 then 0 else 1 := by
  unfold wrongCount
  simp only [List.countP_append, List.countP_singleton]
  split_ifs <;> simp_all

/-- A fake fix comes from a probe that can read a string wrong, at the last of `recordsToMove`
records at a target that is not real. -/
theorem fake_facts {s : Level4State α} (hs : Inv read C s) {x : FreeMonoid α}
    (hz : FakeFix M side read C s x) :
    x ∈ PotWrong read M side C.walkStart s.tree ∧ C.recordsToMove ≤ wrongCount M side s + 1 := by
  classical
  obtain ⟨p, c, u, t, ho, hc, hr⟩ := hz
  refine ⟨untrue_wrong read C M side hs.edges (τ := (p, c, t)) (by simp [fR, ho, recOf]) hr, ?_⟩
  have h1 : s.records.count (p, c, t) ≤ wrongCount M side s := by
    unfold wrongCount
    rw [List.count_eq_countP]
    refine List.countP_mono_left fun τ _ hτ => ?_
    simp only [beq_iff_eq] at hτ
    subst hτ
    simpa using hr
  rw [List.count_append, List.count_singleton] at hc
  simp only [beq_self_eq_true, if_true] at hc
  omega

open scoped Classical in
/-- What a probe that is not a fake fix leaves of the fake fixes' risk. -/
theorem fake_pt (hm : 1 ≤ C.recordsToMove) {Gw : ℝ} (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1)
    {s : Level4State α} (hs : Inv read C s) (N : ℕ) (x : FreeMonoid α) :
    ((if FakeFix M side read C s x then 1 else 0)
      + if FakeFix M side read C s x then 0
        else Sum.elim (fakeQ C M side Gw N) (fun _ => 0) (applyProbe read C s x))
      ≤ ENNReal.ofReal ((C.maxLeaves - s.tree.leaves.length : ℕ) * binomSfGe N Gw C.recordsToMove
        + if x ∈ PotWrong read M side C.walkStart s.tree
          then binomSfGe N Gw (C.recordsToMove - wrongCount M side s - 1)
          else binomSfGe N Gw (C.recordsToMove - wrongCount M side s)) := by
  classical
  set m := C.recordsToMove
  set R := wrongCount M side s
  set K : ℝ := (C.maxLeaves - s.tree.leaves.length : ℕ) * binomSfGe N Gw m
  have hK : 0 ≤ K := mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _)
  have hAB : binomSfGe N Gw (m - R) ≤ binomSfGe N Gw (m - R - 1) :=
    binomSfGe_antitone' hGw0 hGw1 (by omega)
  have hB0 := binomSfGe_nonneg hGw0 hGw1 (n := N) (m - R)
  have hite : binomSfGe N Gw (m - R) ≤ if x ∈ PotWrong read M side C.walkStart s.tree
      then binomSfGe N Gw (m - R - 1) else binomSfGe N Gw (m - R) := by
    split_ifs
    · exact hAB
    · exact le_rfl
  have hite0 : 0 ≤ if x ∈ PotWrong read M side C.walkStart s.tree
      then binomSfGe N Gw (m - R - 1) else binomSfGe N Gw (m - R) := hB0.trans hite
  by_cases hz : FakeFix M side read C s x
  · obtain ⟨hW, hR⟩ := fake_facts read C M side hs hz
    simp only [hz, if_true, add_zero]
    rw [if_pos hW, show m - R - 1 = 0 by omega, binomSfGe_zero_right]
    rw [← ENNReal.ofReal_one]
    exact ENNReal.ofReal_le_ofReal (by linarith)
  simp only [hz, if_false, zero_add]
  rcases happ : applyProbe read C s x with s' | e
  swap
  · simp
  simp only [Sum.elim_inl, fakeQ]
  refine ENNReal.ofReal_le_ofReal ?_
  rcases step_cases read C hs happ with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, -, -, hc, hfix⟩ | ⟨rfl, -, -⟩
  · show K + binomSfGe N Gw (m - R) ≤ _
    linarith
  · unfold redirectOrSplit at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold splitLeaf at hfix
      simp only at hfix
      split_ifs at hfix with hlt
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      have hlen := DTree.length_leaves_splitAt
        (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
      show ((C.maxLeaves - (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt
          (commonPrefix t t₀)) p).leaves.length : ℕ) : ℝ) * binomSfGe N Gw m
          + binomSfGe N Gw (m - 0) ≤ _
      rw [Nat.sub_zero, hlen]
      rw [hlen] at hlt
      have hKeq : K = ((C.maxLeaves - (s.tree.leaves.length + 1) : ℕ) : ℝ) * binomSfGe N Gw m
          + binomSfGe N Gw m := by
        show ((C.maxLeaves - s.tree.leaves.length : ℕ) : ℝ) * binomSfGe N Gw m = _
        rw [show C.maxLeaves - s.tree.leaves.length
          = C.maxLeaves - (s.tree.leaves.length + 1) + 1 by omega]
        push_cast; ring
      linarith
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      have hreal : RealT M side s.tree p c t := by
        by_contra hr
        exact hz ⟨p, c, u, t, ho, hc, hr⟩
      have : wrongCount M side (newHypothesis s.tree
          (Function.update s.edges (p, c) (some (t, u))) (s.records ++ [(p, c, t)])) = R := by
        have := wrongCount_append M side s (p, c, t)
        simp only [hreal, if_true, add_zero] at this
        exact this
      show K + binomSfGe N Gw (m - wrongCount M side (newHypothesis s.tree
        (Function.update s.edges (p, c) (some (t, u))) (s.records ++ [(p, c, t)]))) ≤ _
      rw [this]
      linarith
  · have hR' : wrongCount M side (chg read C s x)
        ≤ R + if x ∈ PotWrong read M side C.walkStart s.tree then 1 else 0 := by
      rcases hr : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) with _ | ⟨p, c, t⟩
      · have : chg read C s x = countProbe read C s x := chg_of_none read C hr
        rw [this]
        show wrongCount M side s ≤ _
        omega
      · have hw := wrongCount_append M side s (p, c, t)
        have : wrongCount M side (chg read C s x)
            = wrongCount M side { s with records := s.records ++ [(p, c, t)] } := by
          simp [chg, countProbe, hr, wrongCount]
        rw [this, hw]
        split_ifs with h1 h2 h2
        · omega
        · omega
        · omega
        · exact absurd (untrue_wrong read C M side hs.edges (τ := (p, c, t))
            (by simp [fR, hr]) h1) h2
    have hmono : binomSfGe N Gw (m - wrongCount M side (chg read C s x))
        ≤ if x ∈ PotWrong read M side C.walkStart s.tree
          then binomSfGe N Gw (m - R - 1) else binomSfGe N Gw (m - R) := by
      split_ifs with hW
      · rw [if_pos hW] at hR'
        exact binomSfGe_antitone' hGw0 hGw1 (by omega)
      · rw [if_neg hW] at hR'
        exact binomSfGe_antitone' hGw0 hGw1 (by omega)
    show K + binomSfGe N Gw (m - wrongCount M side (chg read C s x)) ≤ _
    linarith

open scoped Classical in
/-- The fake fixes' risk is a supermartingale over the draws. -/
theorem fake_step (hm : 1 ≤ C.recordsToMove) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {Gw : ℝ} (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1) {s : Level4State α}
    (hs : Inv read C s) (hsz : s.tree.leaves.length ≤ C.maxLeaves)
    (hW : D.real (PotWrong read M side C.walkStart s.tree) ≤ Gw) (N : ℕ) :
    ∫⁻ x, ((if FakeFix M side read C s x then 1 else 0)
      + if FakeFix M side read C s x then 0
        else Sum.elim (fakeQ C M side Gw N) (fun _ => 0) (applyProbe read C s x)) ∂D
      ≤ fakeQ C M side Gw (N + 1) s := by
  classical
  set m := C.recordsToMove
  set R := wrongCount M side s
  set W := PotWrong read M side C.walkStart s.tree
  have hWm : MeasurableSet W := (Set.to_countable _).measurableSet
  set K : ℝ := (C.maxLeaves - s.tree.leaves.length : ℕ) * binomSfGe N Gw m
  set A := binomSfGe N Gw (m - R - 1)
  set B := binomSfGe N Gw (m - R)
  have hK : 0 ≤ K := mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _)
  have hA0 : 0 ≤ A := binomSfGe_nonneg hGw0 hGw1 _ _
  have hB0 : 0 ≤ B := binomSfGe_nonneg hGw0 hGw1 _ _
  have hAB : B ≤ A := binomSfGe_antitone' hGw0 hGw1 (by omega)
  have hg : ∀ x, ENNReal.ofReal (K + if x ∈ W then A else B)
      = ENNReal.ofReal K + W.indicator (fun _ => ENNReal.ofReal A) x
        + Wᶜ.indicator (fun _ => ENNReal.ofReal B) x := by
    intro x
    by_cases hx : x ∈ W
    · simp [hx, ENNReal.ofReal_add hK hA0]
    · simp [hx, ENNReal.ofReal_add hK hB0]
  set w := D.real W
  have hw0 : 0 ≤ w := measureReal_nonneg
  have hw1 : w ≤ 1 := measureReal_le_one
  have hDW : D W = ENNReal.ofReal w := (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hDWc : D Wᶜ = ENNReal.ofReal (1 - w) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top D Wᶜ), ← measureReal_def, measureReal_compl hWm,
      probReal_univ]
  calc _ ≤ ∫⁻ x, ENNReal.ofReal (K + if x ∈ W then A else B) ∂D :=
        lintegral_mono fun x => fake_pt read C M side hm hGw0 hGw1 hs N x
    _ = ENNReal.ofReal K + ENNReal.ofReal A * D W + ENNReal.ofReal B * D Wᶜ := by
        simp_rw [hg]
        rw [lintegral_add_right _ ((measurable_const.indicator hWm.compl)),
          lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_const hWm, lintegral_indicator_const hWm.compl]
    _ = ENNReal.ofReal (K + A * w + B * (1 - w)) := by
        rw [hDW, hDWc, ← ENNReal.ofReal_mul hA0, ← ENNReal.ofReal_mul hB0,
          ← ENNReal.ofReal_add hK (by positivity), ← ENNReal.ofReal_add (by positivity)
            (mul_nonneg hB0 (by linarith))]
    _ ≤ fakeQ C M side Gw (N + 1) s := by
        refine ENNReal.ofReal_le_ofReal ?_
        have hβ : binomSfGe N Gw m ≤ binomSfGe (N + 1) Gw m :=
          binomSfGe_succ_n hGw0 hGw1 N m
        have hK' : K ≤ (C.maxLeaves - s.tree.leaves.length : ℕ) * binomSfGe (N + 1) Gw m :=
          mul_le_mul_of_nonneg_left hβ (Nat.cast_nonneg _)
        have hstep : A * w + B * (1 - w) ≤ binomSfGe (N + 1) Gw (m - R) := by
          rcases Nat.eq_zero_or_pos (m - R) with h0 | hpos
          · have hA1 : A = 1 := by
              show binomSfGe N Gw (m - R - 1) = 1
              rw [show m - R - 1 = 0 by omega, binomSfGe_zero_right]
            have hB1 : B = 1 := by
              show binomSfGe N Gw (m - R) = 1
              rw [h0, binomSfGe_zero_right]
            rw [hA1, hB1, h0, binomSfGe_zero_right]
            linarith
          · obtain ⟨j, hj⟩ : ∃ j, m - R = j + 1 := ⟨m - R - 1, by omega⟩
            have hA' : A = binomSfGe N Gw j := by
              show binomSfGe N Gw (m - R - 1) = _
              rw [show m - R - 1 = j by omega]
            have hB' : B = binomSfGe N Gw (j + 1) := by
              show binomSfGe N Gw (m - R) = _
              rw [hj]
            rw [hj, binomSfGe_succ, ← hA', ← hB']
            nlinarith
        linarith

theorem gInv_stretch {s s' : Level4State α} (hs : Inv read C s) (hg : GInv M side C s)
    {ys : List (FreeMonoid α)} (hst : StInv read C s.tree s.edges s.records ys s') :
    GInv M side C s' := by
  refine ⟨by rw [hst.tree]; exact hg.leaves, fun p c t₀ w hE hle => ?_⟩
  rw [hst.edges] at hE
  rw [hst.tree]
  refine hg.real p c t₀ w hE ?_
  rw [hst.records, List.countP_eq_zero.2 fun z _ => by
    rw [fR_target read C s.tree s.edges hs.edges (by simp [hE]) z]; simp] at hle
  simpa using hle

theorem probe_level_eps [IsProbabilityMeasure D] (L P N₁ : ℕ) (G Gw θr εd' θpt' : ℝ)
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hNw : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotWrong read M side C.walkStart T) ≤ Gw) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    (hm : 1 ≤ C.recordsToMove) (hn₀ : 1 ≤ C.firstLook) (hN₁ : C.firstLook ≤ N₁) (hN₁s : N₁ ≤ Ns)
    (hG0 : 0 ≤ G) (hG1 : 2 * G ≤ 1) (hy1 : y ≤ 1)
    (hGt : 4 * (L + 1) * G ≤ C.edgeRate * C.edgeReadShare)
    (hyt : 2 * y ≤ C.edgeRate * C.edgeReadShare) (hGw0 : 0 ≤ Gw) (hGw1 : Gw ≤ 1)
    (ha : 0 ≤ C.testLevel)
    (hθs0 : 0 ≤ C.startRate) (hθs1 : C.startRate ≤ 1) (hθe : 0 ≤ C.edgeRate)
    (hθpt0 : 0 ≤ C.middleRate) (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate)
    (hεd1 : C.disagreementRate ≤ 1) (hεd : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    (hP : stretches C (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := runProbes read C initialState (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees read ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
            C.walkStart (harvestLaw read D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
      ≤ ENNReal.ofReal (stretches C (Fintype.card α)
          * stretchRisk C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt'
        + C.maxLeaves * binomSfGe P Gw C.recordsToMove) := by
  classical
  have h2 : 2 ≤ C.maxLeaves := by omega
  have hNs : 0 < Ns := by omega
  have hst := inv_start read C hm h2 (α := α)
  have hΨ := Ψr_start C h2 (α := α)
  set Z := FakeFix M side read C
  set B := BadStep read C M U θ D ε Ns y (fun _ _ => False)
  set b := ENNReal.ofReal (stretchRisk C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt')
  set I : Level4State α → Prop := fun s => Inv read C s ∧ GInv M side C s ∧ s.probeCount < Ns
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges s.records → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegBZ (applyProbe read C) B Z Same s (List.ofFn xs)} ≤ b := by
    rintro s ⟨hs, hg, -⟩ hf N
    have hT := gInv_cls M side read C hs hg
    have := seg_le read C M U θ D ε Ns y (fun _ _ => False) (fun _ => False) 0 L N₁ G θr εd' θpt'
      hL Z hs hf hT hN (fun _ _ _ _ _ hx => hx)
      (fun ys s' x hs' hst' _ hx => step_ne_tooBig_g M side read C hcap hs'
        (gInv_stretch read C M side hs hg hst') hx)
      (fun N => by simp) hm hn₀ hN₁ hN₁s hG0 hG1 hy1 hGt hyt ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0
      hεd1 hεd hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    rwa [add_zero] at this
  have hI0 : I initialState := ⟨hst, gInv_start M side C,
      by simp only [initialState, newHypothesis]; omega⟩
  have hany := anyBZ_le (applyProbe read C) B Z Same D I
    (fun s => s = newHypothesis s.tree s.edges s.records) (Ψr C) b (fakeQ C M side Gw)
    (fun s x s' hI h hsm hb _ => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_same read C hs h hsm.1 hsm.2
      obtain ⟨rfl, -, -⟩ := step_same_chg read C hs h hsm
      refine ⟨⟨h1, gInv_chg M side read C hs hg x, ?_⟩, h2⟩
      have : (chg read C s x).probeCount ≠ Ns := fun hc => hb (Or.inr ⟨_, h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm _ hz => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_change read C hm hs h hsm
      refine ⟨⟨h1, gInv_change M side read C hm hs hg h hsm hz, ?_⟩, h2, h3⟩
      rw [h2]; simp only [newHypothesis]; omega)
    hseg
    (fun N s ⟨hs, hg, _⟩ => fake_step read C M side hm D hGw0 hGw1 hs hs.size
      (hNw _ (gInv_cls M side read C hs hg)) N) P initialState hI0
  have hQ0 : fakeQ C M side Gw P (initialState : Level4State α)
      ≤ ENNReal.ofReal (C.maxLeaves * binomSfGe P Gw C.recordsToMove) := by
    unfold fakeQ
    refine ENNReal.ofReal_le_ofReal ?_
    have h0 : wrongCount M side (initialState : Level4State α) = 0 := by
      simp [wrongCount, initialState, newHypothesis]
    rw [h0, Nat.sub_zero]
    simp only [initialState, newHypothesis, leaves_root, List.length_cons, List.length_nil]
    have hβ := binomSfGe_nonneg hGw0 hGw1 (n := P) C.recordsToMove
    have : ((C.maxLeaves - (0 + 1 + 1) : ℕ) : ℝ) + 1 ≤ C.maxLeaves := by
      rw [Nat.cast_sub (by omega)]; push_cast; linarith
    nlinarith
  have hmono : ∀ s x, BadStep read C M U θ D ε Ns y Z s x → B s x ∨ Z s x := by
    rintro s x (⟨hn, ⟨e, he, hbad⟩ | hx⟩ | ⟨s', h', hsm, hNs'⟩)
    · exact Or.inl (Or.inl ⟨hn, Or.inl ⟨e, he, hbad⟩⟩)
    · exact Or.inr hx
    · exact Or.inl (Or.inr ⟨s', h', hsm, hNs'⟩)
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (applyProbe read C) (fun s x => B s x ∨ Z s x) initialState
            (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (initialState : Level4State α).probeCount = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns y Z hm _ initialState hst
          (by rw [hn0]; exact hNs) ?_ fun h => hb (anyB_mono (applyProbe read C) _ hmono _ _ h))
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr C (initialState : Level4State α) + 1) * Ns
            ≤ stretches C (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b + Ψr C (initialState : Level4State α) * b + fakeQ C M side Gw P initialState :=
        hany.trans (add_le_add (add_le_add (hseg _ hI0 rfl P) le_rfl) le_rfl)
    _ = ((Ψr C (initialState : Level4State α) + 1 : ℕ) : ENNReal) * b
          + fakeQ C M side Gw P initialState := by push_cast; ring
    _ ≤ (stretches C (Fintype.card α) : ENNReal) * b
          + ENNReal.ofReal (C.maxLeaves * binomSfGe P Gw C.recordsToMove) := by
        gcongr
    _ = _ := by
        have := stretchRisk_nonneg C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt'
          (le_max_of_le_left (by positivity)) (max_le hG1 hy1) ha hθr0 hθr1 hεd'0 hεd'1 hθpt'0
          hθpt'1
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity),
          ← ENNReal.ofReal_add (by positivity)
            (mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _))]

theorem epsRoundCorrect_holds : EpsRoundCorrect := by
  intro α _ _ σ _ M side U θ εw Ω _ μ _ R P C ε y
  obtain ⟨D, _, L, hL⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  have hεw : 0 ≤ εw := measureReal_nonneg.trans (hwrong 1)
  simp only [epsRoundRisk]
  refine le_iInf₂ fun kn hA => ?_
  obtain ⟨⟨Ns, N₁, G, θr, εd', θpt'⟩, Gw⟩ := kn
  obtain ⟨⟨hcap, hm, hn₀, hN₁, hN₁s, hθ, hG0, ha, hθs0, hθs1, hθe, hθpt0, hθpt1, hεd0, hεd1,
    hεd, hθr0, hθr1, hεd'0, hεd'1, hθpt'0, hθpt'1, hsep, hP, hG1, hy1, hGt, hyt⟩, hGw0, hGw1⟩ := hA
  simp only at hN₁ hN₁s hG0 hG1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP hGw0 hGw1 hGt
  unfold epsRoundBound roundBound
  simp only
  set P := C.probeBudget
  set p₀ := prefixMax D C.walkStart
  have hp₀ : 0 < p₀ := prefixMax_pos D C.walkStart
  have hpre : ∀ u : FreeMonoid α, u.toList.length = C.walkStart
      → D.real {x | prefixOf x C.walkStart = u} ≤ p₀ :=
    fun u _ => le_prefixMax D C.walkStart u
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.walkStart (read · ω) T) ≤ G}
  set Nw := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotWrong (read · ω) M side C.walkStart T) ≤ Gw}
  have hNn := noise_le M U θ μ read D C.walkStart L p₀ G hmeas hind hU hL hp₀ hpre hθ
  have hNw : μ Nw ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw) :=
    noise_hit μ read D C.walkStart L (Fintype.card σ) p₀ Gw εw
      (fun z r => r = if side (M.eval z.toList) then .reject else .accept) hmeas hind hwrong hεw
      hL hp₀ hpre
  set c := ENNReal.ofReal (stretches C (Fintype.card α)
    * stretchRisk C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt' + fakeRisk C Gw)
  set B := toMeasurable μ (Nn ∪ Nw)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
          C.walkStart (harvestLaw (read · ω) D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ Nn ∪ Nw := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨hωN, hωW⟩ := hω'
      simp only [Nn, Nw, Set.mem_ofPred_eq, not_not] at hωN hωW
      exact probe_level_eps (read · ω) C M side U θ D ε Ns y L P N₁ G Gw θr εd' θpt' hL hωN hωW
        hcap hm hn₀ hN₁ hN₁s hG0 hG1 hy1 hGt hyt hGw0 hGw1 ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0
        hεd1 hεd hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP
  have hf0 : 0 ≤ fakeRisk C Gw :=
    mul_nonneg (Nat.cast_nonneg _) (binomSfGe_nonneg hGw0 hGw1 _ _)
  have hnn : 0 ≤ stretches C (Fintype.card α)
      * stretchRisk C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt' + fakeRisk C Gw :=
    add_nonneg (mul_nonneg (Nat.cast_nonneg _) (stretchRisk_nonneg C (Fintype.card α) L Ns N₁ _ θr
      εd' θpt' (le_max_of_le_left (by positivity)) (max_le hG1 hy1) ha hθr0 hθr1 hεd'0 hεd'1
      hθpt'0 hθpt'1)) hf0
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ Nn + μ Nw) := by
        gcongr
        rw [measure_toMeasurable]
        exact measure_union_le _ _
    _ ≤ c + (ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G)
          + ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw)) := by
        gcongr
    _ = _ := by
        rw [add_comm c, ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by unfold noiseRisk; positivity), ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by positivity)]
        congr 1
        ring

end Eps

end Random

end OrthoDFA
