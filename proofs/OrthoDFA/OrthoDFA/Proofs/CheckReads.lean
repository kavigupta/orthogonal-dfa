import OrthoDFA.Proofs.CheckRound
import OrthoDFA.Proofs.EpsReads
import OrthoDFA.Proofs.KappaCount

/-!
# The checked round's wrong reads

Given the draws, the checked round's reads come in rounds (`roundsC`), three per draw: what its
probe's step can read, then, where the success test fires, what the check's strings' probes from
`0` can read, then what the probe's split and the fixes' splits re-sift. While nothing read so
far is wrong the steps keep `WitInv`, so each round is fixed by what the earlier ones read and has
at most `(L + 1)|Q|`, `n (L_c + 1)|Q|` or `(n + 1)(|Q| + 1)²|Σ|` strings, `n` the check's size,
and `AdaptiveReadsBound` bounds the chance some step reads a string wrong (`wrongC_le`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (C : Level3Settings) (K : CheckSettings)

/-- The state before the `j`-th draw of `ds`, from `s`, while the round runs. -/
noncomputable def stateAtC (read : FreeMonoid α → ARU) :
    Level3State α → List (CheckDraw α K) → ℕ → Option (Level3State α)
  | s, _, 0 => some s
  | _, [], _ + 1 => none
  | s, d :: ds, j + 1 =>
    match applyCheck read C K s d with
    | .inl s' => stateAtC read s' ds j
    | .inr _ => none

omit [Fintype σ] [Fintype α] in
theorem stateAtC_succ (read : FreeMonoid α → ARU) :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α) (j : ℕ),
      stateAtC C K read s ds (j + 1) = (stateAtC C K read s ds j).bind fun s' =>
        ds[j]?.bind fun d => (applyCheck read C K s' d).elim some fun _ => none
  | [], s, j => by cases j <;> simp [stateAtC]
  | d :: ds, s, 0 => by
    simp only [stateAtC, Option.bind_some, List.getElem?_cons_zero]
    rcases applyCheck read C K s d with s' | e <;> rfl
  | d :: ds, s, j + 1 => by
    simp only [stateAtC, List.getElem?_cons_succ]
    rcases applyCheck read C K s d with s' | e
    · exact stateAtC_succ read ds s' j
    · rfl

/-- What the check's strings' probes from `0` can read. -/
noncomputable def checkPots (s : Level3State α) (b : Fin K.size → FreeMonoid α) : Finset (FreeMonoid α) :=
  Finset.univ.biUnion fun i => pot 0 (b i) s.tree

/-- What the check's strings' splits re-sift. -/
noncomputable def fixReads (read : FreeMonoid α → ARU) (s : Level3State α)
    (b : Fin K.size → FreeMonoid α) : Finset (FreeMonoid α) :=
  Finset.univ.biUnion fun i => splitReads read C.atZero s (b i)

open scoped Classical in
/-- Round `3j` is what the `j`-th draw's probe can read, round `3j + 1` what its check's strings
can, where the success test fires, and round `3j + 2` what its splits re-sift. -/
noncomputable def roundsC (ds : List (CheckDraw α K)) (i : ℕ) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  match stateAtC C K read initialState ds (i / 3), ds[i / 3]? with
  | some s, some d =>
    if i % 3 = 0 then pot 0 d.1 s.tree
    else if i % 3 = 1 then
      (if applyProbe read C s d.1 = .inr .consistent then checkPots K s d.2 else ∅)
    else splitReads read C s d.1
      ∪ (if applyProbe read C s d.1 = .inr .consistent then fixReads C K read s d.2 else ∅)
  | _, _ => ∅

/-- The step reads a string wrong: one its probe reads, or where the check runs, one the check
reads. -/
def WrongC (read : FreeMonoid α → ARU) (s : Level3State α) (d : CheckDraw α K) : Prop :=
  WrongStep M side read C s d.1
    ∨ (applyProbe read C s d.1 = .inr .consistent ∧ CheckWrong read C K M side s d.2)

/-- No string the rounds before `n` read is read wrong. -/
def GoodC (ds : List (CheckDraw α K)) (read : FreeMonoid α → ARU) (n : ℕ) : Prop :=
  ∀ i < n, ∀ z ∈ roundsC C K ds i read,
    read z ≠ if side (M.eval z.toList) then .reject else .accept

section Rounds

variable (ds : List (CheckDraw α K)) (read : FreeMonoid α → ARU) {j : ℕ} {s : Level3State α}
  {d : CheckDraw α K} (hs : stateAtC C K read initialState ds j = some s) (hd : ds[j]? = some d)
include hs hd

omit [Fintype σ] in
theorem roundsC_0 : roundsC C K ds (3 * j) read = pot 0 d.1 s.tree := by
  have h1 : 3 * j / 3 = j := by omega
  have h2 : 3 * j % 3 = 0 := by omega
  simp only [roundsC, h1, hs, hd, h2, if_true]

omit [Fintype σ] in
open scoped Classical in
theorem roundsC_1 : roundsC C K ds (3 * j + 1) read
    = if applyProbe read C s d.1 = .inr .consistent then checkPots K s d.2 else ∅ := by
  have h1 : (3 * j + 1) / 3 = j := by omega
  have h2 : (3 * j + 1) % 3 = 1 := by omega
  simp only [roundsC, h1, hs, hd, h2, one_ne_zero, if_false, if_true]

omit [Fintype σ] in
open scoped Classical in
theorem roundsC_2 : roundsC C K ds (3 * j + 2) read = splitReads read C s d.1
    ∪ (if applyProbe read C s d.1 = .inr .consistent then fixReads C K read s d.2 else ∅) := by
  have h1 : (3 * j + 2) / 3 = j := by omega
  have h2 : (3 * j + 2) % 3 = 2 := by omega
  simp only [roundsC, h1, hs, hd, h2, two_ne_zero, if_false, show (2 : ℕ) ≠ 1 by omega]

end Rounds

omit [Fintype σ] in
/-- A step reading nothing wrong over its three rounds avoids `WrongC`. -/
theorem not_wrongC {ds : List (CheckDraw α K)} {read : FreeMonoid α → ARU} {j : ℕ}
    {s : Level3State α} {d : CheckDraw α K} (hs : stateAtC C K read initialState ds j = some s)
    (hd : ds[j]? = some d) (hg : GoodC M side C K ds read (3 * j + 3)) :
    ¬ WrongC M side C K read s d := by
  classical
  rintro (⟨z, hz, hw⟩ | ⟨hf, i, z, hz, hw⟩)
  · rcases Finset.mem_union.1 hz with hz | hz
    · exact hg (3 * j) (by omega) z (by rw [roundsC_0 C K ds read hs hd]; exact hz) hw
    · exact hg (3 * j + 2) (by omega) z (by
        rw [roundsC_2 C K ds read hs hd]; exact Finset.mem_union_left _ hz) hw
  · rcases Finset.mem_union.1 hz with hz | hz
    · exact hg (3 * j + 1) (by omega) z (by
        rw [roundsC_1 C K ds read hs hd, if_pos hf]
        exact Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩) hw
    · exact hg (3 * j + 2) (by omega) z (by
        rw [roundsC_2 C K ds read hs hd, if_pos hf]
        exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩)) hw

/-- A step changing the hypothesis that reads nothing wrong keeps the invariants. -/
theorem check_keeps' {read : FreeMonoid α → ARU} {s s' : Level3State α} {d : CheckDraw α K}
    (hs : Inv read s) (hg : WitInv M side s) (hx : ¬ WrongC M side C K read s d)
    (h : applyCheck read C K s d = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ WitInv M side s' := by
  rcases applyCheck_cases read C K s d with ⟨he, -⟩ | ⟨hf, he⟩
  · rw [he] at h
    obtain ⟨h1, h2, -⟩ := eps_step_change M side read C hs hg (fun hw => hx (Or.inl hw)) h hne
    exact ⟨h1, h2⟩
  · rw [he] at h
    obtain ⟨i, hi⟩ := checkStep_inl_fix read K h
    obtain ⟨h1, h2, -⟩ := fix_keeps read C M side hs hg (fun hw => hx (Or.inr ⟨hf, i, hw⟩)) hi
    exact ⟨h1, h2⟩

/-- While nothing read is wrong, the steps keep the invariants. -/
theorem inv_of_goodC (ds : List (CheckDraw α K)) (read : FreeMonoid α → ARU) :
    ∀ (j : ℕ) (s : Level3State α), stateAtC C K read initialState ds j = some s →
      GoodC M side C K ds read (3 * j) → Inv read s ∧ WitInv M side s
  | 0, s, h, _ => by
    simp only [stateAtC, Option.some.injEq] at h
    subst h
    exact ⟨inv_start read, witInv_start M side⟩
  | j + 1, s, h, hg => by
    rw [stateAtC_succ] at h
    rcases hs₁ : stateAtC C K read initialState ds j with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hd : ds[j]? with _ | d <;> simp only [hd, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyCheck read C K s₁ d with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨hI, hW⟩ := inv_of_goodC ds read j s₁ hs₁ fun i hi => hg i (by omega)
    have hx := not_wrongC M side C K hs₁ hd fun i hi => hg i (by omega)
    by_cases hsm : s'.tree = s₁.tree ∧ s'.edges = s₁.edges
    · have ha' := applyCheck_same read C K hI ha hsm
      exact ⟨(step_same read C hI ha' hsm.1 hsm.2).1, witInv_congr M side hW hsm.1 hsm.2⟩
    · exact check_keeps' M side C K hI hW hx ha hsm

omit [Fintype α] in
/-- Reads giving the same probe end level 3's step the same way. -/
theorem applyProbe_inr_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (ho : probeOrAgree r s.tree s.edges C.walkStart x
      = probeOrAgree r' s.tree s.edges C.walkStart x) (e : RoundEnd Level3Site) :
    applyProbe r C s x = .inr e ↔ applyProbe r' C s x = .inr e := by
  have hc : countProbe r C s x = countProbe r' C s x := by simp only [countProbe, ho]
  unfold applyProbe
  rw [← ho, hc]
  rcases probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _ <;>
    simp only
  rcases s.edges (p, c) with _ | ⟨t₀, w₀⟩ <;> simp

omit [DecidableEq α] in
theorem startFails_congr {r r' : FreeMonoid α → ARU} {T : DTree α} {E : Edges α}
    {x : FreeMonoid α} (h : ∀ z ∈ pot 0 x T, r z = r' z) :
    startFails r T E x = startFails r' T E x := by
  have := probeOrAgree_congr (E := E) (k := 0) h
  simp only [probeOrAgree, Nat.not_lt_zero, if_false] at this
  simp only [startFails, this]

theorem fixOf_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : ∀ z ∈ pot 0 x s.tree ∪ splitReads r C.atZero s x, r z = r' z) :
    fixOf r s x = fixOf r' s x := by
  have hp : ∀ rd : FreeMonoid α → ARU,
      probeOrAgree rd s.tree s.edges C.atZero.walkStart x = probe rd s.tree s.edges 0 x :=
    fun rd => by simp [probeOrAgree, Level3Settings.atZero]
  have ho : probe r s.tree s.edges 0 x = probe r' s.tree s.edges 0 x := by
    rw [← hp, ← hp]; exact probeOrAgree_congr fun z hz => h z (Finset.mem_union_left _ hz)
  have ha := applyProbe_congr C.atZero hlv h
  unfold applyProbe at ha
  rw [hp, hp, ← ho] at ha
  unfold fixOf
  rw [← ho]
  generalize probe r s.tree s.edges 0 x = o at ha ⊢
  cases o with
  | wrongEdge p c u t =>
    simp only at ha ⊢
    rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩
    · rfl
    · rw [hE] at ha
      simp only [Sum.inl.injEq] at ha
      simp only [ha]
  | _ => rfl

theorem firstFix_congr {r r' : FreeMonoid α → ARU} {s : Level3State α}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves) :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, ∀ z ∈ pot 0 x s.tree ∪ splitReads r C.atZero s x,
      r z = r' z) → firstFix r s xs = firstFix r' s xs
  | [], _ => rfl
  | x :: xs, h => by
    simp only [firstFix]
    rw [fixOf_congr C hlv (h x List.mem_cons_self),
      firstFix_congr hlv xs fun y hy => h y (List.mem_cons_of_mem _ hy)]

/-- Reads agreeing on what a step reads give the same step, its learned edges leaving leaves. -/
theorem applyCheck_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {d : CheckDraw α K}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : ∀ z ∈ pot 0 d.1 s.tree ∪ splitReads r C s d.1, r z = r' z)
    (hc : applyProbe r C s d.1 = .inr .consistent →
      ∀ z ∈ checkPots K s d.2 ∪ fixReads C K r s d.2, r z = r' z) :
    applyCheck r C K s d = applyCheck r' C K s d := by
  classical
  have ha := applyProbe_congr C hlv h
  unfold applyCheck
  rw [← ha]
  rcases hf : applyProbe r C s d.1 with t | e
  · rfl
  rcases e with _ | e | _
  · have hc' := hc hf
    have hP : ∀ i, ∀ z ∈ pot 0 (d.2 i) s.tree, r z = r' z := fun i z hz =>
      hc' z (Finset.mem_union_left _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩))
    have hpass : checkPasses r K s.hypotheses s.tree s.edges (List.ofFn d.2)
        ↔ checkPasses r' K s.hypotheses s.tree s.edges (List.ofFn d.2) := by
      have hcount : ∀ n, ((List.ofFn d.2).take n).countP (startFails r s.tree s.edges)
          = ((List.ofFn d.2).take n).countP (startFails r' s.tree s.edges) := fun n =>
        List.countP_congr fun x hx => by
          obtain ⟨i, rfl⟩ := List.mem_ofFn.1 (List.mem_of_mem_take hx)
          rw [startFails_congr (hP i)]
      simp only [checkPasses, hcount]
    have hfix : firstFix r s (List.ofFn d.2) = firstFix r' s (List.ofFn d.2) :=
      firstFix_congr C hlv _ fun x hx z hz => by
        obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
        rcases Finset.mem_union.1 hz with hz | hz
        · exact hP i z hz
        · exact hc' z (Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩))
    simp only [checkStep, hpass, hfix]
  · rfl
  · rfl

/-- Reads agreeing with good ones on what the rounds before `3j` read reach the same state. -/
theorem stateAtC_congr (ds : List (CheckDraw α K)) {r r' : FreeMonoid α → ARU} :
    ∀ j, GoodC M side C K ds r (3 * j) → (∀ i < 3 * j, ∀ z ∈ roundsC C K ds i r, r z = r' z) →
      stateAtC C K r initialState ds j = stateAtC C K r' initialState ds j
  | 0, _, _ => by simp only [stateAtC]
  | j + 1, hg, ha => by
    classical
    rw [stateAtC_succ, stateAtC_succ, ← stateAtC_congr ds j (fun i hi => hg i (by omega))
      fun i hi => ha i (by omega)]
    rcases hs₁ : stateAtC C K r initialState ds j with _ | s₁
    · rfl
    rcases hd : ds[j]? with _ | d
    · rfl
    simp only [Option.bind_some]
    have hW := (inv_of_goodC M side C K ds r j s₁ hs₁ fun i hi => hg i (by omega)).2
    rw [applyCheck_congr C K (fun e t w h => edge_leaf M side hW h) (fun z hz => ?_)
      (fun hf z hz => ?_)]
    · rcases Finset.mem_union.1 hz with hz | hz
      · exact ha (3 * j) (by omega) z (by rw [roundsC_0 C K ds r hs₁ hd]; exact hz)
      · exact ha (3 * j + 2) (by omega) z (by
          rw [roundsC_2 C K ds r hs₁ hd]; exact Finset.mem_union_left _ hz)
    · rcases Finset.mem_union.1 hz with hz | hz
      · exact ha (3 * j + 1) (by omega) z (by rw [roundsC_1 C K ds r hs₁ hd, if_pos hf]; exact hz)
      · exact ha (3 * j + 2) (by omega) z (by
          rw [roundsC_2 C K ds r hs₁ hd, if_pos hf]; exact Finset.mem_union_right _ hz)

/-- While nothing read is wrong, each round is fixed by what the earlier ones read. -/
theorem roundsC_local (ds : List (CheckDraw α K)) (i : ℕ) (r r' : FreeMonoid α → ARU)
    (hg : GoodC M side C K ds r i) (ha : ∀ l < i, ∀ z ∈ roundsC C K ds l r, r z = r' z) :
    roundsC C K ds i r = roundsC C K ds i r' := by
  classical
  have hj : 3 * (i / 3) ≤ i := Nat.mul_div_le i 3
  have hst := stateAtC_congr M side C K ds (i / 3) (fun l hl => hg l (by omega))
    fun l hl => ha l (by omega)
  rcases hs : stateAtC C K r initialState ds (i / 3) with _ | s
  · unfold roundsC; rw [← hst, hs]
  rcases hd : ds[i / 3]? with _ | d
  · unfold roundsC; rw [← hst, hs, hd]
  have hs' : stateAtC C K r' initialState ds (i / 3) = some s := by rw [← hst, hs]
  set m := i / 3 with hm
  have hi3 : i = 3 * m + i % 3 := by omega
  have hpot : i % 3 ≠ 0 → ∀ z ∈ pot 0 d.1 s.tree, r z = r' z := fun h0 z hz =>
    ha (3 * m) (by omega) z (by rw [roundsC_0 C K ds r hs hd]; exact hz)
  have hfire : i % 3 ≠ 0 → (applyProbe r C s d.1 = .inr .consistent
      ↔ applyProbe r' C s d.1 = .inr .consistent) := fun h0 =>
    applyProbe_inr_congr C (probeOrAgree_congr (hpot h0)) _
  rcases (by omega : i % 3 = 0 ∨ i % 3 = 1 ∨ i % 3 = 2) with h | h | h
  · rw [show i = 3 * m by omega, roundsC_0 C K ds r hs hd, roundsC_0 C K ds r' hs' hd]
  · rw [show i = 3 * m + 1 by omega, roundsC_1 C K ds r hs hd,
      roundsC_1 C K ds r' hs' hd]
    by_cases hf : applyProbe r C s d.1 = .inr .consistent
    · rw [if_pos hf, if_pos ((hfire (by omega)).1 hf)]
    · rw [if_neg hf, if_neg fun h' => hf ((hfire (by omega)).2 h')]
  · rw [show i = 3 * m + 2 by omega, roundsC_2 C K ds r hs hd,
      roundsC_2 C K ds r' hs' hd, splitReads_congr C (hpot (by omega))]
    by_cases hf : applyProbe r C s d.1 = .inr .consistent
    · rw [if_pos hf, if_pos ((hfire (by omega)).1 hf)]
      congr 1
      unfold fixReads
      refine Finset.biUnion_congr rfl fun i' _ => splitReads_congr C.atZero fun z hz =>
        ha (3 * m + 1) (by omega) z ?_
      rw [roundsC_1 C K ds r hs hd, if_pos hf]
      exact Finset.mem_biUnion.2 ⟨i', Finset.mem_univ _, hz⟩
    · rw [if_neg hf, if_neg fun h' => hf ((hfire (by omega)).2 h')]

omit [Fintype σ] in
/-- A run with a step reading a string wrong has it at some step reached. -/
theorem anyB_stepsC (read : FreeMonoid α → ARU) :
    ∀ (ds : List (CheckDraw α K)) (s₀ : Level3State α),
      AnyB (applyCheck read C K) (WrongC M side C K read) s₀ ds →
      ∃ j < ds.length, ∃ s d, stateAtC C K read s₀ ds j = some s ∧ ds[j]? = some d
        ∧ WrongC M side C K read s d
  | [], _, h => by simp [AnyB] at h
  | d :: ds, s₀, h => by
    rcases h with h | ⟨s', hs', h⟩
    · exact ⟨0, by simp, s₀, d, rfl, rfl, h⟩
    · obtain ⟨j, hj, s, e, h1, h2, h3⟩ := anyB_stepsC read ds s' h
      refine ⟨j + 1, by simp; omega, s, e, ?_, by simpa using h2, h3⟩
      simp only [stateAtC, hs']
      exact h1

/-- What a round may read: `(L + 1)|Q|` for a probe, `n (L_c + 1)|Q|` for a check, `(n + 1)`
splits' re-sifts. -/
def roundBoundC (nQ nα n L Lc i : ℕ) : ℕ :=
  if i % 3 = 0 then (L + 1) * nQ else if i % 3 = 1 then n * ((Lc + 1) * nQ)
  else (n + 1) * ((nQ + 1) * nα * (nQ + 1))

/-- While nothing read is wrong, a round has at most `roundBoundC` strings. -/
theorem roundsC_card (ds : List (CheckDraw α K)) (L Lc : ℕ)
    (hL : ∀ d ∈ ds, d.1.toList.length ≤ L) (hLc : ∀ d ∈ ds, ∀ i, (d.2 i).toList.length ≤ Lc)
    (i : ℕ) (r : FreeMonoid α → ARU) (hg : GoodC M side C K ds r i) :
    (roundsC C K ds i r).card ≤ roundBoundC (Fintype.card σ) (Fintype.card α) K.size L Lc i := by
  classical
  have hj : 3 * (i / 3) ≤ i := Nat.mul_div_le i 3
  rcases hs : stateAtC C K r initialState ds (i / 3) with _ | s
  · unfold roundsC; rw [hs]; simp
  rcases hd : ds[i / 3]? with _ | d
  · unfold roundsC; rw [hs, hd]; simp
  have hW := (inv_of_goodC M side C K ds r (i / 3) s hs fun l hl => hg l (by omega)).2
  have hmid : (mids s.tree).card ≤ Fintype.card σ := by
    have := mids_card s.tree
    have := hW.leaves; have := img_card_le M side s.tree; omega
  have hdm : d ∈ ds := List.mem_of_getElem? hd
  have hsplit : ∀ (C' : Level3Settings) x, (splitReads r C' s x).card
      ≤ (Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1) :=
    fun C' x => Kappa.splitReads_card M side r C' hW x
  set m := i / 3
  rcases (by omega : i % 3 = 0 ∨ i % 3 = 1 ∨ i % 3 = 2) with h | h | h
  · rw [show i = 3 * m by omega, roundsC_0 C K ds r hs hd]
    simp only [roundBoundC, show 3 * m % 3 = 0 by omega, if_true]
    calc _ ≤ (d.1.toList.length + 1) * (mids s.tree).card := card_pot 0 d.1 _
      _ ≤ _ := Nat.mul_le_mul (by have := hL d hdm; omega) hmid
  · rw [show i = 3 * m + 1 by omega, roundsC_1 C K ds r hs hd]
    simp only [roundBoundC, show (3 * m + 1) % 3 = 1 by omega, one_ne_zero, if_false, if_true]
    split_ifs
    · unfold checkPots
      refine Finset.card_biUnion_le.trans ?_
      calc _ ≤ ∑ _i : Fin K.size, (Lc + 1) * Fintype.card σ := Finset.sum_le_sum fun i' _ =>
            (card_pot 0 _ _).trans (Nat.mul_le_mul (by have := hLc d hdm i'; omega) hmid)
        _ = _ := by simp
    · simp
  · rw [show i = 3 * m + 2 by omega, roundsC_2 C K ds r hs hd]
    simp only [roundBoundC, show (3 * m + 2) % 3 = 2 by omega, two_ne_zero, if_false,
      show (2 : ℕ) ≠ 1 by omega]
    refine (Finset.card_union_le _ _).trans ?_
    have h1 := hsplit C d.1
    have h2 : (if applyProbe r C s d.1 = .inr .consistent then fixReads C K r s d.2 else ∅).card
        ≤ K.size * ((Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1)) := by
      split_ifs
      · unfold fixReads
        refine Finset.card_biUnion_le.trans ?_
        calc _ ≤ ∑ _i : Fin K.size, (Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1) :=
              Finset.sum_le_sum fun i' _ => hsplit C.atZero _
          _ = _ := by simp
      · simp
    rw [add_mul, one_mul]
    omega

theorem sum_alt3 (n a b c : ℕ) :
    ∑ i ∈ Finset.range (3 * n),
      ((if i % 3 = 0 then a else if i % 3 = 1 then b else c : ℕ) : ℝ) = n * (a + b + c) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 3 * (n + 1) = 3 * n + 1 + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, ih]
    have h0 : 3 * n % 3 = 0 := by omega
    have h1 : (3 * n + 1) % 3 = 1 := by omega
    have h2 : (3 * n + 1 + 1) % 3 = 2 := by omega
    simp only [h0, h1, h2, if_true, one_ne_zero, if_false, two_ne_zero, show (2 : ℕ) ≠ 1 by omega]
    push_cast
    ring

/-- Given draws no longer than `L` and check strings no longer than `L_c`, some step reads a
string wrong with chance at most `εw · |ds| · ((L + 1)|Q| + n (L_c + 1)|Q| + (n + 1)(|Q| + 1)²|Σ|)`.
-/
theorem wrongC_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw : ℝ) (R : EpsReadOracle μ M side U εw) (ds : List (CheckDraw α K)) (L Lc : ℕ)
    (hL : ∀ d ∈ ds, d.1.toList.length ≤ L) (hLc : ∀ d ∈ ds, ∀ i, (d.2 i).toList.length ≤ Lc) :
    μ {ω | AnyB (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω)) initialState ds}
      ≤ ENNReal.ofReal (εw * (ds.length * ((L + 1) * Fintype.card σ
        + K.size * ((Lc + 1) * Fintype.card σ)
        + (K.size + 1) * ((Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1))))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {if side (M.eval z.toList) then .reject else .accept}
  have hmain := adaptiveReadsBound_holds μ R.read B εw (fun i r => roundsC C K ds i r)
    (roundBoundC (Fintype.card σ) (Fintype.card α) K.size L Lc) (3 * ds.length)
    R.measurable R.indep (fun z => R.wrong z)
    (fun i r r' hg ha => roundsC_local M side C K ds i r r' (fun l hl z hz => hg l hl z hz) ha)
    (fun i r hg => roundsC_card M side C K ds L Lc hL hLc i r fun l hl z hz => hg l hl z hz)
  simp only [roundBoundC] at hmain
  rw [sum_alt3] at hmain
  refine (measure_mono fun ω hω => ?_).trans (hmain.trans (le_of_eq ?_))
  · obtain ⟨j, hj, s, d, h1, h2, hw⟩ := anyB_stepsC M side C K (R.read · ω) ds _ hω
    by_contra hno
    simp only [Set.mem_ofPred_eq, not_exists, not_and] at hno
    exact not_wrongC M side C K h1 h2 (fun i hi z hz hzw => hno i (by omega) z hz hzw) hw
  · push_cast
    ring_nf

/-- Over the draws and the reads, some step reads a string wrong with chance at most the same
bound over `P`'s probes and `L_c`-long check strings. -/
theorem pi_wrongC_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw : ℝ) (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc] (Lc : ℕ)
    (hDc : ∀ᵐ x ∂Dc, x.toList.length ≤ Lc) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget =>
        P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
        {ds | AnyB (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω)) initialState
          (List.ofFn ds)} ∂μ
      ≤ ENNReal.ofReal (εw * (C.probeBudget * ((P.L + 1) * Fintype.card σ
        + K.size * ((Lc + 1) * Fintype.card σ)
        + (K.size + 1) * ((Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1))))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points μ _ _ _ fun ds hd => ?_
  have hpi : (Measure.pi fun _ : Fin C.probeBudget =>
      P.D.prod (Measure.pi fun _ : Fin K.size => Dc)) {ds}
      = ∏ j, (P.D.prod (Measure.pi fun _ : Fin K.size => Dc)) {ds j} := by
    rw [← Measure.pi_pi]
    congr 1
    ext d'
    simp [funext_iff]
  have hpt : ∀ j, (P.D.prod (Measure.pi fun _ : Fin K.size => Dc)) {ds j} ≠ 0 := fun j h0 =>
    hd (by rw [hpi]; exact Finset.prod_eq_zero (Finset.mem_univ j) h0)
  have hL : ∀ d ∈ List.ofFn ds, d.1.toList.length ≤ P.L := by
    intro d hdm
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hdm
    by_contra hlt
    apply hpt j
    have : {ds j} ⊆ {x : FreeMonoid α | ¬ x.toList.length ≤ P.L} ×ˢ Set.univ := by
      intro d hd'; rw [Set.mem_singleton_iff] at hd'; subst hd'; exact ⟨hlt, trivial⟩
    refine measure_mono_null this ?_
    rw [Measure.prod_prod]
    simp only [measure_univ, mul_one]
    exact ae_iff.1 P.length_le
  have hLc : ∀ d ∈ List.ofFn ds, ∀ i, (d.2 i).toList.length ≤ Lc := by
    intro d hdm i
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hdm
    by_contra hlt
    apply hpt j
    have : {ds j} ⊆ Set.univ ×ˢ {b : Fin K.size → FreeMonoid α | ¬ (b i).toList.length ≤ Lc} := by
      intro d hd'; rw [Set.mem_singleton_iff] at hd'; subst hd'; exact ⟨trivial, hlt⟩
    refine measure_mono_null this ?_
    rw [Measure.prod_prod]
    have hnull : (Measure.pi fun _ : Fin K.size => Dc)
        {b : Fin K.size → FreeMonoid α | ¬ (b i).toList.length ≤ Lc} = 0 := by
      have : {b : Fin K.size → FreeMonoid α | ¬ (b i).toList.length ≤ Lc}
          = Function.eval i ⁻¹' {x | ¬ x.toList.length ≤ Lc} := rfl
      rw [this, ← Measure.map_apply (measurable_pi_apply i) (Set.to_countable _).measurableSet,
        Measure.pi_map_eval]
      simp only [Finset.prod_const_one, measure_univ, one_smul]
      exact ae_iff.1 hDc
    rw [hnull, mul_zero]
  have := wrongC_le M side C K μ U εw R (List.ofFn ds) P.L Lc hL hLc
  rw [List.length_ofFn] at this
  exact this

end Eps

end Random

end OrthoDFA
