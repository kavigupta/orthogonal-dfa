import OrthoDFA.Proofs.Allowance

/-!
# A step's reads, one round for its probe and one for each edge its split re-sifts

The step at `s` with probe `x` reads in `R + 1` rounds (`stepRound`): its probe's reads, then,
for each edge `s` can hold in a fixed order (`edgeList`), the sift of the edge's successor
through the split tree, where the split re-sifts that edge (`resiftOf`). The run's rounds
(`rounds6`) stop at the first state with more than `n + 2` leaves, so that while they run every
edge has its round, `|edgeList| ≤ (n + 2)|Σ| ≤ R`. Each round is then fixed by what it and the
earlier rounds read (`rounds6_local`).
-/

namespace OrthoDFA

namespace Random

namespace Spur

open Eps Kappa

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- Every letter, once. -/
noncomputable def letters : List α := (Finset.univ : Finset α).toList

/-- The edges a state can hold, in a fixed order. -/
noncomputable def edgeList (s : Level3State α) : List (List Bool × α) :=
  s.tree.leaves ×ˢ letters

/-- Where the step splits, its leaf and the tree after. -/
noncomputable def splitOf (s : Level3State α) (x : FreeMonoid α) : Option (List Bool × DTree α) :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) =>
      some (p, s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
    | none => none
  | _ => none

/-- What the split re-sifts of the edge `e`: its successor, through the new tree, where `e`
points into the split leaf from outside it. -/
noncomputable def resiftOf (s : Level3State α) (x : FreeMonoid α) (e : List Bool × α) :
    Finset (FreeMonoid α) :=
  match splitOf read C s x, s.edges e with
  | some (p, T'), some (t, w) =>
    if p.isPrefixOf e.1 = false ∧ t = p then siftReads read T' (w * FreeMonoid.of e.2) else ∅
  | _, _ => ∅

/-- Round `i` of the step at `s` with probe `x`: the probe's reads, then the re-sifts in turn. -/
noncomputable def stepRound (s : Level3State α) (x : FreeMonoid α) (i : ℕ) :
    Finset (FreeMonoid α) :=
  if i = 0 then probeReads read s.tree s.edges C.walkStart x
  else match (edgeList s)[i - 1]? with
    | some e => resiftOf read C s x e
    | none => ∅


/-! ## Every step keeps the invariant and its edges at leaves -/

/-- Every learned edge leaves a leaf. -/
def Keys (s : Level3State α) : Prop := ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves

theorem inv_keys_step {s s' : Level3State α} {x : FreeMonoid α} (hs : Inv read s) (hk : Keys s)
    (h : applyProbe read C s x = .inl s') : Inv read s' ∧ Keys s' := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · refine ⟨⟨edgesOK_upd read hs.edges hu huc, hs.cls, hs.two⟩, fun e t' w he => ?_⟩
    simp only [newHypothesis] at he ⊢
    by_cases hpc : e = (p, c)
    · subst hpc; exact hp
    · rw [Function.update_of_ne hpc] at he; exact hk e t' w he
  · have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have ht := DTree.sift_inl_mem read _ _ t huc
    have ht₀ := DTree.sift_inl_mem read _ _ t₀ (hs.edges p hp c t₀ w₀ hE).2
    refine ⟨⟨edgesOK_retarget read hs.edges hp _, ?_, ?_⟩, fun e t' w he => ?_⟩
    · simp only [splitLeaf, newHypothesis]
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hs.two; omega]
      exact classSet_split hs.cls hp ht ht₀
    · simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega
    · simp only [splitLeaf, newHypothesis, retarget] at he ⊢
      split_ifs at he with hpre
      rcases hE' : s.edges e with _ | ⟨t₁, w₁⟩ <;> rw [hE'] at he
      · simp at he
      refine DTree.mem_leaves_splitAt_of_ne _ _ _ _ (hk e t₁ w₁ hE') ?_
      rintro rfl
      simp at hpre
  · exact ⟨⟨hs.edges, hs.cls, hs.two⟩, hk⟩

theorem inv_keys_stateAt : ∀ (xs : List (FreeMonoid α)) (j : ℕ) (s : Level3State α),
    stateAt C read initialState xs j = some s → Inv read s ∧ Keys s
  | xs, 0, s, h => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    exact ⟨inv_start read, fun _ _ _ h => by simp [initialState, newHypothesis] at h⟩
  | xs, j + 1, s, h => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C read initialState xs j with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[j]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe read C s₁ x with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨h1, h2⟩ := inv_keys_stateAt xs j s₁ hs₁
    exact inv_keys_step read C h1 h2 ha

/-! ## Reads agreeing on a step's rounds give the same step -/

omit [DecidableEq α] in
theorem mem_edgeList {s : Level3State α} {e : List Bool × α} (he : e.1 ∈ s.tree.leaves) :
    e ∈ edgeList s := by
  unfold edgeList letters
  exact List.mem_product.2 ⟨he, Finset.mem_toList.2 (Finset.mem_univ _)⟩

omit [DecidableEq α] in
theorem length_edgeList (s : Level3State α) :
    (edgeList s).length = s.tree.leaves.length * Fintype.card α := by
  simp [edgeList, letters, List.length_product]

theorem applyProbe_congr6 {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hk : Keys s) (hp : ∀ z ∈ probeReads r s.tree s.edges C.walkStart x, r z = r' z)
    (hres : ∀ e ∈ edgeList s, ∀ z ∈ resiftOf r C s x e, r z = r' z) :
    applyProbe r C s x = applyProbe r' C s x := by
  have ho : probeOrAgree r s.tree s.edges C.walkStart x
      = probeOrAgree r' s.tree s.edges C.walkStart x :=
    (probeReads_congr s.edges C.walkStart hp).1.symm
  have hc : countProbe r C s x = countProbe r' C s x := by simp only [countProbe, ho]
  unfold applyProbe
  rw [← ho, hc]
  rcases hO : probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
  all_goals try rfl
  dsimp only
  rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩
  · rfl
  dsimp only
  simp only [splitLeaf, Sum.inl.injEq]
  refine congrArg (fun E => newHypothesis _ E _) ?_
  funext e
  unfold retarget
  split_ifs with hpre
  · rfl
  rcases hE' : s.edges e with _ | ⟨t', w⟩
  · rfl
  simp only
  split_ifs with htp
  · have hmem := mem_edgeList (hk e t' w hE')
    have hro : resiftOf r C s x e = siftReads r (s.tree.splitAt
        (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) (w * FreeMonoid.of e.2) := by
      unfold resiftOf splitOf
      rw [hO]
      simp only [hE, hE']
      rw [if_pos ⟨Bool.eq_false_iff.2 hpre, htp⟩]
    rw [(siftReads_congr _ _ fun z hz => hres e hmem z (hro ▸ hz)).1]
  · rfl

variable (xs : List (FreeMonoid α)) (n R : ℕ)

/-- The states of the first `j + 1` steps have at most `n + 2` leaves. -/
def Small (j : ℕ) : Prop :=
  ∀ i ≤ j, ∀ s, stateAt C read initialState xs i = some s → s.tree.leaves.length ≤ n + 2

open scoped Classical in
/-- Round `ρ` of the run: round `ρ mod (R + 1)` of step `⌊ρ/(R + 1)⌋`, while the states are
small. -/
noncomputable def rounds6 (ρ : ℕ) (read : FreeMonoid α → ARU) : Finset (FreeMonoid α) :=
  if Small read C xs n (ρ / (R + 1)) then
    match stateAt C read initialState xs (ρ / (R + 1)), xs[ρ / (R + 1)]? with
    | some s, some x => stepRound read C s x (ρ % (R + 1))
    | _, _ => ∅
  else ∅

variable {C xs n R}

theorem rounds6_eq {ρ : ℕ} {r : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hS : Small r C xs n (ρ / (R + 1)))
    (hs : stateAt C r initialState xs (ρ / (R + 1)) = some s) (hx : xs[ρ / (R + 1)]? = some x) :
    rounds6 C xs n R ρ r = stepRound r C s x (ρ % (R + 1)) := by
  simp only [rounds6, if_pos hS, hs, hx]

theorem rounds6_of_not_small {ρ : ℕ} {r : FreeMonoid α → ARU}
    (hS : ¬ Small r C xs n (ρ / (R + 1))) : rounds6 C xs n R ρ r = ∅ := by
  simp only [rounds6, if_neg hS]

omit [DecidableEq α] in
theorem resiftOf_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    {e : List Bool × α}
    (hpo : probeOrAgree r' s.tree s.edges C.walkStart x
      = probeOrAgree r s.tree s.edges C.walkStart x)
    (h : ∀ z ∈ resiftOf r C s x e, r z = r' z) : resiftOf r C s x e = resiftOf r' C s x e := by
  have hsplit : splitOf r' C s x = splitOf r C s x := by unfold splitOf; rw [hpo]
  unfold resiftOf at h ⊢
  rw [hsplit]
  rcases hso : splitOf r C s x with _ | ⟨p, T'⟩
  · rfl
  rcases hE : s.edges e with _ | ⟨t, w⟩
  · rfl
  rw [hso, hE] at h
  dsimp only at h ⊢
  split_ifs with hc
  · rw [if_pos hc] at h
    exact (siftReads_congr _ _ h).2.symm
  · rfl

/-- Reads agreeing on a step's rounds, its state small, give the same next state. -/
theorem stateAt_congr6 (hR : (n + 2) * Fintype.card α ≤ R) {r r' : FreeMonoid α → ARU}
    {J : ℕ} (h : ∀ i < J * (R + 1), ∀ z ∈ rounds6 C xs n R i r, r z = r' z) :
    ∀ j ≤ J, (∀ i < j, ∀ s, stateAt C r initialState xs i = some s →
      s.tree.leaves.length ≤ n + 2) →
      stateAt C r initialState xs j = stateAt C r' initialState xs j
  | 0, _, _ => by simp only [stateAt]
  | j + 1, hj, hsm => by
    have ih := stateAt_congr6 hR h j (by omega) fun i hi => hsm i (by omega)
    rw [stateAt_succ, stateAt_succ, ← ih]
    rcases hs : stateAt C r initialState xs j with _ | s
    · rfl
    rcases hx : xs[j]? with _ | x
    · rfl
    simp only [Option.bind_some]
    have hS : Small r C xs n j := fun i hi s' hs' => hsm i (by omega) s' hs'
    have hdiv : ∀ i < R + 1, (j * (R + 1) + i) / (R + 1) = j := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
    have hmod : ∀ i < R + 1, (j * (R + 1) + i) % (R + 1) = i := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    have hround : ∀ i < R + 1, rounds6 C xs n R (j * (R + 1) + i) r = stepRound r C s x i :=
      fun i hi => by
        rw [rounds6_eq (s := s) (x := x) (by rw [hdiv i hi]; exact hS) (by rw [hdiv i hi]; exact hs)
          (by rw [hdiv i hi]; exact hx), hmod i hi]
    have hlt : ∀ i < R + 1, j * (R + 1) + i < J * (R + 1) := fun i hi => by
      have : (j + 1) * (R + 1) ≤ J * (R + 1) := Nat.mul_le_mul_right _ hj
      nlinarith
    have hk := (inv_keys_stateAt r C xs j s hs).2
    rw [applyProbe_congr6 C hk (fun z hz => h _ (hlt 0 (by omega)) z (by
      rw [hround 0 (by omega)]; simpa [stepRound] using hz)) fun e he z hz => ?_]
    obtain ⟨i, hi, hie⟩ := List.getElem_of_mem he
    have hilen : i < R := by
      have := length_edgeList s
      have := hsm j (by omega) s hs
      have : s.tree.leaves.length * Fintype.card α ≤ (n + 2) * Fintype.card α :=
        Nat.mul_le_mul_right _ this
      omega
    refine h _ (hlt (i + 1) (by omega)) z ?_
    rw [hround (i + 1) (by omega)]
    simp only [stepRound, Nat.add_sub_cancel, if_neg (Nat.succ_ne_zero i),
      List.getElem?_eq_getElem hi, hie]
    exact hz

/-- Each round is fixed by what it and the rounds before it read. -/
theorem rounds6_local (hR : (n + 2) * Fintype.card α ≤ R) (ρ : ℕ) (r r' : FreeMonoid α → ARU)
    (h : ∀ i ≤ ρ, ∀ z ∈ rounds6 C xs n R i r, r z = r' z) :
    rounds6 C xs n R ρ r = rounds6 C xs n R ρ r' := by
  classical
  set J := ρ / (R + 1)
  have hJ : J * (R + 1) ≤ ρ := Nat.div_mul_le_self ρ (R + 1)
  have h' : ∀ i < J * (R + 1), ∀ z ∈ rounds6 C xs n R i r, r z = r' z :=
    fun i hi => h i (by omega)
  by_cases hS : Small r C xs n J
  · have hst : ∀ j ≤ J, stateAt C r initialState xs j = stateAt C r' initialState xs j :=
      fun j hj => stateAt_congr6 hR h' j hj fun i hi s hs => hS i (by omega) s hs
    have hS' : Small r' C xs n J := fun i hi s hs => hS i hi s (by rw [hst i hi]; exact hs)
    rcases hs : stateAt C r initialState xs J with _ | s
    · simp only [rounds6, if_pos hS, if_pos hS', ← hst J le_rfl, hs, J]
    rcases hx : xs[J]? with _ | x
    · simp only [rounds6, if_pos hS, if_pos hS', ← hst J le_rfl, hs, hx, J]
    rw [rounds6_eq hS hs hx, rounds6_eq hS' (by rw [← hst J le_rfl]; exact hs) hx]
    have hround := rounds6_eq (n := n) (R := R) (ρ := ρ) hS hs hx
    have hρ := h ρ le_rfl
    rw [hround] at hρ
    have hpr : ∀ z ∈ probeReads r s.tree s.edges C.walkStart x, r z = r' z := by
      intro z hz
      refine h (J * (R + 1)) hJ z ?_
      rw [rounds6_eq (s := s) (x := x) (by rw [Nat.mul_div_cancel _ (by omega)]; exact hS)
        (by rw [Nat.mul_div_cancel _ (by omega)]; exact hs)
        (by rw [Nat.mul_div_cancel _ (by omega)]; exact hx), Nat.mul_mod_left]
      simpa [stepRound] using hz
    obtain ⟨hpo, hpr'⟩ := probeReads_congr s.edges C.walkStart hpr
    unfold stepRound at hρ ⊢
    split_ifs with hi
    · exact hpr'.symm
    rcases he : (edgeList s)[ρ % (R + 1) - 1]? with _ | e
    · rfl
    rw [if_neg hi, he] at hρ
    exact resiftOf_congr hpo hρ
  · have hS' : ¬ Small r' C xs n J := by
      intro hS'
      apply hS
      intro i hi s hs
      by_contra hlt
      push Not at hlt
      have hP : ∃ i, i ≤ J ∧ ∃ s, stateAt C r initialState xs i = some s
          ∧ n + 2 < s.tree.leaves.length := ⟨i, hi, s, hs, hlt⟩
      obtain ⟨hi0, s0, hs0, hlt0⟩ := Nat.find_spec hP
      have hmin : ∀ i' < Nat.find hP, ∀ s', stateAt C r initialState xs i' = some s' →
          s'.tree.leaves.length ≤ n + 2 := fun i' hi' s' hs' => by
        by_contra hc
        exact Nat.find_min hP hi' ⟨by omega, s', hs', by omega⟩
      have := stateAt_congr6 hR h' (Nat.find hP) hi0 hmin
      rw [hs0] at this
      have := hS' (Nat.find hP) hi0 s0 this.symm
      omega
    rw [rounds6_of_not_small hS, rounds6_of_not_small hS']

end Spur

end Random

end OrthoDFA
