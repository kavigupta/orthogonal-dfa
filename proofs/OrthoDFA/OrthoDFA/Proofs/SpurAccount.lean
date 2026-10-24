import OrthoDFA.Proofs.SpurRounds
import OrthoDFA.Proofs.RaceTail
set_option profiler true
set_option profiler.threshold 2000

/-!
# Spurious leaves cost fresh-bad rounds

Where no two probes share their first `k` letters, every string a round reads belongs to one
probe, the one whose first `k` letters it starts with (`Belongs`): a probe's own reads to it, an
edge's re-sifts to the probe that learned the edge. So the rounds of a step read disjoint strings,
and no round of one probe can taint another's edge.

An edge is bad when its witness is off its true leaf or its successor off its true target, or a
string read so far is wrong and extends its successor (it may yet be re-read). Then

    leaves + bad edges ≤ true leaves + 1 + fresh-bad rounds

holds along the run (`account`): a step's probe round, if wrong somewhere, adds at most one leaf
or one bad edge; a re-sift round, wrong somewhere new, at most one bad edge; a split on a bad edge
drops it; and every other split is genuine, adding a true leaf with the leaf.
-/

namespace OrthoDFA

namespace Random

namespace Spur

open Eps Kappa AdaptiveReads RaceTail

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## Strings belong to the probe whose first `k` letters they start with -/

section Belongs

variable (xs : List (FreeMonoid α)) (k : ℕ)

/-- No two probes long enough to be walked share their first `k` letters. -/
def Distinct : Prop :=
  ∀ (i j : ℕ) (x y : FreeMonoid α), i ≠ j → xs[i]? = some x → xs[j]? = some y →
    k ≤ x.toList.length →
    k ≤ y.toList.length → x.toList.take k ≠ y.toList.take k

/-- `z` starts with the first `k` letters of the `i`-th probe. -/
def Belongs (i : ℕ) (z : FreeMonoid α) : Prop :=
  ∃ x : FreeMonoid α, xs[i]? = some x ∧ k ≤ z.toList.length ∧ z.toList.take k = x.toList.take k

variable {xs k}

omit [Fintype α] [DecidableEq α] in
theorem belongs_unique (hd : Distinct xs k) {i j : ℕ} {z : FreeMonoid α} (hi : Belongs xs k i z)
    (hj : Belongs xs k j z) : i = j := by
  obtain ⟨x, hx, hz, hzx⟩ := hi
  obtain ⟨y, hy, -, hzy⟩ := hj
  by_contra hne
  have hlx : k ≤ x.toList.length := by
    have := congrArg List.length hzx
    simp only [List.length_take] at this
    omega
  have hly : k ≤ y.toList.length := by
    have := congrArg List.length hzy
    simp only [List.length_take] at this
    omega
  exact hd i j x y hne hx hy hlx hly (hzx.symm.trans hzy)

omit [DecidableEq α] in
theorem belongs_mul {i : ℕ} {w : FreeMonoid α} (m : FreeMonoid α) (hw : Belongs xs k i w) :
    Belongs xs k i (w * m) := by
  obtain ⟨x, hx, hl, ht⟩ := hw
  refine ⟨x, hx, by simp only [FreeMonoid.toList_mul, List.length_append]; omega, ?_⟩
  rw [FreeMonoid.toList_mul, List.take_append_of_le_length hl, ht]

omit [DecidableEq α] in
theorem belongs_of_mul {i : ℕ} {w m : FreeMonoid α} (hl : k ≤ w.toList.length)
    (hw : Belongs xs k i (w * m)) : Belongs xs k i w := by
  obtain ⟨x, hx, -, ht⟩ := hw
  refine ⟨x, hx, hl, ?_⟩
  rw [FreeMonoid.toList_mul, List.take_append_of_le_length hl] at ht
  exact ht

omit [Fintype α] [DecidableEq α] in
theorem belongs_prefixOf {i p : ℕ} {x : FreeMonoid α} (hx : xs[i]? = some x) (hkp : k ≤ p)
    (hp : p ≤ x.toList.length) : Belongs xs k i (prefixOf x p) := by
  refine ⟨x, hx, by simp [prefixOf]; omega, ?_⟩
  simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take, min_eq_left hkp]

omit [DecidableEq α] in
theorem belongs_probeReads {i : ℕ} {x : FreeMonoid α} (hx : xs[i]? = some x)
    {r : FreeMonoid α → ARU} {T : DTree α} {E : Edges α} {z : FreeMonoid α}
    (hz : z ∈ probeReads r T E k x) : Belongs xs k i z := by
  classical
  have := probeReads_pot (rd := r) (T := T) E k (x := x) hz
  unfold pot at this
  obtain ⟨p, hp, hz'⟩ := Finset.mem_biUnion.1 this
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz'
  obtain ⟨h1, h2⟩ := Finset.mem_Icc.1 hp
  exact belongs_mul m (belongs_prefixOf hx h1 h2)

end Belongs

/-! ## A block of rounds -/

section Block

variable {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B : Z → Set R) (r : Z → R)

open scoped Classical in
theorem badCount_add (a : ℕ) : ∀ m, badCount Q B (a + m) r
    = badCount Q B a r + ((Finset.range m).filter fun i => FreshBad Q B (a + i) r).card
  | 0 => by simp
  | m + 1 => by
    rw [← Nat.add_assoc, badCount_succ, badCount_add a m, Finset.range_add_one,
      Finset.filter_insert]
    split_ifs with h
    · rw [Finset.card_insert_of_notMem (by simp)]; ring
    · rfl

theorem mem_before_add {a m : ℕ} {z : Z} :
    z ∈ before Q (a + m) r ↔ z ∈ before Q a r ∨ ∃ i < m, z ∈ Q (a + i) r := by
  simp only [mem_before]
  constructor
  · rintro ⟨i, hi, hz⟩
    by_cases hia : i < a
    · exact Or.inl ⟨i, hia, hz⟩
    · exact Or.inr ⟨i - a, by omega, by rwa [show a + (i - a) = i by omega]⟩
  · rintro (⟨i, hi, hz⟩ | ⟨i, hi, hz⟩)
    · exact ⟨i, by omega, hz⟩
    · exact ⟨a + i, by omega, hz⟩

theorem freshBad_of {ρ : ℕ} {z : Z} (hz : z ∈ Q ρ r) (hB : r z ∈ B z)
    (hnew : ∀ i < ρ, z ∉ Q i r) : FreshBad Q B ρ r := by
  classical
  exact ⟨z, Finset.mem_filter.2 ⟨hz, hnew⟩, hB⟩

end Block

/-! ## Bad edges -/

section Bad

variable {σ : Type*} (M : DFA α σ) (side : σ → Bool) (r : FreeMonoid α → ARU)

/-- `z` reads on its state's wrong side. -/
def Wrong (z : FreeMonoid α) : Prop := r z = if side (M.eval z.toList) then .reject else .accept

/-- The edge `e`, to `t` with witness `w`, has its witness at its true leaf and its successor at
its true target. -/
def Sound (T : DTree α) (e : List Bool × α) (t : List Bool) (w : FreeMonoid α) : Prop :=
  leafOf M side T (M.eval w.toList) = e.1
    ∧ leafOf M side T (M.eval (w * FreeMonoid.of e.2).toList) = t

/-- A string read so far is wrong and extends the edge's successor. -/
def Tainted (Qd : Finset (FreeMonoid α)) (e : List Bool × α) (w : FreeMonoid α) : Prop :=
  ∃ z ∈ Qd, Wrong M side r z ∧ ∃ m, z = w * FreeMonoid.of e.2 * m

/-- The edge is learned and unsound or tainted. -/
def BadE (s : Level3State α) (Qd : Finset (FreeMonoid α)) (e : List Bool × α) : Prop :=
  ∃ t w, s.edges e = some (t, w) ∧ (¬ Sound M side s.tree e t w ∨ Tainted M side r Qd e w)

open scoped Classical in
/-- The bad edges. -/
noncomputable def bad (s : Level3State α) (Qd : Finset (FreeMonoid α)) : ℕ :=
  ((s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter (BadE M side r s Qd)).card

end Bad

omit [DecidableEq α] in
theorem retarget_some {r : FreeMonoid α → ARU} {T' : DTree α} {p : List Bool} {E : Edges α}
    {e : List Bool × α} {t' : List Bool} {w : FreeMonoid α}
    (h : retarget r T' p E e = some (t', w)) :
    p.isPrefixOf e.1 = false ∧ ∃ t₁, E e = some (t₁, w)
      ∧ ((t₁ ≠ p ∧ t' = t₁) ∨ (t₁ = p ∧ T'.sift r (w * FreeMonoid.of e.2) = .inl t')) := by
  unfold retarget at h
  split_ifs at h with hpre
  rcases hE : E e with _ | ⟨t₁, w₁⟩ <;> rw [hE] at h
  · simp at h
  simp only at h
  refine ⟨Bool.eq_false_iff.2 hpre, t₁, ?_⟩
  split_ifs at h with ht
  · rcases hs : T'.sift r (w₁ * FreeMonoid.of e.2) with a | z <;> rw [hs] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, Prod.mk.injEq,
        reduceCtorEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, Or.inr ⟨ht, hs⟩⟩
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, Or.inl ⟨ht, rfl⟩⟩

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

omit [Fintype α] [DecidableEq α] in
theorem img_card_mono {T : DTree α} {p : List Bool} (hp : p ∈ T.leaves) (d : FreeMonoid α) :
    (img M side T).card ≤ (img M side (T.splitAt d p)).card := by
  classical
  refine Finset.card_le_card_of_injOn
    (fun l => if h : ∃ q, leafOf M side T q = l then leafOf M side (T.splitAt d p) h.choose else l)
    (fun l hl => ?_) ?_
  · obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hl
    have h : ∃ q', leafOf M side T q' = leafOf M side T q := ⟨q, rfl⟩
    simp only [dif_pos h]
    exact Finset.mem_image.2 ⟨_, Finset.mem_univ _, rfl⟩
  · intro l₁ hl₁ l₂ hl₂ heq
    obtain ⟨q₁, -, rfl⟩ := Finset.mem_image.1 hl₁
    obtain ⟨q₂, -, rfl⟩ := Finset.mem_image.1 hl₂
    have h₁ : ∃ q', leafOf M side T q' = leafOf M side T q₁ := ⟨q₁, rfl⟩
    have h₂ : ∃ q', leafOf M side T q' = leafOf M side T q₂ := ⟨q₂, rfl⟩
    simp only [dif_pos h₁, dif_pos h₂] at heq
    have e₁ := h₁.choose_spec
    have e₂ := h₂.choose_spec
    rw [leafOf_splitAt M side d T p hp, leafOf_splitAt M side d T p hp] at heq
    rw [← e₁, ← e₂]
    split_ifs at heq with ha hb hb
    · rw [ha, hb]
    · exfalso
      have := leafOf_mem M side T h₂.choose
      rw [← heq] at this
      exact DTree.snoc_not_mem_leaves T p hp _ this
    · exfalso
      have := leafOf_mem M side T h₁.choose
      rw [heq] at this
      exact DTree.snoc_not_mem_leaves T p hp _ this
    · exact heq

/-! ## One step -/

section Step

variable (r : FreeMonoid α → ARU) (C : Level3Settings) (xs : List (FreeMonoid α))

/-- Before the `j`-th probe: every edge's witness belongs to an earlier probe, no two edges to
the same one, and so do the strings read so far. -/
structure Owned (j : ℕ) (s : Level3State α) (Qd : Finset (FreeMonoid α)) : Prop where
  wit : ∀ e t w, s.edges e = some (t, w) → ∃ i < j, Belongs xs C.walkStart i w
  inj : ∀ e e' t w t' w', s.edges e = some (t, w) → s.edges e' = some (t', w') →
    w.toList.take C.walkStart = w'.toList.take C.walkStart → e = e'
  read : ∀ z ∈ Qd, ∃ i < j, Belongs xs C.walkStart i z

variable {r C xs}

omit [DecidableEq α] in
theorem mem_resiftOf {s : Level3State α} {x : FreeMonoid α} {e : List Bool × α}
    {z : FreeMonoid α} (hz : z ∈ resiftOf r C s x e) :
    ∃ p T' t w, splitOf r C s x = some (p, T') ∧ s.edges e = some (t, w)
      ∧ p.isPrefixOf e.1 = false ∧ t = p ∧ z ∈ siftReads r T' (w * FreeMonoid.of e.2) := by
  unfold resiftOf at hz
  rcases hso : splitOf r C s x with _ | ⟨p, T'⟩ <;> rw [hso] at hz
  · simp at hz
  rcases hE : s.edges e with _ | ⟨t, w⟩ <;> rw [hE] at hz
  · simp at hz
  dsimp only at hz
  split_ifs at hz with hc
  · exact ⟨p, T', t, w, rfl, rfl, hc.1, hc.2, hz⟩
  · simp at hz

omit [DecidableEq α] in
theorem resiftOf_eq {s : Level3State α} {x : FreeMonoid α} {e : List Bool × α} {p : List Bool}
    {T' : DTree α} {t : List Bool} {w : FreeMonoid α} (hso : splitOf r C s x = some (p, T'))
    (hE : s.edges e = some (t, w)) (hpre : p.isPrefixOf e.1 = false) (ht : t = p) :
    resiftOf r C s x e = siftReads r T' (w * FreeMonoid.of e.2) := by
  unfold resiftOf
  rw [hso, hE]
  dsimp only
  rw [if_pos ⟨hpre, ht⟩]

omit [Fintype α] [DecidableEq α] in
theorem take_eq_of_belongs {i : ℕ} {w w' : FreeMonoid α} {k : ℕ} (h : Belongs xs k i w)
    (h' : Belongs xs k i w') : w.toList.take k = w'.toList.take k := by
  obtain ⟨x, hx, -, hw⟩ := h
  obtain ⟨x', hx', -, hw'⟩ := h'
  rw [hx] at hx'
  cases hx'
  rw [hw, hw']

omit [DecidableEq α] in
/-- A string this step reads that extends an edge's successor is that edge's re-sift. -/
theorem taint_new (hd : Distinct xs C.walkStart) {j : ℕ} {s : Level3State α}
    {x : FreeMonoid α} (hx : xs[j]? = some x) {Qd : Finset (FreeMonoid α)}
    (hown : Owned C xs j s Qd) {e : List Bool × α} {t : List Bool} {w m z : FreeMonoid α}
    (he : s.edges e = some (t, w)) (hz : z = w * FreeMonoid.of e.2 * m)
    (hmem : z ∈ probeReads r s.tree s.edges C.walkStart x
      ∨ ∃ e', z ∈ resiftOf r C s x e') : z ∈ resiftOf r C s x e := by
  obtain ⟨i, hi, hbw⟩ := hown.wit e t w he
  have hbz : Belongs xs C.walkStart i z := by
    rw [hz, mul_assoc]; exact belongs_mul _ hbw
  rcases hmem with hmem | ⟨e', hmem⟩
  · exact absurd (belongs_unique hd hbz (belongs_probeReads hx hmem)) (by omega)
  · obtain ⟨p, T', t', w', hso, hE', hpre, htp, hzs⟩ := mem_resiftOf hmem
    obtain ⟨m', -, hzm⟩ := siftReads_mids T' _ z hzs
    obtain ⟨i', -, hbw'⟩ := hown.wit e' t' w' hE'
    have hbz' : Belongs xs C.walkStart i' z := by
      rw [hzm, mul_assoc]; exact belongs_mul _ hbw'
    have hii := belongs_unique hd hbz hbz'
    subst hii
    have := hown.inj e e' t w t' w' he hE' (take_eq_of_belongs hbw hbw')
    subst this
    exact hmem

end Step

section StepAccount

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)
  {r : FreeMonoid α → ARU} {C : Level3Settings} {xs : List (FreeMonoid α)}

omit [DecidableEq α] in
theorem mem_K {s : Level3State α} {e : List Bool × α} (he : e.1 ∈ s.tree.leaves) :
    e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α) :=
  Finset.mem_product.2 ⟨List.mem_toFinset.2 he, Finset.mem_univ _⟩

open scoped Classical in
/-- A step adds to the leaves and bad edges at most what it adds to the true leaves, plus one for
a wrong probe round and one for each re-sift wrong somewhere new. -/
theorem step_account (hd : Distinct xs C.walkStart) {j : ℕ} {s s' : Level3State α}
    {x : FreeMonoid α} (hx : xs[j]? = some x) (hs : Inv r s) (hk : Keys s)
    {Qd : Finset (FreeMonoid α)} (hown : Owned C xs j s Qd)
    (h : applyProbe r C s x = .inl s') :
    s'.tree.leaves.length
        + bad M side r s' (Qd ∪ probeReads r s.tree s.edges C.walkStart x
          ∪ (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).biUnion (resiftOf r C s x))
        + (img M side s.tree).card
      ≤ (img M side s'.tree).card + s.tree.leaves.length + bad M side r s Qd
        + (if ∃ z ∈ probeReads r s.tree s.edges C.walkStart x, Wrong M side r z then 1 else 0)
        + ((s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter fun e =>
            ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd).card := by
  set K := s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)
  set PR := probeReads r s.tree s.edges C.walkStart x
  set Qd' := Qd ∪ PR ∪ K.biUnion (resiftOf r C s x)
  have hnew : ∀ e t w m z, s.edges e = some (t, w) → z = w * FreeMonoid.of e.2 * m →
      z ∈ Qd' → z ∈ Qd ∨ z ∈ resiftOf r C s x e := by
    intro e t w m z he hz hzQ
    rcases Finset.mem_union.1 hzQ with hzQ | hzQ
    · rcases Finset.mem_union.1 hzQ with hzQ | hzQ
      · exact Or.inl hzQ
      · exact Or.inr (taint_new hd hx hown he hz (Or.inl hzQ))
    · obtain ⟨e', -, hz'⟩ := Finset.mem_biUnion.1 hzQ
      exact Or.inr (taint_new hd hx hown he hz (Or.inr ⟨e', hz'⟩))
  -- the strings of an edge the probe learns or splits at are its reads
  have hclean : ∀ {p c u t}, (probeOrAgree r s.tree s.edges C.walkStart x = .wrongEdge p c u t
      ∨ probeOrAgree r s.tree s.edges C.walkStart x = .unlearnedEdge p c u t) →
      s.tree.sift r u = .inl p → s.tree.sift r (u * FreeMonoid.of c) = .inl t →
      ¬ (∃ z ∈ PR, Wrong M side r z) →
      leafOf M side s.tree (M.eval u.toList) = p
        ∧ leafOf M side s.tree (M.eval (u * FreeMonoid.of c).toList) = t := by
    intro p c u t ho hu huc hcl
    have hR := probeReads_edge s.edges C.walkStart ho
    exact ⟨(sift_true_reads M side r _ _ _ hu fun w hw hw' =>
        hcl ⟨w, hR w (Finset.mem_union_left _ hw), hw'⟩).symm,
      (sift_true_reads M side r _ _ _ huc fun w hw hw' =>
        hcl ⟨w, hR w (Finset.mem_union_right _ hw), hw'⟩).symm⟩
  rcases step_cases r C hs h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, hch⟩
  · -- a learned edge is bad only where the probe read wrong
    have hso : splitOf r C s x = none := by unfold splitOf; rw [ho]
    have hRR : ∀ e, resiftOf r C s x e = ∅ := fun e => by unfold resiftOf; rw [hso]
    obtain ⟨i₀, hki, hil, rfl, hci⟩ := probe_prefix r s.tree s.edges C.walkStart x (Or.inr ho)
    have hbu : Belongs xs C.walkStart j (prefixOf x i₀) := belongs_prefixOf hx hki hil.le
    have hsub : (K.filter (BadE M side r (newHypothesis s.tree
          (Function.update s.edges (p, c) (some (t, prefixOf x i₀))) (s.hypotheses + 1)) Qd'))
        ⊆ K.filter (BadE M side r s Qd)
          ∪ (if (∃ z ∈ PR, Wrong M side r z) then {(p, c)} else ∅) := by
      intro e he
      obtain ⟨heK, t', w', he', hbad⟩ := Finset.mem_filter.1 he
      simp only [newHypothesis] at he' hbad
      by_cases hpc : e = (p, c)
      · subst hpc
        by_cases hd' : ∃ z ∈ PR, Wrong M side r z
        · simp [hd']
        exfalso
        simp only [Function.update_self, Option.some.injEq, Prod.mk.injEq] at he'
        obtain ⟨rfl, rfl⟩ := he'
        obtain ⟨h1, h2⟩ := hclean (Or.inr ho) hu huc hd'
        rcases hbad with hbad | ⟨z, hzQ, hzw, m, hz⟩
        · exact hbad ⟨h1, h2⟩
        · rcases Finset.mem_union.1 hzQ with hzQ | hzQ
          · rcases Finset.mem_union.1 hzQ with hzQ | hzQ
            · obtain ⟨i, hi, hbz⟩ := hown.read z hzQ
              have : Belongs xs C.walkStart j z := by
                rw [hz, mul_assoc]; exact belongs_mul _ hbu
              exact absurd (belongs_unique hd hbz this) (by omega)
            · exact hd' ⟨z, hzQ, hzw⟩
          · obtain ⟨e', -, hz'⟩ := Finset.mem_biUnion.1 hzQ
            simp [hRR] at hz'
      · refine Finset.mem_union_left _ (Finset.mem_filter.2 ⟨heK, t', w', ?_, ?_⟩)
        · rw [Function.update_of_ne hpc] at he'; exact he'
        · rw [Function.update_of_ne hpc] at he'
          rcases hbad with hbad | ⟨z, hzQ, hzw, m, hz⟩
          · exact Or.inl hbad
          · rcases hnew e t' w' m z he' hz hzQ with hzQ | hzQ
            · exact Or.inr ⟨z, hzQ, hzw, m, hz⟩
            · simp [hRR] at hzQ
    have hcard := (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
    have hif : (if (∃ z ∈ PR, Wrong M side r z) then ({(p, c)} : Finset (List Bool × α))
        else ∅).card
        = if (∃ z ∈ PR, Wrong M side r z) then 1 else 0 := by split_ifs <;> simp
    rw [hif] at hcard
    have hb' : bad M side r (newHypothesis s.tree
        (Function.update s.edges (p, c) (some (t, prefixOf x i₀))) (s.hypotheses + 1)) Qd'
        ≤ bad M side r s Qd + (if (∃ z ∈ PR, Wrong M side r z) then 1 else 0) := hcard
    have ht' : (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, prefixOf x i₀)))
        (s.hypotheses + 1)).tree = s.tree := rfl
    rw [ht']
    omega
  · -- a split
    set d := FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)
    set T' := s.tree.splitAt d p
    have hso : splitOf r C s x = some (p, T') := by
      unfold splitOf; rw [ho]; dsimp only; rw [hE]
    have hlen : T'.leaves.length = s.tree.leaves.length + 1 :=
      DTree.length_leaves_splitAt d s.tree p hp
    have himg : (img M side s.tree).card ≤ (img M side T').card := img_card_mono M side hp d
    have he0K : (p, c) ∈ K := mem_K hp
    -- every bad edge after was bad before, or its re-sift read wrong somewhere new
    have hsub : (T'.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter
          (BadE M side r (splitLeaf r s p c t t₀) Qd')
        ⊆ ((K.filter (BadE M side r s Qd)).erase (p, c))
          ∪ K.filter fun e => ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd := by
      intro e he
      obtain ⟨-, t', w', he', hbad⟩ := Finset.mem_filter.1 he
      change retarget r T' p s.edges e = some (t', w') at he'
      change ¬ Sound M side T' e t' w' ∨ Tainted M side r Qd' e w' at hbad
      obtain ⟨hpre, t₁, he₁, hcase⟩ := retarget_some he'
      have heK : e ∈ K := mem_K (hk e t₁ w' he₁)
      have hep : e.1 ≠ p := by
        rintro h
        rw [h, List.isPrefixOf_iff_prefix.2 (List.prefix_refl p)] at hpre
        exact absurd hpre (by decide)
      have hne : e ≠ (p, c) := fun h => hep (by rw [h])
      by_contra hno
      simp only [Finset.mem_union, Finset.mem_erase, Finset.mem_filter, not_or, not_and] at hno
      have hgood : ¬ BadE M side r s Qd e := fun hb => hno.1 hne heK hb
      have hfb : ¬ ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd := fun hb => hno.2 heK hb
      have hS : Sound M side s.tree e t₁ w' := by
        by_contra hc; exact hgood ⟨t₁, w', he₁, Or.inl hc⟩
      have hT : ¬ Tainted M side r Qd e w' := fun hc => hgood ⟨t₁, w', he₁, Or.inr hc⟩
      have hwit : leafOf M side T' (M.eval w'.toList) = e.1 := by
        rw [leafOf_splitAt M side d s.tree p hp, hS.1, if_neg hep]
      rcases hcase with ⟨htp, ht'⟩ | ⟨htp, hsift⟩
      · -- not into the split leaf: as before
        rcases hbad with hbad | ⟨z, hzQ, hzw, m, hz⟩
        · refine hbad ⟨hwit, ?_⟩
          rw [leafOf_splitAt M side d s.tree p hp, hS.2, if_neg htp, ht']
        · rcases hnew e t₁ w' m z he₁ hz hzQ with hzQ | hzQ
          · exact hT ⟨z, hzQ, hzw, m, hz⟩
          · obtain ⟨p₂, T₂, t₂, w₂, hso₂, he₂, hpre₂, ht₂, hz₂⟩ := mem_resiftOf hzQ
            rw [he₁] at he₂
            simp only [Option.some.injEq, Prod.mk.injEq] at he₂
            have hpp : p₂ = p := by
              rw [hso] at hso₂; exact (Prod.mk.inj (Option.some.inj hso₂)).1.symm
            exact htp (he₂.1.trans (ht₂.trans hpp))
      · -- into the split leaf: its re-sift read right
        have hro := resiftOf_eq hso he₁ hpre htp
        have hcl : ∀ z ∈ siftReads r T' (w' * FreeMonoid.of e.2), ¬ Wrong M side r z := by
          intro z hz hzw
          by_cases hzQ : z ∈ Qd
          · obtain ⟨m, -, hzm⟩ := siftReads_mids T' _ z hz
            exact hT ⟨z, hzQ, hzw, m, hzm⟩
          · exact hfb ⟨z, hro ▸ hz, hzw, hzQ⟩
        rcases hbad with hbad | ⟨z, hzQ, hzw, m, hz⟩
        · exact hbad ⟨hwit, (sift_true_reads M side r T' _ t' hsift fun z hz => hcl z hz).symm⟩
        · rcases hnew e t₁ w' m z he₁ hz hzQ with hzQ' | hzQ'
          · exact hT ⟨z, hzQ', hzw, m, hz⟩
          · exact hcl z (hro ▸ hzQ') hzw
    have hcard := (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
    have hb' : bad M side r (splitLeaf r s p c t t₀) Qd'
        ≤ ((K.filter (BadE M side r s Qd)).erase (p, c)).card
          + (K.filter fun e => ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd).card :=
      hcard
    have htree : (splitLeaf r s p c t t₀).tree = T' := rfl
    rw [htree, hlen]
    have hbad0 : bad M side r s Qd = (K.filter (BadE M side r s Qd)).card := rfl
    by_cases hb0 : BadE M side r s Qd (p, c)
    · have := Finset.card_erase_add_one (Finset.mem_filter.2 ⟨he0K, hb0⟩)
      omega
    · rw [Finset.erase_eq_of_notMem (fun hm => hb0 (Finset.mem_filter.1 hm).2)] at hb'
      by_cases hdirty : ∃ z ∈ PR, Wrong M side r z
      · rw [if_pos hdirty]; omega
      · -- a clean probe at a good edge splits genuinely
        rw [if_neg hdirty]
        obtain ⟨h1, h2⟩ := hclean (Or.inl ho) hu huc hdirty
        have hS0 : Sound M side s.tree (p, c) t₀ w₀ := by
          by_contra hc; exact hb0 ⟨t₀, w₀, hE, Or.inl hc⟩
        have ht : RealT M side s.tree p c t := ⟨M.eval u.toList, h1, by rw [← h2, eval_mul]; rfl⟩
        have ht₀ : RealT M side s.tree p c t₀ :=
          ⟨M.eval w₀.toList, hS0.1, by rw [← hS0.2, eval_mul]; rfl⟩
        have hgen : (img M side s.tree).card + 1 ≤ (img M side T').card :=
          img_card_splitAt M side hp (genuine_of_real M side ht ht₀ (Ne.symm htt))
        omega
  · -- a probe that keeps the hypothesis taints nothing
    have hso : splitOf r C s x = none := by
      unfold splitOf
      rcases hO : probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
      all_goals try rfl
      rw [hO] at hch; simp [changes] at hch
    have hRR : ∀ e, resiftOf r C s x e = ∅ := fun e => by unfold resiftOf; rw [hso]
    have hsub : K.filter (BadE M side r (countProbe r C s x) Qd')
        ⊆ K.filter (BadE M side r s Qd) := by
      intro e he
      obtain ⟨heK, t', w', he', hbad⟩ := Finset.mem_filter.1 he
      refine Finset.mem_filter.2 ⟨heK, t', w', he', ?_⟩
      rcases hbad with hbad | ⟨z, hzQ, hzw, m, hz⟩
      · exact Or.inl hbad
      · rcases hnew e t' w' m z he' hz hzQ with hzQ | hzQ
        · exact Or.inr ⟨z, hzQ, hzw, m, hz⟩
        · simp [hRR] at hzQ
    have hb' : bad M side r (countProbe r C s x) Qd' ≤ bad M side r s Qd := Finset.card_le_card hsub
    have htree : (countProbe r C s x).tree = s.tree := rfl
    rw [htree]
    omega

omit [DecidableEq α] [Fintype σ] in
theorem bad_mono {r : FreeMonoid α → ARU} {s : Level3State α} {Qd Qd' : Finset (FreeMonoid α)}
    (h : Qd ⊆ Qd') : bad M side r s Qd ≤ bad M side r s Qd' := by
  classical
  refine Finset.card_le_card fun e he => ?_
  obtain ⟨heK, t, w, he', hb⟩ := Finset.mem_filter.1 he
  refine Finset.mem_filter.2 ⟨heK, t, w, he', ?_⟩
  rcases hb with hb | ⟨z, hz, hzw, m, hzm⟩
  · exact Or.inl hb
  · exact Or.inr ⟨z, h hz, hzw, m, hzm⟩

/-- A step keeps every edge's witness, and every string read, with an earlier probe. -/
theorem owned_step (hd : Distinct xs C.walkStart) {j : ℕ} {s s' : Level3State α}
    {x : FreeMonoid α} (hx : xs[j]? = some x) (hs : Inv r s)
    {Qd : Finset (FreeMonoid α)} (hown : Owned C xs j s Qd)
    (h : applyProbe r C s x = .inl s') {Qd'' : Finset (FreeMonoid α)}
    (hQd : ∀ z ∈ Qd'', z ∈ Qd ∨ z ∈ probeReads r s.tree s.edges C.walkStart x
      ∨ ∃ e, z ∈ resiftOf r C s x e) : Owned C xs (j + 1) s' Qd'' := by
  have hread : ∀ z ∈ Qd'', ∃ i < j + 1, Belongs xs C.walkStart i z := by
    intro z hz
    rcases hQd z hz with hz | hz | ⟨e, hz⟩
    · obtain ⟨i, hi, hb⟩ := hown.read z hz; exact ⟨i, by omega, hb⟩
    · exact ⟨j, by omega, belongs_probeReads hx hz⟩
    · obtain ⟨p, T', t, w, -, hE, -, -, hzs⟩ := mem_resiftOf hz
      obtain ⟨m, -, hzm⟩ := siftReads_mids T' _ z hzs
      obtain ⟨i, hi, hb⟩ := hown.wit e t w hE
      exact ⟨i, by omega, by rw [hzm, mul_assoc]; exact belongs_mul _ hb⟩
  rcases step_cases r C hs h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · obtain ⟨i₀, hki, hil, rfl, -⟩ := probe_prefix r s.tree s.edges C.walkStart x (Or.inr ho)
    have hbu : Belongs xs C.walkStart j (prefixOf x i₀) := belongs_prefixOf hx hki hil.le
    have hwit : ∀ e t' w, (Function.update s.edges (p, c) (some (t, prefixOf x i₀))) e
        = some (t', w) → (e = (p, c) ∧ w = prefixOf x i₀) ∨ s.edges e = some (t', w) := by
      intro e t' w he
      by_cases hpc : e = (p, c)
      · subst hpc
        simp only [Function.update_self, Option.some.injEq, Prod.mk.injEq] at he
        exact Or.inl ⟨rfl, he.2.symm⟩
      · rw [Function.update_of_ne hpc] at he; exact Or.inr he
    refine ⟨fun e t' w he => ?_, fun e e' t₁ w₁ t₂ w₂ h₁ h₂ htk => ?_, hread⟩
    · rcases hwit e t' w he with ⟨-, rfl⟩ | he
      · exact ⟨j, by omega, hbu⟩
      · obtain ⟨i, hi, hb⟩ := hown.wit e t' w he; exact ⟨i, by omega, hb⟩
    · have hnew : ∀ e₁ t₃ w₃, s.edges e₁ = some (t₃, w₃) →
          w₃.toList.take C.walkStart ≠ (prefixOf x i₀).toList.take C.walkStart := by
        intro e₁ t₃ w₃ he₁ htk'
        obtain ⟨i, hi, hb⟩ := hown.wit e₁ t₃ w₃ he₁
        have : Belongs xs C.walkStart j w₃ := by
          obtain ⟨y, hy, -, hy''⟩ := hbu
          obtain ⟨-, -, hl₃, -⟩ := hb
          exact ⟨y, hy, hl₃, htk'.trans hy''⟩
        exact absurd (belongs_unique hd hb this) (by omega)
      rcases hwit e t₁ w₁ h₁ with ⟨rfl, rfl⟩ | h₁ <;> rcases hwit e' t₂ w₂ h₂ with ⟨rfl, rfl⟩ | h₂
      · rfl
      · exact absurd htk.symm (hnew e' t₂ w₂ h₂)
      · exact absurd htk (hnew e t₁ w₁ h₁)
      · exact hown.inj e e' t₁ w₁ t₂ w₂ h₁ h₂ htk
  · refine ⟨fun e t' w he => ?_, fun e e' t₁ w₁ t₂ w₂ h₁ h₂ htk => ?_, hread⟩
    · obtain ⟨-, t₁, he₁, -⟩ := retarget_some (show retarget r _ p s.edges e = some (t', w) from he)
      obtain ⟨i, hi, hb⟩ := hown.wit e t₁ w he₁; exact ⟨i, by omega, hb⟩
    · obtain ⟨-, t₃, he₁, -⟩ :=
        retarget_some (show retarget r _ p s.edges e = some (t₁, w₁) from h₁)
      obtain ⟨-, t₄, he₂, -⟩ :=
        retarget_some (show retarget r _ p s.edges e' = some (t₂, w₂) from h₂)
      exact hown.inj e e' t₃ w₁ t₄ w₂ he₁ he₂ htk
  · exact ⟨fun e t' w he => by
      obtain ⟨i, hi, hb⟩ := hown.wit e t' w he; exact ⟨i, by omega, hb⟩, hown.inj, hread⟩

end StepAccount

/-! ## The run -/

section Run

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (r : FreeMonoid α → ARU)
  (C : Level3Settings) (xs : List (FreeMonoid α)) (n R : ℕ)

/-- The wrong reads of each string. -/
def wrongSet (z : FreeMonoid α) : Set ARU :=
  {v | v = if side (M.eval z.toList) then .reject else .accept}

omit [DecidableEq α] in
theorem edgeList_nodup (s : Level3State α) : (edgeList s).Nodup :=
  List.Nodup.product (DTree.leaves_nodup _) (Finset.nodup_toList _)

variable {M side r C xs n R}

open scoped Classical in
/-- Along the run, while its states are small, the leaves and bad edges are at most the true
leaves, one, and the fresh-bad rounds so far. -/
theorem account (hd : Distinct xs C.walkStart) (hR : (n + 2) * Fintype.card α ≤ R) :
    ∀ j s, stateAt C r initialState xs j = some s →
      (∀ i < j, ∀ s', stateAt C r initialState xs i = some s' →
        s'.tree.leaves.length ≤ n + 2) →
      Inv r s ∧ Keys s ∧ Owned C xs j s (before (rounds6 C xs n R) (j * (R + 1)) r)
        ∧ s.tree.leaves.length + bad M side r s (before (rounds6 C xs n R) (j * (R + 1)) r)
          ≤ (img M side s.tree).card + 1
            + badCount (rounds6 C xs n R) (wrongSet M side) (j * (R + 1)) r
  | 0, s, h, _ => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    have hnone : ∀ e, (initialState : Level3State α).edges e = none := fun _ => rfl
    have hbad : bad M side r (initialState : Level3State α)
        (before (rounds6 C xs n R) (0 * (R + 1)) r) = 0 := by
      unfold bad
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      rintro e - ⟨t, w, he, -⟩
      rw [hnone] at he; exact absurd he (by simp)
    have himg : 0 < (img M side (initialState : Level3State α).tree).card :=
      Finset.card_pos.2 ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ M.start)⟩
    refine ⟨inv_start r, fun e t w he => by rw [hnone] at he; exact absurd he (by simp),
      ⟨fun e t w he => by rw [hnone] at he; exact absurd he (by simp),
        fun e _ t w _ _ he => by rw [hnone] at he; exact absurd he (by simp),
        fun z hz => by simp [before] at hz⟩, ?_⟩
    rw [hbad]
    simp only [zero_mul, badCount, Finset.range_zero, Finset.filter_empty, Finset.card_empty,
      initialState, newHypothesis, DTree.leaves] at himg ⊢
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.length_cons,
      List.length_nil]
    omega
  | j + 1, s', h, hsm => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C r initialState xs j with _ | s <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[j]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe r C s x with s'' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨hinv, hk, hown, hacc⟩ := account hd hR j s hs₁ fun i hi => hsm i (by omega)
    have hS : Small r C xs n j := fun i hi s₂ hs₂ => hsm i (by omega) s₂ hs₂
    have hsmall := hsm j (by omega) s hs₁
    have hdiv : ∀ i < R + 1, (j * (R + 1) + i) / (R + 1) = j := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
    have hmod : ∀ i < R + 1, (j * (R + 1) + i) % (R + 1) = i := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    have hround : ∀ i < R + 1, rounds6 C xs n R (j * (R + 1) + i) r = stepRound r C s x i :=
      fun i hi => by
        rw [rounds6_eq (s := s) (x := x) (by rw [hdiv i hi]; exact hS)
          (by rw [hdiv i hi]; exact hs₁) (by rw [hdiv i hi]; exact hx), hmod i hi]
    have hlenE : (edgeList s).length ≤ R := by
      rw [length_edgeList]
      exact (Nat.mul_le_mul_right _ hsmall).trans hR
    set K := s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)
    set PR := probeReads r s.tree s.edges C.walkStart x
    set Qd := before (rounds6 C xs n R) (j * (R + 1)) r
    have hjj : (j + 1) * (R + 1) = j * (R + 1) + (R + 1) := by ring
    -- what the step reads
    have hQd : ∀ z ∈ before (rounds6 C xs n R) ((j + 1) * (R + 1)) r,
        z ∈ Qd ∨ z ∈ PR ∨ ∃ e, z ∈ resiftOf r C s x e := by
      intro z hz
      rw [hjj] at hz
      rcases (mem_before_add _ r).1 hz with hz | ⟨i, hi, hzi⟩
      · exact Or.inl hz
      rw [hround i hi] at hzi
      unfold stepRound at hzi
      split_ifs at hzi with hi0
      · exact Or.inr (Or.inl hzi)
      rcases he : (edgeList s)[i - 1]? with _ | e <;> rw [he] at hzi
      · simp at hzi
      · exact Or.inr (Or.inr ⟨e, hzi⟩)
    have hsub : before (rounds6 C xs n R) ((j + 1) * (R + 1)) r
        ⊆ Qd ∪ PR ∪ K.biUnion (resiftOf r C s x) := by
      intro z hz
      rcases hQd z hz with hz | hz | ⟨e, hz⟩
      · exact Finset.mem_union_left _ (Finset.mem_union_left _ hz)
      · exact Finset.mem_union_left _ (Finset.mem_union_right _ hz)
      · obtain ⟨p, T', t, w, -, hE, -, -, -⟩ := mem_resiftOf hz
        exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨e, mem_K (hk e t w hE), hz⟩)
    have hstep : s''.tree.leaves.length
          + bad M side r s'' (Qd ∪ PR ∪ K.biUnion (resiftOf r C s x)) + (img M side s.tree).card
        ≤ (img M side s''.tree).card + s.tree.leaves.length + bad M side r s Qd
          + (if ∃ z ∈ PR, Wrong M side r z then 1 else 0)
          + (K.filter fun e => ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd).card :=
      step_account M side hd hx hinv hk hown ha
    have hbad := bad_mono M side (r := r) (s := s'') hsub
    have hown' := owned_step hd hx hinv hown ha hQd
    obtain ⟨hinv', hk'⟩ := inv_keys_step r C hinv hk ha
    -- each wrong round is fresh-bad, and has its own index
    have hnewR : ∀ i < j * (R + 1), ∀ z, z ∈ rounds6 C xs n R i r → z ∈ Qd :=
      fun i hi z hz => (mem_before _).2 ⟨i, hi, hz⟩
    have hbelR : ∀ e z, z ∈ resiftOf r C s x e →
        ∃ i < j, Belongs xs C.walkStart i z ∧ ∃ t w, s.edges e = some (t, w)
          ∧ Belongs xs C.walkStart i w := by
      intro e z hz
      obtain ⟨p, T', t, w, -, hE, -, -, hzs⟩ := mem_resiftOf hz
      obtain ⟨m, -, hzm⟩ := siftReads_mids T' _ z hzs
      obtain ⟨i, hi, hb⟩ := hown.wit e t w hE
      exact ⟨i, hi, by rw [hzm, mul_assoc]; exact belongs_mul _ hb, t, w, hE, hb⟩
    set F := (Finset.range (R + 1)).filter fun i =>
      FreshBad (rounds6 C xs n R) (wrongSet M side) (j * (R + 1) + i) r
    have hidx : ∀ e ∈ edgeList s, ∃ i, ∃ h : i < (edgeList s).length, (edgeList s)[i] = e :=
      fun e he => List.getElem_of_mem he
    set g : List Bool × α → ℕ := fun e =>
      if he : e ∈ edgeList s then (hidx e he).choose + 1 else 0
    have hg : ∀ e (he : e ∈ edgeList s), ∃ h : g e - 1 < (edgeList s).length,
        (edgeList s)[g e - 1] = e ∧ 1 ≤ g e := by
      intro e he
      obtain ⟨h1, h2⟩ := (hidx e he).choose_spec
      refine ⟨by simpa [g, he] using h1, by simpa [g, he] using h2, by simp [g, he]⟩
    have hFB : (if ∃ z ∈ PR, Wrong M side r z then 1 else 0)
        + (K.filter fun e => ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd).card
        ≤ F.card := by
      set A := K.filter fun e => ∃ z ∈ resiftOf r C s x e, Wrong M side r z ∧ z ∉ Qd
      have hAe : ∀ e ∈ A, e ∈ edgeList s := fun e heA => by
        obtain ⟨-, z, hz, -⟩ := Finset.mem_filter.1 heA
        obtain ⟨p, T', t, w, -, hE, -, -, -⟩ := mem_resiftOf hz
        exact mem_edgeList (hk e t w hE)
      have hgA : ∀ e ∈ A, g e ∈ F := by
        intro e heA
        obtain ⟨heK, z, hz, hzw, hzQ⟩ := Finset.mem_filter.1 heA
        obtain ⟨hlt, hget, hg1⟩ := hg e (hAe e heA)
        have hgR : g e < R + 1 := by omega
        refine Finset.mem_filter.2 ⟨Finset.mem_range.2 hgR, ?_⟩
        have hzr : z ∈ rounds6 C xs n R (j * (R + 1) + g e) r := by
          rw [hround (g e) hgR]
          unfold stepRound
          rw [if_neg (by omega), List.getElem?_eq_getElem hlt, hget]
          exact hz
        refine freshBad_of _ _ r hzr hzw fun i hi hzi => ?_
        by_cases hij : i < j * (R + 1)
        · exact hzQ (hnewR i hij z hzi)
        obtain ⟨i₀, hi₀, hbz, t₀, w₀, hE₀, hbw₀⟩ := hbelR e z hz
        set i' := i - j * (R + 1)
        have hii : i = j * (R + 1) + i' := by omega
        rw [hii, hround i' (by omega)] at hzi
        unfold stepRound at hzi
        split_ifs at hzi with hi0
        · exact absurd (belongs_unique hd hbz (belongs_probeReads hx hzi)) (by omega)
        have hlt' : i' - 1 < (edgeList s).length := by omega
        rw [List.getElem?_eq_getElem hlt'] at hzi
        obtain ⟨i₁, -, hbz₁, t₁, w₁, hE₁, hbw₁⟩ := hbelR _ z hzi
        have hi01 := belongs_unique hd hbz hbz₁
        subst hi01
        have hsame := hown.inj _ _ t₁ w₁ t₀ w₀ hE₁ hE₀ (take_eq_of_belongs hbw₁ hbw₀)
        have := (edgeList_nodup s).getElem_inj_iff.1 (hsame.trans hget.symm)
        omega
      have hginj : Set.InjOn g A := by
        intro e₁ he₁ e₂ he₂ heq
        obtain ⟨hl₁, h₁, -⟩ := hg e₁ (hAe e₁ he₁)
        obtain ⟨hl₂, h₂, -⟩ := hg e₂ (hAe e₂ he₂)
        rw [← h₁, ← h₂]
        simp only [heq]
      have hcardA : A.card = (A.image g).card := (Finset.card_image_of_injOn hginj).symm
      by_cases hdirty : ∃ z ∈ PR, Wrong M side r z
      · rw [if_pos hdirty]
        have h0 : 0 ∈ F := by
          obtain ⟨z, hz, hzw⟩ := hdirty
          refine Finset.mem_filter.2 ⟨Finset.mem_range.2 (by omega), ?_⟩
          have hzr : z ∈ rounds6 C xs n R (j * (R + 1) + 0) r := by
            rw [hround 0 (by omega)]; simpa [stepRound] using hz
          refine freshBad_of _ _ r hzr hzw fun i hi hzi => ?_
          obtain ⟨i₀, hi₀, hb⟩ := hown.read z (hnewR i (by omega) z hzi)
          exact absurd (belongs_unique hd hb (belongs_probeReads hx hz)) (by omega)
        have h0A : 0 ∉ A.image g := by
          intro h0
          obtain ⟨e, heA, he0⟩ := Finset.mem_image.1 h0
          obtain ⟨-, -, hg1⟩ := hg e (hAe e heA)
          omega
        have hsubF : insert 0 (A.image g) ⊆ F := by
          intro i hi
          rcases Finset.mem_insert.1 hi with rfl | hi
          · exact h0
          · obtain ⟨e, heA, rfl⟩ := Finset.mem_image.1 hi; exact hgA e heA
        have := Finset.card_le_card hsubF
        rw [Finset.card_insert_of_notMem h0A] at this
        omega
      · rw [if_neg hdirty, zero_add, hcardA]
        exact Finset.card_le_card fun i hi => by
          obtain ⟨e, heA, rfl⟩ := Finset.mem_image.1 hi; exact hgA e heA
    have hcount : badCount (rounds6 C xs n R) (wrongSet M side) ((j + 1) * (R + 1)) r
        = badCount (rounds6 C xs n R) (wrongSet M side) (j * (R + 1)) r + F.card := by
      rw [hjj]; exact badCount_add (rounds6 C xs n R) (wrongSet M side) r (j * (R + 1)) (R + 1)
    refine ⟨hinv', hk', hown', ?_⟩
    rw [hcount]
    have hb := hbad
    omega

end Run

end Spur

end Random

end OrthoDFA
