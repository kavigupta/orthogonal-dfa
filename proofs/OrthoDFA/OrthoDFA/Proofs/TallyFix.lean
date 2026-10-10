import OrthoDFA.TallyLoop
import OrthoDFA.Proofs.FixTime

/-!
# The tally loop fixes a pair with clean records in time

`fix_in_time` for the tally loop, at any reads: the hypothesis is the tree, the edges and their
version, and a hypothesis is significant where some edge out of a leaf and a target it does not
point at get clean records at rate at least `q`. Each such record raises that pair's tally while
the hypothesis stays, and a settled state's tally there is below `m`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Settle

variable (cut : FreeMonoid α → Option Bool) (m Lmax : ℕ)

theorem fixEdge_version (s : TState α) (p : List Bool) (c : α) (t : List Bool) :
    (fixEdge cut m s p c t).version = s.version + 1 := by
  unfold fixEdge
  simp only []
  split
  · split_ifs <;> rfl
  · rfl

theorem settle_version : ∀ (fuel : ℕ) (s s' : TState α),
    settleEdges cut m Lmax fuel s = .inl s' →
    s.version ≤ s'.version ∧ (s'.version = s.version → s' = s)
  | 0, s, s', h => by
    unfold settleEdges at h
    split_ifs at h
    simp only [Sum.inl.injEq] at h
    subst h
    exact ⟨le_rfl, fun _ => rfl⟩
  | fuel + 1, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hv hL
    · obtain ⟨h1, -⟩ := settle_version fuel _ s' h
      rw [fixEdge_version] at h1
      exact ⟨by omega, fun he => by omega⟩
    · simp only [Sum.inl.injEq] at h
      subst h
      exact ⟨le_rfl, fun _ => rfl⟩

theorem settle_settled : ∀ (fuel : ℕ) (s s' : TState α),
    settleEdges cut m Lmax fuel s = .inl s' → Settled m s'
  | 0, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hs
    simp only [Sum.inl.injEq] at h
    exact h ▸ hs
  | fuel + 1, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hv hL
    · exact settle_settled fuel _ s' h
    · simp only [Sum.inl.injEq] at h
      subst h
      intro p c t hpct
      exact hv ⟨p, c, t, hpct⟩

end Settle

section Step

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

theorem tallyStep_settled {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') : Settled C.m s' := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  · exact settle_settled cut C.m C.Lmax C.fuel _ _ h

theorem tally_addRec_ge (s : TState α) (p : List Bool) (c : α) (r : FreeMonoid α × List Bool)
    (p' : List Bool) (c' : α) (t' : List Bool) :
    s.tally p' c' t' + (if (p, c, r.2) = (p', c', t') then 1 else 0)
      ≤ (s.addRec p c r).tally p' c' t' := by
  unfold TState.tally TState.addRec
  by_cases hpc : p' = p ∧ c' = c
  · obtain ⟨rfl, rfl⟩ := hpc
    simp only [Function.update_self, List.filter_append, List.length_append, Prod.mk.injEq,
      true_and]
    by_cases ht : r.2 = t' <;> simp [ht]
  · have hrow : (Function.update s.recs p (Function.update (s.recs p) c (s.recs p c ++ [r]))) p' c'
        = s.recs p' c' := by
      by_cases hp : p' = p
      · subst hp
        have hc : c' ≠ c := fun h => hpc ⟨rfl, h⟩
        simp [hc]
      · simp [hp]
    simp only [hrow]
    have : ¬ (p, c, r.2) = (p', c', t') := by
      simp only [Prod.mk.injEq]; exact fun h => hpc ⟨h.1.symm, h.2.1.symm⟩
    simp [this]

theorem recordBy_none {s : TState α} {x : FreeMonoid α}
    (h : ∀ ps fd, probeBy cut s.tree s.edges C.k x ≠ .edge ps fd) :
    recordBy cut C.k (s.tree, s.edges) x = none := by
  unfold recordBy
  split
  · rename_i ps fd ho; exact absurd ho (h ps fd)
  · rfl

@[simp] theorem charge_tree (k : ℕ) (s : TState α) (x : FreeMonoid α) :
    (s.charge cut k x).tree = s.tree := rfl

@[simp] theorem charge_edges (k : ℕ) (s : TState α) (x : FreeMonoid α) :
    (s.charge cut k x).edges = s.edges := rfl

@[simp] theorem charge_version (k : ℕ) (s : TState α) (x : FreeMonoid α) :
    (s.charge cut k x).version = s.version := rfl

@[simp] theorem charge_recs (k : ℕ) (s : TState α) (x : FreeMonoid α) :
    (s.charge cut k x).recs = s.recs := rfl

theorem tallyPre_version_ge (s : TState α) (x : FreeMonoid α) :
    s.version ≤ (tallyPre C cut s x).version := by
  unfold tallyPre
  split
  · split
    · split <;> simp [TState.setEdge]
    · simp
  · split <;> simp [TState.addRec]
  all_goals simp

/-- With the version unchanged, a probe's outcome only adds its record and counts. -/
theorem tallyPre_same (s : TState α) (x : FreeMonoid α)
    (hv : (tallyPre C cut s x).version = s.version) :
    (tallyPre C cut s x).tree = s.tree ∧ (tallyPre C cut s x).edges = s.edges ∧ ∀ p c t,
      s.tally p c t + (if ((recordBy cut C.k (s.tree, s.edges) x).map Prod.fst) = some (p, c, t)
        then 1 else 0) ≤ (tallyPre C cut s x).tally p c t := by
  have hnr : (∀ ps fd, probeBy cut s.tree s.edges C.k x ≠ .edge ps fd) →
      recordBy cut C.k (s.tree, s.edges) x = none := recordBy_none C cut
  unfold tallyPre at hv ⊢
  split
  · rename_i u ho
    have hr := hnr (by rw [ho]; intro ps fd h; cases h)
    rw [ho] at hv
    simp only [] at hv
    split at hv
    · split at hv
      · simp [TState.setEdge] at hv
      · simp_all [TState.tally]
    · simp_all [TState.tally]
  · rename_i ps fd ho
    rcases hr : recordBy cut C.k (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩
    · simp [TState.tally]
    · refine ⟨rfl, rfl, fun p' c' t' => ?_⟩
      have := tally_addRec_ge { s with n := s.n + 1, dis := s.n + 1 - s.n + s.dis } p c (sp, t)
        p' c' t'
      simp only [Option.map_some, Option.some.injEq] at this ⊢
      simpa [TState.tally, TState.addRec] using this
  all_goals
    have hr : recordBy cut C.k (s.tree, s.edges) x = none := hnr fun ps fd h => by simp_all
    simp [hr, TState.tally]

/-- With the version unchanged, a step only adds its record and counts. -/
theorem tallyStep_same {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') (hv : s'.version = s.version) :
    s'.tree = s.tree ∧ s'.edges = s.edges ∧ ∀ p c t,
      s.tally p c t + (if ((recordBy cut C.k (s.tree, s.edges) x).map Prod.fst) = some (p, c, t)
        then 1 else 0) ≤ s'.tally p c t := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  obtain ⟨hle, heq⟩ := settle_version cut C.m C.Lmax C.fuel _ s' h
  have hge := tallyPre_version_ge C cut s x
  have hs' : s' = tallyPre C cut s x := heq (by omega)
  subst hs'
  exact tallyPre_same C cut s x (by omega)

theorem tallyStep_version_ge {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') : (tallyPre C cut s x).version ≤ s'.version := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  · exact (settle_version cut C.m C.Lmax C.fuel _ s' h).1

/-- A probe that learns an unlearned edge starts a new version. -/
theorem learnsBy_version {s : TState α} {x : FreeMonoid α} {p : List Bool} {c : α}
    (h : LearnsBy cut C.k (s.tree, s.edges) p c x) :
    (tallyPre C cut s x).version = s.version + 1 := by
  obtain ⟨he, u, t, ho, hc, hp, ht⟩ := h
  simp only [] at he ho hp ht
  unfold tallyPre
  rw [ho]
  simp only [hc, hp, ht, he]
  rfl

end Step

section Fix

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (q : ℝ)

/-- The loop's step, `none` once the round has ended. -/
noncomputable def tallyStep' (s : TState α) (x : FreeMonoid α) : Option (TState α) :=
  (tallyStep C cut s x).getLeft?

/-- The hypothesis: the tree, the edges and their version. -/
def tallyKey (s : TState α) : DTree α × Edges α × ℕ := (s.tree, s.edges, s.version)

/-- The edge `(p, c)` out of a leaf and a target `t` it does not point at get clean records, or
probes learning the edge, at rate `q`. -/
def CleanSig (h : DTree α × Edges α × ℕ) (pct : List Bool × α × List Bool) : Prop :=
  pct.1 ∈ h.1.paths ∧ (h.2.1 pct.1 pct.2.1).map Prod.fst ≠ some pct.2.2
    ∧ q ≤ D.real {x | (recordBy cut C.k (h.1, h.2.1) x).map Prod.fst = some pct
      ∨ LearnsBy cut C.k (h.1, h.2.1) pct.1 pct.2.1 x}

/-- Some edge and a target it does not point at get clean records at rate `q`. -/
def Significant (h : DTree α × Edges α × ℕ) : Prop := ∃ pct, CleanSig C cut D q h pct

open scoped Classical in
/-- The clean records of a significant pair, where `h` is significant. -/
noncomputable def sigRecs (h : DTree α × Edges α × ℕ) : Set (FreeMonoid α) :=
  if hs : Significant C cut D q h then
    {x | (recordBy cut C.k (h.1, h.2.1) x).map Prod.fst = some hs.choose
      ∨ LearnsBy cut C.k (h.1, h.2.1) hs.choose.1 hs.choose.2.1 x}
  else ∅

open scoped Classical in
/-- That pair's tally, where the state is settled and its hypothesis significant. -/
noncomputable def sigCount (s : TState α) : ℕ :=
  if hs : Settled C.m s ∧ Significant C cut D q (tallyKey s) then
    s.tally hs.2.choose.1 hs.2.choose.2.1 hs.2.choose.2.2
  else 0

/-- `fix_in_time` for the tally loop: over `T` fresh draws, at some point a hypothesis with an
edge and a target it does not point at getting clean records at rate `q` outlasts the next `n`
draws, the round still running, with chance at most `T · P(Bin(n, q) < m)`. -/
theorem tally_fix_in_time (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hm : 0 < C.m) (n T : ℕ)
    (s₀ : TState α) :
    (Measure.pi fun _ : Fin T => D)
        {xs | Lingers (tallyStep' C cut) tallyKey (Significant C cut D q) n s₀ T xs}
      ≤ T * ENNReal.ofReal (1 - binomSfGe n q C.m) := by
  classical
  refine fix_in_time D (tallyStep' C cut) tallyKey (sigRecs C cut D q) (sigCount C cut D q)
    (Significant C cut D q) q C.m hq0 hq1 ?_ ?_ ?_ n T s₀
  · intro h hs
    simp only [sigRecs, dif_pos hs]
    exact hs.choose_spec.2.2
  · intro s x s' hst hk
    simp only [tallyStep', Sum.getLeft?_eq_some_iff] at hst
    simp only [tallyKey, Prod.mk.injEq] at hk
    obtain ⟨-, -, hv⟩ := hk
    obtain ⟨htree, hedges, htal⟩ := tallyStep_same C cut hst hv
    have hset := tallyStep_settled C cut hst
    have hkey : tallyKey s' = tallyKey s := by simp only [tallyKey, htree, hedges, hv]
    by_cases hs : Significant C cut D q (tallyKey s)
    · have h1 : Settled C.m s' ∧ Significant C cut D q (tallyKey s') := ⟨hset, hkey ▸ hs⟩
      have hch : h1.2.choose = hs.choose := by
        have : ∀ (h₁ : Significant C cut D q (tallyKey s'))
            (h₂ : Significant C cut D q (tallyKey s)), h₁.choose = h₂.choose := by
          rw [hkey]; intro _ _; rfl
        exact this _ _
      have hs' : sigCount C cut D q s' = s'.tally hs.choose.1 hs.choose.2.1 hs.choose.2.2 := by
        unfold sigCount; rw [dif_pos h1, hch]
      have hle : sigCount C cut D q s ≤ s.tally hs.choose.1 hs.choose.2.1 hs.choose.2.2 := by
        unfold sigCount; split_ifs <;> simp
      have hnl : ¬ LearnsBy cut C.k (s.tree, s.edges) hs.choose.1 hs.choose.2.1 x := fun hl =>
        by have := learnsBy_version C cut hl; have := tallyStep_version_ge C cut hst; omega
      have hA : (if x ∈ sigRecs C cut D q (tallyKey s) then 1 else 0)
          = (if (recordBy cut C.k (s.tree, s.edges) x).map Prod.fst
              = some (hs.choose.1, hs.choose.2.1, hs.choose.2.2) then 1 else 0) := by
        rw [sigRecs, dif_pos hs]
        by_cases hx : (recordBy cut C.k (s.tree, s.edges) x).map Prod.fst = some hs.choose
        · rw [if_pos hx]; exact if_pos (Or.inl hx)
        · rw [if_neg hx]; exact if_neg fun h => h.elim hx hnl
      rw [hs', hA]
      exact le_trans (Nat.add_le_add_right hle _) (htal _ _ _)
    · simp [sigCount, hs, sigRecs]
  · intro s
    unfold sigCount
    split_ifs with hs
    · have hsp := hs.2.choose_spec
      by_contra hc
      exact hs.1 _ _ _ ⟨hsp.1, hsp.2.1, by omega⟩
    · exact hm

end Fix

end OrthoDFA
