import OrthoDFA.TallyLoop
import OrthoDFA.Proofs.FixTime

/-!
# The tally loop fixes a pair with clean records in time

`fix_in_time` for the tally loop, at any reads: the states are the settled ones, the hypothesis is
the tree, the edges and their version, and a hypothesis is significant where some edge and a
target it does not point at get clean records at rate at least `q`. Each such record raises that
pair's tally while the hypothesis stays, and a settled state's tally there is below `m`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Settle

variable (R : CutReads α) (m Lmax : ℕ)

theorem fixEdge_version (s : TState α) (p : List Bool) (c : α) (t : List Bool) :
    (fixEdge R m s p c t).version = s.version + 1 := by
  unfold fixEdge
  simp only []
  split
  · split_ifs <;> rfl
  · rfl

theorem settle_version : ∀ (fuel : ℕ) (s s' : TState α), settleEdges R m Lmax fuel s = some s' →
    s.version ≤ s'.version ∧ (s'.version = s.version → s' = s)
  | 0, s, s', h => by
    unfold settleEdges at h
    split_ifs at h
    simp only [Option.some.injEq] at h
    subst h
    exact ⟨le_rfl, fun _ => rfl⟩
  | fuel + 1, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hv hL
    · obtain ⟨h1, -⟩ := settle_version fuel _ s' h
      rw [fixEdge_version] at h1
      exact ⟨by omega, fun he => by omega⟩
    · simp only [Option.some.injEq] at h
      subst h
      exact ⟨le_rfl, fun _ => rfl⟩

theorem settle_settled : ∀ (fuel : ℕ) (s s' : TState α), settleEdges R m Lmax fuel s = some s' →
    Settled m s'
  | 0, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hs
    simp only [Option.some.injEq] at h
    exact h ▸ hs
  | fuel + 1, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hv hL
    · exact settle_settled fuel _ s' h
    · simp only [Option.some.injEq] at h
      subst h
      intro p c t hpct
      exact hv ⟨p, c, t, hpct⟩

end Settle

section Step

variable (C : TallyCfg) (R : CutReads α) (ends : TState α → FreeMonoid α → Prop)

theorem tallyStep_settled {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C R ends s x = some s') : Settled C.m s' := by
  unfold tallyStep at h
  split_ifs at h
  exact settle_settled R C.m C.Lmax C.fuel _ _ h

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

theorem recordOf_none {s : TState α} {x : FreeMonoid α}
    (h : ∀ ps fd, probeOutcome R s.tree s.edges C.k x ≠ .edge ps fd) :
    recordOf R C.k (s.tree, s.edges) x = none := by
  unfold recordOf
  split
  · rename_i ps fd ho; exact absurd ho (h ps fd)
  · rfl

theorem tallyPre_version_ge (s : TState α) (x : FreeMonoid α) :
    s.version ≤ (tallyPre C R s x).version := by
  unfold tallyPre
  split
  · split
    · split <;> simp [TState.setEdge]
    · simp
  · split <;> simp [TState.addRec]
  · simp
  · simp

/-- With the version unchanged, a probe's outcome only counts as undecided or adds its record. -/
theorem tallyPre_same (s : TState α) (x : FreeMonoid α)
    (hv : (tallyPre C R s x).version = s.version) :
    (tallyPre C R s x).tree = s.tree ∧ (tallyPre C R s x).edges = s.edges ∧ ∀ p c t,
      s.tally p c t + (if ((recordOf R C.k (s.tree, s.edges) x).map Prod.fst) = some (p, c, t)
        then 1 else 0) ≤ (tallyPre C R s x).tally p c t := by
  unfold tallyPre at hv ⊢
  split
  · rename_i u ho
    have hr := recordOf_none C R (s := s) (x := x) (by rw [ho]; intro ps fd h; cases h)
    rw [ho] at hv
    simp only [] at hv
    split at hv
    · split at hv
      · simp [TState.setEdge] at hv
      · simp_all [TState.tally]
    · simp_all [TState.tally]
  · rename_i ps fd ho
    rcases hr : recordOf R C.k (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩
    · simp [TState.tally]
    · refine ⟨rfl, rfl, fun p' c' t' => ?_⟩
      have := tally_addRec_ge s p c (sp, t) p' c' t'
      simp only [Option.map_some, Option.some.injEq] at this ⊢
      exact this
  · rename_i ho
    have hr := recordOf_none C R (s := s) (x := x) (by rw [ho]; intro ps fd h; cases h)
    simp [hr]
  · rename_i ho1 ho2 ho3
    have hr := recordOf_none C R (s := s) (x := x) (fun ps fd h => ho2 ps fd h)
    simp [hr, TState.tally]

/-- With the version unchanged, a step only counts an undecided outcome or adds its record. -/
theorem tallyStep_same {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C R ends s x = some s') (hv : s'.version = s.version) :
    s'.tree = s.tree ∧ s'.edges = s.edges ∧ ∀ p c t,
      s.tally p c t + (if ((recordOf R C.k (s.tree, s.edges) x).map Prod.fst) = some (p, c, t)
        then 1 else 0) ≤ s'.tally p c t := by
  unfold tallyStep at h
  split_ifs at h
  obtain ⟨hle, heq⟩ := settle_version R C.m C.Lmax C.fuel _ s' h
  have hge := tallyPre_version_ge C R s x
  have hs' : s' = tallyPre C R s x := heq (by omega)
  subst hs'
  exact tallyPre_same C R s x (by omega)

end Step

section Fix

variable (C : TallyCfg) (R : CutReads α) (ends : TState α → FreeMonoid α → Prop)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (q : ℝ)

/-- A settled state of the loop. -/
abbrev TallyS (C : TallyCfg) (α : Type*) [Fintype α] [DecidableEq α] :=
  {s : TState α // Settled C.m s}

/-- The loop's step on settled states. -/
noncomputable def tallyStep' (s : TallyS C α) (x : FreeMonoid α) : Option (TallyS C α) :=
  (tallyStep C R ends s.1 x).attach.map fun s' => ⟨s'.1, tallyStep_settled C R ends s'.2⟩

/-- The hypothesis: the tree, the edges and their version. -/
def tallyKey (s : TallyS C α) : DTree α × Edges α × ℕ := (s.1.tree, s.1.edges, s.1.version)

/-- The edge `(p, c)` and a target `t` it does not point at get clean records at rate `q`. -/
def CleanSig (h : DTree α × Edges α × ℕ) (pct : List Bool × α × List Bool) : Prop :=
  (h.2.1 pct.1 pct.2.1).map Prod.fst ≠ some pct.2.2
    ∧ q ≤ D.real {x | (recordOf R C.k (h.1, h.2.1) x).map Prod.fst = some pct}

/-- Some edge and a target it does not point at get clean records at rate `q`. -/
def Significant (h : DTree α × Edges α × ℕ) : Prop := ∃ pct, CleanSig C R D q h pct

open scoped Classical in
/-- The clean records of a significant pair, where `h` is significant. -/
noncomputable def sigRecs (h : DTree α × Edges α × ℕ) : Set (FreeMonoid α) :=
  if hs : Significant C R D q h then
    {x | (recordOf R C.k (h.1, h.2.1) x).map Prod.fst = some hs.choose}
  else ∅

open scoped Classical in
/-- That pair's tally, where the hypothesis is significant. -/
noncomputable def sigCount (s : TallyS C α) : ℕ :=
  if hs : Significant C R D q (tallyKey C s) then
    s.1.tally hs.choose.1 hs.choose.2.1 hs.choose.2.2
  else 0

/-- `fix_in_time` for the tally loop: over `T` fresh draws, at some point a hypothesis with an
edge and a target it does not point at getting clean records at rate `q` outlasts the next `n`
draws, the round still running, with chance at most `T · P(Bin(n, q) < m)`. -/
theorem tally_fix_in_time (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hm : 0 < C.m) (n T : ℕ)
    (s₀ : TallyS C α) :
    (Measure.pi fun _ : Fin T => D)
        {xs | Lingers (tallyStep' C R ends) (tallyKey C) (Significant C R D q) n s₀ T xs}
      ≤ T * ENNReal.ofReal (1 - binomSfGe n q C.m) := by
  classical
  refine fix_in_time D (tallyStep' C R ends) (tallyKey C) (sigRecs C R D q) (sigCount C R D q)
    (Significant C R D q) q C.m hq0 hq1 ?_ ?_ ?_ n T s₀
  · intro h hs
    simp only [sigRecs, dif_pos hs]
    exact hs.choose_spec.2
  · intro s x s' hst hk
    simp only [tallyStep', Option.map_eq_some_iff] at hst
    obtain ⟨⟨s'', hs''⟩, -, rfl⟩ := hst
    simp only [tallyKey, Prod.mk.injEq] at hk
    obtain ⟨-, -, hv⟩ := hk
    obtain ⟨htree, hedges, htal⟩ := tallyStep_same C R ends hs'' hv
    have hkey : tallyKey C ⟨s'', tallyStep_settled C R ends hs''⟩ = tallyKey C s := by
      simp only [tallyKey, htree, hedges, hv]
    unfold sigCount
    rw [hkey]
    by_cases hs : Significant C R D q (tallyKey C s)
    · simp only [dif_pos hs, sigRecs, Set.mem_ofPred_eq]
      have := htal hs.choose.1 hs.choose.2.1 hs.choose.2.2
      simp only [tallyKey] at this ⊢
      exact this
    · simp [hs, sigRecs]
  · intro s
    unfold sigCount
    split_ifs with hs
    · have hsp := hs.choose_spec
      by_contra hc
      exact s.2 _ _ _ ⟨hsp.1, by omega⟩
    · exact hm

end Fix

end OrthoDFA
