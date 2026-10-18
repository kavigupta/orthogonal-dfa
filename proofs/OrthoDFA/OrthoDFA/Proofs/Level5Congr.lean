import OrthoDFA.Proofs.Level5Noise
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# What a probe depends on

A probe of `x` reads the cut only through the sifts of `x`'s prefixes from `k` on, and the edges
only through their targets at the tree's leaves. So two cuts placing those prefixes alike, or two
edge maps with the same targets out of every leaf, give the same probe.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Edges

/-- The two edge maps point every leaf's edges at the same targets. -/
def TgtAgree (T : DTree α) (e e' : Edges α) : Prop :=
  ∀ p ∈ T.leaves, ∀ c, (e (p, c)).map Prod.fst = (e' (p, c)).map Prod.fst

variable {cut : FreeMonoid α → Option Bool} {T : DTree α} {e e' : Edges α} {k : ℕ}
  {x : FreeMonoid α}

theorem probeOrAgree_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    probeOrAgree (ofCut cut) T e k x = probeOrAgree (ofCut cut) T e' k x :=
  probe_econgr _ T k x he h

theorem middleHarvestBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    middleHarvestBy cut T e k x = middleHarvestBy cut T e' k x := by
  simp only [middleHarvestBy, probeOrAgree_econgr (x := x) (k := k) (cut := cut) he h]

open scoped Classical in
/-- Edge maps out of the leaves into the leaves, each witness `1`. -/
noncomputable def edgeMaps (T : DTree α) : Finset (Edges α) :=
  (Finset.univ : Finset (↥(T.leaves.toFinset ×ˢ (Finset.univ : Finset α))
    → Option ↥T.leaves.toFinset)).image fun f pc =>
      if h : pc ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
        (f ⟨pc, h⟩).map fun t => (t.1, 1)
      else none

theorem edgeMaps_card (T : DTree α) :
    (edgeMaps T).card ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := by
  classical
  refine Finset.card_image_le.trans ?_
  simp only [Finset.card_univ, Fintype.card_fun, Fintype.card_option, Fintype.card_coe,
    Finset.card_product, Finset.card_univ]
  have h := List.toFinset_card_le T.leaves
  calc _ ≤ (T.leaves.length + 1) ^ (T.leaves.toFinset.card * Fintype.card α) := by gcongr
    _ ≤ _ := by
      apply Nat.pow_le_pow_right (by omega)
      gcongr

theorem exists_edgeMaps {T : DTree α} {e : Edges α} (he : EdgesInto T e) :
    ∃ e' ∈ edgeMaps T, TgtAgree T e e' ∧ EdgesInto T e' := by
  classical
  refine ⟨fun pc => if h : pc ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
      (e pc).map fun q => (q.1, 1) else none, ?_, ?_, ?_⟩
  · refine Finset.mem_image.2 ⟨fun pc => (e pc.1).bind fun q =>
      if hq : q.1 ∈ T.leaves.toFinset then some ⟨q.1, hq⟩ else none, Finset.mem_univ _, ?_⟩
    funext ⟨p, c⟩
    split_ifs with h
    · rcases hpc : e (p, c) with _ | ⟨q, w⟩
      · simp [hpc]
      · have hq : q ∈ T.leaves := he _ _ _ _ hpc
        simp [hpc, hq]
    · rfl
  · intro p hp c
    have h : (p, c) ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) := by simp [hp]
    simp only [dif_pos h]
    rcases e (p, c) with _ | ⟨q, w⟩ <;> simp
  · intro p c q w hpc
    simp only at hpc
    split_ifs at hpc with h
    · rcases he' : e (p, c) with _ | ⟨q', w'⟩ <;> rw [he'] at hpc
      · simp at hpc
      · simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hpc
        obtain ⟨rfl, -⟩ := hpc
        exact he _ _ _ _ he'

end Edges

section CutK

/-- The sifts of `x`'s prefixes from position `k` on agree. -/
def PrefAgreeK (cut cut' : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) (k : ℕ) :
    Prop :=
  ∀ j, k ≤ j → t.sift (ofCut cut) (prefixOf x j) = t.sift (ofCut cut') (prefixOf x j)

variable {cut cut' : FreeMonoid α → Option Bool} {t : DTree α} {edges : Edges α} {k : ℕ}
  {x : FreeMonoid α}

theorem probeOrAgree_congrK (h : PrefAgreeK cut cut' t x k) :
    probeOrAgree (ofCut cut) t edges k x = probeOrAgree (ofCut cut') t edges k x :=
  probe_congrK _ t edges k x h

theorem probeOrAgree_congrK_rd {rd rd' : FreeMonoid α → ARU}
    (h : PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) t x k) :
    probeOrAgree rd t edges k x = probeOrAgree rd' t edges k x := by
  have := probeOrAgree_congrK (edges := edges) h
  rwa [ofCut_rd, ofCut_rd] at this

theorem middleHarvestBy_congrK (h : PrefAgreeK cut cut' t x k) :
    middleHarvestBy cut t edges k x = middleHarvestBy cut' t edges k x := by
  simp only [middleHarvestBy, probeOrAgree_congrK (edges := edges) h]

end CutK

end OrthoDFA
