import OrthoDFA.Proofs.PowerCase

/-!
# The power of the round's split tests

At each key, the round's first test that is either in the power case or splits is a power-case
test that does not split with chance at most `e^{−τ²}` times the chance there is such a test.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

section Strings

variable (K : StageKnobs α) (R : CutReads α)

theorem testStrings_nodup {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {d : FreeMonoid α} {skip : FreeMonoid α → Prop} :
    ((testStrings K R t pool path d skip).map Prod.fst).Nodup := by
  classical
  unfold testStrings
  have key : ∀ (l acc : List (FreeMonoid α × Bool)), (acc.map Prod.fst).Nodup →
      ((l.foldl (fun acc p => if p.1 ∈ acc.map Prod.fst ∨ skip p.1 then acc
        else acc ++ [p]) acc).map Prod.fst).Nodup := by
    intro l
    induction l with
    | nil => exact fun acc h => h
    | cons p l ih =>
      intro acc h
      simp only [List.foldl_cons]
      refine ih _ ?_
      split_ifs with hc
      · exact h
      · rw [List.map_append, List.nodup_append]
        refine ⟨h, List.nodup_singleton _, ?_⟩
        simp only [List.map_cons, List.map_nil, List.mem_singleton]
        rintro a ha b rfl rfl
        exact hc (.inl ha)
  exact key _ [] List.nodup_nil

end Strings

section Mask

variable (C : StrongCfg α) {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem stepTest_mask {A : RoundAcc α} {x : FreeMonoid α} (hw : WitPool A.s)
    (h : ∀ z ∈ stepPre C F A x, f₁ z = f₂ z) :
    stepTest C (rd B F f₁) A x = stepTest C (rd B F f₂) A x := by
  have hag : ∀ b, (b ∈ A.s.pool ∨ ∃ i, C.k ≤ i ∧ b = prefixOf x i) →
      AgreeOne C.K F f₁ f₂ A.s.tree b := fun b hb e m hm v hv =>
    h _ (mem_stepReads e hb (.inl hm) hv)
  have hpool : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ A.s.tree b := fun b hb => hag b (.inl hb)
  have hwit : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ A.s.tree y :=
    fun p c q y he => hpool y (hw p c q y he)
  have hpo : probeOutcome (rd B F f₁) A.s.tree A.s.edges C.k x
      = probeOutcome (rd B F f₂) A.s.tree A.s.edges C.k x :=
    probeOutcome_congr fun i hi m hm => (hag _ (.inr ⟨i, hi, rfl⟩)).tree m hm
  have hsk : stepSkip C.K (rd B F f₁) C.k A.s x = stepSkip C.K (rd B F f₂) C.k A.s x := rfl
  unfold stepTest
  rw [hpo, hsk]
  split
  · rename_i ps fd ho
    have hfd : C.k ≤ fd - 1 := by have := probeOutcome_edge_gt _ ho; omega
    rw [seedStep_key_congr hwit (hag _ (.inr ⟨fd - 1, hfd, rfl⟩))]
    rcases hk : (seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
      (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd).key with _ | ⟨κ1, κ2⟩
    · rfl
    · have hr : (∃ s1 y sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
          (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .split κ2 s1 y sprime)
          ∨ ∃ s1 sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
            (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .member s1 sprime κ2 := by
        revert hk
        rcases seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
          (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd with
          ⟨d, s1, y, sp⟩ | ⟨s1, sp, d⟩ | b | _ <;> intro hk <;>
          simp only [SeedResult.key, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at hk
        · exact .inl ⟨s1, y, sp, by rw [hk.2]⟩
        · exact .inr ⟨s1, sp, by rw [hk.2]⟩
      obtain ⟨c, m, hm, rfl⟩ := seedStep_dist C.K _ hr
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
      exact testStrings_congr (fun b hb => (hpool b hb).tree) fun b hb => (hpool b hb).letter' c m hm
  · rfl

end Mask

end OrthoDFA
