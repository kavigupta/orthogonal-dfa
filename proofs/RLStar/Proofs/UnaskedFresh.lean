import RLStar.LowLevelClaims.UnaskedFresh
import RLStar.Proofs.AdaptiveReads
import RLStar.Proofs.Query

/-!
# An unasked string's read is independent of the run: the proof

A run that does not ask `z` is fixed by the reads of the strings it asks (`Query.eval_congr`), so
that it does not ask `z` and returns in `A` is measurable in the other strings' reads.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory AdaptiveReads

variable {Z Ω R β : Type*} [Countable Z] [DecidableEq Z] [MeasurableSpace R] [Countable R]
  [MeasurableSingletonClass R]

omit [Countable Z] [MeasurableSpace R] [Countable R] [MeasurableSingletonClass R] in
private theorem unasked_eq (read : Z → Ω → R) (p : Query Z R β) (z : Z) (A : Set β) :
    {ω | z ∉ p.reads (read · ω) ∧ p.eval (read · ω) ∈ A}
      = ⋃ (S : Finset Z) (_ : z ∉ S), (fun ω (y : S) => read y ω) ⁻¹'
        {a | ∃ r, (∀ y : S, r y = a y) ∧ p.reads r = S ∧ p.eval r ∈ A} := by
  ext ω
  simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_preimage, exists_prop]
  constructor
  · rintro ⟨hz, hA⟩
    exact ⟨p.reads (read · ω), hz, (read · ω), fun _ => rfl, rfl, hA⟩
  · rintro ⟨S, hzS, r, hr, hS, hA⟩
    obtain ⟨he, hre⟩ := p.eval_congr (o' := (read · ω)) r fun y hy => (hr ⟨y, hS ▸ hy⟩).symm
    exact ⟨hre ▸ hS ▸ hzS, he ▸ hA⟩

private theorem unasked_measurable (read : Z → Ω → R) (p : Query Z R β) (z : Z) (A : Set β) :
    MeasurableSet[others read z] {ω | z ∉ p.reads (read · ω) ∧ p.eval (read · ω) ∈ A} := by
  rw [unasked_eq]
  refine MeasurableSet.iUnion fun S => MeasurableSet.iUnion fun hzS => ?_
  have hm : Measurable[others read z] fun ω (y : S) => read y ω := by
    let : MeasurableSpace Ω := others read z
    refine measurable_pi_iff.2 fun y => ?_
    refine Measurable.of_comap_le ?_
    refine le_trans ?_ (le_iSup₂ (f := fun y (_ : y ∈ ({z}ᶜ : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) y.1 ?_)
    · exact le_rfl
    · exact fun h => hzS (h ▸ y.2)
  exact hm ((Set.to_countable _).measurableSet)

theorem unaskedFresh_holds : UnaskedFresh := by
  intro Z Ω R β _ _ _ _ _ _ μ _ read p z A B hmeas hind
  have hle : ∀ z,
      MeasurableSpace.comap (read z) inferInstance ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => (hmeas z).comap_le
  have hI := indep_iSup_of_disjoint hle ((iIndepFun_iff_iIndep _ read μ).1 hind) (S := {z}ᶜ)
    (T := {z}) (Set.disjoint_compl_left_iff_subset.2 subset_rfl)
  have hB : MeasurableSet[⨆ y ∈ ({z} : Set Z), MeasurableSpace.comap (read y) inferInstance]
      (read z ⁻¹' B) := by
    refine le_iSup₂ (f := fun y (_ : y ∈ ({z} : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) z rfl _ ?_
    exact ⟨B, (Set.to_countable _).measurableSet, rfl⟩
  convert (Indep_iff _ _ μ).1 hI _ _ (unasked_measurable read p z A) hB using 2
  · ext ω
    simp [and_assoc]
  · rfl

end OrthoDFA
