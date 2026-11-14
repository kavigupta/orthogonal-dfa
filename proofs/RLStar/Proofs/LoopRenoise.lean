import RLStar.Proofs.FamilyOracle
import RLStar.Proofs.Level5
import RLStar.Proofs.LoopAvoid
import RLStar.Proofs.Renoise

/-!
# A family read through re-noised strings

`SuffixFamily.renoise` reads the family through `O.renoise R`: a suffix family on two copies of
the noise, with the same read distributions, which agrees with the family off `R` at the first
copy (`renoise_read_of_notMem`). Strings past different first `k` letters, followed by a family's
suffixes, differ (`disjoint_readStrings_of_keys`). A second copy can be fixed at a point where a
bound over both still holds (`exists_noise_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}

/-- The family read through `O.renoise R`. -/
def SuffixFamily.renoise (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    SuffixFamily (μ.prod μ) α σ :=
  ⟨fam.O.renoise R, fam.M, fam.F, fam.kl, fam.kh, fam.κ, fam.lang, fam.suffixFree, fam.band⟩

lemma SuffixFamily.renoise_readDist (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    (fam.renoise R).readDist = fam.readDist := by
  classical
  funext q
  unfold SuffixFamily.readDist
  show (if h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q then
      readProb (fam.O.renoise R) fam.F fam.kl fam.kh h.choose else fun _ => 0)
    = (if h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q then
      readProb fam.O fam.F fam.kl fam.kh h.choose else fun _ => 0)
  split_ifs with h
  · funext rd
    rw [readProb_eq_readLaw, readProb_eq_readLaw]
    rfl
  · rfl

lemma renoise_read_of_notMem (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α))
    {z : FreeMonoid α} (hz : ∀ v ∈ fam.F, z * v ∉ R) (p : Ω × Ω) :
    (fam.renoise R).read p z = fam.read p.1 z := by
  (unfold SuffixFamily.read SuffixFamily.renoise; simp only [familyRead_eq]; unfold voteCount)
  congr 2
  refine Finset.filter_congr fun v hv => ?_
  rw [renoise_mq_of_notMem fam.O (hz v hv)]

section Zone

/-- The first `k` letters of the probes. -/
def probeKeys (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) : Finset (FreeMonoid α) :=
  Finset.univ.image fun i => prefixOf (xs i) k

theorem prefixOf_of_mem_readStrings {k : ℕ} {Y F : Finset (FreeMonoid α)} {w : FreeMonoid α}
    (hw : w ∈ readStrings Y F) (hY : ∀ z ∈ Y, k ≤ z.toList.length) :
    ∃ z ∈ Y, prefixOf w k = prefixOf z k := by
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hz, -⟩ := Finset.mem_product.1 hzv
  exact ⟨z, hz, prefixOf_mul_of_le (hY z hz) v⟩

/-- Strings past different first `k` letters, followed by the family's suffixes, differ. -/
theorem disjoint_readStrings_of_keys (k : ℕ) {Y Y' : Finset (FreeMonoid α)}
    (F : Finset (FreeMonoid α)) (hY : ∀ z ∈ Y, k ≤ z.toList.length)
    (hY' : ∀ z ∈ Y', k ≤ z.toList.length)
    (hne : ∀ z ∈ Y, ∀ z' ∈ Y', prefixOf z k ≠ prefixOf z' k) :
    Disjoint (readStrings Y F) (readStrings Y' F) := by
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨z, hz, he⟩ := prefixOf_of_mem_readStrings hw hY
  obtain ⟨z', hz', he'⟩ := prefixOf_of_mem_readStrings hw' hY'
  exact hne z hz z' hz' (he.symm.trans he')

end Zone

/-- A second copy of the noise fixed at a point where a set's chance, over the first copy and the
draws, is still at most its chance over both. -/
theorem exists_noise_le {X : Type*} [MeasurableSpace X] (π : Measure X) [IsProbabilityMeasure π]
    (S : Set ((Ω × Ω) × X)) (hS : MeasurableSet S) :
    ∃ ω₀ : Ω, (μ.prod π) {q | ((q.1, ω₀), q.2) ∈ S} ≤ ((μ.prod μ).prod π) S := by
  set g : Ω → ENNReal := fun ω₀ => (μ.prod π) {q | ((q.1, ω₀), q.2) ∈ S}
  have hmeasS' : ∀ ω₀ : Ω, MeasurableSet {q : Ω × X | ((q.1, ω₀), q.2) ∈ S} := fun ω₀ =>
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hS
  -- the second copy, integrated out last
  set e : Ω × (Ω × X) → (Ω × Ω) × X := fun t => ((t.2.1, t.1), t.2.2)
  have he : MeasurePreserving e (μ.prod (μ.prod π)) ((μ.prod μ).prod π) := by
    have h1 : MeasurePreserving (fun t : Ω × (Ω × X) => ((t.1, t.2.1), t.2.2))
        (μ.prod (μ.prod π)) ((μ.prod μ).prod π) := (measurePreserving_prodAssoc μ μ π).symm
    have h2 : MeasurePreserving (fun t : (Ω × Ω) × X => ((t.1.2, t.1.1), t.2))
        ((μ.prod μ).prod π) ((μ.prod μ).prod π) :=
      ((Measure.measurePreserving_swap (μ := μ) (ν := μ)).prod (MeasurePreserving.id π))
    exact h2.comp h1
  have hint : ∫⁻ ω₀, g ω₀ ∂μ = ((μ.prod μ).prod π) S := by
    rw [← he.measure_preimage hS.nullMeasurableSet, Measure.prod_apply (he.measurable hS)]
    rfl
  have hgm : Measurable g := by
    exact measurable_measure_prodMk_left (ν := μ.prod π) (he.measurable hS)
  obtain ⟨ω₀, hω₀⟩ := exists_le_lintegral (μ := μ) hgm.aemeasurable
  exact ⟨ω₀, hω₀.trans hint.le⟩

end OrthoDFA
