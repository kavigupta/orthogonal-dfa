import OrthoDFA.Proofs.LoopAvoid
import OrthoDFA.Proofs.Renoise

/-!
# Level 4 read through a re-noised family

`SuffixFamily.renoise` reads the family through `O.renoise R`: a suffix family on two copies of
the noise, with the same read distributions, which agrees with the family off `R` at the first
copy (`renoise_read_of_notMem`). Level 4's guarantee holds of it, and a second copy can be fixed at
a point where the bound still holds (`exists_noise_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}

/-- The family read through `O.renoise R`. -/
def SuffixFamily.renoise (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    SuffixFamily (μ.prod μ) α σ :=
  ⟨fam.O.renoise R, fam.M, fam.F, fam.kl, fam.kh, fam.ε, fam.ε₂, fam.κ, fam.lang, fam.suffixFree,
    fam.band⟩

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

lemma SuffixFamily.renoise_good (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) (θ : ℝ)
    (q : σ) : (fam.renoise R).oracle.Good θ q ↔ fam.oracle.Good θ q := by
  unfold TriOracle.Good SuffixFamily.oracle
  dsimp only
  rw [SuffixFamily.renoise_readDist]

lemma renoise_read_of_notMem (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α))
    {z : FreeMonoid α} (hz : ∀ v ∈ fam.F, z * v ∉ R) (p : Ω × Ω) :
    (fam.renoise R).read p z = fam.read p.1 z := by
  unfold SuffixFamily.read SuffixFamily.renoise familyRead voteCount
  dsimp only
  congr 2
  refine Finset.filter_congr fun v hv => ?_
  rw [renoise_mq_of_notMem fam.O (hz v hv)]

section Fail

variable {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'} [IsProbabilityMeasure μ']

/-- Where level 4, reading `fam` over probes from `D'`, fails its claim. -/
def l4Fail (fam : SuffixFamily μ' α σ) (C : Level4Settings) (D' : Measure (FreeMonoid α))
    (θ y : ℝ) : Set (Ω' × (Fin C.probeBudget → FreeMonoid α)) :=
  {d | let result := indecisiveLStarLevel4 C (fun z => fam.read d.1 z) d.2
    ¬ EndsWell fam.M (fun q => ¬ fam.oracle.Good θ q) D' C.disagreementRate
      (searchingProbes (fun z => fam.read d.1 z) C.walkStart result.hypothesis) C.walkStart
      (harvestLaw D' (fun z => (fam.read d.1 z).cut) C.walkStart) y result.ending}

theorem measurableSet_l4Fail (fam : SuffixFamily μ' α σ) (C : Level4Settings)
    (hC : 2 ≤ C.maxLeaves) (D' : Measure (FreeMonoid α)) (θ y : ℝ) :
    MeasurableSet (l4Fail fam C D' θ y) := by
  classical
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ'
  refine measurableSet_prod_countable_right _ fun xs => ?_
  set A := readStrings (shortStrings (probesLen xs + 1 + C.maxLeaves)) fam.F
  set res : Ω' → Level4Result α := fun ω =>
    indecisiveLStarLevel4 C (fun z => familyRead fam.O.mq fam.F fam.kl fam.kh z ω) xs
  have hres : ∀ ω ω', (∀ p ∈ A, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → res ω = res ω' :=
    fun ω ω' h => level4_result_congr fam.O C hC fam.F _ _ xs h
  set rep : Finset (FreeMonoid α) → Level4Result α := fun U =>
    if h : ∃ ω, A.filter (fun p => fam.O.mq p ω = 1) = U then res h.choose else res ω₀
  have hrepω : ∀ ω, res ω = rep (A.filter (fun p => fam.O.mq p ω = 1)) := by
    intro ω
    have h : ∃ ω', A.filter (fun p => fam.O.mq p ω' = 1) = A.filter (fun p => fam.O.mq p ω = 1) :=
      ⟨ω, rfl⟩
    simp only [rep, dif_pos h]
    refine hres _ _ fun p hp => ?_
    have := Finset.ext_iff.1 h.choose_spec p
    simp only [Finset.mem_filter, hp, true_and] at this
    exact this.symm
  have hset : {ω | (ω, xs) ∈ l4Fail fam C D' θ y}
      = ⋃ U ∈ A.powerset, ({ω | A.filter (fun p => fam.O.mq p ω = 1) = U}
        ∩ {ω | ¬ EndsWell fam.M (fun q => ¬ fam.oracle.Good θ q) D' C.disagreementRate
          (searchingProbes (fun z => familyRead fam.O.mq fam.F fam.kl fam.kh z ω) C.walkStart
            (rep U).hypothesis) C.walkStart
          (harvestLaw D' (fun z => (familyRead fam.O.mq fam.F fam.kl fam.kh z ω).cut) C.walkStart)
          y (rep U).ending}) := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop, l4Fail]
    constructor
    · intro h
      refine ⟨_, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hrepω ω]
      exact h
    · rintro ⟨U, -, hU, h⟩
      rw [← hU, ← hrepω ω] at h
      exact h
  rw [hset]
  refine Finset.measurableSet_biUnion _ fun U _ => MeasurableSet.inter ?_ ?_
  · exact measurableSet_of_reads fam.O A _ fun ω ω' h (hω : A.filter _ = U) => by
      show A.filter _ = U
      rw [← hω]
      exact Finset.filter_congr fun p hp => (h p hp).symm
  · exact measurableSet_not_endsWell fam.O fam.F _ _ C.walkStart fam.M _ D' _ y _ _

end Fail

/-- A second copy of the noise fixed at a point where level 4's failure, over the first copy and
the probes, is still at most its bound over both. -/
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
