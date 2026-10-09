import OrthoDFA.Loop

/-!
# The L\* loop: the skeleton

`loop_succeeds` by a union bound over five events, each bounded by a lemma `sorry` for now:
* `loop_false_end`: a consistent ending or a harvest whose claim fails;
* `loop_decided_wrong`: a read outside the band off its side, among the round's reads;
* `loop_inband_wrong`: more than `j` reads inside the band off their side;
* `loop_heavy_hits`: one of at most `j` misread strings drawn `m / j` times in a window of
  `nmax + 1` probes;
* `loop_unsettled`: a stretch left unsettled with none of the others.
`loop_cover` says nothing else fails: with no misread past these, every split is genuine, so the
tree stays within `|Q| + 2` leaves, the loop runs at most `stretchGood` stretches within the
probes, and its ending is genuine unless one of the events holds.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- The loop's run on the noise and draws `p`. -/
noncomputable def loopOut (C : LoopCfg) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) {P : ℕ} (p : Ω × (Fin P → FreeMonoid α)) :
    LoopState α × Option LoopEnd :=
  loopRun C (readsAt O B F p.1) loopStart (List.ofFn p.2)

variable (C : LoopCfg) (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (P j : ℕ)

/-- The cut at `ω` decides `z` off its side. -/
def Misread (ω : Ω) (z : FreeMonoid α) : Prop :=
  (readsAt O B F ω).cut z = some (!refSide O B F z)

/-- A consistent ending or a harvest whose claim fails. -/
def FalseEnd : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | ((loopOut C O B F p).2 = some .agree ∨ ∃ c, (loopOut C O B F p).2 = some (.harvest c))
    ∧ ¬ LoopGenuine C (readsAt O B F p.1) D (loopOut C O B F p).1 (loopOut C O B F p).2}

/-- A read outside the band off its side. -/
def DecidedWrong : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | ∃ z ∈ (loopOut C O B F p).1.log, ¬ InBand O B F z ∧ Misread O B F p.1 z}

/-- More than `j` reads inside the band off their side. -/
def InbandWrong : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | ∃ S : Finset (FreeMonoid α), S.card = j + 1
    ∧ ∀ z ∈ S, z ∈ (loopOut C O B F p).1.log ∧ InBand O B F z ∧ Misread O B F p.1 z}

open scoped Classical in
/-- A misread string beginning `m / j` of the draws in a window of `nmax + 1`. -/
def HeavyHits : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | ∃ z ∈ (loopOut C O B F p).1.log, Misread O B F p.1 z ∧ ∃ i : ℕ,
    C.m / j ≤ (Finset.univ.filter fun l : Fin P =>
      i ≤ l ∧ (l : ℕ) ≤ i + C.nmax ∧ z.toList <+: (p.2 l).toList).card}

/-- A stretch left unsettled with none of the misreads above. -/
def Unsettled : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | (loopOut C O B F p).2 = some .unsettled}
    ∩ (DecidedWrong C O B F P ∪ InbandWrong C O B F P j ∪ HeavyHits C O B F P j)ᶜ

/-- With no misread past the events, the tree stays within `|Q| + 2` leaves (every split
separates two states' placements), the loop runs at most `stretchGood` stretches of at most
`nmax + 1` probes, and its ending is genuine but for the events. -/
theorem loop_cover {Q : Type*} [Fintype Q] (A : DFA (FreeMonoid α) Q)
    (hA : O.L = {w | A.state w ∈ A.accept}) (hF : SuffixFree F) (hk : C.k ≤ C.L)
    (hm : 2 ≤ C.m / j) (hQ : Fintype.card Q + 3 ≤ C.Lmax) (hT : BelowTau C)
    (hP : stretchGood (Fintype.card α) (Fintype.card Q) * (C.nmax + 1) ≤ P) :
    {p | ¬ LoopGenuine C (readsAt O B F p.1) D (loopOut C O B F p).1 (loopOut C O B F p).2}
      ⊆ FalseEnd C O B F D P ∪ DecidedWrong C O B F P ∪ InbandWrong C O B F P j
        ∪ HeavyHits C O B F P j ∪ Unsettled C O B F P j := by
  sorry

/-- A stretch's draws are fresh given what came before, so each test settles on the wrong side
of its threshold with chance at most `a` per probe, over the round's `P` probes. -/
theorem loop_false_end :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real (FalseEnd C O B F D P) ≤ 5 * P * C.a := by
  sorry

/-- Each string read is read first on fresh bits (the family is suffix-free), so outside the
band it lands off its side with chance at most `crossB` given the past, over at most `readMax`
strings. -/
theorem loop_decided_wrong (hF : SuffixFree F) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real (DecidedWrong C O B F P)
      ≤ readMax C (Fintype.card α) P * crossB O B F := by
  sorry

/-- Below `τ₀` an undecided read ends the loop at once, and inside the band a read lands off its
side at most `κ` times as often as it is undecided: more than `j` such before the first
undecided one has chance at most `κ^(j+1)`. -/
theorem loop_inband_wrong (hF : SuffixFree F) (hT : BelowTau C) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real (InbandWrong C O B F P j)
      ≤ kappa O B F ^ (j + 1) := by
  sorry

/-- Each misread string is fixed by the draw that first reads it, and the later draws are
fresh: one of at most `j` of them beginning `r` of a window's `nmax + 1` draws has chance at most
`(e·(nmax + 1)·prefixMax/(r − 1))^(r − 1)` per window and string. -/
theorem loop_heavy_hits (hm : 2 ≤ C.m / j) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real
        (HeavyHits C O B F P j ∩ (InbandWrong C O B F P j)ᶜ ∩ (DecidedWrong C O B F P)ᶜ)
      ≤ hitBound C D P j (C.m / j) := by
  sorry

/-- With no misread past the events, correct records fall on at most `|Σ|·|Q|` edges and targets
(each state's strings sift to one leaf), so edges at `1 − acc − η` or more reach `m` records at
one of them within `nmax` probes, and edges below settle the disagreement test; at most
`stretchGood` stretches. -/
theorem loop_unsettled {Q : Type*} [Fintype Q] (A : DFA (FreeMonoid α) Q)
    (hA : O.L = {w | A.state w ∈ A.accept}) (hT : BelowTau C) (η : ℝ) :
    (μ.prod (Measure.pi fun _ : Fin P => D)).real (Unsettled C O B F P j)
      ≤ stretchGood (Fintype.card α) (Fintype.card Q)
        * unsettledAt C (Fintype.card α) (Fintype.card Q) η := by
  sorry

theorem loop_succeeds : LoopSucceeds := by
  intro α _ _ Ω _ μ _ Q _ C A O B F D _ P j η hA hF hL hk hj hm hQ hT hP
  set M := μ.prod (Measure.pi fun _ : Fin P => D)
  set G := {p : Ω × (Fin P → FreeMonoid α) | LoopGenuine C (readsAt O B F p.1) D
    (loopOut C O B F p).1 (loopOut C O B F p).2}
  have hsub : Gᶜ ⊆ FalseEnd C O B F D P ∪ DecidedWrong C O B F P ∪ InbandWrong C O B F P j
      ∪ (HeavyHits C O B F P j ∩ (InbandWrong C O B F P j)ᶜ ∩ (DecidedWrong C O B F P)ᶜ)
      ∪ Unsettled C O B F P j := by
    intro p hp
    rcases loop_cover C O B F D P j A hA hF hk hm hQ hT hP hp with
      (((h | h) | h) | h) | h
    · exact .inl (.inl (.inl (.inl h)))
    · exact .inl (.inl (.inl (.inr h)))
    · exact .inl (.inl (.inr h))
    · by_cases hi : p ∈ InbandWrong C O B F P j
      · exact .inl (.inl (.inr hi))
      by_cases hd : p ∈ DecidedWrong C O B F P
      · exact .inl (.inl (.inl (.inr hd)))
      exact .inl (.inr ⟨⟨h, hi⟩, hd⟩)
    · exact .inr h
  have hcov : (1 : ℝ) ≤ M.real G + M.real Gᶜ := by
    rw [← probReal_univ (μ := M), ← Set.union_compl_self G]
    exact measureReal_union_le _ _
  set E₁ := FalseEnd C O B F D P
  set E₂ := DecidedWrong C O B F P
  set E₃ := InbandWrong C O B F P j
  set E₄ := HeavyHits C O B F P j ∩ (InbandWrong C O B F P j)ᶜ ∩ (DecidedWrong C O B F P)ᶜ
  set E₅ := Unsettled C O B F P j
  have hc := measureReal_mono (μ := M) hsub (measure_ne_top _ _)
  have u₁ := measureReal_union_le (μ := M) (E₁ ∪ E₂ ∪ E₃ ∪ E₄) E₅
  have u₂ := measureReal_union_le (μ := M) (E₁ ∪ E₂ ∪ E₃) E₄
  have u₃ := measureReal_union_le (μ := M) (E₁ ∪ E₂) E₃
  have u₄ := measureReal_union_le (μ := M) E₁ E₂
  have h1 := loop_false_end C O B F D P
  have h2 := loop_decided_wrong C O B F D P hF
  have h3 := loop_inband_wrong C O B F D P j hF hT
  have h4 := loop_heavy_hits C O B F D P j hm
  have h5 := loop_unsettled C O B F D P j A hA hT η
  change _ ≤ M.real G
  linarith

end OrthoDFA
