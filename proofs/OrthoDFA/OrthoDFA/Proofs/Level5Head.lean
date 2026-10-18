import OrthoDFA.Proofs.Level5RareReads
import OrthoDFA.Proofs.Level5Deep

/-!
# Level 5's reads before `K`, at the round's end

The run's end is fixed by what its rounds read, so a last round reading `headStrings` of the end
state's tree is fixed by the earlier ones while nothing they read is on the rarer side
(`headRounds_local`). Counting a good read-state read undecided as bad besides one on the rarer
side, `AdaptiveReadsBound` bounds the chance that some step reads on the rarer side or that a
string the end state's tree reads before `K` at a good read-state is undecided (`wrongHead_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Run

variable (C : Level5Settings)

/-- Reads agreeing on what the steps read, at the states the first reads reach, end the run in the
same state. -/
theorem runProbes_congr_steps {r r' : FreeMonoid α → ARU} :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level5State α),
      (∀ j < xs.length, ∀ s x, stateAt C r s₀ xs j = some s → xs[j]? = some x →
        applyProbe C (fun z => (r z).cut) s x = applyProbe C (fun z => (r' z).cut) s x) →
      runProbes C (fun z => (r z).cut) s₀ xs = runProbes C (fun z => (r' z).cut) s₀ xs
  | [], _, _ => rfl
  | x :: xs, s₀, h => by
    have h0 := h 0 (by simp) s₀ x rfl rfl
    simp only [runProbes]
    rw [← h0]
    rcases ha : applyProbe C (fun z => (r z).cut) s₀ x with s' | ⟨e, s'⟩
    · refine runProbes_congr_steps xs s' fun j hj s y hs hy => ?_
      exact h (j + 1) (by simp; omega) s y (by simp only [stateAt, ha]; exact hs) (by simpa using hy)
    · rfl

/-- The run's end state, once it ends, is the state `RunEnds` checks. -/
theorem runEnds_end {cut : FreeMonoid α → Option Bool} {Q : Level5Ending α → Level5State α → Prop} :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level5State α),
      RunEnds (applyProbe C cut) Q s₀ xs →
      RunEnds (applyProbe C cut) (fun e s => Q e s ∧ s = (runProbes C cut s₀ xs).1) s₀ xs
  | [], _, h => h.elim
  | x :: xs, s₀, h => by
    simp only [RunEnds, runProbes] at h ⊢
    rcases ha : applyProbe C cut s₀ x with s' | ⟨e, s'⟩ <;> rw [ha] at h
    · simpa only [ha] using runEnds_end xs s' h
    · exact ⟨h, rfl⟩

end Run

section Head

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)

/-- The rounds, and last the strings the end state's tree reads before `K`. -/
noncomputable def headRounds (K : ℕ) (xs : List (FreeMonoid α)) (i : ℕ)
    (read : FreeMonoid α → ARU) : Finset (FreeMonoid α) :=
  if i < 2 * xs.length then rounds C xs i read
  else if i = 2 * xs.length then
    headStrings (runProbes C (fun z => (read z).cut) initialState xs).1.tree C.walkStart K
  else ∅

variable (hσ : Fintype.card σ + 2 ≤ C.maxLeaves)
include hσ

/-- While nothing read is on the rarer side, the run's end state is true. -/
theorem inv_end (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level5State α),
      TruthInv G C.maxLeaves s₀ →
      (∀ j < xs.length, ∀ s x, stateAt C read s₀ xs j = some s → xs[j]? = some x →
        ¬ WrongStep G read C s x) →
      TruthInv G C.maxLeaves (runProbes C (fun z => (read z).cut) s₀ xs).1
  | [], _, ht, _ => ht
  | x :: xs, s₀, ht, hw => by
    have hx0 : ¬ WrongStep G read C s₀ x := hw 0 (by simp) s₀ x rfl rfl
    obtain ⟨hinl, hinr⟩ := truthInv_step G read C hσ ht hx0
    simp only [runProbes]
    rcases ha : applyProbe C (fun z => (read z).cut) s₀ x with s' | ⟨e, s'⟩
    · refine inv_end read xs s' (hinl s' ha) fun j hj s y hs hy => ?_
      exact hw (j + 1) (by simp; omega) s y (by simp only [stateAt, ha]; exact hs)
        (by simpa using hy)
    · exact (hinr e s' ha).2

omit hσ in
theorem steps_not_wrong (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU)
    (hg : Good G C xs read (2 * xs.length)) :
    ∀ j < xs.length, ∀ s x, stateAt C read initialState xs j = some s → xs[j]? = some x →
      ¬ WrongStep G read C s x := by
  rintro j hj s x hs hx ⟨z, hz, hw⟩
  rcases Finset.mem_union.1 hz with hz | hz
  · exact hg (2 * j) (by omega) z (by rw [rounds_even C xs read hs hx]; exact hz) hw
  · exact hg (2 * j + 1) (by omega) z (by rw [rounds_odd C xs read hs hx]; exact hz) hw

/-- While nothing read is on the rarer side, each round, the last included, is fixed by what the
earlier ones read. -/
theorem headRounds_local (K : ℕ) (xs : List (FreeMonoid α)) (i : ℕ) (r r' : FreeMonoid α → ARU)
    (hg : Good G C xs r (min i (2 * xs.length)))
    (ha : ∀ l < i, ∀ z ∈ headRounds C K xs l r, r z = r' z) :
    headRounds C K xs i r = headRounds C K xs i r' := by
  unfold headRounds
  by_cases hi : i < 2 * xs.length
  · simp only [if_pos hi]
    refine rounds_local G C hσ xs i r r' (by rwa [min_eq_left hi.le] at hg) fun l hl z hz => ?_
    exact ha l hl z (by simp only [headRounds, if_pos (show l < 2 * xs.length by omega)]; exact hz)
  simp only [if_neg hi]
  by_cases hi' : i = 2 * xs.length
  swap
  · simp only [if_neg hi']
  simp only [if_pos hi']
  subst hi'
  rw [min_self] at hg
  have ha' : ∀ l < 2 * xs.length, ∀ z ∈ rounds C xs l r, r z = r' z := fun l hl z hz =>
    ha l hl z (by simp only [headRounds, if_pos hl]; exact hz)
  congr 2
  refine congrArg Prod.fst (runProbes_congr_steps C xs initialState fun j hj s x hs hx => ?_)
  obtain ⟨-, -, ht, -⟩ := inv_of_good G C hσ xs r j s hs fun l hl => hg l (by omega)
  refine applyProbe_congr C (fun q e t w h => edge_leaf G ht h) fun z hz => ?_
  rcases Finset.mem_union.1 hz with hz | hz
  · simp only [ha' (2 * j) (by omega) z (by rw [rounds_even C xs r hs hx]; exact hz)]
  · simp only [ha' (2 * j + 1) (by omega) z (by rw [rounds_odd C xs r hs hx]; exact hz)]

/-- While nothing read is on the rarer side, the last round has at most `H maxLeaves` strings,
`H` the strings of length `k` to `K − 1`. -/
theorem headRounds_last_card (K : ℕ) (xs : List (FreeMonoid α)) (r : FreeMonoid α → ARU)
    (hg : Good G C xs r (2 * xs.length)) :
    (headRounds C K xs (2 * xs.length) r).card
      ≤ (∑ i ∈ Finset.Ico C.walkStart K, Fintype.card α ^ i) * C.maxLeaves := by
  simp only [headRounds, lt_irrefl, if_false, if_true]
  refine (headStrings_card_le _ _ _).trans (Nat.mul_le_mul_left _ ?_)
  obtain ⟨-, -, -, hL⟩ := inv_end G C hσ r xs initialState (truthInv_start G (by omega))
    (steps_not_wrong G C xs r hg)
  have := midfixes_card (runProbes C (fun z => (r z).cut) initialState xs).1.tree
  omega

end Head

section Bound

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

open scoped Classical in
/-- Given probes no longer than `L`, some step reads on the rarer side, or a string the end
state's tree reads before `K` at a good read-state is undecided, with chance at most
`(ε + κ + 1.5θ) (|xs| ((L + 1) maxLeaves + maxLeaves² |Σ|) + H maxLeaves)`, `H` the strings of
length `k` to `K − 1`. -/
theorem wrongHead_le (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (hind : iIndepFun read μ) (hκ : 0 ≤ G.κ) (hε : 0 ≤ G.ε) (hθ : 0 ≤ G.θ) (K : ℕ)
    (xs : List (FreeMonoid α)) (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L) :
    μ {ω | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C)
          initialState xs
        ∨ ¬ G.Decided (read · ω) (headStrings
          (runProbes C (fun z => (read z ω).cut) initialState xs).1.tree C.walkStart K)}
      ≤ ENNReal.ofReal ((G.ε + G.κ + 3 / 2 * G.θ) * (xs.length
        * ((L + 1) * C.maxLeaves + C.maxLeaves * Fintype.card α * C.maxLeaves)
        + (∑ i ∈ Finset.Ico C.walkStart K, Fintype.card α ^ i) * C.maxLeaves)) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z =>
    {G.minVal (G.M.eval z.toList)} ∪ (if G.Good (G.M.eval z.toList) then {.undecided} else ∅)
  have hWB : ∀ (rd : FreeMonoid α → ARU) z, G.WrongRead rd z → rd z ∈ B z := fun rd z h =>
    Or.inl ((cut_minVal G _ _).1 h)
  have hB : ∀ z, μ.real {ω | read z ω ∈ B z} ≤ G.ε + G.κ + 3 / 2 * G.θ := fun z => by
    have hw : μ.real {ω | read z ω = G.minVal (G.M.eval z.toList)} ≤ G.ε + G.κ := by
      have h := wrongRead_le G hmeas hlaw hκ hε z
      have he : {ω | G.WrongRead (read · ω) z}
          = {ω | read z ω = G.minVal (G.M.eval z.toList)} := by
        ext ω; simp only [Set.mem_ofPred_eq]; exact cut_minVal G _ _
      rw [he] at h
      exact ENNReal.toReal_le_of_le_ofReal (by linarith) h
    have hu : μ.real {ω | read z ω ∈ (if G.Good (G.M.eval z.toList) then {ARU.undecided}
        else (∅ : Set ARU))} ≤ 3 / 2 * G.θ := by
      split_ifs with hg
      · simp only [Set.mem_singleton_iff]
        rw [hlaw]; exact hg
      · simp only [Set.mem_empty_iff_false, Set.ofPred_false, measureReal_empty]
        positivity
    have hu' := measureReal_union_le (μ := μ) {ω | read z ω = G.minVal (G.M.eval z.toList)}
      {ω | read z ω ∈ (if G.Good (G.M.eval z.toList) then {ARU.undecided} else (∅ : Set ARU))}
    have hset : {ω | read z ω ∈ B z} = {ω | read z ω = G.minVal (G.M.eval z.toList)}
        ∪ {ω | read z ω ∈ (if G.Good (G.M.eval z.toList) then {ARU.undecided}
          else (∅ : Set ARU))} := by
      ext ω; simp only [B, Set.mem_union, Set.mem_singleton_iff, Set.mem_ofPred_eq]
    rw [hset]
    linarith
  have hgood : ∀ (r : FreeMonoid α → ARU) (n : ℕ),
      (∀ i < n, ∀ z ∈ headRounds C K xs i r, r z ∉ B z) →
      Good G C xs r (min n (2 * xs.length)) := by
    intro r n hg i hi z hz hw
    have hz' : z ∈ headRounds C K xs i r := by
      simp only [headRounds, if_pos (show i < 2 * xs.length by omega)]; exact hz
    exact hg i (by omega) z hz' (hWB r z hw)
  have hmain := adaptiveReadsBound_holds μ read B (G.ε + G.κ + 3 / 2 * G.θ)
    (fun i r => headRounds C K xs i r)
    (fun i => if i < 2 * xs.length then (if i % 2 = 0 then (L + 1) * C.maxLeaves
      else C.maxLeaves * Fintype.card α * C.maxLeaves)
      else (∑ i ∈ Finset.Ico C.walkStart K, Fintype.card α ^ i) * C.maxLeaves)
    (2 * xs.length + 1) hmeas hind hB
    (fun i r r' hg ha => headRounds_local G C hσ K xs i r r' (hgood r i hg) ha)
    (fun i r hg => by
      by_cases hi : i < 2 * xs.length
      · simp only [if_pos hi, headRounds]
        have := rounds_card G C hσ xs L hL i r (by
          have := hgood r i hg; rwa [min_eq_left hi.le] at this)
        exact this
      · simp only [if_neg hi]
        by_cases hi' : i = 2 * xs.length
        · subst hi'
          refine headRounds_last_card G C hσ K xs r ?_
          have := hgood r (2 * xs.length) hg; rwa [min_self] at this
        · simp [headRounds, hi, hi'])
  refine (measure_mono fun ω hω => ?_).trans (hmain.trans (le_of_eq ?_))
  · rcases hω with hω | hω
    · obtain ⟨j, hj, s, x, h1, h2, z, hz, hw⟩ := anyStep_rounds G C (read · ω) xs _ hω
      rcases Finset.mem_union.1 hz with hz | hz
      · refine ⟨2 * j, by omega, z, ?_, hWB _ z hw⟩
        simp only [headRounds, if_pos (show 2 * j < 2 * xs.length by omega)]
        rw [rounds_even C xs _ h1 h2]; exact hz
      · refine ⟨2 * j + 1, by omega, z, ?_, hWB _ z hw⟩
        simp only [headRounds, if_pos (show 2 * j + 1 < 2 * xs.length by omega)]
        rw [rounds_odd C xs _ h1 h2]; exact hz
    · simp only [ReadModel.Decided, not_forall, not_not] at hω
      obtain ⟨z, hz, hg, hu⟩ := hω
      refine ⟨2 * xs.length, by omega, z, ?_, Or.inr ?_⟩
      · simp only [headRounds, lt_irrefl, if_false, if_true]; exact hz
      · simp only [if_pos hg, Set.mem_singleton_iff]; exact hu
  · set a := (L + 1) * C.maxLeaves
    set b := C.maxLeaves * Fintype.card α * C.maxLeaves
    set c := (∑ i ∈ Finset.Ico C.walkStart K, Fintype.card α ^ i) * C.maxLeaves
    have e1 : ∀ i ∈ Finset.range (2 * xs.length),
        ((if i < 2 * xs.length then (if i % 2 = 0 then a else b) else c : ℕ) : ℝ)
          = ((if i % 2 = 0 then a else b : ℕ) : ℝ) := fun i hi => by
      rw [if_pos (Finset.mem_range.1 hi)]
    rw [Finset.sum_range_succ, Finset.sum_congr rfl e1, sum_alt, if_neg (lt_irrefl _)]
    simp only [a, b, c]
    push_cast
    ring

/-- Over the probes and the reads, `wrongHead_le` at `probeBudget` probes. -/
theorem pi_wrongHead_le (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (hind : iIndepFun read μ) (hκ : 0 ≤ G.κ) (hε : 0 ≤ G.ε) (hθ : 0 ≤ G.θ) (K : ℕ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => D)
        {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C)
            initialState (List.ofFn xs)
          ∨ ¬ G.Decided (read · ω) (headStrings
            (runProbes C (fun z => (read z ω).cut) initialState (List.ofFn xs)).1.tree
              C.walkStart K)} ∂μ
      ≤ ENNReal.ofReal ((G.ε + G.κ + 3 / 2 * G.θ) * (C.probeBudget
        * ((L + 1) * C.maxLeaves + C.maxLeaves * Fintype.card α * C.maxLeaves)
        + (∑ i ∈ Finset.Ico C.walkStart K, Fintype.card α ^ i) * C.maxLeaves)) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points _ _ _ fun d hd => ?_
  have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ L := by
    intro x hx
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    by_contra hlt
    apply hd
    have hpi : (Measure.pi fun _ : Fin C.probeBudget => D) {d} = ∏ j, D {d j} := by
      rw [← Measure.pi_pi]
      congr 1
      ext d'
      simp [funext_iff]
    rw [hpi]
    refine Finset.prod_eq_zero (Finset.mem_univ i) (measure_mono_null ?_ (ae_iff.1 hlen))
    intro x hx
    rw [Set.mem_singleton_iff] at hx
    subst hx
    exact hlt
  have := wrongHead_le G C hσ hmeas hlaw hind hκ hε hθ K (List.ofFn d) L hL
  rw [List.length_ofFn] at this
  exact this

end Bound

end OrthoDFA
