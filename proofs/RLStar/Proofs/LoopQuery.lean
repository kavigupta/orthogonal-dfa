import RLStar.Proofs.QueryLevel3
import RLStar.Proofs.ClusterCongr
import RLStar.Proofs.LoopGood
import RLStar.Proofs.LazyZone
import RLStar.Proofs.TrieTable

/-!
# The loop program

`loopQ`'s run against the answers at one noise is `loopRounds` at that noise in every round, each
round with its family read dropped (`indecisiveLStarLoop_eq`), and each of those rounds reads its
family at that noise (`loopRounds_read`). Against any answers, each round's clustering reading at
most `a` prefixes and `b` suffixes, each family at most `f` members and each lazy round at most
`Nr` strings, a round asks at most `a (b + 1) + f Nr` strings (`card_reads_roundQ_le`), and the
loop `K` times that (`card_reads_loopRoundsQ_le`). The learner adds the last round's table, `f`
reads for each of its sifts' questions (`card_reads_learnerQ_le`).
-/

noncomputable section

namespace OrthoDFA
open MeasureTheory Query

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

/-! ## The program's rounds as the functions' -/

section Bridge

variable (O : Oracle μ (FreeMonoid α)) (ω : Ω)

/-- A round without its family read. -/
def SpecRound.data (o : SpecRound α) : LoopRound α :=
  ⟨o.state, o.family, o.result, o.prefs, o.suffs⟩

/-- The round reads its family at the noise `ω`. -/
def SpecRound.ReadsAt (o : SpecRound α) : Prop :=
  o.read = o.data.read O ω

lemma specHistOf_eq (rs : List (SpecRound α)) :
    specHistOf rs = histOf (rs.map SpecRound.data) := by
  sorry -- closes by: unfold specHistOf histOf, List.filterMap_map, List.foldr_map

lemma allSome_eq_some {ι β : Type*} {f : ι → Option β} {g : ι → β} (h : allSome f = some g)
    (i : ι) : f i = some (g i) := by
  unfold allSome at h
  split_ifs at h with hall
  · rw [Option.some.injEq] at h
    subst h
    exact (Option.some_get _).symm

open scoped Classical in
lemma eval_roundQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : settings.RoundDraws α) :
    (roundQ settings D Dsf h d).eval (fun w => O.mq w ω)
      = (loopRound settings O D Dsf h (ω, d)).map SpecRound.data
      ∧ ∀ o ∈ loopRound settings O D Dsf h (ω, d), o.ReadsAt O ω := by
  sorry -- closes by: eval_bind, eval_table, eval_subst, eval_familyReadQ (main's proof)

/-- The program's first `r` rounds are the functions' at the noise `ω`, and each reads its family
there. -/
theorem eval_loopRoundsQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (rd : Fin K → settings.RoundDraws α) :
    ∀ r, (loopRoundsQ settings D Dsf rd r).eval (fun w => O.mq w ω)
        = (loopRounds settings O D Dsf (persistDraws (ω, rd)) r).map (List.map SpecRound.data)
      ∧ ∀ rs ∈ loopRounds settings O D Dsf (persistDraws (ω, rd)) r, ∀ o ∈ rs, o.ReadsAt O ω
  | 0 => ⟨rfl, by simp [loopRounds]⟩
  | r + 1 => by
    obtain ⟨ih1, ih2⟩ := eval_loopRoundsQ settings D Dsf rd r
    rw [loopRoundsQ, loopRounds, eval_bind, ih1]
    rcases hrs : loopRounds settings O D Dsf (persistDraws (ω, rd)) r with _ | rs
    · simp
    have hR := ih2 rs hrs
    simp only [Option.map_some, Option.bind_some]
    have hcond : (∀ o ∈ rs.map SpecRound.data, o.result.ending.IsHarvest)
        ↔ ∀ o ∈ rs, o.result.ending.IsHarvest := by
      simp [SpecRound.data]
    by_cases hc : r < K ∧ ∀ o ∈ rs, o.result.ending.IsHarvest
    · rw [dif_pos ⟨hc.1, hcond.2 hc.2⟩, dif_pos hc, eval_map]
      obtain ⟨e1, e2⟩ := eval_roundQ O ω settings D Dsf (histOf (rs.map SpecRound.data))
        (rd ⟨r, hc.1⟩)
      rw [← specHistOf_eq rs] at e1 e2
      have hp : persistDraws (ω, rd) ⟨r, hc.1⟩ = (ω, rd ⟨r, hc.1⟩) := rfl
      rw [hp, e1]
      refine ⟨?_, fun rs' hrs' o ho => ?_⟩
      · rcases loopRound settings O D Dsf (specHistOf rs) (ω, rd ⟨r, hc.1⟩) with _ | o <;> simp
      · obtain ⟨o', ho', rfl⟩ := Option.mem_map.1 hrs'
        rcases List.mem_append.1 ho with ho | ho
        · exact hR o ho
        · rw [List.mem_singleton] at ho
          subst ho
          exact e2 _ ho'
    · rw [dif_neg (fun h => hc ⟨h.1, hcond.1 h.2⟩), dif_neg hc]
      refine ⟨rfl, fun rs' hrs' => ?_⟩
      rw [Option.mem_some_iff] at hrs'
      subst hrs'
      exact hR

/-- The loop's rounds at `p` are the functions' at `p.1` in every round. -/
theorem indecisiveLStarLoop_eq (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → settings.RoundDraws α)) :
    indecisiveLStarLoop settings O D Dsf p
      = (loopRounds settings O D Dsf (persistDraws p) K).map (List.map SpecRound.data) :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).1

lemma indecisiveLStarLoop_some {settings : LoopSettings K} {D Dsf : Measure (FreeMonoid α)}
    {p : Ω × (Fin K → settings.RoundDraws α)}
    {rounds : List (LoopRound α)} (h : indecisiveLStarLoop settings O D Dsf p = some rounds) :
    ∃ rs, loopRounds settings O D Dsf (persistDraws p) K = some rs
      ∧ rounds = rs.map SpecRound.data := by
  rw [indecisiveLStarLoop_eq] at h
  obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨rs, hrs, rfl⟩

theorem loopRounds_read (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → settings.RoundDraws α)) :
    ∀ rs ∈ loopRounds settings O D Dsf (persistDraws p) K, ∀ o ∈ rs, o.read = o.data.read O p.1 :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).2

end Bridge

/-! ## What the program asks -/

section Reads

/-- The bound on a round's reads, each lazy round asking at most `Nr` strings. -/
def roundCallBound (a b f Nr : ℕ) : ℕ := a * (b + 1) + f * Nr

/-- The family has at most `k - 1` members. -/
lemma card_clusterAt_le (mq : FreeMonoid α → Ω → ℝ) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) :
    (clusterAt mq pops x B).card ≤ B.k - 1 := by
  unfold clusterAt nearestRows
  exact (Finset.card_le_card (Finset.erase_insert_subset _ _)).trans
    (leastLossSubset_card_le _ _ _)

variable (o : FreeMonoid α → ℝ)

open scoped Classical in
/-- A round that returns, against the answers `o`, a lazy round whose epochs' trees have at most
`n + 2` leaves reads at most `roundCallBound` strings. -/
theorem card_reads_roundQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (h : LoopHist α)
    (d : settings.RoundDraws α) {L a b f : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n Nr : ℕ}
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize L n
      (Fintype.card α) ≤ Nr)
    (hpk : ∀ ro, (roundQ settings D Dsf h d).eval o = some ro → ro.result.peak ≤ n + 2) :
    ((roundQ settings D Dsf h d).reads o).card ≤ roundCallBound a b f Nr := by
  sorry -- closes by: reads_bind, reads_table, card_reads_subst_le, card_reads_lazyRoundQ_le

section Small

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α settings)

/-- Every round the loop runs has a family of at most `f` members. -/
lemma loopRounds_small {f : ℕ} (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1)
    (r : ℕ) : ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs, o.family.card ≤ f := by
  induction r with
  | zero =>
    intro rounds hr
    rw [loopRounds, Option.mem_def, Option.some.injEq] at hr
    subst hr
    simp
  | succ r ih =>
    intro rounds hr
    rw [loopRounds] at hr
    obtain ⟨rs, hrs, hr⟩ := Option.mem_bind_iff.1 hr
    have hrr := ih rs hrs
    split_ifs at hr with hc
    · obtain ⟨o, ho, rfl⟩ := Option.mem_map.1 hr
      intro o' ho'
      rcases List.mem_append.1 ho' with h1 | h1
      · exact hrr o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        unfold loopRound at ho
        obtain ⟨tabs, -, hdr⟩ := Option.mem_bind_iff.1 ho
        dsimp only at hdr
        split_ifs at hdr with hcond
        · rw [← Option.mem_some_iff.1 hdr]
          obtain ⟨B, hB, hr⟩ := hcond.2
          have hBs := (Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
            (loopPopulations K (specHistOf rs).pops.length) ∧ streams (fun _ => ∅) none
              (drawsOf (draws ⟨r, hc.1⟩).1 D Dsf settings.scanCap (specHistOf rs).prefs
                (specHistOf rs).suffs (draws ⟨r, hc.1⟩).2.1 tabs) ∈ ret
              O.mq (loopPopulations K (specHistOf rs).pops.length) none settings.η₀
              settings.indecisionLimit settings.εcov settings.gateLevel B) ⟨B, hB, hr⟩).1
          exact (card_clusterAt_le _ _ _ _).trans (Nat.sub_le_of_le_add (hk _ _ hBs))
        · cases hdr
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact hrr

end Small

section Prefix

variable (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
  (rd : Fin K → settings.RoundDraws α)
  (o : FreeMonoid α → ℝ)

lemma loopRoundsQ_prefix_succ (r : ℕ) {rs' : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd (r + 1)).eval o = some rs') :
    ∃ rs, (loopRoundsQ settings D Dsf rd r).eval o = some rs ∧ rs <+: rs' := by
  rw [loopRoundsQ, eval_bind] at h
  rcases hrs : (loopRoundsQ settings D Dsf rd r).eval o with _ | rs
  · rw [hrs] at h; simp at h
  rw [hrs] at h
  refine ⟨rs, rfl, ?_⟩
  simp only at h
  split_ifs at h with hc
  · rw [eval_map] at h
    obtain ⟨ro, -, rfl⟩ := Option.map_eq_some_iff.1 h
    exact List.prefix_append _ _
  · simp at h
    subst h
    exact List.prefix_refl _

/-- A run's earlier rounds are a prefix of its later ones. -/
lemma loopRoundsQ_prefix {r s : ℕ} (hrs : r ≤ s) {rs' : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd s).eval o = some rs') :
    ∃ rs, (loopRoundsQ settings D Dsf rd r).eval o = some rs ∧ rs <+: rs' := by
  induction s generalizing rs' with
  | zero =>
    obtain rfl : r = 0 := by omega
    exact ⟨rs', h, List.prefix_refl _⟩
  | succ s ih =>
    rcases Nat.lt_or_ge r (s + 1) with hlt | hge
    · obtain ⟨rs, h1, h2⟩ := loopRoundsQ_prefix_succ settings D Dsf rd o s h
      obtain ⟨rs0, h3, h4⟩ := ih (by omega) h1
      exact ⟨rs0, h3, h4.trans h2⟩
    · obtain rfl : r = s + 1 := by omega
      exact ⟨rs', h, List.prefix_refl _⟩

/-- The first `r ≤ K` rounds, all harvesting, are `r` rounds. -/
lemma loopRoundsQ_length {r : ℕ} (hr : r ≤ K) {rs : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd r).eval o = some rs)
    (hall : ∀ ro ∈ rs, ro.result.ending.IsHarvest) : rs.length = r := by
  induction r generalizing rs with
  | zero =>
    simp [loopRoundsQ] at h
    subst h
    rfl
  | succ r ih =>
    rw [loopRoundsQ, eval_bind] at h
    rcases hrs : (loopRoundsQ settings D Dsf rd r).eval o with _ | rs0
    · rw [hrs] at h; simp at h
    rw [hrs] at h
    simp only at h
    split_ifs at h with hc
    · rw [eval_map] at h
      obtain ⟨ro, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.length_append, List.length_singleton,
        ih (by omega) hrs fun x hx => hall x (List.mem_append_left _ hx)]
    · simp at h
      subst h
      exact absurd ⟨by omega, hall⟩ hc

end Prefix

/-- The program's first `r ≤ K` rounds, against the answers at the noise `ω`, where the loop
returns `rounds` whose epochs' trees have at most `n + 2` leaves, read at most `r` times
`roundCallBound` strings. -/
theorem card_reads_loopRoundsQ_le (O : Oracle μ (FreeMonoid α)) (ω : Ω)
    (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : Fin K → settings.RoundDraws α)
    {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n Nr : ℕ}
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize L n
      (Fintype.card α) ≤ Nr)
    {rounds : List (LoopRound α)}
    (hout : (loopQ settings D Dsf rd).eval (fun w => O.mq w ω) = some rounds)
    (hpk : ∀ o ∈ rounds, o.result.peak ≤ n + 2) {r : ℕ} (hr : r ≤ K) :
    ((loopRoundsQ settings D Dsf rd r).reads fun w => O.mq w ω).card
      ≤ r * roundCallBound a b f Nr := by
  sorry -- closes by: main's induction with loopRoundsQ_prefix, card_reads_roundQ_le

end Reads

/-! ## The learner -/

section Learner

variable (O : Oracle μ (FreeMonoid α))

/-- The learner, its rounds' epochs' trees at most `q + 2` leaves and the last round's trie at most
`nX` strings, asks the loop's strings and `f` strings for each of its table's questions. -/
theorem card_reads_learnerQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D]
    (p : Ω × (Fin K → settings.RoundDraws α))
    {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {q Nr nX : ℕ}
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize L q
      (Fintype.card α) ≤ Nr)
    {rounds : List (LoopRound α)}
    (hout : indecisiveLStarLoop settings O D Dsf p = some rounds)
    (hpk : ∀ o ∈ rounds, o.result.peak ≤ q + 2)
    (hX : ∀ o ∈ rounds.getLast?, o.result.expanded.card ≤ nX) :
    ((learnerQ settings D Dsf p.2).reads fun w => O.mq w p.1).card
      ≤ K * roundCallBound a b f Nr + f * ((Fintype.card α * nX + 1) * (q + 2)) := by
  sorry -- closes by: card_reads_loopRoundsQ_le, reads_piList, card_trieFrontier_le, siftQ reads

end Learner

end OrthoDFA

end
