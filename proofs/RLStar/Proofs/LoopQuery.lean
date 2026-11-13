import RLStar.Proofs.QueryLevel3
import RLStar.Proofs.ClusterCongr
import RLStar.Proofs.LoopGood
import RLStar.Proofs.LazyZone
import RLStar.Proofs.LazyTable

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

omit [DecidableEq α] in
lemma specHistOf_eq (rs : List (SpecRound α)) :
    specHistOf rs = histOf (rs.map SpecRound.data) := by
  simp only [specHistOf, histOf, List.filterMap_map, List.foldr_map]
  rfl

open scoped Classical in
lemma eval_roundQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : settings.RoundDraws α) :
    (roundQ settings D Dsf h d).eval (fun w => O.mq w ω)
      = (loopRound settings O D Dsf h (ω, d)).map SpecRound.data
      ∧ ∀ o ∈ loopRound settings O D Dsf h (ω, d), o.ReadsAt O ω := by
  refine ⟨?_, fun o ho => ?_⟩
  swap
  · unfold loopRound at ho
    dsimp only at ho
    split_ifs at ho
    · rw [Option.mem_some_iff] at ho
      subst ho
      rfl
    · cases ho
  unfold roundQ loopRound
  dsimp only
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set tabs := popTables D settings.scanCap h d.1
  set dr := drawsOf ω D Dsf settings.scanCap h.prefs h.suffs d.1 tabs with hdrd
  have hflat : drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs
      = (((), dr.1.2), dr.2) := rfl
  rw [hflat]
  set xs := streams (fun _ => ∅) none dr with hxs
  set xc := streams (fun _ => ∅) none ((((), dr.1.2), dr.2) : Draws Unit _ _) with hxc
  have hx : xc = ((), xs.2) := rfl
  have hxs1 : xs.1 = ω := rfl
  have hscan : ScanWithin sched ps (fun _ => settings.scanCap)
      ((((), dr.1.2), dr.2) : Draws Unit _ _)
      ↔ ScanWithin sched ps (fun _ => settings.scanCap) dr :=
    Iff.rfl
  by_cases hsc : ScanWithin sched ps (fun _ => settings.scanCap) dr
  swap
  · simp only [hscan, hsc, false_and, ↓reduceIte]
    simp
  simp only [hscan, hsc, true_and, ↓reduceIte, eval_bind]
  set t := (Query.table (clusterReads sched ps xc).toList).eval (fun w => O.mq w ω)
  have ht : ∀ w ∈ clusterReads sched ps xc, t w = O.mq w ω := fun w hw =>
    eval_table_of_mem _ (Finset.mem_toList.2 hw)
  have hret : ∀ B ∈ sched, (xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B
      ↔ xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov
        settings.gateLevel B) := by
    intro B hB
    rw [mem_ret_flat O.mq _ _ _ _ _ _ B xs, hxs1, hx]
    refine (ret_eqOn _ _ sched ps none (Finset.mem_insert_self _ _) _ _ _ _ hB _
      fun w hw => ?_).symm
    rw [ht w (by rwa [hx])]
  have hexc : (∃ B ∈ sched, xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B)
      ↔ ∃ B ∈ sched, xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B :=
    exists_congr fun B => ⟨fun h => ⟨h.1, (hret B h.1).1 h.2⟩, fun h => ⟨h.1, (hret B h.1).2 h.2⟩⟩
  have hpred : (fun B => B ∈ sched ∧ xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B)
      = fun B => B ∈ sched ∧ xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B :=
    funext fun B => propext ⟨fun h => ⟨h.1, (hret B h.1).1 h.2⟩,
      fun h => ⟨h.1, (hret B h.1).2 h.2⟩⟩
  by_cases hex : ∃ B ∈ sched, xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
      settings.εcov settings.gateLevel B
  swap
  · simp only [hexc, hex, ↓reduceIte]
    simp
  simp only [hexc, hex, ↓reduceIte]
  rw [hpred]
  set B := Classical.epsilon fun B => B ∈ sched ∧ xs ∈ ret O.mq ps none settings.η₀
    settings.indecisionLimit settings.εcov settings.gateLevel B with hBdef
  have hBs : B ∈ sched := (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ xs ∈ ret O.mq ps
    none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B) hex).1
  have hF : clusterAt (fun w _ => t w) ps xc B = clusterAt O.mq ps xs B := by
    rw [clusterAt_flat O.mq ps B xs, hxs1, ← hx]
    refine clusterAt_eqOn_sched _ _ sched ps hBs xc fun w hw => ?_
    rw [ht w hw]
  rw [hF, eval_map, eval_subst]
  rfl

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
      rw [← specHistOf_eq] at e1 e2 ⊢
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

private lemma epsilon_mem {S : Finset State} {q : State → Prop} (h : ∃ B ∈ S, q B) :
    Classical.epsilon (fun B => B ∈ S ∧ q B) ∈ S :=
  (Classical.epsilon_spec h).1

omit [DecidableEq α] in
private lemma ae_popLaw {ν : Measure (FreeMonoid α)} {P : FreeMonoid α → Prop}
    (hν : ∀ᵐ x ∂ν, P x) (h1 : P 1) : ∀ᵐ x ∂(popLaw ν), P x := by
  unfold popLaw
  split_ifs
  · exact (ae_dirac_iff (MeasurableSet.of_discrete)).2 h1
  · rw [ae_iff, Measure.smul_apply, ae_iff.1 hν, smul_zero]

private lemma drawFrom_length_le (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {L : ℕ}
    (hν : ∀ᵐ x ∂ν, x.toList.length ≤ L) (u : unitInterval) :
    (drawFrom ν u).toList.length ≤ L := by
  rcases drawFrom_pos_or_one ν u with h | h
  · by_contra hlt
    have : ν {drawFrom ν u} = 0 := measure_mono_null (fun z hz => by
      rw [Set.mem_singleton_iff.1 hz]; exact hlt) (ae_iff.1 hν)
    rw [measureReal_def, this, ENNReal.toReal_zero] at h
    exact lt_irrefl _ h
  · rw [h]; simp

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
  unfold roundQ at hpk
  unfold roundQ roundCallBound
  dsimp only at hpk ⊢
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set dr := drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1
    (popTables D settings.scanCap h d.1)
  set x := streams (fun _ => ∅) none dr
  obtain ⟨hpa, hsb⟩ := hab ps x
  split_ifs at hpk ⊢ with hsc
  swap
  · simp
  rw [eval_bind] at hpk
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add ?_ ?_)
  · rw [reads_table]
    refine (List.toFinset_card_le _).trans ?_
    rw [Finset.length_toList]
    refine Finset.card_image_le.trans ?_
    rw [Finset.card_product]
    exact Nat.mul_le_mul hpa ((Finset.card_insert_le _ _).trans (Nat.add_le_add_right hsb 1))
  split_ifs at hpk ⊢ with hex
  swap
  · simp
  simp only [eval_map, eval_subst, Option.some.injEq, forall_eq'] at hpk
  set tb := (Query.table (clusterReads sched ps x).toList).eval o
  have hF := (card_clusterAt_le (fun w _ => tb w) ps x _).trans
    (Nat.sub_le_of_le_add (hk ps _ (epsilon_mem hex)))
  rw [reads_map]
  refine (card_reads_subst_le _ _ _ (n := f) fun q _ =>
    (card_reads_familyReadQ_le _ _ _ q _).trans hF).trans ?_
  rw [Nat.mul_comm _ f]
  refine Nat.mul_le_mul_left _ ((card_reads_lazyRoundQ_le _ _ _ _ _ _ rfl n L
    (fun e => ⟨fun i => ?_, fun j m => ?_⟩) hpk).trans hNr) <;>
  exact drawFrom_length_le _ (ae_popLaw (ae_restrict_of_ae hD) (by simp)) _

section Small

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α settings)

private lemma loopRounds_forall (P : SpecRound α → Prop)
    (hP : ∀ h d, ∀ o ∈ loopRound settings O D Dsf h d, P o) (r : ℕ) :
    ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs, P o := by
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
    split_ifs at hr with hc
    · obtain ⟨o, ho, rfl⟩ := Option.mem_map.1 hr
      intro o' ho'
      rcases List.mem_append.1 ho' with h1 | h1
      · exact ih rs hrs o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        exact hP _ _ _ ho
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact ih rs hrs

/-- Every round the loop runs has a family of at most `f` members. -/
lemma loopRounds_small {f : ℕ} (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1)
    (r : ℕ) : ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs, o.family.card ≤ f := by
  refine loopRounds_forall settings O D Dsf draws _ (fun h d o ho => ?_) r
  unfold loopRound at ho
  dsimp only at ho
  split_ifs at ho with hcond
  · rw [Option.mem_some_iff] at ho
    subst ho
    exact (card_clusterAt_le _ _ _ _).trans
      (Nat.sub_le_of_le_add (hk _ _ (epsilon_mem hcond.2)))
  · cases ho

omit [Fintype α] [DecidableEq α] in
private lemma depth_lt_length_leaves {β : Type*} : ∀ T : DTree β, T.depth < T.leaves.length
  | .leaf => by simp [DTree.depth, DTree.leaves]
  | .node _ r a => by
    have hr := depth_lt_length_leaves r
    have ha := depth_lt_length_leaves a
    simp only [DTree.depth, DTree.leaves, List.length_append, List.length_map]
    rcases le_total r.depth a.depth with h | h
    · rw [max_eq_right h]; omega
    · rw [max_eq_left h]; omega

private lemma lazyEpochsQ_depth_le (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (k nh : ℕ) {J b : ℕ} : ∀ (e pk : ℕ) (ds : List (Random.EpochDraws α C.probeBudget J b))
      (X : Finset (FreeMonoid α)) (ls : Random.LazyLists α),
      ((Random.lazyEpochsQ C k nh e pk ds X ls).eval read).hypothesis.tree.depth
        ≤ ((Random.lazyEpochsQ C k nh e pk ds X ls).eval read).peak
  | _, _, [], X, ls => by simp [Random.lazyEpochsQ, DTree.depth]
  | e, pk, d :: ds, X, ls => by
    simp only [Random.lazyEpochsQ, eval_bind]
    generalize (Random.epochQ (Random.epochSettings C e) k X d.1).eval read = res
    have hres : res.hypothesis.tree.depth ≤ max pk res.hypothesis.tree.leaves.length :=
      (depth_lt_length_leaves _).le.trans (le_max_right _ _)
    rcases res with ⟨hyp, _ | s | _⟩
    · exact hres
    · simp only [eval_bind]
      generalize (Random.topUpQ k nh hyp s (List.ofFn d.2) X (ls s)).eval read = t
      rcases t with _ | ⟨_ | _, rest⟩ <;> dsimp only [eval_pure]
      · exact hres
      · exact hres
      · exact lazyEpochsQ_depth_le read C k nh _ _ ds _ _
    · exact hres

private lemma loopRounds_depth (r : ℕ) : ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs,
    o.result.hypothesis.tree.depth ≤ o.result.peak := by
  refine loopRounds_forall settings O D Dsf draws _ (fun h d o ho => ?_) r
  unfold loopRound at ho
  dsimp only at ho
  split_ifs at ho
  · rw [Option.mem_some_iff] at ho
    subst ho
    exact lazyEpochsQ_depth_le _ _ _ _ _ _ _ _ _
  · cases ho

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
  have hout' : (loopRoundsQ settings D Dsf rd K).eval (fun w => O.mq w ω) = some rounds := hout
  induction r with
  | zero => simp [loopRoundsQ]
  | succ r ih =>
    rw [loopRoundsQ, reads_bind, Nat.succ_mul]
    refine (Finset.card_union_le _ _).trans (Nat.add_le_add (ih (by omega)) ?_)
    obtain ⟨rs', hrs', hpre⟩ := loopRoundsQ_prefix settings D Dsf rd _ hr hout'
    rw [loopRoundsQ, eval_bind] at hrs'
    rcases hrs : (loopRoundsQ settings D Dsf rd r).eval (fun w => O.mq w ω) with _ | rs
    · simp
    rw [hrs] at hrs'
    simp only at hrs' ⊢
    split_ifs with hc
    swap
    · simp
    rw [dif_pos hc, eval_map] at hrs'
    obtain ⟨ro, hoo, rfl⟩ := Option.map_eq_some_iff.1 hrs'
    rw [reads_map]
    refine card_reads_roundQ_le _ settings D Dsf _ _ hD (fun pops x => hab pops (ω, x.2)) hk hNr
      fun x hx => ?_
    rw [hoo, Option.some.injEq] at hx
    subst hx
    exact hpk _ (hpre.subset (List.mem_append_right _ (List.mem_singleton_self _)))

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
  have hloop := card_reads_loopRoundsQ_le O p.1 settings D Dsf p.2 hD hab hk hNr hout hpk le_rfl
  obtain ⟨rs, hrs, rfl⟩ := indecisiveLStarLoop_some O hout
  unfold learnerQ
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add hloop ?_)
  rw [show (loopQ settings D Dsf p.2).eval (fun w => O.mq w p.1) = some (rs.map SpecRound.data)
    from hout]
  dsimp only
  rcases hl : (rs.map SpecRound.data).getLast? with _ | o
  · simp
  dsimp only
  rw [reads_map]
  have ho := List.mem_of_getLast? hl
  obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
  have hT : o'.result.hypothesis.tree.depth ≤ q + 2 :=
    (loopRounds_depth settings O D Dsf _ K rs hrs o' ho').trans (hpk _ ho)
  refine (card_reads_subst_le _ _ _ (n := f) fun q _ => (card_reads_familyReadQ_le _ _ _ q _).trans
    (loopRounds_small settings O D Dsf _ hk K rs hrs o' ho')).trans ?_
  rw [Nat.mul_comm _ f]
  refine Nat.mul_le_mul_left _ ?_
  have ht : ((Random.lazyTableQ settings.level3.walkStart o'.result.hypothesis.tree
      o'.result.expanded).reads fun q => (familyReadQ o'.family o'.state.lo (o'.state.hi + 1)
        q).eval fun w => O.mq w p.1).card
      ≤ (Fintype.card α * o'.result.expanded.card + 1) * o'.result.hypothesis.tree.depth := by
    exact_mod_cast lazyTableReads_holds _ _ _ _
  exact ht.trans (Nat.mul_le_mul (Nat.add_le_add_right (Nat.mul_le_mul_left _ (hX _ hl)) 1) hT)

end Learner

end OrthoDFA

end
