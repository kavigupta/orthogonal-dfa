import OrthoDFA.Proofs.ReadingCongr

/-!
# A step is decided by what it logs

A step of the pass reads only what it adds to the read log, and, at a split test whose key is not
forced, the held-out strings it records as counted. So two read functions agreeing there give the
step alike.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Step

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness lies in the population. -/
def WitPool (s : KState α) : Prop := ∀ p c q y, s.edges p c = some (q, y) → y ∈ s.pool

theorem probeStepK_log (k : ℕ) (s : KState α) (x : FreeMonoid α) :
    (probeStepK K R k s x).log = s.log ∪ stepReads K R.F s.tree (probeStepK K R k s x).tree
      (s.pool ++ (probeStepK K R k s x).pool) k x := by
  unfold probeStepK
  simp only []
  repeat' split
  all_goals simp [closeK]

theorem tested_sub_record (T : Tested α) (κ : TestKey α) (bs : List (FreeMonoid α)) :
    ∀ e ∈ T, e ∈ T.record κ bs := fun e he => List.mem_append_left _ he

theorem probeStepK_tested_mono (k : ℕ) (s : KState α) (x : FreeMonoid α) :
    ∀ e ∈ s.tested, e ∈ (probeStepK K R k s x).tested := by
  intro e he
  unfold probeStepK
  simp only []
  repeat' split
  all_goals first
    | exact he
    | (simp only [closeK, testedAfter]; exact tested_sub_record _ _ _ e he)

theorem testStrings_not_skip {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {d : FreeMonoid α} {skip : FreeMonoid α → Prop} {p : FreeMonoid α × Bool}
    (hp : p ∈ testStrings K R t pool path d skip) : ¬ skip p.1 := by
  classical
  unfold testStrings at hp
  have key : ∀ (l acc : List (FreeMonoid α × Bool)), (∀ q ∈ acc, ¬ skip q.1) →
      ∀ q ∈ l.foldl (fun acc p => if p.1 ∈ acc.map Prod.fst ∨ skip p.1 then acc
        else acc ++ [p]) acc, ¬ skip q.1 := by
    intro l
    induction l with
    | nil => exact fun acc h => h
    | cons p l ih =>
      intro acc h
      simp only [List.foldl_cons]
      refine ih _ fun q hq => ?_
      split_ifs at hq with hc
      · exact h q hq
      · rcases List.mem_append.1 hq with hq | hq
        · exact h q hq
        · rw [List.mem_singleton.1 hq]; exact fun hs => hc (.inr hs)
  exact key _ [] (by simp) p hp

/-- The strings a step's split test counts at an unforced key are recorded as that key's. -/
theorem probeStepK_counted (k : ℕ) (s : KState α) (x : FreeMonoid α) {ps : List (List Bool)}
    {fd : ℕ} {κ : TestKey α} (ho : probeOutcome R s.tree s.edges k x = .edge ps fd)
    (hk : (seedStep K R s.tree s.pool s.edges (stepSkip K R k s x) K.forced k x ps fd).key
      = some κ) :
    ∀ p ∈ testStrings K R s.tree s.pool κ.1 κ.2 (stepSkip K R k s x κ),
      (p.1, κ) ∈ (probeStepK K R k s x).tested := by
  classical
  intro p hp
  have hns := testStrings_not_skip K R hp
  have hown : p.1 ∈ s.tested.map Prod.fst → (p.1, κ) ∈ s.tested := by
    intro hm
    by_contra hn
    exact hns ⟨.inr (.inr hm), hn⟩
  have hcount : ∀ T : Tested α, T = s.tested →
      (p.1, κ) ∈ T.record κ ((testStrings K R s.tree s.pool κ.1 κ.2
        (stepSkip K R k s x κ)).map Prod.fst) := by
    rintro T rfl
    by_cases hm : p.1 ∈ s.tested.map Prod.fst
    · exact List.mem_append_left _ (hown hm)
    · refine List.mem_append_right _ (List.mem_map.2 ⟨p.1, List.mem_filter.2
        ⟨List.mem_map.2 ⟨p, hp, rfl⟩, by simpa using hm⟩, rfl⟩)
  obtain ⟨κ1, κ2⟩ := κ
  unfold probeStepK
  simp only [ho]
  revert hk
  rcases hs : seedStep K R s.tree s.pool s.edges (stepSkip K R k s x) K.forced k x ps fd with
    ⟨d, s1, y, sp⟩ | ⟨s1, sp, d⟩ | b | _ <;>
    intro hk <;> simp only [SeedResult.key, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at hk
  · obtain ⟨rfl, rfl⟩ := hk
    simp only [closeK, testedAfter, counted]
    exact hcount _ rfl
  · obtain ⟨rfl, rfl⟩ := hk
    simp only [closeK, testedAfter, counted]
    exact hcount _ rfl

theorem probeStepK_witPool (k : ℕ) (s : KState α) (x : FreeMonoid α) (h : WitPool s) :
    WitPool (probeStepK K R k s x) := by
  have key : ∀ (t : DTree α) (pool' : List (FreeMonoid α)) (e : Edges α) (st : ℕ) (T : Tested α)
      (lg : Finset (FreeMonoid α)), (∀ b ∈ s.pool, b ∈ pool') →
      (∀ p c q y, e p c = some (q, y) → s.edges p c = some (q, y)) →
      WitPool (closeK K R t pool' e st T lg) := by
    intro t pool' e st T lg hp he p c q y hy
    exact (closeK_poolIn K R (Bs := {b | b ∈ pool'}) (fun b hb => hb)
      (fun p c q y hq => hp _ (h p c q y (he p c q y hq)))).2 p c q y hy
  unfold probeStepK
  simp only []
  split
  · split
    · rename_i d s1 y sp _
      refine key _ _ _ _ _ _ (fun b hb => List.mem_append_left _ hb) fun p c q w hq => ?_
      revert hq
      rcases hE : s.edges p c with _ | ⟨q', w'⟩ <;> simp only []
      · simp
      · split_ifs <;> simp
    · rename_i s1 sp d _
      exact key _ _ _ _ _ _ (fun b hb => by
        by_cases hbs : b = sp
        · exact List.mem_cons.2 (.inl hbs)
        · exact List.mem_cons_of_mem _ (List.mem_filter.2 ⟨hb, by simpa using hbs⟩))
        fun _ _ _ _ hq => hq
    · exact key _ _ _ _ _ _ (fun b hb => hb) fun _ _ _ _ hq => hq
  · exact key _ _ _ _ _ _ (fun b hb => by split_ifs <;> simp [hb]) fun _ _ _ _ hq => hq
  · exact key _ _ _ _ _ _ (fun b hb => hb) fun _ _ _ _ hq => hq

end Step

section Mask

variable {K : StageKnobs α} {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem seedStep_key_congr {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop} {forced : Set (TestKey α)} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (hwit : ∀ p c q y, edges p c = some (q, y) → AgreeOne K F f₁ f₂ t y)
    (hw : AgreeOne K F f₁ f₂ t (prefixOf x (fd - 1))) :
    (seedStep K (rd B F f₁) t pool edges skip forced k x ps fd).key
      = (seedStep K (rd B F f₂) t pool edges skip forced k x ps fd).key := by
  unfold seedStep
  simp only []
  split
  · rfl
  rename_i c hc
  split
  · rfl
  rename_i s2 y he
  split_ifs with h1
  · rfl
  have hy := hwit _ _ _ _ he
  rw [sift_congr (B := B) hw.tree, sift_congr (B := B) hy.tree]
  split
  · rfl
  rename_i p hsp
  split_ifs with h2
  · rfl
  rw [parting_congr t fun m hm => ⟨cut_congr ((hy.letter' c) m hm),
    cut_congr ((hw.letter' c) m hm)⟩]
  split
  · rfl
  · rfl
  split_ifs
  · rfl
  · split <;> split <;> rfl

theorem ext_elim (e : Option α) : ext e = e.elim 1 FreeMonoid.of := by
  rcases e with _ | c <;> rfl

theorem mem_stepReads {F : Finset (FreeMonoid α)} {t t' : DTree α} {pool : List (FreeMonoid α)}
    {k : ℕ} {x b m v : FreeMonoid α} (e : Option α)
    (hb : b ∈ pool ∨ ∃ i, k ≤ i ∧ b = prefixOf x i) (hm : m ∈ t.mids ∨ m ∈ t'.mids)
    (hv : v ∈ F ∪ K.train F) :
    b * ext e * m * v ∈ stepReads K F t t' pool k x := by
  classical
  unfold stepReads
  refine Finset.mem_image.2 ⟨(((b, ext e), m), v), ?_, by simp [ext_elim]⟩
  simp only [Finset.mem_product, List.mem_toFinset, List.mem_append, List.mem_map,
    List.mem_range, Finset.mem_union, Finset.mem_insert, Finset.mem_image, Finset.mem_univ,
    true_and]
  refine ⟨⟨⟨?_, ?_⟩, ?_⟩, Finset.mem_union.1 hv⟩
  · rcases hb with hb | ⟨i, hi, rfl⟩
    · exact .inl hb
    · exact .inr ⟨min i x.toList.length, by omega, (prefixOf_max_min hi).symm⟩
  · rcases e with _ | c
    · exact .inl rfl
    · exact .inr ⟨c, rfl⟩
  · rcases hm with hm | hm
    · exact .inl ((DTree.mem_midfixes_iff _).2 hm)
    · exact .inr ((DTree.mem_midfixes_iff _).2 hm)

/-- A step whose logged reads and recorded test strings two read functions agree on runs alike. -/
theorem strongStep_mask (C : StrongCfg α) {A : RoundAcc α} {x : FreeMonoid α}
    (hwp : WitPool A.s)
    (h : ∀ z ∈ (strongStep C (rd B F f₁) A x).s.log, f₁ z = f₂ z)
    (ht : ∀ b κ, (b, κ) ∈ (strongStep C (rd B F f₁) A x).s.tested → κ ∉ C.K.forced →
      f₁ b = f₂ b) :
    strongStep C (rd B F f₁) A x = strongStep C (rd B F f₂) A x := by
  set s₁ := probeStepK C.K (rd B F f₁) C.k A.s x with hs₁
  have hst := (strongStep_state C (rd B F f₁) A x).1
  rw [hst] at h ht
  set Tf := s₁.tree
  have hlog := probeStepK_log C.K (rd B F f₁) C.k A.s x
  have hT : ∀ m ∈ A.s.tree.mids, m ∈ Tf.mids := by
    have := strongStep_mids C (rd B F f₁) A x
    rwa [hst] at this
  have hag : ∀ b, (b ∈ A.s.pool ∨ ∃ i, C.k ≤ i ∧ b = prefixOf x i) → AgreeOne C.K F f₁ f₂ Tf b := by
    intro b hb e m hm v hv
    refine h _ ?_
    rw [← hs₁] at hlog
    rw [hlog]
    refine Finset.mem_union_right _ (mem_stepReads e ?_ (.inr hm) hv)
    rcases hb with hb | hb
    · exact .inl (List.mem_append_left _ hb)
    · exact .inr hb
  have hpool : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ Tf b := fun b hb => hag b (.inl hb)
  have hwit : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ Tf y :=
    fun p c q y he => hpool y (hwp p c q y he)
  have hw : ∀ i, C.k ≤ i → AgreeOne C.K F f₁ f₂ Tf (prefixOf x i) :=
    fun i hi => hag _ (.inr ⟨i, hi, rfl⟩)
  refine strongStep_congr C Tf hT (by rw [hst]; exact fun m hm => hm) hpool hwit hw ?_
  intro ps fd κ ho hk hf p hp
  have hpo : probeOutcome (rd B F f₁) A.s.tree A.s.edges C.k x
      = probeOutcome (rd B F f₂) A.s.tree A.s.edges C.k x :=
    probeOutcome_congr fun i hi m hm => (hw i hi).mono' hT |>.tree m hm
  have hfd : C.k ≤ fd - 1 := by have := probeOutcome_edge_gt _ ho; omega
  have hpool' : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ A.s.tree b := fun b hb => (hpool b hb).mono' hT
  have hwit' : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ A.s.tree y :=
    fun p c q y he => (hwit p c q y he).mono' hT
  have hsk : stepSkip C.K (rd B F f₁) C.k A.s x = stepSkip C.K (rd B F f₂) C.k A.s x := rfl
  have hk₁ : (seedStep C.K (rd B F f₁) A.s.tree A.s.pool A.s.edges (stepSkip C.K (rd B F f₁) C.k
      A.s x) C.K.forced C.k x ps fd).key = some κ := by
    rw [hsk, seedStep_key_congr hwit' ((hw _ hfd).mono' hT)]; exact hk
  obtain ⟨κ1, κ2⟩ := κ
  have hr : (∃ s1 y sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
      (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .split κ2 s1 y sprime)
      ∨ ∃ s1 sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
        (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .member s1 sprime κ2 := by
    revert hk
    rcases seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges (stepSkip C.K (rd B F f₂) C.k A.s x)
      C.K.forced C.k x ps fd with ⟨d, s1, y, sp⟩ | ⟨s1, sp, d⟩ | b | _ <;> intro hk <;>
      simp only [SeedResult.key, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at hk
    · exact .inl ⟨s1, y, sp, by rw [hk.2]⟩
    · exact .inr ⟨s1, sp, by rw [hk.2]⟩
  obtain ⟨c, m, hm, rfl⟩ := seedStep_dist C.K _ hr
  have hts : testStrings C.K (rd B F f₁) A.s.tree A.s.pool κ1 (FreeMonoid.of c * m)
        (stepSkip C.K (rd B F f₂) C.k A.s x (κ1, FreeMonoid.of c * m))
      = testStrings C.K (rd B F f₂) A.s.tree A.s.pool κ1 (FreeMonoid.of c * m)
        (stepSkip C.K (rd B F f₂) C.k A.s x (κ1, FreeMonoid.of c * m)) :=
    testStrings_congr (fun b hb => (hpool' b hb).tree) fun b hb => (hpool' b hb).letter' c m hm
  rw [← hts] at hp
  rw [← hsk] at hp
  have := probeStepK_counted C.K (rd B F f₁) C.k A.s x (hpo ▸ ho) hk₁ p hp
  exact ht _ _ this hf

end Mask

section Round

variable (C : StrongCfg α)

/-- What the round has read, but the held-out strings the tests at the keys it takes as not
splitting have counted. -/
def readLog (A : RoundAcc α) : Set (FreeMonoid α) :=
  {z | z ∈ A.s.log ∨ z ∈ A.reads ∨ ∃ κ, (z, κ) ∈ A.s.tested ∧ κ ∉ C.K.forced}

theorem strongStep_reads (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    (strongStep C R A x).reads = A.reads := by
  unfold strongStep
  simp only []
  repeat' split
  all_goals rfl

theorem strongStep_readLog (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    readLog C A ⊆ readLog C (strongStep C R A x) := by
  have hs := (strongStep_state C R A x).1
  rintro z (hz | hz | ⟨κ, hz, hf⟩)
  · left; rw [hs, probeStepK_log]; exact Finset.mem_union_left _ hz
  · right; left; rw [strongStep_reads]; exact hz
  · right; right
    refine ⟨κ, ?_, hf⟩
    rw [hs]; exact probeStepK_tested_mono _ _ _ _ _ _ hz

theorem passBody_readLog (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    readLog C A ⊆ readLog C (passBody C R A x) := by
  unfold passBody
  split_ifs
  · exact le_rfl
  · exact strongStep_readLog C R A x

theorem fold_readLog (R : CutReads α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α),
      readLog C A ⊆ readLog C (probes.foldl (passBody C R) A)
  | [], _ => le_rfl
  | x :: xs, A => (passBody_readLog C R A x).trans (fold_readLog R xs _)

theorem passBody_witPool (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) (h : WitPool A.s) :
    WitPool (passBody C R A x).s := by
  unfold passBody
  split_ifs
  · exact h
  · rw [(strongStep_state C R A x).1]; exact probeStepK_witPool C.K R C.k A.s x h

theorem fold_witPool (R : CutReads α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), WitPool A.s →
      WitPool (probes.foldl (passBody C R) A).s
  | [], _, h => h
  | x :: xs, A, h => fold_witPool R xs _ (passBody_witPool C R A x h)

theorem strongPass_witPool (R : CutReads α) (A : RoundAcc α) (probes : List (FreeMonoid α))
    (h : WitPool A.s) : WitPool (strongPass C R A probes).s := by
  rw [strongPass_eq]; exact fold_witPool C R probes _ h

theorem strongPass_readLog (R : CutReads α) (A : RoundAcc α) (probes : List (FreeMonoid α)) :
    readLog C A ⊆ readLog C (strongPass C R A probes) := by
  rw [strongPass_eq]
  refine le_trans ?_ (fold_readLog C R probes _)
  rintro z (hz | hz | hz)
  · left; exact Finset.mem_union_left _ hz
  · right; left; exact hz
  · right; right; exact hz

theorem strongReading_fst_acc (R : CutReads α) (j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (y : C.Draws) :
    readLog C (strongPass C R A (first ++ List.ofFn y.1))
        ⊆ readLog C (strongReading C R j A first y).1
      ∧ ↑(readingReads C R.F (strongPass C R A (first ++ List.ofFn y.1)).s.tree y)
        ⊆ readLog C (strongReading C R j A first y).1 := by
  rw [strongReading_acc]
  constructor
  · rintro z (hz | hz | hz)
    · left; exact hz
    · right; left; exact Finset.mem_union_left _ hz
    · right; right; exact hz
  · intro z hz
    right; left; exact Finset.mem_union_right _ hz

theorem strongReading_witPool (R : CutReads α) (j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (y : C.Draws) (h : WitPool A.s) :
    WitPool (strongReading C R j A first y).1.s := by
  rw [(strongReading_fst C R j A first y).1]
  exact strongPass_witPool C R A _ h

theorem strongRound_readLog (R : CutReads α) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      readLog C A ⊆ readLog C (strongRound C R n j A first d).2.1
  | 0, _, _, _, _ => le_rfl
  | n + 1, j, A, first, d => by
    have h1 := (strongPass_readLog C R A (first ++ List.ofFn (d 0).1)).trans
      (strongReading_fst_acc C R j A first (d 0)).1
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> rw [hR] at h1 <;>
      simp only [strongRound, hR]
    · exact h1
    · exact h1.trans (strongRound_readLog R n (j + 1) A'' lv (Fin.tail d))

variable {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem fold_mask :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), WitPool A.s →
      (∀ z ∈ readLog C (probes.foldl (passBody C (rd B F f₁)) A), f₁ z = f₂ z) →
      probes.foldl (passBody C (rd B F f₁)) A = probes.foldl (passBody C (rd B F f₂)) A
  | [], _, _, _ => by simp only [List.foldl_nil]
  | x :: xs, A, hw, h => by
    simp only [List.foldl_cons] at h ⊢
    have hstep : passBody C (rd B F f₁) A x = passBody C (rd B F f₂) A x := by
      by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
      · unfold passBody; rw [if_pos hg, if_pos hg]
      have e₁ : passBody C (rd B F f₁) A x = strongStep C (rd B F f₁) A x := by
        unfold passBody; rw [if_neg hg]
      have e₂ : passBody C (rd B F f₂) A x = strongStep C (rd B F f₂) A x := by
        unfold passBody; rw [if_neg hg]
      have hsub : readLog C (strongStep C (rd B F f₁) A x)
          ⊆ readLog C (xs.foldl (passBody C (rd B F f₁)) (passBody C (rd B F f₁) A x)) := by
        rw [e₁]; exact fold_readLog C (rd B F f₁) xs _
      rw [e₁, e₂]
      refine strongStep_mask C hw (fun z hz => h z (hsub (.inl hz))) fun b κ hb hf => ?_
      exact h b (hsub (.inr (.inr ⟨κ, hb, hf⟩)))
    rw [← hstep]
    exact fold_mask xs _ (passBody_witPool C _ A x hw) h

theorem strongPass_mask {A : RoundAcc α} {probes : List (FreeMonoid α)} (hw : WitPool A.s)
    (h : ∀ z ∈ readLog C (strongPass C (rd B F f₁) A probes), f₁ z = f₂ z) :
    strongPass C (rd B F f₁) A probes = strongPass C (rd B F f₂) A probes := by
  rw [strongPass_eq] at h ⊢
  rw [strongPass_eq]
  exact fold_mask C probes _ hw h

theorem strongReading_mask {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)} {y : C.Draws}
    (hw : WitPool A.s)
    (h : ∀ z ∈ readLog C (strongReading C (rd B F f₁) j A first y).1, f₁ z = f₂ z) :
    strongReading C (rd B F f₁) j A first y = strongReading C (rd B F f₂) j A first y := by
  obtain ⟨h1, h2⟩ := strongReading_fst_acc C (rd B F f₁) j A first y
  have hpass := strongPass_mask C hw fun z hz => h z (h1 hz)
  exact strongReading_congr C hpass fun z hz => h z (h2 hz)

theorem strongRound_mask :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      WitPool A.s →
      (∀ z ∈ readLog C (strongRound C (rd B F f₁) n j A first d).2.1, f₁ z = f₂ z) →
      strongRound C (rd B F f₁) n j A first d = strongRound C (rd B F f₂) n j A first d
  | 0, _, _, _, _, _, _ => by simp only [strongRound]
  | n + 1, j, A, first, d, hw, h => by
    have hsub : readLog C (strongReading C (rd B F f₁) j A first (d 0)).1
        ⊆ readLog C (strongRound C (rd B F f₁) (n + 1) j A first d).2.1 := by
      rcases hR : strongReading C (rd B F f₁) j A first (d 0) with ⟨A'', e | lv⟩ <;>
        simp only [strongRound, hR]
      · exact le_rfl
      · exact strongRound_readLog C _ n (j + 1) A'' lv (Fin.tail d)
    have hread := strongReading_mask C hw fun z hz => h z (hsub hz)
    have hw' := strongReading_witPool C (rd B F f₁) j A first (d 0) hw
    simp only [strongRound] at h ⊢
    rw [← hread]
    rcases hR : strongReading C (rd B F f₁) j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at h hw'
    all_goals first
      | rfl
      | (simp only [] at h ⊢; rw [strongRound_mask n (j + 1) A'' lv (Fin.tail d) hw' h])

end Round

end OrthoDFA
