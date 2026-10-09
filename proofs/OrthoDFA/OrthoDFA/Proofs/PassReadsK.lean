import OrthoDFA.Proofs.Triple

/-!
# What the pass walked from `k` reads

Every read of the pass is of a string `b·e·m`: `b` a seed string or a probe's prefix, `e` empty
or a letter, and `m` a midfix of the tree it ends with. So two oracles agreeing on the bits those
reads ask take it to the same state (`runPassK_determined`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem DTree.mem_midfixes_iff {m : FreeMonoid α} : ∀ t : DTree α, m ∈ t.midfixes ↔ m ∈ t.mids
  | .leaf => by simp [DTree.midfixes, DTree.mids]
  | .node n r a => by
    simp [DTree.midfixes, DTree.mids, DTree.mem_midfixes_iff r, DTree.mem_midfixes_iff a]

namespace Qry

variable {β γ : Type*}

/-- Whatever it is answered, the computation asks only strings satisfying `S`. -/
def AsksIn (S : FreeMonoid α → Prop) : Qry α β → Prop
  | pure _ => True
  | ask w _ k => S w ∧ ∀ o, (k o).AsksIn S

theorem asksIn_bind {S : FreeMonoid α → Prop} {f : β → Qry α γ} (hf : ∀ b, (f b).AsksIn S) :
    ∀ q : Qry α β, q.AsksIn S → (q.bind f).AsksIn S
  | pure b, _ => hf b
  | ask w g k, h => ⟨h.1, fun o => asksIn_bind hf (k o) (h.2 o)⟩

theorem asksIn_map {S : FreeMonoid α → Prop} (f : β → γ) (q : Qry α β) (h : q.AsksIn S) :
    (q.map f).AsksIn S := asksIn_bind (f := fun b => pure (f b)) (fun _ => trivial) q h

theorem asksIn_mono {S S' : FreeMonoid α → Prop} (hS : ∀ w, S w → S' w) :
    ∀ q : Qry α β, q.AsksIn S → q.AsksIn S'
  | pure _, _ => trivial
  | ask w g k, h => ⟨hS w h.1, fun o => asksIn_mono hS (k o) (h.2 o)⟩

theorem run_congr_of_asksIn {S : FreeMonoid α → Prop} {c₁ c₂ : FreeMonoid α → Option Bool}
    (hc : ∀ w, S w → c₁ w = c₂ w) : ∀ q : Qry α β, q.AsksIn S → q.run c₁ = q.run c₂
  | pure _, _ => rfl
  | ask w g k, h => by
    simp only [run, hc w h.1]
    exact run_congr_of_asksIn hc (k (c₂ w)) (h.2 _)

end Qry

theorem qSift_asksIn (w : FreeMonoid α) (g : Bool) :
    ∀ t : DTree α, (qSift w g t).AsksIn fun y => ∃ m ∈ t.mids, y = w * m
  | .leaf => trivial
  | .node m r a => by
    refine ⟨⟨m, by simp [DTree.mids], rfl⟩, fun o => ?_⟩
    rcases o with _ | _ | _
    · trivial
    · exact Qry.asksIn_map _ _ (Qry.asksIn_mono (fun y ⟨m', hm', he⟩ =>
        ⟨m', by simp [DTree.mids, hm'], he⟩) _ (qSift_asksIn w g r))
    · exact Qry.asksIn_map _ _ (Qry.asksIn_mono (fun y ⟨m', hm', he⟩ =>
        ⟨m', by simp [DTree.mids, hm'], he⟩) _ (qSift_asksIn w g a))

/-- A probe asks only about its prefixes followed by a midfix. -/
def PrefixRead (t : DTree α) (x y : FreeMonoid α) : Prop := ∃ i, ∃ m ∈ t.mids, y = prefixOf x i * m

theorem qAgrees_asksIn (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (i : ℕ)
    (g : Bool) : (qAgrees t x walkAt i g).AsksIn (PrefixRead t x) :=
  Qry.asksIn_map _ _ (Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, m, hm, he⟩) _
    (qSift_asksIn _ g t))

theorem qBracket_asksIn (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) : ∀ fuel lo hi,
    (qBracket (qAgrees t x walkAt) ps fuel lo hi).AsksIn (PrefixRead t x)
  | 0, _, _ => trivial
  | fuel + 1, lo, hi => by
    have hg : ∀ p g, (qGuard t x walkAt lo hi p g).AsksIn (PrefixRead t x) := by
      intro p g; unfold qGuard; split_ifs
      · trivial
      · trivial
      · exact qAgrees_asksIn t x walkAt p g
    by_cases hlh : lo + 1 < hi
    · rw [qBracket_unfold t x walkAt ps fuel lo hi hlh]
      refine Qry.asksIn_bind (fun o => ?_) _ (hg _ _)
      rcases o with _ | _ | _
      · refine Qry.asksIn_bind (fun l => ?_) _ (hg _ _)
        rcases l with _ | lv
        · trivial
        · refine Qry.asksIn_bind (fun r => ?_) _ (hg _ _)
          rcases lv <;> rcases r with _ | _ | _
          all_goals first | trivial | exact qBracket_asksIn t x walkAt ps fuel _ _
      · exact qBracket_asksIn t x walkAt ps fuel _ _
      · exact qBracket_asksIn t x walkAt ps fuel _ _
    · rw [qBracket, if_neg hlh]
      trivial

theorem qWalk_asksIn (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qWalk t edges k x).AsksIn (PrefixRead t x) := by
  have hs : ∀ i g, (qSift (prefixOf x i) g t).AsksIn (PrefixRead t x) := fun i g =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, m, hm, he⟩) _ (qSift_asksIn _ g t)
  have hx : (qSift x false t).AsksIn (PrefixRead t x) :=
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ =>
      ⟨x.toList.length, m, hm, by rw [prefixOf_length]; exact he⟩)
      _ (qSift_asksIn _ false t)
  unfold qWalk
  refine Qry.asksIn_bind (fun s0 => ?_) _ (hs k false)
  rcases s0 with p | _
  · simp only []
    split
    · exact Qry.asksIn_bind (fun s => by rcases s with _ | _ <;> trivial) _ hx
    · refine Qry.asksIn_bind (fun s1 => ?_) _ (hs _ false)
      rcases s1 with _ | _
      · exact Qry.asksIn_bind (fun s2 => by rcases s2 with _ | _ <;> trivial) _ (hs _ false)
      · trivial
  · trivial

theorem qProbe_asksIn (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qProbe t edges k x).AsksIn (PrefixRead t x) :=
  Qry.asksIn_bind (fun d => by
    rcases d with o | d
    · trivial
    · exact qBracket_asksIn t x _ d.1 _ _ _) _ (qWalk_asksIn t edges k x)

theorem qProbe_asksIn_ge (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qProbe t edges k x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := by
  set S : FreeMonoid α → Prop := fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m
  have hs : ∀ i g, k ≤ i → (qSift (prefixOf x i) g t).AsksIn S := fun i g hi =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, hi, m, hm, he⟩) _ (qSift_asksIn _ g t)
  have hx : (qSift x false t).AsksIn S := by
    refine Qry.asksIn_mono (fun y ⟨m, hm, he⟩ =>
      ⟨max k x.toList.length, le_max_left _ _, m, hm, ?_⟩)
      _ (qSift_asksIn _ false t)
    rw [he]
    congr 1
    apply FreeMonoid.toList.injective
    simp [prefixOf, List.take_of_length_le (le_max_right _ _)]
  have hb : ∀ walkAt ps fuel lo hi, k ≤ lo →
      (qBracket (qAgrees t x walkAt) ps fuel lo hi).AsksIn S := by
    intro walkAt ps fuel
    induction fuel with
    | zero => intro lo hi _; trivial
    | succ fuel ih =>
      intro lo hi hlo
      have hg : ∀ p g, lo ≤ p → (qGuard t x walkAt lo hi p g).AsksIn S := by
        intro p g hp; unfold qGuard; split_ifs
        · trivial
        · trivial
        · exact Qry.asksIn_map _ _ (hs p g (hlo.trans hp))
      by_cases hlh : lo + 1 < hi
      · rw [qBracket_unfold t x walkAt ps fuel lo hi hlh]
        refine Qry.asksIn_bind (fun o => ?_) _ (hg _ _ (by omega))
        rcases o with _ | _ | _
        · refine Qry.asksIn_bind (fun l => ?_) _ (hg _ _ (by omega))
          rcases l with _ | lv
          · trivial
          · refine Qry.asksIn_bind (fun r => ?_) _ (hg _ _ (by omega))
            rcases lv <;> rcases r with _ | _ | _
            all_goals first | trivial | exact ih _ _ (by omega)
        · exact ih _ _ (by omega)
        · exact ih _ _ (by omega)
      · rw [qBracket, if_neg hlh]
        trivial
  refine Qry.asksIn_bind (fun d => ?_) _ ?_
  · rcases d with o | d
    · trivial
    · exact hb _ d.1 _ _ _ le_rfl
  · unfold qWalk
    refine Qry.asksIn_bind (fun s0 => ?_) _ (hs k false le_rfl)
    rcases s0 with p | _
    · simp only []
      split
      · exact Qry.asksIn_bind (fun s => by rcases s with _ | _ <;> trivial) _ hx
      · refine Qry.asksIn_bind (fun s1 => ?_) _ (hs _ false (by omega))
        rcases s1 with _ | _
        · exact Qry.asksIn_bind (fun s2 => by rcases s2 with _ | _ <;> trivial) _
            (hs _ false (by omega))
        · trivial
    · trivial

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_edge_range (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi ps' j, lo < hi → bracketAt (α := α) agrees ps fuel lo hi = .edge ps' j →
      lo < j ∧ j ≤ hi
  | 0, lo, hi, ps', j, hlt, h => by
    simp only [bracketAt, Outcome.edge.injEq] at h
    omega
  | fuel + 1, lo, hi, ps', j, hlt, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h
      simp only [Outcome.edge.injEq] at h
      omega
    rw [if_pos hlh] at h
    have hl0 : (lo + hi) / 2 - 1 = lo → (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1))
          = some true := fun e => if_pos e
    have hr0 : (lo + hi) / 2 + 1 = hi → (if (lo + hi) / 2 + 1 = lo then some true
        else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1))
          = some false := fun e => by rw [if_neg (by omega), if_pos e]
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h hl0
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h hr0
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      · -- false, some false
        have hne : (lo + hi) / 2 - 1 ≠ lo := fun e => by simpa using hl0 e
        have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
        omega
      · have hne : (lo + hi) / 2 - 1 ≠ lo := fun e => by simpa using hl0 e
        have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
        omega
      · -- true, some true
        have hne : (lo + hi) / 2 + 1 ≠ hi := fun e => by simpa using hr0 e
        have := bracketAt_edge_range agrees ps fuel _ hi ps' j (by omega) h
        omega
    · have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
      omega
    · have := bracketAt_edge_range agrees ps fuel _ hi ps' j (by omega) h
      omega

section Congr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)
variable {K B F f₁ f₂}

theorem probeOutcome_congr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : ∀ i, k ≤ i → ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf x i * m)) :
    probeOutcome (rd B F f₁) t edges k x = probeOutcome (rd B F f₂) t edges k x := by
  rw [← qProbe_run, ← qProbe_run]
  exact Qry.run_congr_of_asksIn (fun y ⟨i, hi, m, hm, he⟩ => he ▸ cut_congr (h i hi m hm)) _
    (qProbe_asksIn_ge t edges k x)

theorem parting_congr {c₁ c₂ : FreeMonoid α → Option Bool} {x y pre : FreeMonoid α} :
    ∀ t : DTree α, (∀ m ∈ t.mids, c₁ (x * (pre * m)) = c₂ (x * (pre * m))
        ∧ c₁ (y * (pre * m)) = c₂ (y * (pre * m))) →
      t.parting c₁ x y pre = t.parting c₂ x y pre
  | .leaf, _ => rfl
  | .node m r a, h => by
    have hm := h m (by simp [DTree.mids])
    have hr := parting_congr r fun m' hm' => h m' (by simp [DTree.mids, hm'])
    have ha := parting_congr a fun m' hm' => h m' (by simp [DTree.mids, hm'])
    simp only [DTree.parting, hm.1, hm.2, hr, ha]

theorem parting_mids {cut : FreeMonoid α → Option Bool} {x y pre d : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = some (.inl d) → ∃ m ∈ t.mids, d = pre * m
  | .leaf, h => by simp [DTree.parting] at h
  | .node m r a, h => by
    simp only [DTree.parting] at h
    split at h
    · obtain ⟨m', hm', rfl⟩ := parting_mids a h
      exact ⟨m', by simp [DTree.mids, hm'], rfl⟩
    · obtain ⟨m', hm', rfl⟩ := parting_mids r h
      exact ⟨m', by simp [DTree.mids, hm'], rfl⟩
    · exact ⟨m, by simp [DTree.mids], (Sum.inl.inj (Option.some.inj h)).symm⟩
    · simp at h
    · simp at h

/-- Agreement on `b`'s reads against a tree containing every midfix of `t`. -/
theorem AgreeOne.mono' {t t' : DTree α} {b : FreeMonoid α} (h : AgreeOne K F f₁ f₂ t' b)
    (ht : ∀ m ∈ t.mids, m ∈ t'.mids) : AgreeOne K F f₁ f₂ t b :=
  fun e m hm => h e m (ht m hm)

/-- They agree on `b`'s held-out reads at a letter and a midfix of `t`. -/
def AgreeBlk (K : StageKnobs α) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)
    (t : DTree α) (b : FreeMonoid α) : Prop :=
  ∀ c : α, ∀ m ∈ t.mids, ∀ h ∈ K.block F, f₁ (b * (FreeMonoid.of c * m) * h)
    = f₂ (b * (FreeMonoid.of c * m) * h)

/-- They agree on the strings a split test counts at every key `forced` leaves tested. -/
def AgreeTests (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α))
    (f₁ f₂ : FreeMonoid α → ℝ) (t : DTree α) (pool : List (FreeMonoid α))
    (skip : TestKey α → FreeMonoid α → Prop) (forced : Set (TestKey α)) : Prop :=
  ∀ s1 c, ∀ m ∈ t.mids, (s1, FreeMonoid.of c * m) ∉ forced →
    ∀ p ∈ testStrings K (rd B F f₂) t pool s1 (FreeMonoid.of c * m)
      (skip (s1, FreeMonoid.of c * m)), f₁ p.1 = f₂ p.1

theorem agreeTests_of_blk {t t' : DTree α} {pool : List (FreeMonoid α)}
    {skip : TestKey α → FreeMonoid α → Prop} {forced : Set (TestKey α)}
    (ht : ∀ m ∈ t.mids, m ∈ t'.mids) (h : ∀ b ∈ pool, AgreeBlk K F f₁ f₂ t' b) :
    AgreeTests K B F f₁ f₂ t pool skip forced := by
  intro s1 c m hm _ p hp
  obtain ⟨b, hb, v, hv, he⟩ := mem_testStrings hp
  rw [he]
  exact h b (members_mem hb) c m (ht m hm) v hv

/-- The key of the split test a probe's step reaches, if any. -/
def SeedResult.key : SeedResult α → Option (TestKey α)
  | .split d s1 _ _ => some (s1, d)
  | .member s1 _ d => some (s1, d)
  | _ => none

theorem seedStep_key {R : CutReads α} {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : Edges α} {skip : TestKey α → FreeMonoid α → Prop} {forced : Set (TestKey α)} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ} {c : α} {s2 p : List Bool}
    {y d : FreeMonoid α} (hc : x.toList[fd - 1]? = some c)
    (he : edges (ps.getD (fd - 1 - k) []) c = some (s2, y)) (h1 : s2 = ps.getD (fd - k) [])
    (hsp : t.sift R.cut (prefixOf x (fd - 1)) = .inl p)
    (h2 : p = ps.getD (fd - 1 - k) [] ∧ t.sift R.cut y = .inl (ps.getD (fd - 1 - k) []))
    (hd : t.parting R.cut y (prefixOf x (fd - 1)) (FreeMonoid.of c) = some (.inl d)) :
    (seedStep K R t pool edges skip forced k x ps fd).key = some (ps.getD (fd - 1 - k) [], d) := by
  unfold seedStep
  simp only [hc]
  simp only [he]
  rw [if_neg (by rw [h1]; exact fun h => h rfl)]
  simp only [hsp]
  rw [if_neg (by rw [h2.1, h2.2]; simp)]
  simp only [hd]
  split_ifs
  · rfl
  · split <;> rfl

theorem seedStep_congr {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (hpool : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b)
    (hwit : ∀ p c q y, edges p c = some (q, y) → AgreeOne K F f₁ f₂ t y)
    (hw : AgreeOne K F f₁ f₂ t (prefixOf x (fd - 1)))
    (htest : ∀ κ, (seedStep K (rd B F f₂) t pool edges skip forced k x ps fd).key = some κ →
      κ ∉ forced → ∀ p ∈ testStrings K (rd B F f₂) t pool κ.1 κ.2 (skip κ), f₁ p.1 = f₂ p.1) :
    seedStep K (rd B F f₁) t pool edges skip forced k x ps fd
      = seedStep K (rd B F f₂) t pool edges skip forced k x ps fd := by
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
  rename_i d hd
  obtain ⟨m, hm, rfl⟩ := parting_mids t hd
  split_ifs with hf
  · rfl
  push Not at h1 h2
  have hk := seedStep_key (K := K) (R := rd B F f₂) (pool := pool) (skip := skip) (forced := forced)
    hc he h1 hsp h2 hd
  have hf' : ∀ q ∈ testStrings K (rd B F f₂) t pool (ps.getD (fd - 1 - k) [])
      (FreeMonoid.of c * m) (skip (ps.getD (fd - 1 - k) [], FreeMonoid.of c * m)),
      f₁ q.1 = f₂ q.1 := htest _ hk hf
  rw [verdict_congr (fun b hb => (hpool b hb).tree) (fun b hb => (hpool b hb).letter' c m hm) hf']

theorem counted_congr {t : DTree α} {pool : List (FreeMonoid α)}
    {skip : TestKey α → FreeMonoid α → Prop} {s1 : List Bool} {c : α} {m : FreeMonoid α}
    (hm : m ∈ t.mids) (hpool : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b) :
    counted K (rd B F f₁) t pool skip s1 (FreeMonoid.of c * m)
      = counted K (rd B F f₂) t pool skip s1 (FreeMonoid.of c * m) := by
  unfold counted
  rw [testStrings_congr (fun b hb => (hpool b hb).tree) fun b hb => (hpool b hb).letter' c m hm]

theorem testedAfter_congr {t : DTree α} {pool : List (FreeMonoid α)} {T : Tested α}
    {skip : TestKey α → FreeMonoid α → Prop} {s1 : List Bool} {c : α} {m : FreeMonoid α}
    (hm : m ∈ t.mids) (hpool : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b) :
    testedAfter K (rd B F f₁) t pool T skip s1 (FreeMonoid.of c * m)
      = testedAfter K (rd B F f₂) t pool T skip s1 (FreeMonoid.of c * m) := by
  unfold testedAfter
  rw [counted_congr hm hpool]

end Congr

section Shapes

variable (K : StageKnobs α) (R : CutReads α)

theorem seedStep_split_spec {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ} {d : FreeMonoid α} {s1 : List Bool}
    {y sprime : FreeMonoid α}
    (h : seedStep K R t pool edges skip forced k x ps fd = .split d s1 y sprime) :
    (∃ c s2, edges s1 c = some (s2, y)) ∧ sprime = prefixOf x (fd - 1) := by
  unfold seedStep at h
  simp only [] at h
  split at h
  · simp at h
  rename_i c _
  split at h
  · simp at h
  rename_i s2 y' he
  split_ifs at h
  split at h
  · simp at h
  split_ifs at h
  split at h
  · simp at h
  · simp at h
  split_ifs at h
  all_goals try (simp at h; done)
  split at h
  · simp only [SeedResult.split.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    exact ⟨⟨c, s2, he⟩, rfl⟩
  · simp at h

theorem seedStep_member_spec {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)}
    {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ} {s1 : List Bool}
    {sprime d : FreeMonoid α}
    (h : seedStep K R t pool edges skip forced k x ps fd = .member s1 sprime d) :
    sprime = prefixOf x (fd - 1) := by
  unfold seedStep at h
  simp only [] at h
  split at h
  · simp at h
  split at h
  · simp at h
  split_ifs at h
  split at h
  · simp at h
  split_ifs at h
  split at h
  · simp at h
  · simp at h
  split_ifs at h
  · simp only [SeedResult.member.injEq] at h
    exact h.2.1.symm
  split at h
  · simp at h
  · simp only [SeedResult.member.injEq] at h
    exact h.2.1.symm

/-- A split test's distinguisher is a letter and then a midfix of the tree. -/
theorem seedStep_dist {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)} {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    {d : FreeMonoid α}
    (h : (∃ s1 y sprime, seedStep K R t pool edges skip forced k x ps fd = .split d s1 y sprime)
      ∨ ∃ s1 sprime, seedStep K R t pool edges skip forced k x ps fd = .member s1 sprime d) :
    ∃ c, ∃ m ∈ t.mids, d = FreeMonoid.of c * m := by
  have key : ∀ r : SeedResult α, seedStep K R t pool edges skip forced k x ps fd = r →
      (∃ s1 y sprime, r = .split d s1 y sprime) ∨ (∃ s1 sprime, r = .member s1 sprime d) →
      ∃ c, ∃ m ∈ t.mids, d = FreeMonoid.of c * m := by
    intro r hr hrd
    unfold seedStep at hr
    simp only [] at hr
    split at hr
    · subst hr; simp at hrd
    rename_i c _
    split at hr
    · subst hr; simp at hrd
    split_ifs at hr
    · subst hr; simp at hrd
    split at hr
    · subst hr; simp at hrd
    split_ifs at hr
    · subst hr; simp at hrd
    split at hr
    · subst hr; simp at hrd
    · subst hr; simp at hrd
    rename_i d' hd'
    obtain ⟨m, hm, rfl⟩ := parting_mids t hd'
    refine ⟨c, m, hm, ?_⟩
    split_ifs at hr
    · subst hr; simp_all
    split at hr <;> subst hr <;> simp_all
  rcases h with ⟨s1, y, sp, h⟩ | ⟨s1, sp, h⟩
  · exact key _ h (.inl ⟨s1, y, sp, rfl⟩)
  · exact key _ h (.inr ⟨s1, sp, rfl⟩)

theorem kWalk_edge_ge {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {s : List Bool}
    {c : α} {j : ℕ} (h : kWalk R t edges k x = .edge s c j) : k ≤ j := by
  unfold kWalk at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · simp only [KWalk.edge.injEq] at h
      omega

theorem probeOutcome_edge_gt {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} (h : probeOutcome R t edges k x = .edge ps fd) : k < fd := by
  obtain ⟨ps', hi, hw, hbr⟩ := probeOutcome_search R h trivial
  obtain ⟨-, -, -, hkhi, -, -⟩ := walkCheck_inr R hw
  exact (bracketAt_edge_range (α := α) _ ps' _ _ _ _ _ hkhi hbr).1

theorem probeOutcome_member {t : DTree α} {edges : Edges α} {k : ℕ} {x u : FreeMonoid α}
    (h : probeOutcome R t edges k x = .member u) : ∃ j, k ≤ j ∧ u = prefixOf x j := by
  unfold probeOutcome at h
  rcases hw : walkCheck R t edges k x with o | ⟨ps, hi⟩ <;> rw [hw] at h
  swap
  · have := bracketAt_isSearch (α := α) (agreesAt R t x fun j => ps.getD (j - k) []) ps (hi - k)
      k hi
    simp only [Sum.elim_inr] at h
    rw [h] at this
    exact this.elim
  simp only [Sum.elim_inl, id] at h
  subst h
  unfold walkCheck at hw
  split at hw
  · simp at hw
  · rename_i s c j hk
    split at hw
    · simp at hw
    · split at hw
      · simp at hw
      · split_ifs at hw
        · exact ⟨_, kWalk_edge_ge R hk, (Outcome.member.inj (Sum.inl.inj hw)).symm⟩
  · split at hw
    · simp at hw
    · split_ifs at hw <;> simp at hw

/-- The pool and every edge's witness lie in `Bs`. -/
def KPoolIn (Bs : Set (FreeMonoid α)) (s : KState α) : Prop :=
  (∀ b ∈ s.pool, b ∈ Bs) ∧ ∀ p c q y, s.edges p c = some (q, y) → y ∈ Bs

theorem closeK_poolIn {Bs : Set (FreeMonoid α)} {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : Edges α} {st : ℕ} {T : Tested α} {lg : Finset (FreeMonoid α)}
    {fc : Set (TestKey α)} (hp : ∀ b ∈ pool, b ∈ Bs)
    (he : ∀ p c q y, edges p c = some (q, y) → y ∈ Bs) :
    KPoolIn Bs (closeK K R t pool edges st T lg fc) := by
  refine ⟨hp, fun p c q y h => ?_⟩
  simp only [closeK, closeEdges] at h
  rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', y'⟩
  · rw [hd] at h
    exact he _ _ _ _ h
  · rw [hd] at h
    dsimp only at h
    split_ifs at h
    · exact he _ _ _ _ h
    obtain ⟨-, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    exact hp _ (members_mem (decisiveTarget_mem K R hd))

theorem probeStepK_poolIn {Bs : Set (FreeMonoid α)} {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (hs : KPoolIn Bs s) (hw : ∀ i, k ≤ i → prefixOf x i ∈ Bs) :
    KPoolIn Bs (probeStepK K R k s x) := by
  obtain ⟨hp, he⟩ := hs
  unfold probeStepK
  simp only []
  split
  · rename_i ps fd hpo
    have hfd := hw (fd - 1) (by have := probeOutcome_edge_gt R hpo; omega)
    split
    · rename_i d s1 y sprime hss
      obtain ⟨⟨c, s2, hy⟩, rfl⟩ := seedStep_split_spec K R hss
      refine closeK_poolIn K R (mem_append_filter hp (mem_pair (he _ _ _ _ hy) hfd))
        fun p c' q w hq => ?_
      rcases hE : s.edges p c' with _ | ⟨q', w'⟩ <;> simp only [hE] at hq
      · simp at hq
      · split_ifs at hq
        have : w' = w := by simp_all
        exact this ▸ he _ _ _ _ hE
    · rename_i s1 sprime d hss
      rw [seedStep_member_spec K R hss]
      exact closeK_poolIn K R (mem_cons_filter hfd hp) he
    · exact closeK_poolIn K R hp he
  · rename_i u hu
    obtain ⟨j, hj, rfl⟩ := probeOutcome_member R hu
    refine closeK_poolIn K R ?_ he
    split_ifs
    · exact hp
    · exact mem_append_single hp (hw j hj)
  · exact closeK_poolIn K R hp he

theorem probeStepK_tree_cases (k : ℕ) (s : KState α) (x : FreeMonoid α) :
    (probeStepK K R k s x).tree = s.tree
      ∨ ∃ d p, (probeStepK K R k s x).tree = s.tree.splitAt d p := by
  unfold probeStepK
  simp only []
  repeat' split
  all_goals first | exact .inl rfl | exact .inr ⟨_, _, rfl⟩

end Shapes

section StepCongr

variable {K : StageKnobs α} {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

/-- They agree on the strings the split test of `x`'s step from `s` counts, where it tests at a
key `s` leaves tested. -/
def StepTestsAgree (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α))
    (f₁ f₂ : FreeMonoid α → ℝ) (k : ℕ) (s : KState α) (x : FreeMonoid α) : Prop :=
  ∀ ps fd κ, probeOutcome (rd B F f₂) s.tree s.edges k x = .edge ps fd →
    (seedStep K (rd B F f₂) s.tree s.pool s.edges (stepSkip K (rd B F f₂) k s x) s.forced k x
      ps fd).key = some κ → κ ∉ s.forced →
    ∀ p ∈ testStrings K (rd B F f₂) s.tree s.pool κ.1 κ.2 (stepSkip K (rd B F f₂) k s x κ),
      f₁ p.1 = f₂ p.1

theorem stepTests_of_agreeTests {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (h : AgreeTests K B F f₁ f₂ s.tree s.pool (stepSkip K (rd B F f₂) k s x) s.forced) :
    StepTestsAgree K B F f₁ f₂ k s x := by
  intro ps fd κ _ hk hf p hp
  obtain ⟨κ1, κ2⟩ := κ
  have hr : (∃ s1 y sprime, seedStep K (rd B F f₂) s.tree s.pool s.edges
      (stepSkip K (rd B F f₂) k s x) s.forced k x ps fd = .split κ2 s1 y sprime)
      ∨ ∃ s1 sprime, seedStep K (rd B F f₂) s.tree s.pool s.edges
        (stepSkip K (rd B F f₂) k s x) s.forced k x ps fd = .member s1 sprime κ2 := by
    revert hk
    rcases seedStep K (rd B F f₂) s.tree s.pool s.edges (stepSkip K (rd B F f₂) k s x) s.forced
      k x ps fd with ⟨d, s1, y, sp⟩ | ⟨s1, sp, d⟩ | b | _ <;> intro hk <;>
      simp only [SeedResult.key, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at hk
    · exact .inl ⟨s1, y, sp, by rw [hk.2]⟩
    · exact .inr ⟨s1, sp, by rw [hk.2]⟩
  obtain ⟨c, m, hm, rfl⟩ := seedStep_dist K _ hr
  exact h κ1 c m hm hf p hp

theorem probeStepK_congr {k : ℕ} {s : KState α} {x : FreeMonoid α} (Tf : DTree α)
    (hT : ∀ m ∈ s.tree.mids, m ∈ Tf.mids)
    (hT' : ∀ m ∈ (probeStepK K (rd B F f₁) k s x).tree.mids, m ∈ Tf.mids)
    (hpool : ∀ b ∈ s.pool, AgreeOne K F f₁ f₂ Tf b)
    (hwit : ∀ p c q y, s.edges p c = some (q, y) → AgreeOne K F f₁ f₂ Tf y)
    (hw : ∀ i, k ≤ i → AgreeOne K F f₁ f₂ Tf (prefixOf x i))
    (htest : StepTestsAgree K B F f₁ f₂ k s x) :
    probeStepK K (rd B F f₁) k s x = probeStepK K (rd B F f₂) k s x := by
  have hpool' : ∀ b ∈ s.pool, AgreeOne K F f₁ f₂ s.tree b := fun b hb => (hpool b hb).mono' hT
  have hwit' : ∀ p c q y, s.edges p c = some (q, y) → AgreeOne K F f₁ f₂ s.tree y :=
    fun p c q y h => (hwit p c q y h).mono' hT
  have hw' : ∀ i, k ≤ i → AgreeOne K F f₁ f₂ s.tree (prefixOf x i) :=
    fun i hi => (hw i hi).mono' hT
  have hpo : probeOutcome (rd B F f₁) s.tree s.edges k x
      = probeOutcome (rd B F f₂) s.tree s.edges k x :=
    probeOutcome_congr fun i hi m hm => (hw' i hi).tree m hm
  have hsk : stepSkip K (rd B F f₁) k s x = stepSkip K (rd B F f₂) k s x := rfl
  unfold probeStepK at hT' ⊢
  simp only [] at hT' ⊢
  rw [hpo, hsk] at hT' ⊢
  split
  · rename_i ps fd heq
    have hfd : k ≤ fd - 1 := by have := probeOutcome_edge_gt _ heq; omega
    simp only [heq] at hT'
    rw [seedStep_congr hpool' hwit' (hw' _ hfd) fun κ hk hf => htest ps fd κ heq hk hf] at hT' ⊢
    split
    · rename_i d s1 y sprime hsd
      simp only [hsd, closeK] at hT'
      obtain ⟨c', m', hm', rfl⟩ := seedStep_dist K _ (.inl ⟨s1, y, sprime, hsd⟩)
      obtain ⟨⟨c, s2, hy⟩, rfl⟩ := seedStep_split_spec K _ hsd
      simp only [closeK]
      rw [testedAfter_congr hm' hpool']
      rw [closeEdges_congr fun b hb => ?_]
      refine AgreeOne.mono' ?_ hT'
      rcases List.mem_append.1 hb with hb | hb
      · exact hpool b hb
      · rcases List.mem_cons.1 (List.mem_of_mem_filter hb) with rfl | hb
        · exact hwit _ _ _ _ hy
        · rw [List.mem_singleton.1 hb]; exact hw _ hfd
    · rename_i s1 sprime d hsd
      obtain ⟨c', m', hm', rfl⟩ := seedStep_dist K _ (.inr ⟨s1, sprime, hsd⟩)
      rw [seedStep_member_spec K _ hsd]
      simp only [closeK]
      rw [testedAfter_congr hm' hpool']
      rw [closeEdges_congr fun b hb => ?_]
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hw' _ hfd
      · exact hpool' b (List.mem_of_mem_filter hb)
    · simp only [closeK]
      rw [closeEdges_congr hpool']
  · rename_i u hu
    obtain ⟨j, hj, rfl⟩ := probeOutcome_member _ hu
    simp only [closeK]
    rw [closeEdges_congr fun b hb => ?_]
    split_ifs at hb
    · exact hpool' b hb
    · rcases List.mem_append.1 hb with hb | hb
      · exact hpool' b hb
      · rw [List.mem_singleton.1 hb]; exact hw' j hj
  · simp only [closeK]
    rw [closeEdges_congr hpool']

end StepCongr

section PhasesK

variable (K : StageKnobs α) (R : CutReads α) (k : ℕ)

/-- One step of `runPassK`. -/
noncomputable def stepK (s : KState α) (x : FreeMonoid α) : KState α :=
  if K.patience ≤ s.streak then s else probeStepK K R k s x

/-- The pass's state after the first `n` probes. -/
noncomputable def phaseK (seed probes : List (FreeMonoid α)) (n : ℕ) : KState α :=
  runPassK K R k (initialK K R seed) (probes.take n)

theorem phaseK_succ (seed probes : List (FreeMonoid α)) {n : ℕ} (hn : n < probes.length) :
    phaseK K R k seed probes (n + 1) = stepK K R k (phaseK K R k seed probes n) probes[n] := by
  simp only [phaseK, runPassK, List.take_succ, List.getElem?_eq_getElem hn, Option.toList_some,
    List.foldl_append, List.foldl_cons, List.foldl_nil, stepK]
  rfl

theorem phaseK_of_le (seed probes : List (FreeMonoid α)) {n : ℕ} (hn : probes.length ≤ n) :
    phaseK K R k seed probes n = phaseK K R k seed probes probes.length := by
  simp only [phaseK, List.take_of_length_le hn, List.take_length]

theorem stepK_mids_mono (s : KState α) (x : FreeMonoid α) :
    ∀ m ∈ s.tree.mids, m ∈ (stepK K R k s x).tree.mids := fun m hm => by
  unfold stepK
  split
  · exact hm
  · rcases probeStepK_tree_cases K R k s x with h | ⟨d, p, h⟩
    · rw [h]; exact hm
    · rw [h]; exact DTree.mem_mids_splitAt hm

theorem phaseK_mids_mono (seed probes : List (FreeMonoid α)) (n : ℕ) :
    ∀ m ∈ (phaseK K R k seed probes n).tree.mids,
      m ∈ (phaseK K R k seed probes (n + 1)).tree.mids := by
  by_cases hn : n < probes.length
  · rw [phaseK_succ K R k seed probes hn]; exact stepK_mids_mono K R k _ _
  · rw [phaseK_of_le K R k seed probes (n := n + 1) (by omega),
      phaseK_of_le K R k seed probes (n := n) (by omega)]
    exact fun m hm => hm

theorem phaseK_mids_le (seed probes : List (FreeMonoid α)) {n n' : ℕ} (h : n ≤ n') :
    ∀ m ∈ (phaseK K R k seed probes n).tree.mids, m ∈ (phaseK K R k seed probes n').tree.mids := by
  induction h with
  | refl => exact fun m hm => hm
  | step _ ih => exact fun m hm => phaseK_mids_mono K R k seed probes _ m (ih m hm)

theorem phaseK_final_mids (seed probes : List (FreeMonoid α)) (n : ℕ) :
    ∀ m ∈ (phaseK K R k seed probes n).tree.mids,
      m ∈ (runPassK K R k (initialK K R seed) probes).tree.mids := by
  have hfin : runPassK K R k (initialK K R seed) probes
      = phaseK K R k seed probes probes.length := by
    simp [phaseK]
  rw [hfin]
  rcases le_total n probes.length with h | h
  · exact phaseK_mids_le K R k seed probes h
  · rw [phaseK_of_le K R k seed probes h]; exact fun m hm => hm

theorem phaseK_poolIn {Bs : Set (FreeMonoid α)} {seed probes : List (FreeMonoid α)}
    (hseed : ∀ b ∈ seed, b ∈ Bs) (hws : ∀ w ∈ probes, ∀ i, k ≤ i → prefixOf w i ∈ Bs) :
    ∀ n, KPoolIn Bs (phaseK K R k seed probes n)
  | 0 => by
    simp only [phaseK, List.take_zero, runPassK, List.foldl_nil]
    exact closeK_poolIn K R hseed fun _ _ _ _ h => by simp at h
  | n + 1 => by
    by_cases hn : n < probes.length
    · rw [phaseK_succ K R k seed probes hn]
      unfold stepK
      split
      · exact phaseK_poolIn hseed hws n
      · exact probeStepK_poolIn K R (phaseK_poolIn hseed hws n) (hws _ (List.getElem_mem hn))
    · rw [phaseK_of_le K R k seed probes (n := n + 1) (by omega)]
      rw [← phaseK_of_le K R k seed probes (n := n) (by omega)]
      exact phaseK_poolIn hseed hws n

end PhasesK

section Determined

variable {K : StageKnobs α} {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem phaseK_congr (k : ℕ) {Bs : Set (FreeMonoid α)} {seed probes : List (FreeMonoid α)}
    (hseed : ∀ b ∈ seed, b ∈ Bs) (hws : ∀ w ∈ probes, ∀ i, k ≤ i → prefixOf w i ∈ Bs)
    (Tf : DTree α)
    (hTf : ∀ n, ∀ m ∈ (phaseK K (rd B F f₁) k seed probes n).tree.mids, m ∈ Tf.mids)
    (hB : ∀ b ∈ Bs, AgreeOne K F f₁ f₂ Tf b) (hBblk : ∀ b ∈ Bs, AgreeBlk K F f₁ f₂ Tf b) :
    ∀ n, phaseK K (rd B F f₁) k seed probes n = phaseK K (rd B F f₂) k seed probes n
  | 0 => by
    have h0 := hTf 0
    simp only [phaseK, List.take_zero, runPassK, List.foldl_nil, initialK, closeK] at h0 ⊢
    rw [closeEdges_congr fun b hb => (hB b (hseed b hb)).mono' h0]
  | n + 1 => by
    by_cases hn : n < probes.length
    · rw [phaseK_succ K _ k seed probes hn, phaseK_succ K _ k seed probes hn,
        ← phaseK_congr k hseed hws Tf hTf hB hBblk n]
      have hT' := hTf (n + 1)
      rw [phaseK_succ K _ k seed probes hn] at hT'
      unfold stepK at hT' ⊢
      split
      · rfl
      · rename_i hst
        simp only [hst, if_false] at hT'
        obtain ⟨hp, he⟩ := phaseK_poolIn K (rd B F f₁) k hseed hws n
        exact probeStepK_congr Tf (hTf n) hT' (fun b hb => hB b (hp b hb))
          (fun p c q y hy => hB y (he p c q y hy))
          (fun i hi => hB _ (hws _ (List.getElem_mem hn) i hi))
          (stepTests_of_agreeTests (agreeTests_of_blk (hTf n) fun b hb => hBblk b (hp b hb)))
    · rw [phaseK_of_le K _ k seed probes (n := n + 1) (by omega),
        phaseK_of_le K _ k seed probes (n := n + 1) (by omega),
        ← phaseK_of_le K _ k seed probes (n := n) (by omega),
        ← phaseK_of_le K _ k seed probes (n := n) (by omega)]
      exact phaseK_congr k hseed hws Tf hTf hB hBblk n

theorem prefixOf_mem_bases {k : ℕ} {seed probes : List (FreeMonoid α)} {p : FreeMonoid α}
    (hp : p ∈ probes) {i : ℕ} (hi : k ≤ i) :
    prefixOf p i ∈ seed ++ probes.flatMap fun p =>
      (List.range (p.toList.length + 1)).map fun i => prefixOf p (max k i) := by
  refine List.mem_append_right _ (List.mem_flatMap.2 ⟨p, hp, List.mem_map.2
    ⟨min i p.toList.length, List.mem_range.2 (by omega), ?_⟩⟩)
  unfold prefixOf
  congr 1
  rcases le_total i p.toList.length with h | h
  · rw [min_eq_left h, max_eq_right hi]
  · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le (by omega)]

/-- The pass walked from `k` is decided by the oracle's bits at what it reads against the tree it
ends with. -/
theorem runPassK_determined {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω}
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
    (k : ℕ) (seed probes : List (FreeMonoid α)) (ω ω' : Ω)
    (h : ∀ y ∈ vBits (K.suffixes F) (passReadSet k seed probes
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).tree),
      O.noise y ω = O.noise y ω') :
    runPassK K (readsAt O B F ω') k (initialK K (readsAt O B F ω') seed) probes
      = runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes := by
  set Bs : Set (FreeMonoid α) := {b | b ∈ seed ++ probes.flatMap fun p =>
    (List.range (p.toList.length + 1)).map fun i => prefixOf p (max k i)}
  set Tf := (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).tree
  have hseed : ∀ b ∈ seed, b ∈ Bs := fun b hb => List.mem_append_left _ hb
  have hws : ∀ w ∈ probes, ∀ i, k ≤ i → prefixOf w i ∈ Bs := fun w hw i hi =>
    prefixOf_mem_bases hw hi
  have hAll : ∀ b ∈ Bs, ∀ e, ∀ m ∈ Tf.mids,
      AgreeAll K F (fun w => O.mq w ω) (fun w => O.mq w ω') (b * ext e * m) := by
    intro b hb e m hm
    refine agree_of_noise K O F fun v hv => h _ ?_
    simp only [vBits, Finset.mem_biUnion, Finset.mem_image]
    refine ⟨b * ext e * m, ?_, v, hv, rfl⟩
    simp only [passReadSet, Finset.mem_image, Finset.mem_product, Prod.exists]
    refine ⟨b, ext e, m, ⟨⟨List.mem_toFinset.2 hb, ?_⟩, ?_⟩, rfl⟩
    · rcases e with _ | c
      · exact Finset.mem_insert_self _ _
      · exact Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨c, Finset.mem_univ _, rfl⟩)
    · exact Finset.mem_insert_of_mem ((DTree.mem_midfixes_iff _).2 hm)
  have hB : ∀ b ∈ Bs, AgreeOne K F (fun w => O.mq w ω) (fun w => O.mq w ω') Tf b :=
    fun b hb e m hm => (hAll b hb e m hm).agree
  have hBblk : ∀ b ∈ Bs, AgreeBlk K F (fun w => O.mq w ω) (fun w => O.mq w ω') Tf b := by
    intro b hb c m hm v hv
    have := hAll b hb (some c) m hm v (K.block_sub F hv)
    simpa [ext, mul_assoc] using this
  have hTf : ∀ n, ∀ m ∈ (phaseK K (rd B F fun w => O.mq w ω) k seed probes n).tree.mids,
      m ∈ Tf.mids := fun n => phaseK_final_mids K _ k seed probes n
  have := phaseK_congr k hseed hws Tf hTf hB hBblk probes.length
  simp only [phaseK, List.take_length] at this
  exact this.symm

end Determined

end OrthoDFA
