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
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨x.toList.length, m, hm, by rw [prefixOf_length]; exact he⟩)
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

section Congr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)
variable {K B F f₁ f₂}

theorem probeOutcome_congr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf x i * m)) :
    probeOutcome (rd B F f₁) t edges k x = probeOutcome (rd B F f₂) t edges k x := by
  rw [← qProbe_run, ← qProbe_run]
  exact Qry.run_congr_of_asksIn (fun y ⟨i, m, hm, he⟩ => he ▸ cut_congr (h i m hm)) _
    (qProbe_asksIn t edges k x)

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

theorem seedStep_congr {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (hpool : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b)
    (hwit : ∀ p c q y, edges p c = some (q, y) → AgreeOne K F f₁ f₂ t y)
    (hw : ∀ i, AgreeOne K F f₁ f₂ t (prefixOf x i)) :
    seedStep K (rd B F f₁) t pool edges k x ps fd = seedStep K (rd B F f₂) t pool edges k x ps fd := by
  unfold seedStep
  simp only []
  split
  · rfl
  rename_i c _
  split
  · rfl
  rename_i s2 y he
  split_ifs
  · rfl
  have hy := hwit _ _ _ _ he
  rw [sift_congr (B := B) (hw (fd - 1)).tree, sift_congr (B := B) hy.tree]
  split
  · rfl
  rename_i p _
  split_ifs
  · rfl
  rw [parting_congr t fun m hm => ⟨cut_congr ((hy.letter' c) m hm),
    cut_congr (((hw (fd - 1)).letter' c) m hm)⟩]
  split
  · rfl
  · rfl
  rename_i d hd
  obtain ⟨m, hm, rfl⟩ := parting_mids t hd
  rw [verdict_congr (fun b hb => (hpool b hb).tree) fun b hb => (hpool b hb).letter' c m hm]

end Congr

end OrthoDFA
