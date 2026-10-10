import OrthoDFA.Proofs.TallyTrue
import OrthoDFA.Proofs.TallyFix

/-!
# How many versions the round's hypothesis takes

`pot` counts the leaves still to come, weighted by the most edges a tree within `Lmax` leaves can
learn or redirect, plus the edges out of leaves not learned and those not pointing at a target
with `m` records. A learning or a redirect lowers it by one, a split by at least one, and nothing
raises it. So a round within `Lmax` leaves takes at most `versionCap` versions.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `s` is at the start of a stretch. -/
def TState.Fresh (s : TState α) : Prop :=
  s.n = 0 ∧ s.starts = 0 ∧ s.dis = 0 ∧ s.reads = (fun _ _ => 0) ∧ s.und = (fun _ _ => 0)

open scoped Classical in
/-- The edges out of `s`'s leaves. -/
noncomputable def TState.keys (s : TState α) : Finset (List Bool × α) :=
  s.tree.paths.toFinset ×ˢ Finset.univ

open scoped Classical in
/-- The edges out of leaves not learned. -/
noncomputable def TState.unl (s : TState α) : ℕ :=
  (s.keys.filter fun pc => s.edges pc.1 pc.2 = none).card

open scoped Classical in
/-- The edges out of leaves not pointing at a target with `m` records. -/
noncomputable def TState.uns (m : ℕ) (s : TState α) : ℕ :=
  (s.keys.filter fun pc => ¬ ∃ t w, s.edges pc.1 pc.2 = some (t, w)
    ∧ m ≤ s.tally pc.1 pc.2 t).card

/-- The leaves to come, weighted by the most edges a tree within `Lmax` leaves has, and the edges
out of leaves not learned or not at a target with `m` records. -/
noncomputable def TState.pot (m Lmax : ℕ) (s : TState α) : ℕ :=
  (Lmax - s.tree.paths.length) * (2 * Lmax * Fintype.card α + 1) + s.unl + s.uns m

variable (m Lmax : ℕ)

theorem keys_card_le (s : TState α) : s.keys.card ≤ s.tree.paths.length * Fintype.card α := by
  classical
  simp only [TState.keys, Finset.card_product, Finset.card_univ]
  exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)

theorem unl_le (s : TState α) : s.unl ≤ s.tree.paths.length * Fintype.card α :=
  (Finset.card_le_card (Finset.filter_subset _ _)).trans (keys_card_le s)

theorem uns_le (s : TState α) : s.uns m ≤ s.tree.paths.length * Fintype.card α := by
  classical
  exact (Finset.card_le_card (Finset.filter_subset _ _)).trans (keys_card_le s)

theorem setEdge_eq (s : TState α) (p : List Bool) (c : α) (e) (q : List Bool) (c' : α) :
    (s.setEdge p c e).edges q c' = if (q, c') = (p, c) then some e else s.edges q c' := by
  unfold TState.setEdge
  by_cases hq : q = p
  · subst hq
    by_cases hc : c' = c
    · subst hc; simp
    · simp [hc]
  · simp [hq]

theorem card_add_one_le {β : Type*} [DecidableEq β] {A B : Finset β} {x : β}
    (h : A ⊆ B.erase x) (hx : x ∈ B) : A.card + 1 ≤ B.card := by
  have := Finset.card_le_card h
  rw [Finset.card_erase_of_mem hx] at this
  have := Finset.card_pos.2 ⟨x, hx⟩
  omega

theorem unl_setEdge (s : TState α) (p : List Bool) (c : α) (e) :
    (s.setEdge p c e).unl ≤ s.unl := by
  classical
  unfold TState.unl
  apply Finset.card_le_card
  intro pc hpc
  simp only [Finset.mem_filter, setEdge_eq] at hpc ⊢
  refine ⟨hpc.1, ?_⟩
  have := hpc.2
  split_ifs at this <;> first | exact this | simp at this

theorem unl_setEdge_lt (s : TState α) {p : List Bool} {c : α} (e) (hp : p ∈ s.tree.paths)
    (h : s.edges p c = none) : (s.setEdge p c e).unl + 1 ≤ s.unl := by
  classical
  unfold TState.unl
  apply card_add_one_le (x := (p, c))
  · intro pc hpc
    simp only [Finset.mem_filter, setEdge_eq] at hpc
    obtain ⟨h1, h2⟩ := hpc
    split_ifs at h2 with he
    exact Finset.mem_erase.2 ⟨he, Finset.mem_filter.2 ⟨h1, h2⟩⟩
  · simp [TState.keys, hp, h]

theorem uns_setEdge (s : TState α) {p : List Bool} {c : α} {t : List Bool} {w : FreeMonoid α}
    (h : ¬ ∃ t' w', s.edges p c = some (t', w') ∧ m ≤ s.tally p c t') :
    (s.setEdge p c (t, w)).uns m ≤ s.uns m := by
  classical
  unfold TState.uns
  apply Finset.card_le_card
  intro pc hpc
  simp only [Finset.mem_filter, setEdge_eq] at hpc ⊢
  refine ⟨hpc.1, ?_⟩
  have := hpc.2
  split_ifs at this with he
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
    exact h
  · exact this

theorem uns_setEdge_lt (s : TState α) {p : List Bool} {c : α} {t : List Bool} {w : FreeMonoid α}
    (hp : p ∈ s.tree.paths) (h : ¬ ∃ t' w', s.edges p c = some (t', w') ∧ m ≤ s.tally p c t')
    (ht : m ≤ s.tally p c t) : (s.setEdge p c (t, w)).uns m + 1 ≤ s.uns m := by
  classical
  unfold TState.uns
  apply card_add_one_le (x := (p, c))
  · intro pc hpc
    simp only [Finset.mem_filter, setEdge_eq] at hpc
    obtain ⟨h1, h2⟩ := hpc
    split_ifs at h2 with he
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      exact absurd ⟨t, w, rfl, ht⟩ h2
    · exact Finset.mem_erase.2 ⟨he, Finset.mem_filter.2 ⟨h1, h2⟩⟩
  · simp only [Finset.mem_filter]
    exact ⟨by simp [TState.keys, hp], h⟩

/-- A learning lowers `pot` by one. -/
theorem pot_learn (s : TState α) {p : List Bool} {c : α} {t : List Bool} {w : FreeMonoid α}
    (hp : p ∈ s.tree.paths) (h : s.edges p c = none) :
    (s.setEdge p c (t, w)).pot m Lmax + 1 ≤ s.pot m Lmax := by
  have h1 := unl_setEdge_lt s (t, w) hp h
  have h2 := uns_setEdge m s (t := t) (w := w) (by rintro ⟨t', w', he, -⟩; rw [h] at he; cases he)
  unfold TState.pot
  simp only [show (s.setEdge p c (t, w)).tree = s.tree from rfl]
  omega

/-- A redirect, from an edge not at a target with `m` records to one with `m`, lowers `pot` by
one. -/
theorem pot_redirect (s : TState α) {p : List Bool} {c : α} {t : List Bool} {w : FreeMonoid α}
    (hp : p ∈ s.tree.paths) (h : ¬ ∃ t' w', s.edges p c = some (t', w') ∧ m ≤ s.tally p c t')
    (ht : m ≤ s.tally p c t) : (s.setEdge p c (t, w)).pot m Lmax + 1 ≤ s.pot m Lmax := by
  have h1 := unl_setEdge s p c (t, w)
  have h2 := uns_setEdge_lt m s hp h ht (w := w)
  unfold TState.pot
  simp only [show (s.setEdge p c (t, w)).tree = s.tree from rfl]
  omega

/-- A split within `Lmax` leaves lowers `pot` by at least one. -/
theorem pot_split (s s' : TState α) (hp : s'.tree.paths.length = s.tree.paths.length + 1)
    (hL : s'.tree.paths.length ≤ Lmax) : s'.pot m Lmax + 1 ≤ s.pot m Lmax := by
  have h1 := unl_le s'
  have h2 := uns_le m s'
  have hk' : s'.tree.paths.length * Fintype.card α ≤ Lmax * Fintype.card α :=
    Nat.mul_le_mul_right _ hL
  unfold TState.pot
  have : (Lmax - s.tree.paths.length) = (Lmax - s'.tree.paths.length) + 1 := by omega
  rw [this, add_mul, one_mul]
  nlinarith

/-- A record only raises tallies, so it never raises `pot`. -/
theorem pot_addRec (s : TState α) (p : List Bool) (c : α) (r : FreeMonoid α × List Bool) :
    (s.addRec p c r).pot m Lmax ≤ s.pot m Lmax := by
  classical
  have : (s.addRec p c r).uns m ≤ s.uns m := by
    unfold TState.uns
    apply Finset.card_le_card
    intro pc hpc
    simp only [Finset.mem_filter] at hpc ⊢
    refine ⟨hpc.1, fun ⟨t, w, he, ht⟩ => hpc.2 ⟨t, w, he, ?_⟩⟩
    have := tally_addRec_ge s p c r pc.1 pc.2 t
    omega
  unfold TState.pot
  have hu : (s.addRec p c r).unl = s.unl := rfl
  simp only [show (s.addRec p c r).tree = s.tree from rfl, hu]
  omega

section Step

variable (cut : FreeMonoid α → Option Bool)

/-- A fix splits, where the current target has `m` records too, or redirects from an edge not at
a target with `m` records. -/
theorem fixEdge_cases (s : TState α) (p : List Bool) (c : α) (t : List Bool) :
    (∃ t₀ w₀, s.edges p c = some (t₀, w₀) ∧ m ≤ s.tally p c t₀
      ∧ fixEdge cut m s p c t = { s.fresh with
          tree := s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
          version := s.version + 1
          edges := fun q e => if p <+: q then none else retargetBy cut
            (s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p) p s.edges q e
          recs := fun q e => if p <+: q then [] else (s.recs q e).filter (·.2 ≠ p) })
    ∨ ((¬ ∃ t₀ w₀, s.edges p c = some (t₀, w₀) ∧ m ≤ s.tally p c t₀)
      ∧ fixEdge cut m s p c t
        = s.setEdge p c (t, (((s.recs p c).find? (·.2 = t)).map Prod.fst).getD 1)) := by
  unfold fixEdge
  rcases he : s.edges p c with _ | ⟨t₀, w₀⟩
  · right
    refine ⟨(fun ⟨_, _, h, _⟩ => nomatch h), ?_⟩
    rfl
  · simp only []
    by_cases hm : m ≤ s.tally p c t₀
    · left
      exact ⟨t₀, w₀, rfl, hm, by rw [if_pos hm]⟩
    · right
      refine ⟨?_, by rw [if_neg hm]⟩
      rintro ⟨t₁, w₁, h, h'⟩
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      exact hm h'

theorem fixEdge_fresh (s : TState α) (p : List Bool) (c : α) (t : List Bool) :
    (fixEdge cut m s p c t).Fresh := by
  rcases fixEdge_cases m cut s p c t with h | h
  · obtain ⟨t₀, w₀, -, -, h⟩ := h
    rw [h]
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  · rw [h.2]
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- A fix within `Lmax` leaves lowers `pot` by at least one. -/
theorem pot_fixEdge (s : TState α) {p : List Bool} {c : α} {t : List Bool}
    (hv : Violates m s p c t) (hL : (fixEdge cut m s p c t).tree.paths.length ≤ Lmax) :
    (fixEdge cut m s p c t).pot m Lmax + 1 ≤ s.pot m Lmax := by
  obtain ⟨hp, -, ht⟩ := hv
  rcases fixEdge_cases m cut s p c t with ⟨t₀, w₀, -, -, h⟩ | ⟨hn, h⟩
  · rw [h] at hL ⊢
    exact pot_split m Lmax s _ (DTree.splitAt_paths_length _ s.tree p hp) hL
  · rw [h]
    exact pot_redirect m Lmax s hp hn ht

/-- The edges settle within `fuel` fixes, `pot` falling by at least one at each, where every fix
keeps an invariant and stays within `Lmax` leaves. -/
theorem settle_spec (I : TState α → Prop)
    (hI : ∀ s p c t, I s → Violates m s p c t →
      I (fixEdge cut m s p c t) ∧ (fixEdge cut m s p c t).tree.paths.length ≤ Lmax) :
    ∀ (fuel : ℕ) (s : TState α), I s → s.pot m Lmax ≤ fuel →
      ∃ s', settleEdges cut m Lmax fuel s = .inl s' ∧ I s' ∧ s.version ≤ s'.version
        ∧ s'.pot m Lmax + (s'.version - s.version) ≤ s.pot m Lmax
        ∧ (s'.version = s.version → s' = s) ∧ (s'.version ≠ s.version → s'.Fresh)
  | 0, s, hs, hp => by
    by_cases hset : Settled m s
    · refine ⟨s, by simp [settleEdges, hset], hs, le_rfl, by simp, fun _ => rfl, fun h => ?_⟩
      exact absurd rfl h
    · exfalso
      simp only [Settled, not_forall, not_not] at hset
      obtain ⟨p, c, t, hv⟩ := hset
      have := pot_fixEdge m Lmax cut s hv (hI s p c t hs hv).2
      omega
  | fuel + 1, s, hs, hp => by
    by_cases hex : ∃ p c t, Violates m s p c t
    · set p := hex.choose
      set c := hex.choose_spec.choose
      set t := hex.choose_spec.choose_spec.choose
      have hv : Violates m s p c t := hex.choose_spec.choose_spec.choose_spec
      obtain ⟨hI1, hL⟩ := hI s p c t hs hv
      have hpot := pot_fixEdge m Lmax cut s hv hL
      obtain ⟨s', hs', hI', hle, hpot', heq, hfr⟩ :=
        settle_spec I hI fuel (fixEdge cut m s p c t) hI1 (by omega)
      have hver := fixEdge_version cut m s p c t
      refine ⟨s', ?_, hI', by omega, by omega, fun h => by omega, fun _ => ?_⟩
      · rw [settleEdges, dif_pos hex, if_neg (not_lt.2 hL)]
        exact hs'
      · by_cases h' : s'.version = (fixEdge cut m s p c t).version
        · rw [heq h']; exact fixEdge_fresh m cut s p c t
        · exact hfr h'
    · refine ⟨s, by rw [settleEdges, dif_neg hex], hs, le_rfl, by simp, fun _ => rfl,
        fun h => absurd rfl h⟩

theorem pot_congr {s s' : TState α} (ht : s'.tree = s.tree) (he : s'.edges = s.edges)
    (hr : s'.recs = s.recs) : s'.pot m Lmax = s.pot m Lmax := by
  unfold TState.pot TState.unl TState.uns TState.keys TState.tally
  rw [ht, he, hr]

/-- A probe's outcome learns an unlearned edge out of a leaf, at a leaf, or keeps the hypothesis
and the records, adding at most its own record. -/
theorem tallyPre_cases (C : TallyCfg) (s : TState α) (x : FreeMonoid α) :
    (∃ p c t u, p ∈ s.tree.paths ∧ t ∈ s.tree.paths ∧ s.edges p c = none
      ∧ tallyPre C cut s x = s.setEdge p c (t, u))
    ∨ ∃ s₂ : TState α, s₂.tree = s.tree ∧ s₂.edges = s.edges ∧ s₂.recs = s.recs
      ∧ s₂.version = s.version
      ∧ (tallyPre C cut s x = s₂ ∨ ∃ p c r, recordBy cut C.k (s.tree, s.edges) x
          = some ((p, c, r.2), r.1) ∧ tallyPre C cut s x = s₂.addRec p c r) := by
  unfold tallyPre
  split
  · split
    · split
      · rename_i u _ _ _ c p _ hp _ _ t ht he
        exact .inl ⟨p, c, t, u, DTree.sift_mem_paths _ _ _ hp, DTree.sift_mem_paths _ _ _ ht,
          he, rfl⟩
      · refine (.inr ⟨_, ?_, ?_, ?_, ?_, .inl rfl⟩) <;> rfl
    · refine (.inr ⟨_, ?_, ?_, ?_, ?_, .inl rfl⟩) <;> rfl
  · split
    · rename_i p c t sp hr
      refine (.inr ⟨_, ?_, ?_, ?_, ?_, .inr ⟨p, c, (sp, t), hr, rfl⟩⟩) <;> rfl
    · refine (.inr ⟨_, ?_, ?_, ?_, ?_, .inl rfl⟩) <;> rfl
  all_goals refine (.inr ⟨_, ?_, ?_, ?_, ?_, .inl rfl⟩) <;> rfl

/-- A probe's outcome never raises `pot`, and a learning lowers it and starts a stretch. -/
theorem tallyPre_spec (C : TallyCfg) (s : TState α) (x : FreeMonoid α) :
    (tallyPre C cut s x).tree = s.tree ∧ s.version ≤ (tallyPre C cut s x).version
      ∧ (tallyPre C cut s x).pot C.m Lmax + ((tallyPre C cut s x).version - s.version)
        ≤ s.pot C.m Lmax
      ∧ ((tallyPre C cut s x).version ≠ s.version → (tallyPre C cut s x).Fresh) := by
  rcases tallyPre_cases cut C s x with ⟨p, c, t, u, hp, -, he, h⟩ | ⟨s₂, ht, he, hr, hv, h⟩
  · rw [h]
    have := pot_learn C.m Lmax s (t := t) (w := u) hp he
    refine ⟨rfl, by simp [TState.setEdge], ?_, fun _ => ⟨rfl, rfl, rfl, rfl, rfl⟩⟩
    simp only [show (s.setEdge p c (t, u)).version = s.version + 1 from rfl]
    omega
  · have h2 := pot_congr C.m Lmax ht he hr
    rcases h with h | ⟨p, c, r, -, h⟩ <;> rw [h]
    · exact ⟨ht, hv.ge, by rw [hv, h2]; simp, fun h => absurd hv h⟩
    · have := pot_addRec C.m Lmax s₂ p c r
      exact ⟨ht, hv.ge, by
        rw [show (s₂.addRec p c r).version = s₂.version from rfl, hv]; simp; omega,
        fun h => absurd hv h⟩

theorem tallyLook_fresh (C : TallyCfg) (hn₀ : 1 ≤ C.n₀) {s : TState α} (hs : s.Fresh) :
    tallyLook C s = none := by
  obtain ⟨h0, -⟩ := hs
  unfold tallyLook
  have : ∀ θ h, rateSide θ C.a C.n₀ s.n h = none := fun θ h => by
    simp [rateSide, h0]; omega
  rw [this, this, if_neg (by simp), dif_neg (by rintro ⟨e, -, h, -⟩; omega), if_neg (by simp)]

end Step

end OrthoDFA
