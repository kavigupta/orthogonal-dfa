import OrthoDFA.Proofs.Level4RiskDefs
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 4's replay is spread

Every string a probe walked from `k` reads is a prefix of the probe at least `k` long, extended by
a midfix, in every branch of its query tree. So the replay, walked from its anchor, keeps a prefix
of its draw past the anchor, whatever the hypothesis.
-/

namespace OrthoDFA

namespace QTree

variable {Λ ι V β γ : Type*}

/-- Every key read in every branch satisfies `P`. -/
def AllAsk (P : ι → Prop) : QTree Λ ι V β → Prop
  | done _ => True
  | ask _ w next => P w ∧ ∀ v, AllAsk P (next v)

theorem AllAsk.mono {P P' : ι → Prop} (h : ∀ w, P w → P' w) :
    ∀ {t : QTree Λ ι V β}, AllAsk P t → AllAsk P' t
  | done _, _ => trivial
  | ask _ _ _, ⟨hw, hn⟩ => ⟨h _ hw, fun v => AllAsk.mono h (hn v)⟩

theorem AllAsk.bind {P : ι → Prop} {f : β → QTree Λ ι V γ} (hf : ∀ b, AllAsk P (f b)) :
    ∀ {t : QTree Λ ι V β}, AllAsk P t → AllAsk P (t.bind f)
  | done b, _ => hf b
  | ask _ _ _, ⟨hw, hn⟩ => ⟨hw, fun v => AllAsk.bind hf (hn v)⟩

theorem AllAsk.map {P : ι → Prop} (f : β → γ) {t : QTree Λ ι V β} (h : AllAsk P t) :
    AllAsk P (t.map f) := by
  unfold QTree.map
  exact AllAsk.bind (f := fun b => done (f b)) (fun _ => trivial) h

theorem AllAsk.firsts [DecidableEq ι] {P : ι → Prop} (rd : ι → V) :
    ∀ (S : Finset ι) {t : QTree Λ ι V β}, AllAsk P t → ∀ q ∈ firsts rd S t, P q.2
  | _, done _, _, q, hq => by simp [QTree.firsts] at hq
  | S, ask ℓ w next, ⟨hw, hn⟩, q, hq => by
    simp only [QTree.firsts] at hq
    split_ifs at hq
    · exact AllAsk.firsts rd S (hn _) q hq
    · rcases List.mem_cons.1 hq with rfl | hq
      · exact hw
      · exact AllAsk.firsts rd _ (hn _) q hq

end QTree

open QTree

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem siftQuery_allAsk (ℓ : ReadLabel α) : ∀ (T : DTree α) (u : FreeMonoid α),
    AllAsk (fun z => ∃ m, z = u * m) (siftQuery ℓ T u)
  | .leaf, _ => trivial
  | .node m r a, u => ⟨⟨m, rfl⟩, fun v => by
      simp only
      split
      · trivial
      · exact (siftQuery_allAsk ℓ a u).map _
      · exact (siftQuery_allAsk ℓ r u).map _⟩

/-- A read past `k`: a prefix of `x` at least `k` long, extended. -/
def PastK (x : FreeMonoid α) (k : ℕ) (z : FreeMonoid α) : Prop :=
  ∃ i, k ≤ i ∧ ∃ m, z = prefixOf x i * m

theorem siftQuery_pastK (x : FreeMonoid α) {k i : ℕ} (hi : k ≤ i) (ℓ : ReadLabel α) (T : DTree α) :
    AllAsk (PastK x k) (siftQuery ℓ T (prefixOf x i)) :=
  (siftQuery_allAsk ℓ T _).mono fun _ ⟨m, hm⟩ => ⟨i, hi, m, hm⟩

theorem bracketQuery_pastK (x : FreeMonoid α) (k : ℕ) (agreementQuery : ℕ
    → ProbeQuery α (Option Bool))
    (ps : List (List Bool)) :
    ∀ (fuel lo hi : ℕ), k ≤ lo → (∀ p, k ≤ p → AllAsk (PastK x k) (agreementQuery p)) →
      AllAsk (PastK x k) (bracketQuery agreementQuery ps fuel lo hi)
  | 0, _, _, _, _ => trivial
  | fuel + 1, lo, hi, hk, hag => by
    have hat : ∀ p, k ≤ p → AllAsk (PastK x k) (agAt agreementQuery lo hi p) := by
      intro p hp
      unfold agAt
      split_ifs
      · trivial
      · trivial
      · exact hag p hp
    simp only [bracketQuery]
    split_ifs with hlt
    · refine (hat _ (by omega)).bind fun am => ?_
      split
      · exact bracketQuery_pastK x k agreementQuery ps fuel _ hi (by omega) hag
      · exact bracketQuery_pastK x k agreementQuery ps fuel lo _ hk hag
      · refine (hat _ (by omega)).bind fun al => ?_
        split
        · trivial
        · refine (hat _ (by omega)).bind fun ar => ?_
          split
          · trivial
          · trivial
          · exact bracketQuery_pastK x k agreementQuery ps fuel _ hi (by omega) hag
          · exact bracketQuery_pastK x k agreementQuery ps fuel lo _ hk hag
    · trivial

theorem probeQuery_pastK (T : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) :
    AllAsk (PastK x k) (probeQuery T edges k x) := by
  have hag : ∀ p₀ walkAt p, k ≤ p → AllAsk (PastK x k) (agreementQuery T edges k x p₀ walkAt p) :=
    fun p₀ walkAt p hp => (siftQuery_pastK x hp _ T).map _
  unfold probeQuery
  split_ifs
  · trivial
  refine (siftQuery_pastK x le_rfl _ T).bind fun a => ?_
  split
  · trivial
  · split
    · refine (siftQuery_pastK x ?_ _ T).bind fun r => ?_
      · omega
      split
      · trivial
      · split_ifs
        · trivial
        · exact bracketQuery_pastK x k _ _ _ _ _ le_rfl (hag _ _)
    · refine (siftQuery_pastK x (by omega) _ T).bind fun r1 => ?_
      split
      · trivial
      · refine (siftQuery_pastK x (by omega) _ T).bind fun r2 => ?_
        split
        · trivial
        · split_ifs
          · trivial
          · exact bracketQuery_pastK x k _ _ _ _ _ le_rfl (hag _ _)

theorem replayAt_prefix (rd : FreeMonoid α → ARU) (T : DTree α) (edges : EdgeTable α)
    (w : FreeMonoid α) (e : ℕ) (x : FreeMonoid α) (hx : x ∈ replayAt rd T edges w e) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  rw [replayAt_eq] at hx
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hx
  have hq' := (List.mem_filter.1 hq).1
  have hw : e ≤ w.toList.length := by
    by_contra hlt
    simp [firstsBy, probeQuery, show w.toList.length < e by omega, QTree.firsts] at hq'
  obtain ⟨i, hei, m, hm⟩ := (probeQuery_pastK T edges e w).firsts _ ∅ q hq'
  rw [hm]
  have hi : prefixOf w i = prefixOf w (min i w.toList.length) := by
    rcases le_total i w.toList.length with h | h
    · rw [min_eq_left h]
    · rw [min_eq_right h]
      simp only [prefixOf, List.take_of_length_le h, List.take_length]
  have hlen : (prefixOf w (min i w.toList.length)).toList.length = min i w.toList.length := by
    simp [prefixOf, List.length_take]
  refine ⟨min i w.toList.length, by omega, ?_, ?_⟩
  · rw [hi, FreeMonoid.toList_mul, List.length_append, hlen]; omega
  · rw [hi, FreeMonoid.toList_mul, List.take_left' hlen]
    simp only [prefixOf, FreeMonoid.toList_ofList]

/-- The tree's root reads its string alone. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

theorem splitAt_rootOne {T : DTree α} (hT : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := hT
  rcases p with _ | ⟨_ | _, p⟩
  · exact ⟨r, a, rfl⟩
  · exact ⟨_, a, rfl⟩
  · exact ⟨r, _, rfl⟩

theorem recordProbe_tree (C : Level4Settings) (cut : FreeMonoid α → Option Bool) (s : Level4State α)
    (x : FreeMonoid α) : (recordProbe C cut s x).tree = s.tree := by
  unfold recordProbe
  simp only
  repeat' split
  all_goals rfl

theorem fixEdge_rootOne (cut : FreeMonoid α → Option Bool) (m : ℕ) {s : Level4State α}
    (hs : RootOne s.tree) (p : List Bool) (c : α) (t : List Bool) :
    RootOne (fixEdge cut m s p c t).tree := by
  unfold fixEdge
  simp only
  repeat' split
  all_goals first | exact splitAt_rootOne hs _ _ | exact hs

theorem applyProbe_rootOne (C : Level4Settings) (cut : FreeMonoid α → Option Bool)
    {s : Level4State α}
    (hs : RootOne s.tree) (x : FreeMonoid α) :
    (∀ s', applyProbe C cut s x = .inl s' → RootOne s'.tree)
      ∧ ∀ e s', applyProbe C cut s x = .inr (e, s') → RootOne s'.tree := by
  have h1 : RootOne (recordProbe C cut s x).tree := by rw [recordProbe_tree]; exact hs
  unfold applyProbe settleOne
  simp only
  constructor
  · intro s' h
    repeat' split at h
    all_goals first
      | (injection h with h; rw [← h]; exact fixEdge_rootOne cut _ h1 _ _ _)
      | (injection h with h; rw [← h]; exact h1)
      | cases h
  · intro e s' h
    repeat' split at h
    all_goals first
      | (injection h with h; injection h with _ h; rw [← h]; exact fixEdge_rootOne cut _ h1 _ _ _)
      | (injection h with h; injection h with _ h; rw [← h]; exact h1)
      | cases h

/-- A run from a state where `I` holds, `I` kept by every step, ends where `I` holds. -/
theorem runEnds_inv {S X E : Type*} {step : S → X → S ⊕ (E × S)} {Q : E → S → Prop}
    (I : S → Prop) (hI : ∀ s x, I s → (∀ s', step s x = .inl s' → I s')
      ∧ ∀ e s', step s x = .inr (e, s') → I s') :
    ∀ (s : S) (l : List X), I s → RunEnds step Q s l → RunEnds step (fun e s' => Q e s' ∧ I s') s l
  | _, [], _, h => h
  | s, x :: xs, hs, h => by
    simp only [RunEnds] at h ⊢
    rcases hst : step s x with s' | ⟨e, s'⟩ <;> rw [hst] at h
    · exact runEnds_inv I hI s' xs ((hI s x hs).1 s' hst) h
    · exact ⟨h, (hI s x hs).2 e s' hst⟩

theorem replayAt_root (rd : FreeMonoid α → ARU) {T : DTree α} (hT : RootOne T)
    (edges : EdgeTable α) {w : FreeMonoid α} {e : ℕ} (he : e ≤ w.toList.length)
    (hu : rd (prefixOf w e) = .undecided) : replayAt rd T edges w e ≠ [] := by
  obtain ⟨r, a, rfl⟩ := hT
  rw [replayAt_eq]; unfold firstsBy probeQuery
  rw [if_neg (by omega)]
  simp [siftQuery, QTree.bind, QTree.firsts, hu, ARU.cut]

theorem replayAt_spread (rd : FreeMonoid α → ARU) (T : DTree α) (edges : EdgeTable α)
    (D : MeasureTheory.Measure (FreeMonoid α)) [MeasureTheory.IsFiniteMeasure D] (s L : ℕ) :
    Spread D s L (replayAt rd T edges) :=
  spread_of_prefix D s L _ (replayAt_prefix rd T edges)

end OrthoDFA
