import OrthoDFA.TallyRound
import OrthoDFA.Proofs.RoundEnd

/-!
# The tally replay is spread

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

theorem siftQ_allAsk (ℓ : PLabel α) : ∀ (T : DTree α) (u : FreeMonoid α),
    AllAsk (fun z => ∃ m, z = u * m) (siftQ ℓ T u)
  | .leaf, _ => trivial
  | .node m r a, u => ⟨⟨m, rfl⟩, fun v => by
      simp only
      split
      · trivial
      · exact (siftQ_allAsk ℓ a u).map _
      · exact (siftQ_allAsk ℓ r u).map _⟩

/-- A read past `k`: a prefix of `x` at least `k` long, extended. -/
def PastK (x : FreeMonoid α) (k : ℕ) (z : FreeMonoid α) : Prop :=
  ∃ i, k ≤ i ∧ ∃ m, z = prefixOf x i * m

theorem siftQ_pastK (x : FreeMonoid α) {k i : ℕ} (hi : k ≤ i) (ℓ : PLabel α) (T : DTree α) :
    AllAsk (PastK x k) (siftQ ℓ T (prefixOf x i)) :=
  (siftQ_allAsk ℓ T _).mono fun _ ⟨m, hm⟩ => ⟨i, hi, m, hm⟩

theorem bracketQ_pastK (x : FreeMonoid α) (k : ℕ) (agQ : ℕ → PQ α (Option Bool))
    (ps : List (List Bool)) :
    ∀ (fuel lo hi : ℕ), k ≤ lo → (∀ p, k ≤ p → AllAsk (PastK x k) (agQ p)) →
      AllAsk (PastK x k) (bracketQ agQ ps fuel lo hi)
  | 0, _, _, _, _ => trivial
  | fuel + 1, lo, hi, hk, hag => by
    have hat : ∀ p, k ≤ p → AllAsk (PastK x k) (agAt agQ lo hi p) := by
      intro p hp
      unfold agAt
      split_ifs
      · trivial
      · trivial
      · exact hag p hp
    simp only [bracketQ]
    split_ifs with hlt
    · refine (hat _ (by omega)).bind fun am => ?_
      split
      · exact bracketQ_pastK x k agQ ps fuel _ hi (by omega) hag
      · exact bracketQ_pastK x k agQ ps fuel lo _ hk hag
      · refine (hat _ (by omega)).bind fun al => ?_
        split
        · trivial
        · refine (hat _ (by omega)).bind fun ar => ?_
          split
          · trivial
          · trivial
          · exact bracketQ_pastK x k agQ ps fuel _ hi (by omega) hag
          · exact bracketQ_pastK x k agQ ps fuel lo _ hk hag
    · trivial

theorem probeQ_pastK (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    AllAsk (PastK x k) (probeQ T edges k x) := by
  have hag : ∀ p₀ walkAt p, k ≤ p → AllAsk (PastK x k) (agQ T edges k x p₀ walkAt p) :=
    fun p₀ walkAt p hp => (siftQ_pastK x hp _ T).map _
  unfold probeQ
  split_ifs
  · trivial
  refine (siftQ_pastK x le_rfl _ T).bind fun a => ?_
  split
  · trivial
  · split
    · refine (siftQ_pastK x ?_ _ T).bind fun r => ?_
      · omega
      split
      · trivial
      · split_ifs
        · trivial
        · exact bracketQ_pastK x k _ _ _ _ _ le_rfl (hag _ _)
    · refine (siftQ_pastK x (by omega) _ T).bind fun r1 => ?_
      split
      · trivial
      · refine (siftQ_pastK x (by omega) _ T).bind fun r2 => ?_
        split
        · trivial
        · split_ifs
          · trivial
          · exact bracketQ_pastK x k _ _ _ _ _ le_rfl (hag _ _)

theorem tallyReplay_prefix (rd : FreeMonoid α → ARU) (T : DTree α) (edges : Edges α)
    (w : FreeMonoid α) (e : ℕ) (x : FreeMonoid α) (hx : x ∈ tallyReplay rd T edges w e) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  unfold tallyReplay at hx
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hx
  have hq' := (List.mem_filter.1 hq).1
  have hw : e ≤ w.toList.length := by
    by_contra hlt
    simp [firstsBy, probeQ, show w.toList.length < e by omega, QTree.firsts] at hq'
  obtain ⟨i, hei, m, hm⟩ := (probeQ_pastK T edges e w).firsts _ ∅ q hq'
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

theorem tallyPre_tree (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (s : TState α)
    (x : FreeMonoid α) : (tallyPre C cut s x).tree = s.tree := by
  unfold tallyPre
  simp only
  repeat' split
  all_goals rfl

theorem fixEdge_rootOne (cut : FreeMonoid α → Option Bool) (m : ℕ) {s : TState α}
    (hs : RootOne s.tree) (p : List Bool) (c : α) (t : List Bool) :
    RootOne (fixEdge cut m s p c t).tree := by
  unfold fixEdge
  simp only
  repeat' split
  all_goals first | exact splitAt_rootOne hs _ _ | exact hs

theorem tallyStep_rootOne (C : TallyCfg) (cut : FreeMonoid α → Option Bool) {s : TState α}
    (hs : RootOne s.tree) (x : FreeMonoid α) :
    (∀ s', tallyStep C cut s x = .inl s' → RootOne s'.tree)
      ∧ ∀ e s', tallyStep C cut s x = .inr (e, s') → RootOne s'.tree := by
  have h1 : RootOne (tallyPre C cut s x).tree := by rw [tallyPre_tree]; exact hs
  unfold tallyStep settleOne
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

theorem tallyReplay_root (rd : FreeMonoid α → ARU) {T : DTree α} (hT : RootOne T)
    (edges : Edges α) {w : FreeMonoid α} {e : ℕ} (he : e ≤ w.toList.length)
    (hu : rd (prefixOf w e) = .undecided) : tallyReplay rd T edges w e ≠ [] := by
  obtain ⟨r, a, rfl⟩ := hT
  unfold tallyReplay firstsBy probeQ
  rw [if_neg (by omega)]
  simp [siftQ, QTree.bind, QTree.firsts, hu, ARU.cut]

theorem tallyReplay_spread (rd : FreeMonoid α → ARU) (T : DTree α) (edges : Edges α)
    (D : MeasureTheory.Measure (FreeMonoid α)) [MeasureTheory.IsFiniteMeasure D] (L : ℕ) :
    Spread D L (tallyReplay rd T edges) :=
  spread_of_prefix D L _ (tallyReplay_prefix rd T edges)

end OrthoDFA
