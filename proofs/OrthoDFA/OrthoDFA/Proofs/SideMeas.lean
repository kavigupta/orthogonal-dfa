import OrthoDFA.Proofs.SplitLaw

/-!
# The sides are measurable in the hash

A side's subsequence reads the hash only at the draws it has passed, so it is measurable in the
hash and the draws jointly.
-/

namespace OrthoDFA

open MeasureTheory Thin

variable {S : Type*} [Stringlike S]

lemma entries_congr_set {A A' : Set S} {w : ℕ → S} {n : ℕ} (h : ∀ i < n, (w i ∈ A ↔ w i ∈ A')) :
    drawsIn A w n = drawsIn A' w n := by
  classical
  unfold drawsIn
  refine List.filter_congr (fun s hs => ?_)
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hs
  simp [h i (List.mem_range.1 hi)]

/-- Every event about a side's first drawsIn among `n` draws is measurable in hash and draws. -/
lemma measurableSet_entries_onSide (H : Finset S) (b : Bool) (n : ℕ) (P : List S → Prop) :
    MeasurableSet {p : (S → Bool) × (ℕ → S) | P (drawsIn (onSide H b p.1) p.2 n)} := by
  classical
  rcases isEmpty_or_nonempty S with hS | ⟨⟨s₀⟩⟩
  · have : IsEmpty ((S → Bool) × (ℕ → S)) := by
      refine ⟨fun p => ?_⟩
      exact hS.false (p.2 0)
    exact Set.Subsingleton.measurableSet (fun a _ => isEmptyElim a)
  have hcov : {p : (S → Bool) × (ℕ → S) | P (drawsIn (onSide H b p.1) p.2 n)}
      = ⋃ t : Fin n → S, ({p : (S → Bool) × (ℕ → S) | ∀ i : Fin n, p.2 i = t i}
        ∩ {p | P (drawsIn (onSide H b p.1)
            (fun k => if h : k < n then t ⟨k, h⟩ else s₀) n)}) := by
    ext p
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    constructor
    · intro hp
      refine ⟨fun i => p.2 i, fun _ => rfl, ?_⟩
      rwa [entries_congr (onSide H b p.1) (w := p.2)
          (w' := fun k => if h : k < n then (fun i : Fin n => p.2 i) ⟨k, h⟩ else s₀)
          (fun k hk => by beta_reduce; rw [dif_pos hk])] at hp
    · rintro ⟨t, ht, hp⟩
      rwa [entries_congr (onSide H b p.1) (w' := p.2) (fun k hk => by
        simp only [dif_pos hk]; exact (ht ⟨k, hk⟩).symm)] at hp
  rw [hcov]
  refine MeasurableSet.iUnion (fun t => MeasurableSet.inter ?_ ?_)
  · have hm : Measurable (fun (p : (S → Bool) × (ℕ → S)) (i : Fin n) => p.2 i.val) := by
      fun_prop
    exact hm (measurableSet_singleton t) |>.congr (by ext p; simp [funext_iff])
  · set y : ℕ → S := fun k => if h : k < n then t ⟨k, h⟩ else s₀ with hy
    have he : ∀ g : S → Bool, drawsIn (onSide H b g) y n
        = drawsIn (onSide H b (fun s => if s ∈ Finset.univ.image t then g s else false)) y n := by
      intro g
      refine entries_congr_set (fun k hk => ?_)
      have hmem : y k ∈ Finset.univ.image t := by
        simp only [hy, dif_pos hk]; exact Finset.mem_image_of_mem _ (Finset.mem_univ _)
      simp only [onSide, Set.mem_setOf_eq, if_pos hmem]
    have hm : Measurable (fun (p : (S → Bool) × (ℕ → S)) (i : Fin n) => p.1 (t i)) := by fun_prop
    have hset : {p : (S → Bool) × (ℕ → S) | P (drawsIn (onSide H b p.1) y n)}
        = (fun (p : (S → Bool) × (ℕ → S)) (i : Fin n) => p.1 (t i)) ⁻¹'
          {c : Fin n → Bool | ∃ g : S → Bool, (∀ i, g (t i) = c i)
            ∧ P (drawsIn (onSide H b g) y n)} := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_preimage]
      refine ⟨fun h => ⟨p.1, fun _ => rfl, h⟩, ?_⟩
      rintro ⟨g, hg, hP⟩
      rw [he] at hP ⊢
      convert hP using 4
      split_ifs with hs
      · obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hs
        exact (hg i).symm
      · rfl
    rw [hset]
    exact hm (Set.to_countable _).measurableSet

/-- A side's subsequence is measurable in hash, pad and draws. -/
lemma measurable_sub_onSide (H : Finset S) (b : Bool) :
    Measurable (fun p : ((S → Bool) × (ℕ → S)) × S => nthDrawIn (onSide H b p.1.1) p.2 p.1.2) := by
  classical
  refine measurable_from_prod_countable_left (fun d => ?_)
  refine measurable_pi_lambda _ (fun k => measurable_to_countable' (fun s => ?_))
  have hcov : (fun p : (S → Bool) × (ℕ → S) => nthDrawIn (onSide H b p.1) d p.2 k) ⁻¹' {s}
      = (⋃ n : ℕ, {p : (S → Bool) × (ℕ → S) | (fun E : List S => k < E.length ∧ E.getD k d = s)
          (drawsIn (onSide H b p.1) p.2 n)})
        ∪ ((⋂ n : ℕ, {p : (S → Bool) × (ℕ → S) | (fun E : List S => ¬ k < E.length)
          (drawsIn (onSide H b p.1) p.2 n)}) ∩ {_p | d = s}) := by
    ext p
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_union, Set.mem_iUnion,
      Set.mem_inter_iff, Set.mem_iInter, Set.mem_setOf_eq]
    constructor
    · intro hp
      by_cases h : ∃ n, k < (drawsIn (onSide H b p.1) p.2 n).length
      · obtain ⟨n, hn⟩ := h
        exact Or.inl ⟨n, hn, (sub_eq_getD _ d p.2 hn).symm.trans hp⟩
      · refine Or.inr ⟨fun n hn => h ⟨n, hn⟩, ?_⟩
        rw [← hp, nthDrawIn, dif_neg h]
    · rintro (⟨n, hn, he⟩ | ⟨hno, he⟩)
      · rw [sub_eq_getD _ d p.2 hn, he]
      · rw [nthDrawIn, dif_neg (fun ⟨n, hn⟩ => hno n hn)]
        exact he
  rw [hcov]
  refine (MeasurableSet.iUnion (fun n => measurableSet_entries_onSide H b n
    (fun E => k < E.length ∧ E.getD k d = s))).union
    ((MeasurableSet.iInter (fun n => measurableSet_entries_onSide H b n
      (fun E => ¬ k < E.length))).inter ?_)
  by_cases hds : d = s
  · simp [hds]
  · simp [hds]

end OrthoDFA
