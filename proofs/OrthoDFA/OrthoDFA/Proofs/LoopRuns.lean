import OrthoDFA.Proofs.KappaCount
import OrthoDFA.Proofs.ProbeKeys

/-!
# Level 3's run, whatever it reads

Every step keeps `Inv`, adds at most one leaf, and keeps each learned edge at a leaf with a witness
a probe's prefix past its first `k` letters (`RunInv`). So after `T` probes the tree has at most
`T + 2` leaves and midfixes at most `T` long, and the run reads only a probe's prefix past its
first `k` letters followed by a midfix: reads agreeing there give the same run
(`runProbes_congr_zone`).
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## Trees in the class -/

/-- Every midfix of `T` is at most `n` long. -/
def MidsLe (n : ℕ) (T : DTree α) : Prop := ∀ m ∈ mids T, m.toList.length ≤ n

theorem MidsLe.mono {n n' : ℕ} {T : DTree α} (h : MidsLe n T) (hn : n ≤ n') : MidsLe n' T :=
  fun m hm => (h m hm).trans hn

theorem midsLe_splitAt {n : ℕ} {T : DTree α} (hT : MidsLe n T) {d : FreeMonoid α}
    (hd : d.toList.length ≤ n + 1) (p : List Bool) : MidsLe (n + 1) (T.splitAt d p) := by
  classical
  intro m hm
  rcases Finset.mem_insert.1 (mids_splitAt d T p hm) with rfl | hm
  · exact hd
  · exact (hT m hm).trans (Nat.le_succ n)

theorem midfixAt_length_le {n : ℕ} {T : DTree α} (hT : MidsLe n T) (q : List Bool) :
    (T.midfixAt q).toList.length ≤ n := by
  rcases midfixAt_mem T q with h | h
  · rw [h]; simp
  · exact hT _ h

theorem classSet_props : ∀ (n : ℕ) (T : DTree α), T ∈ classSet n →
    MidsLe n T ∧ T.leaves.length ≤ n + 2
  | 0, T, h => by
    classical
    simp only [classSet, Finset.mem_singleton] at h
    subst h
    refine ⟨fun m hm => ?_, by simp [DTree.leaves]⟩
    simp only [mids, Finset.mem_insert, Finset.notMem_empty, Finset.empty_union, or_false] at hm
    subst hm
    simp
  | n + 1, T, h => by
    classical
    simp only [classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T₀, hT₀, ⟨p, c, t, t₀⟩, hq, rfl⟩
    · obtain ⟨h1, h2⟩ := classSet_props n T h
      exact ⟨h1.mono (Nat.le_succ n), by omega⟩
    · obtain ⟨h1, h2⟩ := classSet_props n T₀ hT₀
      simp only [Finset.mem_product, List.mem_toFinset, Finset.mem_univ, true_and] at hq
      refine ⟨midsLe_splitAt h1 ?_ p, ?_⟩
      · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
          List.length_singleton]
        have := midfixAt_length_le h1 (commonPrefix t t₀)
        omega
      · rw [DTree.length_leaves_splitAt _ _ _ hq.1]
        omega

/-! ## The run's invariant -/

section Run

variable (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- `Inv`, with every learned edge at a leaf and its witness, followed by its letter, a prefix of
one of the probes `X` past its first `walkStart` letters. -/
structure RunInv (X : List (FreeMonoid α)) (s : Level3State α) : Prop where
  inv : Inv read s
  keys : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves
  wit : ∀ e t w, s.edges e = some (t, w) → ∃ x ∈ X, ∃ i, C.walkStart ≤ i ∧
    i < x.toList.length ∧ w = prefixOf x i ∧ x.toList[i]? = some e.2

theorem runInv_start (X : List (FreeMonoid α)) : RunInv read C X initialState where
  inv := inv_start read
  keys := fun _ _ _ h => by simp [initialState, newHypothesis] at h
  wit := fun _ _ _ h => by simp [initialState, newHypothesis] at h

theorem retarget_some {T' : DTree α} {p : List Bool} {E : Edges α} {e : List Bool × α}
    {t : List Bool} {w : FreeMonoid α} (h : retarget read T' p E e = some (t, w)) :
    ¬ p.isPrefixOf e.1 = true ∧ ∃ t₀, E e = some (t₀, w) := by
  unfold retarget at h
  split_ifs at h with hp
  rcases he : E e with _ | ⟨t₀, w₀⟩ <;> rw [he] at h
  · cases h
  · refine ⟨hp, t₀, ?_⟩
    dsimp only at h
    split_ifs at h
    · rcases hs : T'.sift read (w₀ * FreeMonoid.of e.2) with t' | z <;>
        rw [hs] at h <;> simp at h
      rw [h.2]
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      rw [h.2]

/-- A step keeps the invariant and adds at most one leaf. -/
theorem runInv_step {X : List (FreeMonoid α)} {s s' : Level3State α} {x : FreeMonoid α}
    (hx : x ∈ X) (hs : RunInv read C X s) (h : applyProbe read C s x = .inl s') :
    RunInv read C X s' ∧ s'.tree.leaves.length ≤ s.tree.leaves.length + 1 := by
  rcases step_cases read C hs.inv h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · refine ⟨⟨⟨edgesOK_upd read hs.inv.edges hu huc, hs.inv.cls, hs.inv.two⟩, ?_, ?_⟩, by
      simp [newHypothesis]⟩
    · intro e t' w he
      simp only [newHypothesis] at he ⊢
      by_cases hep : e = (p, c)
      · subst hep; exact hp
      · rw [Function.update_of_ne hep] at he
        exact hs.keys e t' w he
    · intro e t' w he
      simp only [newHypothesis] at he
      by_cases hep : e = (p, c)
      · subst hep
        rw [Function.update_self, Option.some.injEq, Prod.mk.injEq] at he
        obtain ⟨-, rfl⟩ := he
        obtain ⟨i, hki, hi, rfl, hc⟩ := probe_prefix read s.tree s.edges C.walkStart x (Or.inr ho)
        exact ⟨x, hx, i, hki, hi, rfl, hc⟩
      · rw [Function.update_of_ne hep] at he
        exact hs.wit e t' w he
  · have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have ht := DTree.sift_inl_mem read _ _ t huc
    obtain ⟨hw₀, hw₀c⟩ := hs.inv.edges p hp c t₀ w₀ hE
    have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
    refine ⟨⟨⟨edgesOK_retarget read hs.inv.edges hp _, ?_, ?_⟩, ?_, ?_⟩, ?_⟩
    · simp only [splitLeaf, newHypothesis]
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hs.inv.two; omega]
      exact classSet_split hs.inv.cls hp ht ht₀
    · simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.inv.two; omega
    · intro e t' w he
      simp only [splitLeaf, newHypothesis] at he ⊢
      obtain ⟨hpe, t₁, he₁⟩ := retarget_some read he
      have hq := hs.keys e t₁ w he₁
      refine DTree.mem_leaves_splitAt_of_ne _ s.tree p e.1 hq fun hpq => hpe ?_
      rw [hpq]
      exact List.isPrefixOf_iff_prefix.2 (List.prefix_refl _)
    · intro e t' w he
      simp only [splitLeaf, newHypothesis] at he
      obtain ⟨-, t₁, he₁⟩ := retarget_some read he
      exact hs.wit e t₁ w he₁
    · simp only [splitLeaf, newHypothesis]; rw [hlen]
  · exact ⟨⟨⟨hs.inv.edges, hs.inv.cls, hs.inv.two⟩, hs.keys, hs.wit⟩, by simp [countProbe]⟩

theorem runInv_runProbes {X : List (FreeMonoid α)} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv read C X s →
      RunInv read C X (runProbes read C s xs).1
        ∧ (runProbes read C s xs).1.tree.leaves.length ≤ s.tree.leaves.length + xs.length
  | [], s, _, hs => ⟨hs, by simp [runProbes]⟩
  | x :: xs, s, hX, hs => by
    simp only [runProbes]
    rcases ha : applyProbe read C s x with s' | e
    · obtain ⟨hs', hl⟩ := runInv_step read C (hX x (by simp)) hs ha
      obtain ⟨h1, h2⟩ := runInv_runProbes xs s' (fun y hy => hX y (by simp [hy])) hs'
      refine ⟨h1, ?_⟩
      simp only [List.length_cons]
      omega
    · exact ⟨hs, by simp⟩

/-- After `T` probes the tree has at most `T + 2` leaves and midfixes at most `T` long. -/
theorem runProbes_tree (xs : List (FreeMonoid α)) :
    RunInv read C xs (runProbes read C initialState xs).1
      ∧ (runProbes read C initialState xs).1.tree.leaves.length ≤ xs.length + 2
      ∧ MidsLe xs.length (runProbes read C initialState xs).1.tree := by
  obtain ⟨h1, h2⟩ := runInv_runProbes read C xs initialState (fun _ h => h)
    (runInv_start read C xs)
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  rw [h0] at h2
  refine ⟨h1, by omega, ?_⟩
  exact (classSet_props _ _ h1.inv.cls).1.mono (by omega)

/-! ## What the run reads -/

/-- Reads agreeing on every prefix of a probe of `X` past its first `walkStart` letters, followed
by at most `B` letters, agree on what a step reads, where the step keeps the midfixes within
`B`. -/
theorem step_reads_agree {r r' : FreeMonoid α → ARU} {X : List (FreeMonoid α)} {B : ℕ}
    (h : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ B → r (prefixOf x j * m) = r' (prefixOf x j * m))
    {s : Level3State α} {x : FreeMonoid α} (hxX : x ∈ X) (hs : RunInv r C X s)
    (hB : s.tree.leaves.length - 2 + 1 ≤ B) :
    ∀ z ∈ Kappa.probeReads r s.tree s.edges C.walkStart x ∪ Eps.splitReads r C s x, r z = r' z := by
  classical
  have hmid := (classSet_props _ _ hs.inv.cls).1
  intro z hz
  rcases Finset.mem_union.1 hz with hz | hz
  · obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc C.walkStart x.toList.length,
        ∃ m ∈ mids s.tree, prefixOf x p * m = z := by
      have := Kappa.probeReads_pot s.edges C.walkStart hz
      simpa [pot] using this
    obtain ⟨hp1, hp2⟩ := Finset.mem_Icc.1 hp
    exact h x hxX p hp1 hp2 m ((hmid m hm).trans (by omega))
  · unfold Eps.splitReads at hz
    rcases ho : probeOrAgree r s.tree s.edges C.walkStart x with
      _ | ⟨p, c, u, t⟩ | _ | _ | _ <;> rw [ho] at hz <;> simp only [Finset.notMem_empty] at hz
    rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩ <;> rw [hE] at hz
    · simp at hz
    simp only [Eps.resifts, Finset.mem_biUnion] at hz
    obtain ⟨e, -, hz⟩ := hz
    rcases he : s.edges e with _ | ⟨t', w⟩ <;> rw [he] at hz
    · simp at hz
    obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
    obtain ⟨x', hx', i, hki, hi, rfl, hc⟩ := hs.wit e t' w he
    rw [prefixOf_mul_of hc]
    refine h x' hx' (i + 1) (by omega) (by omega) m ?_
    have hd : (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)).toList.length
        ≤ s.tree.leaves.length - 2 + 1 := by
      simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
        List.length_singleton]
      have := midfixAt_length_le hmid (commonPrefix t t₀)
      omega
    exact (midsLe_splitAt hmid hd p m hm).trans (by omega)

/-- Reads agreeing on every prefix of a probe of `X` past its first `walkStart` letters, followed
by at most `B` letters, give the same run, where the steps left keep the midfixes within `B`. -/
theorem runProbes_congr_zone {r r' : FreeMonoid α → ARU} {X : List (FreeMonoid α)} {B : ℕ}
    (h : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ B → r (prefixOf x j * m) = r' (prefixOf x j * m)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv r C X s →
      s.tree.leaves.length - 2 + xs.length ≤ B → runProbes r C s xs = runProbes r' C s xs
  | [], s, _, _, _ => rfl
  | x :: xs, s, hX, hs, hB => by
    have hxX := hX x (by simp)
    simp only [List.length_cons] at hB
    have hstep : applyProbe r C s x = applyProbe r' C s x :=
      Kappa.applyProbe_congr' C (fun e t w he => hs.keys e t w he)
        (step_reads_agree C h hxX hs (by omega))
    simp only [runProbes]
    rw [← hstep]
    rcases ha : applyProbe r C s x with s' | e
    · obtain ⟨hs', hl⟩ := runInv_step r C hxX hs ha
      exact runProbes_congr_zone h xs s' (fun y hy => hX y (by simp [hy])) hs' (by omega)
    · rfl

/-- Reads agreeing on every prefix of a probe past its first `walkStart` letters, followed by at
most `T` letters, give the same result. -/
theorem level3_congr_zone {r r' : FreeMonoid α → ARU} {T : ℕ} (xs : Fin T → FreeMonoid α)
    (h : ∀ i j, C.walkStart ≤ j → j ≤ (xs i).toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ T → r (prefixOf (xs i) j * m) = r' (prefixOf (xs i) j * m)) :
    indecisiveLStarLevel3 C r xs = indecisiveLStarLevel3 C r' xs := by
  have hrun := runProbes_congr_zone C (X := List.ofFn xs) (B := T)
    (fun x hx j hj hjx m hm => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact h i j hj hjx m hm)
    (List.ofFn xs) initialState (fun _ h => h) (runInv_start r C _) (by
      simp [initialState, newHypothesis, DTree.leaves])
  simp only [indecisiveLStarLevel3, hrun]

/-- A probe reads only a prefix past its first `k` letters followed by a midfix. -/
theorem probeOrAgree_congr_zone {r r' : FreeMonoid α → ARU} {T : DTree α} (E : Edges α) (k : ℕ)
    {x : FreeMonoid α}
    (h : ∀ j, k ≤ j → j ≤ x.toList.length → ∀ m ∈ mids T,
      r (prefixOf x j * m) = r' (prefixOf x j * m)) :
    probeOrAgree r T E k x = probeOrAgree r' T E k x := by
  classical
  refine probeOrAgree_congr r T E k x fun z hz => ?_
  obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc k x.toList.length,
      ∃ m ∈ mids T, prefixOf x p * m = z := by simpa [pot] using hz
  obtain ⟨hp1, hp2⟩ := Finset.mem_Icc.1 hp
  exact h p hp1 hp2 m hm

end Run

/-! ## The sampler -/

section Harvest

open MeasureTheory

theorem harvestLaw_univ_le (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) :
    (harvestLaw read D k H site).real Set.univ ≤ 1 := by
  rw [harvestLaw_real_univ]
  refine (integral_mono (by simpa using integrable_siteCount read D k H site fun _ => true)
    (integrable_const 1) fun x => ?_).trans
    (le_of_eq (by simp : ∫ _ : FreeMonoid α, (1 : ℝ) ∂D = 1))
  have h1 : ((siteHarvest read k H site x).length : ℝ) ≤ 1 := by
    exact_mod_cast siteHarvest_length read k H site x
  convert h1

theorem harvestLaw_univ_ne_top (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) :
    harvestLaw read D k H site Set.univ ≠ ⊤ := by
  classical
  rw [harvestLaw_apply]
  refine ne_top_of_le_ne_top (b := ∫⁻ _, 1 ∂D) (by simp) (lintegral_mono fun x => ?_)
  have := (List.length_filter_le (· ∈ (Set.univ : Set (FreeMonoid α))) _).trans
    (siteHarvest_length read k H site x)
  calc (((siteHarvest read k H site x).filter (· ∈ (Set.univ : Set (FreeMonoid α)))).length
        : ENNReal) ≤ ((1 : ℕ) : ENNReal) := Nat.cast_le.2 this
    _ = 1 := Nat.cast_one

/-- The sampler finds only strings at most `L + n` long, over probes at most `L` long and a tree
whose midfixes are at most `n` long. -/
theorem harvestLaw_short (read : FreeMonoid α → ARU) {D : Measure (FreeMonoid α)} {L n : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) {H : HypothesisDFA α} (hT : MidsLe n H.tree)
    (site : Level3Site) : harvestLaw read D k H site {z | L + n < z.toList.length} = 0 := by
  classical
  rw [harvestLaw_apply]
  refine (lintegral_congr_ae ?_).trans (lintegral_zero (μ := D))
  filter_upwards [hlen] with x hx
  have : (siteHarvest read k H site x).filter (fun z => decide (L + n < z.toList.length))
      = [] := by
    rw [List.filter_eq_nil_iff]
    intro z hz
    obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc k x.toList.length,
        ∃ m ∈ mids H.tree, prefixOf x p * m = z := by
      simpa [pot] using (siteHarvest_pot read k H site x z hz).1
    have h1 := hT m hm
    have h2 : (prefixOf x p).toList.length ≤ x.toList.length := by simp [prefixOf]
    simp only [FreeMonoid.toList_mul, List.length_append, decide_eq_true_eq, not_lt]
    omega
  rw [this]
  simp

/-- Reads agreeing on every prefix of a probe past its first `k` letters followed by a midfix of
the tree give the same sampler. -/
theorem harvestLaw_congr_zone {r r' : FreeMonoid α → ARU} (D : Measure (FreeMonoid α)) (k : ℕ)
    (H : HypothesisDFA α) (P : Set (FreeMonoid α)) (hP : D Pᶜ = 0)
    (h : ∀ x ∈ P, ∀ j, k ≤ j → j ≤ x.toList.length → ∀ m ∈ mids H.tree,
      r (prefixOf x j * m) = r' (prefixOf x j * m)) (site : Level3Site) :
    harvestLaw r D k H site = harvestLaw r' D k H site := by
  unfold harvestLaw
  ext A hA
  rw [Measure.bind_apply hA (measurable_of_countable _).aemeasurable,
    Measure.bind_apply hA (measurable_of_countable _).aemeasurable]
  refine lintegral_congr_ae ?_
  filter_upwards [compl_mem_ae_iff.2 hP] with x hx
  have : probeOrAgree r H.tree H.edges k x = probeOrAgree r' H.tree H.edges k x :=
    probeOrAgree_congr_zone H.edges k (h x (by simpa using hx))
  cases site <;> simp only [siteHarvest, this]

end Harvest

end Random

end OrthoDFA
