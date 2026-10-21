import OrthoDFA.Proofs.CheckRoundMain
import OrthoDFA.Proofs.LoopRuns

/-!
# The checked round's run, whatever it reads

Every step keeps `Inv`, adds at most one leaf, keeps each learned edge at a leaf with a witness a
prefix of a probe past its first `walkStart` letters or of a check string (`CheckRunInv`), and
only adds midfixes. So where the run ends with its midfixes at most `n` long, every step read only
a probe's prefix past its first `walkStart` letters, or a check string's prefix, followed by at most
`n + 1` letters, and reads agreeing there give the same run (`runChecked_congr_zone`).
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
theorem mids_sub_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), mids T ⊆ mids (T.splitAt d p)
  | .leaf, _ => by simp [mids]
  | .node m r a, [] => by simp [DTree.splitAt]
  | .node m r a, false :: p => by
    classical
    intro m' hm
    have := mids_sub_splitAt d r p
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hm ⊢
    rcases hm with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl (this h))
    · exact Or.inr (Or.inr h)
  | .node m r a, true :: p => by
    classical
    intro m' hm
    have := mids_sub_splitAt d a p
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hm ⊢
    rcases hm with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (this h))

variable (read : FreeMonoid α → ARU) (C : Level3Settings) (K : CheckSettings)

/-- A step of level 3 at any walk start keeps `Inv` and its learned edges at leaves, adds at most
one leaf and only midfixes, and each edge it learns has a witness a prefix of its probe past the
walk start. -/
theorem step_edges (C' : Level3Settings) {s s' : Level3State α} {x : FreeMonoid α}
    (hs : Inv read s) (hk : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : applyProbe read C' s x = .inl s') :
    Inv read s' ∧ (∀ e t w, s'.edges e = some (t, w) → e.1 ∈ s'.tree.leaves)
      ∧ s'.tree.leaves.length ≤ s.tree.leaves.length + 1 ∧ mids s.tree ⊆ mids s'.tree
      ∧ ∀ e t w, s'.edges e = some (t, w) → (∃ t₀, s.edges e = some (t₀, w))
        ∨ ∃ i, C'.walkStart ≤ i ∧ i < x.toList.length ∧ w = prefixOf x i
          ∧ x.toList[i]? = some e.2 := by
  rcases step_cases read C' hs h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · refine ⟨⟨edgesOK_upd read hs.edges hu huc, hs.cls, hs.two⟩, ?_, by simp [newHypothesis],
      by simp [newHypothesis], ?_⟩
    · intro e t' w he
      simp only [newHypothesis] at he ⊢
      by_cases hep : e = (p, c)
      · subst hep; exact hp
      · rw [Function.update_of_ne hep] at he
        exact hk e t' w he
    · intro e t' w he
      simp only [newHypothesis] at he
      by_cases hep : e = (p, c)
      · subst hep
        rw [Function.update_self, Option.some.injEq, Prod.mk.injEq] at he
        obtain ⟨-, rfl⟩ := he
        obtain ⟨i, hki, hi, rfl, hc⟩ := probe_prefix read s.tree s.edges C'.walkStart x (Or.inr ho)
        exact Or.inr ⟨i, hki, hi, rfl, hc⟩
      · rw [Function.update_of_ne hep] at he
        exact Or.inl ⟨t', he⟩
  · have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have ht := DTree.sift_inl_mem read _ _ t huc
    obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
    have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
    refine ⟨⟨edgesOK_retarget read hs.edges hp _, ?_, ?_⟩, ?_, ?_, ?_, ?_⟩
    · simp only [splitLeaf, newHypothesis]
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hs.two; omega]
      exact classSet_split hs.cls hp ht ht₀
    · simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega
    · intro e t' w he
      simp only [splitLeaf, newHypothesis] at he ⊢
      obtain ⟨hpe, t₁, he₁⟩ := retarget_some read he
      have hq := hk e t₁ w he₁
      refine DTree.mem_leaves_splitAt_of_ne _ s.tree p e.1 hq fun hpq => hpe ?_
      rw [hpq]
      exact List.isPrefixOf_iff_prefix.2 (List.prefix_refl _)
    · simp only [splitLeaf, newHypothesis]; rw [hlen]
    · simp only [splitLeaf, newHypothesis]; exact mids_sub_splitAt _ _ _
    · intro e t' w he
      simp only [splitLeaf, newHypothesis] at he
      obtain ⟨-, t₁, he₁⟩ := retarget_some read he
      exact Or.inl ⟨t₁, he₁⟩
  · exact ⟨⟨hs.edges, hs.cls, hs.two⟩, hk, by simp [countProbe], by simp [countProbe],
      fun e t w he => Or.inl ⟨t, he⟩⟩

/-- `Inv`, with every learned edge at a leaf and its witness, followed by its letter, a prefix of
a probe of `X` past its first `walkStart` letters or of a check string of `Y`. -/
structure CheckRunInv (X Y : List (FreeMonoid α)) (s : Level3State α) : Prop where
  inv : Inv read s
  keys : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves
  wit : ∀ e t w, s.edges e = some (t, w) → ∃ x i, ((x ∈ X ∧ C.walkStart ≤ i) ∨ x ∈ Y)
    ∧ i < x.toList.length ∧ w = prefixOf x i ∧ x.toList[i]? = some e.2

omit [DecidableEq α] in
theorem checkRunInv_start (X Y : List (FreeMonoid α)) : CheckRunInv read C X Y initialState where
  inv := inv_start read
  keys := fun _ _ _ h => by simp [initialState, newHypothesis] at h
  wit := fun _ _ _ h => by simp [initialState, newHypothesis] at h

/-- A step of the checked round keeps the invariant, adds at most one leaf and only midfixes. -/
theorem checkRunInv_step {X Y : List (FreeMonoid α)} {s s' : Level3State α} {d : CheckDraw α K}
    (hx : d.1 ∈ X) (hy : ∀ i, d.2 i ∈ Y) (hs : CheckRunInv read C X Y s)
    (h : applyCheck read C K s d = .inl s') :
    CheckRunInv read C X Y s' ∧ s'.tree.leaves.length ≤ s.tree.leaves.length + 1
      ∧ mids s.tree ⊆ mids s'.tree := by
  rcases applyCheck_cases read C K s d with ⟨he, -⟩ | ⟨-, he⟩
  · rw [he] at h
    obtain ⟨h1, h2, h3, h4, h5⟩ := step_edges read C hs.inv hs.keys h
    refine ⟨⟨h1, h2, fun e t w hw => ?_⟩, h3, h4⟩
    rcases h5 e t w hw with ⟨t₀, hw₀⟩ | ⟨i, hki, hi, rfl, hc⟩
    · exact hs.wit e t₀ w hw₀
    · exact ⟨d.1, i, Or.inl ⟨hx, hki⟩, hi, rfl, hc⟩
  · rw [he] at h
    obtain ⟨i₀, hi₀⟩ := checkStep_inl_fix read K h
    have ha := fixOf_apply read C hi₀
    obtain ⟨h1, h2, h3, h4, h5⟩ := step_edges read C.atZero hs.inv hs.keys ha
    refine ⟨⟨h1, h2, fun e t w hw => ?_⟩, h3, h4⟩
    rcases h5 e t w hw with ⟨t₀, hw₀⟩ | ⟨i, -, hi, rfl, hc⟩
    · exact hs.wit e t₀ w hw₀
    · exact ⟨d.2 i₀, i, Or.inr (hy i₀), hi, rfl, hc⟩

/-- Reads giving the same probe and agreeing on what its split, and where the check runs what the
check reads, give the same step. -/
theorem applyCheck_congr_zone {r r' : FreeMonoid α → ARU} {s : Level3State α} {d : CheckDraw α K}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : ∀ z ∈ Kappa.probeReads r s.tree s.edges C.walkStart d.1 ∪ Eps.splitReads r C s d.1,
      r z = r' z)
    (hc : applyProbe r C s d.1 = .inr .consistent →
      ∀ z ∈ Eps.checkPots K s d.2 ∪ Eps.fixReads C K r s d.2, r z = r' z) :
    applyCheck r C K s d = applyCheck r' C K s d := by
  classical
  have ha := Kappa.applyProbe_congr' C hlv h
  unfold applyCheck
  rw [← ha]
  rcases hf : applyProbe r C s d.1 with t | e
  · rfl
  rcases e with _ | e | _
  · have hc' := hc hf
    have hP : ∀ i, ∀ z ∈ pot 0 (d.2 i) s.tree, r z = r' z := fun i z hz =>
      hc' z (Finset.mem_union_left _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩))
    have hpass : checkPasses r K s.hypotheses s.tree s.edges (List.ofFn d.2)
        ↔ checkPasses r' K s.hypotheses s.tree s.edges (List.ofFn d.2) := by
      have hcount : ∀ n, ((List.ofFn d.2).take n).countP (startFails r s.tree s.edges)
          = ((List.ofFn d.2).take n).countP (startFails r' s.tree s.edges) := fun n =>
        List.countP_congr fun x hx => by
          obtain ⟨i, rfl⟩ := List.mem_ofFn.1 (List.mem_of_mem_take hx)
          rw [Eps.startFails_congr (hP i)]
      simp only [checkPasses, hcount]
    have hfix : firstFix r s (List.ofFn d.2) = firstFix r' s (List.ofFn d.2) :=
      Eps.firstFix_congr C hlv _ fun x hx z hz => by
        obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
        rcases Finset.mem_union.1 hz with hz | hz
        · exact hP i z hz
        · exact hc' z (Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, hz⟩))
    simp only [checkStep, hpass, hfix]
  · rfl
  · rfl

/-- What a split re-sifts is a witness's prefix followed by at most `n + 1` letters, where the
midfixes are at most `n` long. -/
theorem splitReads_zone {X Y : List (FreeMonoid α)} {s : Level3State α} (hs : CheckRunInv read C X Y s)
    {n : ℕ} (hn : ∀ m ∈ mids s.tree, m.toList.length ≤ n) (C' : Level3Settings) (x : FreeMonoid α)
    {z : FreeMonoid α} (hz : z ∈ Eps.splitReads read C' s x) :
    ∃ y i, ((y ∈ X ∧ C.walkStart ≤ i) ∨ y ∈ Y) ∧ i ≤ y.toList.length ∧ ∃ m : FreeMonoid α,
      m.toList.length ≤ n + 1 ∧ z = prefixOf y i * m := by
  classical
  unfold Eps.splitReads at hz
  rcases ho : probeOrAgree read s.tree s.edges C'.walkStart x with
    _ | ⟨p, c, u, t⟩ | _ | _ | _ <;> rw [ho] at hz <;> simp only [Finset.notMem_empty] at hz
  rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩ <;> rw [hE] at hz
  · simp at hz
  simp only [Eps.resifts, Finset.mem_biUnion] at hz
  obtain ⟨e, -, hz⟩ := hz
  rcases he : s.edges e with _ | ⟨t', w⟩ <;> rw [he] at hz
  · simp at hz
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨y, i, hyi, hi, rfl, hc⟩ := hs.wit e t' w he
  refine ⟨y, i + 1, ?_, by omega, m, ?_, by rw [prefixOf_mul_of hc]⟩
  · rcases hyi with ⟨hy, hk⟩ | hy
    · exact Or.inl ⟨hy, by omega⟩
    · exact Or.inr hy
  · rcases Finset.mem_insert.1 (mids_splitAt _ s.tree p hm) with rfl | hm'
    · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
        List.length_singleton]
      rcases midfixAt_mem s.tree (commonPrefix t t₀) with h1 | h1
      · rw [h1]; simp
      · have := hn _ h1; omega
    · have := hn m hm'; omega

/-- Along a run, the midfixes only grow. -/
theorem mids_sub_runChecked {X Y : List (FreeMonoid α)} :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α), (∀ d ∈ ds, d.1 ∈ X ∧ ∀ i, d.2 i ∈ Y) →
      CheckRunInv read C X Y s → mids s.tree ⊆ mids (runChecked read C K s ds).1.tree
  | [], s, _, _ => by simp [runChecked]
  | d :: ds, s, hd, hs => by
    simp only [runChecked]
    rcases ha : applyCheck read C K s d with s' | e
    · obtain ⟨hs', -, hm⟩ := checkRunInv_step read C K (hd d (by simp)).1 (hd d (by simp)).2 hs ha
      exact hm.trans (mids_sub_runChecked ds s' (fun d' hd' => hd d' (by simp [hd'])) hs')
    · exact subset_rfl

/-- Reads agreeing with `r` on every probe's prefix past its first `walkStart` letters and every
check string's prefix, each followed by at most `n + 1` letters, give `r`'s run, where `r`'s run
ends with its midfixes at most `n` long. -/
theorem runChecked_congr_zone {r r' : FreeMonoid α → ARU} {X Y : List (FreeMonoid α)} {n : ℕ}
    (hX : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ n + 1 → r (prefixOf x j * m) = r' (prefixOf x j * m))
    (hY : ∀ y ∈ Y, ∀ j, j ≤ y.toList.length → ∀ m : FreeMonoid α,
      m.toList.length ≤ n + 1 → r (prefixOf y j * m) = r' (prefixOf y j * m)) :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α), (∀ d ∈ ds, d.1 ∈ X ∧ ∀ i, d.2 i ∈ Y) →
      CheckRunInv r C X Y s →
      (∀ m ∈ mids (runChecked r C K s ds).1.tree, m.toList.length ≤ n) →
      runChecked r' C K s ds = runChecked r C K s ds
  | [], _, _, _, _ => rfl
  | d :: ds, s, hd, hs, hfin => by
    classical
    have hd0 := hd d (by simp)
    have hn : ∀ m ∈ mids s.tree, m.toList.length ≤ n := fun m hm =>
      hfin m (mids_sub_runChecked r C K (d :: ds) s hd hs hm)
    have hzone : ∀ z, (∃ y i, ((y ∈ X ∧ C.walkStart ≤ i) ∨ y ∈ Y) ∧ i ≤ y.toList.length ∧
        ∃ m : FreeMonoid α, m.toList.length ≤ n + 1 ∧ z = prefixOf y i * m) → r z = r' z := by
      rintro z ⟨y, i, hyi | hyi, hi, m, hm, rfl⟩
      · exact hX y hyi.1 i hyi.2 hi m hm
      · exact hY y hyi i hi m hm
    have hstep : applyCheck r C K s d = applyCheck r' C K s d := by
      refine applyCheck_congr_zone C K hs.keys (fun z hz => hzone z ?_) (fun _ z hz => hzone z ?_)
      · rcases Finset.mem_union.1 hz with hz | hz
        · obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc C.walkStart d.1.toList.length,
              ∃ m ∈ mids s.tree, prefixOf d.1 p * m = z := by
            have := Kappa.probeReads_pot s.edges C.walkStart hz
            simpa [pot] using this
          obtain ⟨hp1, hp2⟩ := Finset.mem_Icc.1 hp
          exact ⟨d.1, p, Or.inl ⟨hd0.1, hp1⟩, hp2, m, by have := hn m hm; omega, rfl⟩
        · exact splitReads_zone r C hs hn C d.1 hz
      · rcases Finset.mem_union.1 hz with hz | hz
        · obtain ⟨i, -, hz⟩ := Finset.mem_biUnion.1 hz
          obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc 0 (d.2 i).toList.length,
              ∃ m ∈ mids s.tree, prefixOf (d.2 i) p * m = z := by simpa [pot] using hz
          exact ⟨d.2 i, p, Or.inr (hd0.2 i), (Finset.mem_Icc.1 hp).2, m,
            by have := hn m hm; omega, rfl⟩
        · obtain ⟨i, -, hz⟩ := Finset.mem_biUnion.1 hz
          exact splitReads_zone r C hs hn C.atZero (d.2 i) hz
    simp only [runChecked]
    rw [← hstep]
    rcases ha : applyCheck r C K s d with s' | e
    · obtain ⟨hs', -, -⟩ := checkRunInv_step r C K hd0.1 hd0.2 hs ha
      simp only [runChecked, ha] at hfin
      exact runChecked_congr_zone hX hY ds s' (fun d' hd' => hd d' (by simp [hd'])) hs' hfin
    · rfl

end Random

end OrthoDFA
