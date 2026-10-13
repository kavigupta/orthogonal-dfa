import OrthoDFA.Proofs.LoopRenoise

/-!
# Level 4's claim is fixed by the short reads

Over probes at most `L` long, level 4's claim reads the family only at strings at most
`max (L + maxLeaves) (probes + 1 + maxLeaves)` long (`endsWell_congr_reads`). The clustering's
prefixes are at most three per population and count, and their set has measurable level sets.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

section Reads

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

theorem runProbes_readInv {Lx : ℕ} :
    ∀ (s : Level4State α) (xs : List (FreeMonoid α)), ReadInv C Lx s →
      (∀ x ∈ xs, x.toList.length ≤ Lx) → (runProbes C cut s xs).2 ≠ .failed →
      ReadInv C Lx (runProbes C cut s xs).1
  | s, [], _, _, h => absurd rfl h
  | s, x :: xs, hs, hxs, h => by
    have hx := hxs x List.mem_cons_self
    simp only [runProbes] at h ⊢
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> simp only [ha] at h ⊢
    · exact runProbes_readInv s' xs (applyProbe_readInv C cut (recordProbe_readInv C cut hs hx) ha)
        (fun y hy => hxs y (List.mem_cons_of_mem _ hy)) h
    · have he : e ≠ .tooBig := by rintro rfl; simp [toRoundEnd] at h
      obtain ⟨rfl, -⟩ := applyProbe_inr_pre C cut ha he
      exact recordProbe_readInv C cut hs hx

theorem level4_tree_midLe (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (h : (indecisiveLStarLevel4 C read xs).ending ≠ .failed) :
    MidLe C.maxLeaves (indecisiveLStarLevel4 C read xs).hypothesis.tree := by
  have hinv := runProbes_readInv C (fun z => (read z).cut) (Lx := probesLen xs) initialState
    (List.ofFn xs) ⟨by simp [initialState, DTree.leaves], by simpa [initialState, DTree.leaves]
      using hC, by simp [initialState, DTree.leaves, DTree.midfixes], by simp [initialState],
      by simp [initialState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact le_probesLen xs i) h
  obtain ⟨-, hL, hm, -, -⟩ := hinv
  intro m hmm
  have := hm m hmm
  show m.toList.length ≤ C.maxLeaves
  omega

/-- Over probes at most `L` long, level 4's claim reads only the short strings. -/
theorem endsWell_congr_reads (hC : 2 ≤ C.maxLeaves) {σ : Type*} (M : _root_.DFA α σ)
    (bad : σ → Prop) (D' : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D', x.toList.length ≤ L)
    (y : ℝ) {T : ℕ} (xs : Fin T → FreeMonoid α) {read read' : FreeMonoid α → ARU}
    (h : ∀ z, z.toList.length ≤ max (L + C.maxLeaves) (probesLen xs + 1 + C.maxLeaves) →
      read z = read' z) :
    EndsWell M bad D' C.disagreementRate (searchingProbes read C.walkStart
        (indecisiveLStarLevel4 C read xs).hypothesis) C.walkStart
        (harvestLaw D' (fun z => (read z).cut) C.walkStart) y
        (indecisiveLStarLevel4 C read xs).ending
      ↔ EndsWell M bad D' C.disagreementRate (searchingProbes read' C.walkStart
        (indecisiveLStarLevel4 C read' xs).hypothesis) C.walkStart
        (harvestLaw D' (fun z => (read' z).cut) C.walkStart) y
        (indecisiveLStarLevel4 C read' xs).ending := by
  have hres : indecisiveLStarLevel4 C read xs = indecisiveLStarLevel4 C read' xs :=
    level4_congr_up C hC xs (le_probesLen xs) fun z hz => h z (le_max_of_le_right hz)
  rw [← hres]
  set res := indecisiveLStarLevel4 C read xs
  have hcut : AgreeUpTo (L + C.maxLeaves) (fun z => (read z).cut) (fun z => (read' z).cut) :=
    fun z hz => by dsimp only; rw [h z (le_max_of_le_left hz)]
  rcases he : res.ending with _ | hv | _
  · -- consistent: the searching probes, at every short probe
    have hmid := level4_tree_midLe C hC read xs (by rw [he]; simp)
    have hae : searchingProbes read C.walkStart res.hypothesis
        =ᵐ[D'] searchingProbes read' C.walkStart res.hypothesis := by
      filter_upwards [hlen] with x hx
      have hpb := probeBy_congr_up hcut hmid (x := x) (by omega)
        (Function.curry res.hypothesis.edges) C.walkStart
      apply propext
      show (match probeBy (fun z => (read z).cut) res.hypothesis.tree
          (Function.curry res.hypothesis.edges) C.walkStart x with
        | .undecidedMid _ | .edge _ _ => True
        | _ => False) ↔ (match probeBy (fun z => (read' z).cut) res.hypothesis.tree
          (Function.curry res.hypothesis.edges) C.walkStart x with
        | .undecidedMid _ | .edge _ _ => True
        | _ => False)
      rw [hpb]
    simp only [EndsWell]
    rw [measureReal_congr hae]
  · have hshort := harvest_short C hC read xs he
    have hlaw := harvestLaw_congr_up (fun z => (read z).cut) (L := L) (M := C.maxLeaves) hlen le_rfl
      hcut C.walkStart (h := hv) fun v hvm m hm => by
        obtain ⟨-, hL, hmid⟩ := hshort v hvm
        have := hmid m hm
        show m.toList.length ≤ C.maxLeaves
        omega
    simp only [EndsWell]
    rw [hlaw]
  · simp [EndsWell]

end Reads

section Prefs

variable (settings : LoopSettings K)

lemma card_prefsOf_le (harvests : List (Measure (FreeMonoid α))) (c : ClusterRest α K) {NP : ℕ}
    (hNP : schedNP (settings.schedule (loopPopulations K harvests.length)) ≤ NP) :
    (prefsOf settings harvests c).card ≤ 3 * (K + 1) * NP := by
  classical
  set pops := loopPopulations K harvests.length
  have hpops : pops.card ≤ K + 1 := by
    have := Finset.card_le_univ pops
    simpa using this
  unfold prefsOf schedPrefs
  refine (Finset.card_union_le _ _).trans ?_
  have h1 : (prefixesAt pops (schedNP (settings.schedule pops))
      (((), runOf harvests c) : Run Unit (FreeMonoid α) (Option (Fin K)))).card
      ≤ pops.card * schedNP (settings.schedule pops) := by
    unfold prefixesAt
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, (prefixesOf j (schedNP (settings.schedule pops))
          (((), runOf harvests c) : Run Unit (FreeMonoid α) (Option (Fin K)))).card
        ≤ ∑ _j ∈ pops, schedNP (settings.schedule pops) :=
          Finset.sum_le_sum fun j _ => Finset.card_image_le.trans (by simp)
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have h2 : (pops.biUnion fun j => certOf j (2 * schedNP (settings.schedule pops))
      (((), runOf harvests c) : Run Unit (FreeMonoid α) (Option (Fin K)))).card
      ≤ pops.card * (2 * schedNP (settings.schedule pops)) := by
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, (certOf j (2 * schedNP (settings.schedule pops))
          (((), runOf harvests c) : Run Unit (FreeMonoid α) (Option (Fin K)))).card
        ≤ ∑ _j ∈ pops, 2 * schedNP (settings.schedule pops) :=
          Finset.sum_le_sum fun j _ => Finset.card_image_le.trans (by simp)
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have h3 : pops.card * schedNP (settings.schedule pops) ≤ (K + 1) * NP :=
    Nat.mul_le_mul hpops hNP
  have h4 : pops.card * (2 * schedNP (settings.schedule pops)) ≤ 2 * ((K + 1) * NP) := by
    nlinarith
  refine le_trans (add_le_add h1 h2) ?_
  rw [show 3 * (K + 1) * NP = 3 * ((K + 1) * NP) by ring]
  omega

end Prefs

end OrthoDFA
