import OrthoDFA.Proofs.TallyKeep

/-!
# The round from its sub-rounds, with its records that are not true as one budget

A split that is not genuine needs `m` records that are not true made in its own sub-round
(`badRecs_of_fake`), so while the round's such records stay below `(S + 1) m` its trees stay in
the class and at most `S + |Q|` of its sub-rounds end in a split. `fail_split`: the round fails to
end at a tree of the class only if those records reach `(S + 1) m` within its first `W` probes, or
it fails with them below (`FailB`). `round_budget`: the latter has chance at most `S + |Q| + 1`
times a sub-round's chance of keeping its tree over `subT` probes with fewer than `(S + 1) m`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Budget

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (D : Measure (FreeMonoid α))
  [IsProbabilityMeasure D] (C : TallyCfg) (rd : FreeMonoid α → ARU) (S : ℕ)

open scoped Classical in
/-- The run from `s`, having made `u` records that are not true, fails to end at a tree of the
class, with those records below `(S + 1) m` after each of its first `W` probes it goes on from. -/
def FailB : TState α → ℕ → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, _, _, 0, _ => True
  | s, u, W, T + 1, xs => match tallyStep C (fun z => (rd z).cut) s (xs 0) with
    | .inl s' =>
      (W = 0 ∨ u + (if xs 0 ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0) < (S + 1) * C.m)
        ∧ FailB s' (u + if xs 0 ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0) (W - 1) T
          (Fin.tail xs)
    | .inr (e, s') => ¬ OkEnd G S e s'

theorem failB_zero : ∀ (T : ℕ) (s : TState α) (u : ℕ) (xs : Fin T → FreeMonoid α),
    ¬ RunEnds (tallyStep C fun z => (rd z).cut) (OkEnd G S) s (List.ofFn xs) →
      FailB G C rd S s u 0 T xs
  | 0, _, _, _, _ => trivial
  | T + 1, s, u, xs, h => by
    simp only [List.ofFn_succ, RunEnds] at h
    simp only [FailB]
    rcases hst : tallyStep C (fun z => (rd z).cut) s (xs 0) with s' | ⟨e, s'⟩ <;>
      rw [hst] at h
    · exact ⟨by simp, failB_zero T s' _ (Fin.tail xs) h⟩
    · exact h

open scoped Classical in
/-- The round fails to end at a tree of the class only with its records that are not true below
`(S + 1) m` over its first `W` probes, or reaching it there. -/
theorem fail_split : ∀ (T : ℕ) (s : TState α) (u W : ℕ) (xs : Fin T → FreeMonoid α),
    u < (S + 1) * C.m →
    ¬ RunEnds (tallyStep C fun z => (rd z).cut) (OkEnd G S) s (List.ofFn xs) →
    FailB G C rd S s u W T xs ∨ UntrueHit G C rd s ((S + 1) * C.m - u) W T xs
  | 0, _, _, _, _, _, _ => .inl trivial
  | T + 1, s, u, W, xs, hu, h => by
    rcases W with _ | W
    · exact .inl (failB_zero G C rd S _ s u xs h)
    simp only [List.ofFn_succ, RunEnds] at h
    obtain ⟨n, hn⟩ : ∃ n, (S + 1) * C.m - u = n + 1 := ⟨(S + 1) * C.m - u - 1, by omega⟩
    rw [hn]
    simp only [FailB, UntrueHit]
    rcases hst : tallyStep C (fun z => (rd z).cut) s (xs 0) with s' | ⟨e, s'⟩ <;>
      rw [hst] at h
    · simp only [Sum.inl.injEq, exists_eq_left', add_tsub_cancel_right]
      by_cases hx : xs 0 ∈ G.untrueAt rd C.k s.tree s.edges
      · simp only [if_pos hx]
        by_cases hlt : u + 1 < (S + 1) * C.m
        · rcases fail_split T s' (u + 1) W _ hlt h with h' | h'
          · exact .inl ⟨.inr hlt, h'⟩
          · right
            rwa [show (S + 1) * C.m - (u + 1) = n by omega] at h'
        · right
          rw [show n = 0 by omega]
          simp [UntrueHit]
      · simp only [if_neg hx, add_zero]
        rcases fail_split T s' u W _ hu h with h' | h'
        · exact .inl ⟨.inr hu, h'⟩
        · right
          rwa [hn] at h'
    · exact .inl h

/-- A sub-round may start at `s`, the round having made `u` records that are not true, with `k`
more splits that are not genuine within the budget, at most `g` genuine ones to come, and `W`
probes of the window enough for `k + g + 1` sub-rounds. -/
def AdmB (Tsub : ℕ) (s : TState α) (u k g W T : ℕ) : Prop :=
  SubStart G S s ∧ (∃ f, G.Grown s.tree f ∧ f + k ≤ S)
    ∧ Fintype.card σ ≤ (G.trueLeaves s.tree).card + g ∧ u < (S + 1) * C.m
    ∧ (S + 1) * C.m ≤ u + (k + 1) * C.m ∧ (k + g + 1) * Tsub ≤ W ∧ W ≤ T

open scoped Classical in
/-- A probe of a sub-round: it continues at the same tree, or ends the round (`some (inl e)`), or
the tree changes (`some (inr ())`), or the round's records that are not true reach the budget
within the window (`none`). -/
noncomputable def segB : TState α × ℕ × ℕ → FreeMonoid α →
    (TState α × ℕ × ℕ) ⊕ (Option (TEnd α ⊕ Unit) × (TState α × ℕ × ℕ))
  | (s, u, W), x =>
    match tallyStep C (fun z => (rd z).cut) s x with
    | .inr (e, s') => .inr (some (.inl e),
        (s', u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0, W - 1))
    | .inl s' =>
      if W = 0 ∨ u + (if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0) < (S + 1) * C.m then
        if s'.tree = s.tree then
          .inl (s', u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0, W - 1)
        else .inr (some (.inr ()),
          (s', u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0, W - 1))
      else .inr (none, (s', u + if x ∈ G.untrueAt rd C.k s.tree s.edges then 1 else 0, W - 1))

open scoped Classical in
/-- From a sub-round's start the round fails with its records that are not true below the budget
over the window with chance at most `k + g + 1` times `δ'`, a sub-round's chance of keeping its
tree over `Tsub` probes with fewer than the budget. -/
theorem round_budget (hm : 0 < C.m) (hLmax : Fintype.card σ + S + 3 ≤ C.Lmax) {δ' : ℝ}
    {Tsub : ℕ} (hδ' : 0 ≤ δ')
    (hsub : ∀ s B T, SubStart G S s → B ≤ (S + 1) * C.m → Tsub ≤ T →
      (Measure.pi fun _ : Fin T => D) {xs | SubOpen G C rd s B Tsub T xs} ≤ ENNReal.ofReal δ') :
    ∀ N (s : TState α) (u k g W T : ℕ), k + g = N → AdmB G C S Tsub s u k g W T →
      (Measure.pi fun _ : Fin T => D) {xs | FailB G C rd S s u W T xs}
        ≤ ENNReal.ofReal ((k + g + 1) * δ') := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  set U : TState α → Set (FreeMonoid α) := fun s' => G.untrueAt rd C.k s'.tree s'.edges with hU
  intro N
  induction N using Nat.strong_induction_on with
  | _ N ih =>
  intro s u k g W T hN hadm
  obtain ⟨hs, ⟨f, hf, hfk⟩, hg, hu, hku, hW, hWT⟩ := hadm
  set w : Option (TEnd α ⊕ Unit) → TState α × ℕ × ℕ → ℕ → ENNReal := fun kd sb T' =>
    match kd with
    | none => 0
    | some (.inl e) => if OkEnd G S e sb.1 then 0 else 1
    | some (.inr _) => min 1 (⨅ (k' : ℕ) (g' : ℕ)
        (_ : k' + g' < N ∧ AdmB G C S Tsub sb.1 sb.2.1 k' g' sb.2.2 T'),
        ENNReal.ofReal ((k' + g' + 1) * δ'))
  set F : TState α × ℕ × ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop := fun sb T xs =>
    FailB G C rd S sb.1 sb.2.1 sb.2.2 T xs
  set V : Option (TEnd α ⊕ Unit) → TState α × ℕ × ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop :=
    fun kd sb T xs => match kd with
    | none => False
    | some (.inl e) => ¬ OkEnd G S e sb.1
    | some (.inr _) => F sb T xs
  have hF : ∀ sb T x xs, F sb (T + 1) (Fin.cons x xs)
      ↔ (segB G C rd S sb x).elim (fun sb' => F sb' T xs) fun q => V q.1 q.2 T xs := by
    rintro ⟨s', u', W'⟩ T' x xs
    change FailB G C rd S s' u' W' (T' + 1) (Fin.cons x xs) ↔ _
    simp only [FailB, segB, Fin.cons_zero, Fin.tail_cons]
    rcases tallyStep C (fun z => (rd z).cut) s' x with s'' | ⟨e, s''⟩
    · dsimp only
      by_cases hc : W' = 0
          ∨ u' + (if x ∈ G.untrueAt rd C.k s'.tree s'.edges then 1 else 0) < (S + 1) * C.m
      · rw [if_pos hc]
        by_cases htr : s''.tree = s'.tree
        · rw [if_pos htr]; exact ⟨fun h => h.2, fun h => ⟨hc, h⟩⟩
        · rw [if_neg htr]; exact ⟨fun h => h.2, fun h => ⟨hc, h⟩⟩
      · rw [if_neg hc]; exact ⟨fun h => absurd h.1 hc, fun h => h.elim⟩
    · rfl
  have hV : ∀ kd sb T', (Measure.pi fun _ : Fin T' => D) {xs | V kd sb T' xs} ≤ w kd sb T' := by
    rintro (_ | e | _) ⟨s', u', W'⟩ T'
    · simp [V, w]
    · simp only [V, w]
      split_ifs with hok
      · simp [hok]
      · simp only [hok, not_false_eq_true, Set.setOf_true]; exact prob_le_one
    · simp only [V, w]
      refine le_min prob_le_one (le_iInf fun k' => le_iInf fun g' => le_iInf fun h => ?_)
      exact ih (k' + g') (hN ▸ h.1) s' u' k' g' W' T' rfl h.2
  have hseg := seg_le D (segB G C rd S) F V w hF hV T (s, u, W) Tsub
  have hfpaths : s.tree.paths.length ≤ Fintype.card σ + 2 + S := by
    have := G.grown_paths hf; omega
  have hpt : ∀ j T' (s' : TState α) (u' W' : ℕ) (xs : Fin T' → FreeMonoid α), s'.tree = s.tree →
      EdgesInto s'.tree s'.edges → RecsInto s' → u ≤ u' → u' < (S + 1) * C.m →
      (∀ pct, badRecs G s' pct ≤ u' - u) → (k + g) * Tsub + j ≤ W' → W' ≤ T' →
      segVal (segB G C rd S) w (s', u', W') j T' xs
        ≤ ENNReal.ofReal ((k + g) * δ')
          + (if SubOpen G C rd s' ((S + 1) * C.m - u') j T' xs then 1 else 0) := by
    intro j
    induction j with
    | zero =>
      intro T' s' u' W' xs _ _ _ _ hu' _ _ _
      have : SubOpen G C rd s' ((S + 1) * C.m - u') 0 T' xs := by
        simp only [SubOpen]; omega
      simp [segVal, this]
    | succ j ihj =>
      intro T' s' u' W' xs htr he hr huu hu' hbad hjW hWT'
      rcases T' with _ | T'
      · have : SubOpen G C rd s' ((S + 1) * C.m - u') (j + 1) 0 xs := by
          simp only [SubOpen]; omega
        simp [segVal, this]
      rw [segVal_succ']
      have hspec := tallyStep_spec cut C hm (x := xs 0) he hr
      rcases hst : tallyStep C cut s' (xs 0) with s'' | ⟨e, s''⟩
      · set u'' := u' + if xs 0 ∈ U s' then 1 else 0 with hu''
        by_cases hbud : u'' < (S + 1) * C.m
        swap
        · have hc : ¬(W' = 0 ∨ u' + (if xs 0 ∈ G.untrueAt rd C.k s'.tree s'.edges then 1 else 0)
              < (S + 1) * C.m) := by
            rintro (h | h)
            · omega
            · exact hbud h
          have hsv : segB G C rd S (s', u', W') (xs 0) = .inr (none, (s'', u'', W' - 1)) := by
            simp only [segB]; rw [hst]; dsimp only; rw [if_neg hc]
          rw [hsv]
          simp [w]
        have hc : W' = 0 ∨ u' + (if xs 0 ∈ G.untrueAt rd C.k s'.tree s'.edges then 1 else 0)
            < (S + 1) * C.m := .inr hbud
        by_cases htr' : s''.tree = s'.tree
        · have hsv : segB G C rd S (s', u', W') (xs 0) = .inl (s'', u'', W' - 1) := by
            simp only [segB]; rw [hst]; dsimp only; rw [if_pos hc, if_pos htr']
          rw [hsv]
          simp only [Sum.elim_inl]
          rcases hspec.1 _ hst with ⟨-, he', hr'⟩ | ⟨p, c, t, t₀, -, -, hp, hT', -⟩
          swap
          · exact absurd (htr'.symm.trans hT') (DTree.splitAt_ne hp).symm
          refine (ihj T' s'' u'' (W' - 1) (Fin.tail xs) (htr'.trans htr) he' hr' (by omega) hbud
            (fun pct => ?_) (by omega) (by omega)).trans (add_le_add le_rfl ?_)
          · have h1 := badRecs_pre G C cut s' (xs 0) pct
            have h2 : badRecs G s'' pct = badRecs G (tallyPre C cut s' (xs 0)) pct := by
              unfold badRecs
              rw [tallyStep_recs C cut hst htr', htr', (tallyPre_cases cut C s' (xs 0)).1]
            have h3 : xs 0 ∈ untrueAt G C cut s' pct → xs 0 ∈ U s' :=
              fun ⟨sp, h, hn⟩ => ⟨pct, sp, h, hn⟩
            have := hbad pct
            split_ifs at h1 hu'' with hx hx'
            · omega
            · exact absurd (h3 hx) hx'
            · omega
            · omega
          · split_ifs with h1 h2
            · exact le_rfl
            · refine absurd ⟨s'', hst, htr', ?_⟩ h2
              convert h1 using 2
              simp only [U] at hu''
              split_ifs at hu'' ⊢ <;> omega
            · exact zero_le
            · exact le_rfl
        · have hsv : segB G C rd S (s', u', W') (xs 0)
              = .inr (some (.inr ()), (s'', u'', W' - 1)) := by
            simp only [segB]; rw [hst]; dsimp only; rw [if_pos hc, if_neg htr']
          rw [hsv]
          simp only [Sum.elim_inr]
          refine le_trans ?_ le_self_add
          obtain ⟨p, c, t, t₀, hp, ht, ht₀, hne, hm₁, hm₂, hT⟩ :=
            tallyStep_split C cut hm he hr hst htr'
          rcases hspec.1 _ hst with ⟨htr'', -, -⟩ |
            ⟨-, -, -, -, -, -, -, -, he'', hr'', hn'', hd''⟩
          · exact absurd htr'' htr'
          have hstart : ∀ f', G.Grown s''.tree f' → f' ≤ S → SubStart G S s'' :=
            fun f' h hle => ⟨⟨f', hle, h⟩, he'', fun q e => by
              have := (hspec.1 _ hst).resolve_left (fun h => htr' h.1)
              obtain ⟨-, -, -, -, -, -, -, -, -, hrr, -⟩ := this
              exact hrr q e, hn'', hd''⟩
          have hf' : G.Grown s'.tree f := htr ▸ hf
          have hg' : Fintype.card σ ≤ (G.trueLeaves s'.tree).card + g := htr ▸ hg
          have htrP := (tallyPre_cases cut C s' (xs 0)).1
          by_cases hgen : G.GenuineSplit s'.tree p (FreeMonoid.of c * s'.tree.midAt (lcp t t₀))
          · obtain ⟨hg₂, hc₂⟩ := split_lands G hf' ht ht₀ hp hT ⟨_, _, hT, hgen⟩
            have hle := G.trueLeaves_card_le s''.tree
            have huu'' : u' ≤ u'' := by rw [hu'']; exact Nat.le_add_right _ _
            have hadm : k + (g - 1) < N ∧ AdmB G C S Tsub s'' u'' k (g - 1) (W' - 1) T' := by
              refine ⟨by omega, hstart f hg₂ (by omega), ⟨f, hg₂, hfk⟩, by omega, hbud,
                by omega, ?_, by omega⟩
              rw [show k + (g - 1) + 1 = k + g by omega]
              omega
            refine (min_le_right _ _).trans ((iInf_le _ k).trans ((iInf_le _ (g - 1)).trans
              ((iInf_le _ hadm).trans (ENNReal.ofReal_le_ofReal ?_))))
            have : ((k + (g - 1) : ℕ) : ℝ) + 1 = (k + g : ℕ) := by
              have : 1 ≤ g := by omega
              push_cast [Nat.cast_sub this]; ring
            push_cast at this ⊢
            rw [this]
          · have hngP : ¬ G.GenuineSplit (tallyPre C cut s' (xs 0)).tree p
                (FreeMonoid.of c * (tallyPre C cut s' (xs 0)).tree.midAt (lcp t t₀)) := by
              rw [htrP]; exact hgen
            have hbig : u + C.m ≤ u'' := by
              have h3 : ∀ pct, xs 0 ∈ untrueAt G C cut s' pct → xs 0 ∈ U s' :=
                fun pct ⟨sp, h, hn⟩ => ⟨pct, sp, h, hn⟩
              have key : ∀ pct, C.m ≤ badRecs G (tallyPre C cut s' (xs 0)) pct → u + C.m ≤ u'' := by
                intro pct hb
                have h1 := badRecs_pre G C cut s' (xs 0) pct
                have := hbad pct
                split_ifs at h1 hu'' with hx hx'
                · omega
                · exact absurd (h3 pct hx) hx'
                · omega
                · omega
              rcases badRecs_of_fake G C hne hm₁ hm₂ hngP with hb | hb
              · exact key _ hb
              · exact key _ hb
            rcases Nat.eq_zero_or_pos k with rfl | hk
            · exfalso
              have : (S + 1) * C.m ≤ u + C.m := by simpa using hku
              omega
            have hg₂ : G.Grown s''.tree (f + 1) := hT ▸ ReadModel.Grown.fake hf' hp ht ht₀ hgen
            have hc₂ := (G.trueLeaves_splitAt hp (FreeMonoid.of c * s'.tree.midAt (lcp t t₀))).1
            rw [← hT] at hc₂
            have hadm : (k - 1) + g < N ∧ AdmB G C S Tsub s'' u'' (k - 1) g (W' - 1) T' := by
              refine ⟨by omega, hstart (f + 1) hg₂ (by omega), ⟨f + 1, hg₂, by omega⟩, by omega,
                hbud, ?_, ?_, by omega⟩
              · rw [show k - 1 + 1 = k by omega]
                have : (S + 1) * C.m ≤ u + (k - 1 + 1 + 1) * C.m := by
                  rw [show k - 1 + 1 = k by omega]; exact hku
                nlinarith
              · rw [show k - 1 + g + 1 = k + g by omega]
                omega
            refine (min_le_right _ _).trans ((iInf_le _ (k - 1)).trans ((iInf_le _ g).trans
              ((iInf_le _ hadm).trans (ENNReal.ofReal_le_ofReal ?_))))
            have : ((k - 1 + g : ℕ) : ℝ) + 1 = (k + g : ℕ) := by
              push_cast [Nat.cast_sub (show 1 ≤ k by omega)]; ring
            push_cast at this ⊢
            rw [this]
      · have hsv : segB G C rd S (s', u', W') (xs 0) = .inr (some (.inl e), (s'',
            u' + if xs 0 ∈ U s' then 1 else 0, W' - 1)) := by
          simp only [segB]; rw [hst]
        rw [hsv]
        simp only [Sum.elim_inr, w]
        have hok : OkEnd G S e s'' := by
          rcases hspec.2 _ _ hst with h | h
          · subst h
            obtain ⟨hL, hle⟩ := tallyStep_tooBig cut C hst
            rw [htr] at hle
            omega
          · refine ⟨fun he' => ?_, by rw [h, htr]; exact hs.1⟩
            subst he'
            obtain ⟨hL, hle⟩ := tallyStep_tooBig cut C hst
            rw [htr] at hle
            omega
        rw [if_pos hok]
        exact zero_le
  have hT1 : (k + g) * Tsub + Tsub ≤ W := by
    have : (k + g + 1) * Tsub = (k + g) * Tsub + Tsub := by ring
    omega
  set P := Measure.pi fun _ : Fin T => D
  have hr : RecsInto s := fun q c r h => by rw [hs.2.2.1 q c] at h; cases h
  calc P {xs | FailB G C rd S s u W T xs} = P {xs | F (s, u, W) T xs} := rfl
    _ ≤ _ := hseg
    _ ≤ ∫⁻ xs, (ENNReal.ofReal ((k + g) * δ')
          + (if SubOpen G C rd s ((S + 1) * C.m - u) Tsub T xs then 1 else 0)) ∂P :=
        lintegral_mono fun xs => hpt Tsub T s u W xs rfl hs.2.1 hr le_rfl hu
          (fun pct => by simp [badRecs, hs.2.2.1]) hT1 hWT
    _ = ENNReal.ofReal ((k + g) * δ') + P {xs | SubOpen G C rd s ((S + 1) * C.m - u) Tsub T xs} := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_ite_one]
    _ ≤ ENNReal.ofReal ((k + g) * δ') + ENNReal.ofReal δ' :=
        add_le_add le_rfl (hsub s _ T hs (by omega) (by nlinarith))
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) hδ']
        congr 1
        ring

end Budget

universe u v w in
/-- `TallyRound` from `SubRound`, `FakeRace`, `HarvestGood` and `SuccessSound`. -/
theorem tally_round_of (hsub : SubRound.{u, v}) (hrace : FakeRace.{u, v})
    (hharv : HarvestGood.{u, v}) (hsucc : SuccessSound.{u}) : TallyRound.{u, v, w} := by
  intro α _ _ σ _ Ω _ μ _ G read D _ C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt' θr εd' Xe θs' Xs
    η ν hlen hρ0 hm hLmax hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond
    hhP hhS hhS' hφn hL1 ha ha1 hθg0 hθg hφe0 hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hφ0 hφ1
    hlow hθe hXe0 hXe hθs' hXs hlin hη hν hside hT
  set Tsub := subT C (Fintype.card α) nEnd nRec
  set W := (S + Fintype.card σ + 1) * Tsub
  set U := (S + 1) * C.m
  set δ' := subOpen C (Fintype.card α) (Fintype.card σ) nRec U θr (termLevel nEnd hP hS θpt' εd')
  set φ := fakeRun C (Fintype.card α) S L W ρ Xe θs' Xs η ν
  set c := ENNReal.ofReal φ + ENNReal.ofReal ((S + Fintype.card σ + 1) * δ')
    + ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 3 * (T * T)) * C.a)
  set M := toMeasurable μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)}
  have hτ0 := termLevel_nonneg (nEnd := nEnd) (hP := hP) (hS := hS) hθpt'0 hθpt'1 hεd'0 hεd'1
  have hδ'0 : 0 ≤ δ' := by
    have := binomSfGe_le_one (n := nRec) hθr0 hθr1
      (Fintype.card σ * Fintype.card α * C.m + U)
    simp only [δ', subOpen]
    have : 0 ≤ 1 - binomSfGe nRec θr (Fintype.card σ * Fintype.card α * C.m + U) := by linarith
    positivity
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
        tallyStart (List.ofFn xs)} ≤ M.indicator 1 ω + c := by
    intro ω
    by_cases hω : TallyE G D C S ρ θg θgs θgpt (read · ω)
    · refine le_trans ?_ le_add_self
      set cut : FreeMonoid α → Option Bool := fun z => (read z ω).cut
      have hsb : ∀ s B T', SubStart G S s → B ≤ U → Tsub ≤ T' →
          (Measure.pi fun _ : Fin T' => D) {xs | SubOpen G C (read · ω) s B Tsub T' xs}
            ≤ ENNReal.ofReal δ' := by
        intro s B T' hs hB hT'
        refine (hsub G (read · ω) D C S nEnd nRec hP hS B θpt' θr εd' hm hLmax hn₀ hn₀' hθpt0
          hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS hhS' hφn s hs T'
          hT').trans (ENNReal.ofReal_le_ofReal ?_)
        simp only [δ', subOpen]
        have := binomSfGe_antitone' (n := nRec) hθr0 hθr1
          (show Fintype.card σ * Fintype.card α * C.m + B
            ≤ Fintype.card σ * Fintype.card α * C.m + U by omega)
        linarith
      have hstart₀ : AdmB G C S Tsub (tallyStart : TState α) 0 S (Fintype.card σ) W T :=
        ⟨⟨⟨0, Nat.zero_le _, .start⟩, fun _ _ _ _ h => by simp [tallyStart] at h,
          fun _ _ => rfl, rfl, rfl⟩, ⟨0, .start, by omega⟩, by omega,
          Nat.mul_pos (Nat.succ_pos _) hm, by simp [U], le_rfl, hT⟩
      have h1 := round_budget G D C (read · ω) S hm hLmax hδ'0 hsb _ tallyStart 0 S
        (Fintype.card σ) W T rfl hstart₀
      have h1' := hrace G (read · ω) D C S L W ρ θg θgs θgpt Xe θs' Xs η ν hlen hm hLmax hθe hφe0
        hρ0 hXe0 hXe hθs' hXs hlin hη hν hside hω T
      have hokb : {xs : Fin T → FreeMonoid α | ¬ RunEnds (tallyStep C cut) (OkEnd G S)
          tallyStart (List.ofFn xs)}
          ⊆ {xs | FailB G C (read · ω) S tallyStart 0 W T xs}
            ∪ {xs | UntrueHit G C (read · ω) tallyStart U W T xs} := by
        intro xs hxs
        have := fail_split G C (read · ω) S T tallyStart 0 W xs
          (Nat.mul_pos (Nat.succ_pos _) hm) hxs
        simpa [U] using this
      have h2 := hharv G (read · ω) D C S L T ρ θg θgs θgpt hlen hL1 hm ha ha1 hθg0 hθg hφe0
        (by omega) hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hφ0 hφ1 hlow hω
      have hsub' : {xs : Fin T → FreeMonoid α | ¬ RunEnds (tallyStep C cut) (GoodEnd G)
          tallyStart (List.ofFn xs)}
          ⊆ {xs | ¬ RunEnds (tallyStep C cut) (OkEnd G S) tallyStart (List.ofFn xs)}
            ∪ {xs | RunEnds (tallyStep C cut)
              (fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig ∧ ¬ GoodEnd G e s') tallyStart
              (List.ofFn xs)} := by
        intro xs hxs
        by_cases hok : RunEnds (tallyStep C cut) (OkEnd G S) tallyStart (List.ofFn xs)
        · rcases runEnds_or (Q := GoodEnd G) _ _ hok with h | h
          · exact absurd h hxs
          · exact .inr (runEnds_mono (fun e s' h => ⟨h.1.2, h.1.1, h.2⟩) _ _ h)
        · exact .inl hok
      have h3 := hsucc (read · ω) D C T hεd0 hεd1 ha1
      have hsnd : {xs : Fin T → FreeMonoid α | ¬ RunEnds (tallyStep C cut)
          (TallyEndsWell G D C (read · ω)) tallyStart (List.ofFn xs)}
          ⊆ {xs | ¬ RunEnds (tallyStep C cut) (GoodEnd G) tallyStart (List.ofFn xs)}
            ∪ {xs | RunEnds (tallyStep C cut) (fun e s' => e = .success
              ∧ C.εd ≤ D.real (ReadModel.searchAt (read · ω) C.k s'.tree s'.edges)) tallyStart
              (List.ofFn xs)} := by
        intro xs hxs
        by_cases hg : RunEnds (tallyStep C cut) (GoodEnd G) tallyStart (List.ofFn xs)
        · rcases runEnds_or (Q := TallyEndsWell G D C (read · ω)) _ _ hg with h | h
          · exact absurd h hxs
          · refine .inr (runEnds_mono (fun e s' h => ?_) _ _ h)
            obtain ⟨hge, hns⟩ := h
            rcases e with _ | _ | e | _ | _
            · simp only [TallyEndsWell, tallyEnd, EndsWell, not_le] at hns
              exact ⟨rfl, hns.le⟩
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact hge.elim
        · exact .inl hg
      have hsplit : ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a)
          + ENNReal.ofReal (T * T * C.a)
          = ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 3 * (T * T)) * C.a) := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        ring
      have hok : (Measure.pi fun _ : Fin T => D) {xs | ¬ RunEnds (tallyStep C cut) (OkEnd G S)
          tallyStart (List.ofFn xs)}
          ≤ ENNReal.ofReal φ + ENNReal.ofReal ((S + Fintype.card σ + 1) * δ') := by
        refine (measure_mono hokb).trans ((measure_union_le _ _).trans ?_)
        rw [add_comm]
        refine add_le_add h1' (h1.trans (le_of_eq ?_))
        congr 1
      refine (measure_mono hsnd).trans ((measure_union_le _ _).trans ((add_le_add
        ((measure_mono hsub').trans ((measure_union_le _ _).trans (add_le_add hok h2))) h3).trans
        (le_of_eq ?_)))
      simp only [c]
      rw [add_assoc, hsplit]
    · have hM : ω ∈ M := subset_toMeasurable _ _ hω
      rw [Set.indicator_of_mem hM, Pi.one_apply]
      exact prob_le_one.trans le_self_add
  calc ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ ∫⁻ ω, (M.indicator 1 ω + c) ∂μ := lintegral_mono hpt
    _ = μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)} + c := by
      rw [lintegral_add_right _ measurable_const, lintegral_indicator_one
        (measurableSet_toMeasurable _ _), measure_toMeasurable, lintegral_const, measure_univ,
        mul_one]
    _ = _ := by simp only [c]; ring

end OrthoDFA
