import OrthoDFA.Proofs.RoundStrong
import OrthoDFA.HalvingStrong

/-!
# What an attempt on an edge comes to

An attempt reached from a probe's search never stops: the search ends between two decided sifts,
the witness sifts decided to the edge's source and, followed by the letter, to its target, and the
parting reads only along those decided paths. So it splits or adds a member.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem parting_ne_inr {cut : FreeMonoid α → Option Bool} {x y pre b : FreeMonoid α} :
    ∀ (t : DTree α) {p p' : List Bool}, t.sift cut (x * pre) = .inl p →
      t.sift cut (y * pre) = .inl p' → t.parting cut x y pre ≠ some (.inr b)
  | .leaf, _, _, _, _ => by simp [DTree.parting]
  | .node m r a, p, p', hx, hy => by
    unfold DTree.sift DTree.route at hx hy
    simp only [mul_assoc] at hx hy
    simp only [DTree.parting]
    rcases hcx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hcy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hcx, hcy, reduceCtorEq] at hx hy ⊢ <;> try simp
    · rcases hrx : (r.route cut (x * pre)).2 with q | q <;> rw [hrx] at hx
      · rcases hry : (r.route cut (y * pre)).2 with q' | q' <;> rw [hry] at hy
        · exact parting_ne_inr r hrx hry
        · simp at hy
      · simp at hx
    · rcases hrx : (a.route cut (x * pre)).2 with q | q <;> rw [hrx] at hx
      · rcases hry : (a.route cut (y * pre)).2 with q' | q' <;> rw [hry] at hy
        · exact parting_ne_inr a hrx hry
        · simp at hy
      · simp at hx

variable (K : StageKnobs α) (R : CutReads α)

/-- An attempt on the edge a probe's search ends at never stops. -/
theorem seedStep_ne_stopped {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (h : probeOutcome R t edges k x = .edge ps fd) (b : FreeMonoid α) :
    seedStep K R t pool edges k x ps fd ≠ .stopped b := by
  obtain ⟨ps₀, hi, hw, hb⟩ := probeOutcome_search R h trivial
  obtain ⟨p₀, hk, hf, hkh, hhn, hpn⟩ := walkCheck_inr R hw
  set walkAt : ℕ → List Bool := fun j => ps₀.getD (j - k) [] with hwalk
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  obtain ⟨rfl, hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) ps₀ (hi - k) k hi ps fd hkh le_rfl hpk hpn hb
  unfold seedStep
  simp only []
  have hfdn : fd - 1 < x.toList.length := by omega
  rw [List.getElem?_eq_getElem hfdn]
  simp only []
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : ((x.toList.drop k).take (hi - k))[fd - 1 - k]'(by simp; omega)
      = x.toList[fd - 1] := by
    simp only [List.getElem_take, List.getElem_drop]
    congr 1
    omega
  rw [hidx, show fd - 1 - k + 1 = fd - k by omega] at hy
  rw [hy]
  simp only [ne_eq, not_true_eq_false, if_false]
  obtain ⟨hy1, hy2⟩ := hl _ _ _ _ hy
  rcases hsp : t.sift R.cut (prefixOf x (fd - 1)) with p | b'
  · simp only []
    have hp : p = ps.getD (fd - 1 - k) [] := by
      have := hfd3
      simp only [agreesAt, hsp, Sum.elim_inl, hwalk, Option.some.injEq, decide_eq_true_eq] at this
      exact this
    subst hp
    simp only [ne_eq, not_true_eq_false, false_or, hy1, not_true_eq_false, if_false]
    rcases hpt : t.parting R.cut y (prefixOf x (fd - 1)) (FreeMonoid.of x.toList[fd - 1]) with
      _ | d | b''
    · simp
    · simp only []
      split <;> simp
    · exfalso
      have hs4 : ∃ p', t.sift R.cut (prefixOf x (fd - 1) * FreeMonoid.of x.toList[fd - 1])
          = .inl p' := by
        rw [← prefixOf_succ hfdn, show fd - 1 + 1 = fd by omega]
        rcases hs : t.sift R.cut (prefixOf x fd) with p' | _
        · exact ⟨p', rfl⟩
        · simp [agreesAt, hs] at hfd4
      obtain ⟨p', hp'⟩ := hs4
      exact parting_ne_inr t hy2 hp' hpt
  · exfalso
    unfold agreesAt at hfd3
    rw [hsp] at hfd3
    simp at hfd3

theorem strongStep_stopped (C : StrongCfg α) (A : RoundAcc α) (x : FreeMonoid α)
    (hl : Learned R A.s.tree A.s.edges) : (strongStep C R A x).stopped = A.stopped := by
  unfold strongStep
  simp only []
  split
  · rename_i ps fd ho
    have hns := seedStep_ne_stopped C.K R (pool := A.s.pool) hl ho
    rcases hs : seedStep C.K R A.s.tree A.s.pool A.s.edges C.k x ps fd with
      ⟨d, s1, y, sp⟩ | ⟨s1, sp⟩ | b | _
    case stopped => exact absurd hs (hns b)
    all_goals first
      | rfl
      | (split <;> (try split_ifs) <;> simp [SeedResult.stoppedAt])
  · rfl

theorem fold_stopped (C : StrongCfg α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), StrongInv R A →
      (probes.foldl (passBody C R) A).stopped = A.stopped := by
  intro probes
  induction probes with
  | nil => intro A _; rfl
  | cons x xs ih =>
    intro A hI
    simp only [List.foldl_cons]
    by_cases hg : C.K.patience ≤ A.s.streak ∨ budgetOf C A.s.tree.paths.length ≤ A.used
    · have h1 : passBody C R A x = A := by unfold passBody; rw [if_pos hg]
      rw [h1]; exact ih A hI
    · have h1 : passBody C R A x = strongStep C R A x := by unfold passBody; rw [if_neg hg]
      rw [h1, ih _ (strongStep_inv C R A x hI), strongStep_stopped R C A x hI.1]

theorem fold_strongInv (C : StrongCfg α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), StrongInv R A →
      StrongInv R (probes.foldl (passBody C R) A) := by
  intro probes
  induction probes with
  | nil => exact fun A h => h
  | cons x xs ih =>
    intro A hI
    simp only [List.foldl_cons]
    by_cases hg : C.K.patience ≤ A.s.streak ∨ budgetOf C A.s.tree.paths.length ≤ A.used
    · have h1 : passBody C R A x = A := by unfold passBody; rw [if_pos hg]
      rw [h1]; exact ih A hI
    · have h1 : passBody C R A x = strongStep C R A x := by unfold passBody; rw [if_neg hg]
      rw [h1]; exact ih _ (strongStep_inv C R A x hI)

theorem round_strong_no_stop : RoundStrongNoStop := by
  intro α _ _ C R seed Rmax d
  suffices h : ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      StrongInv R A → (strongRound C R n j A first d).2.1.stopped = A.stopped by
    exact h Rmax 0 _ [] d (startAcc_inv C R seed)
  intro n
  induction n with
  | zero => intro j A first d _; rfl
  | succ n ih =>
    intro j A first d hI
    have hacc := strongReading_acc C R j A first (d 0)
    obtain ⟨hs, -, hsp, -⟩ := strongReading_fst C R j A first (d 0)
    have hp : (strongPass C R A (first ++ List.ofFn (d 0).1)).stopped = A.stopped := by
      rw [strongPass_eq]
      exact fold_stopped R C _ _ hI
    have hIR : StrongInv R (strongReading C R j A first (d 0)).1 := by
      unfold StrongInv
      rw [hs, hsp, strongPass_eq]
      exact fold_strongInv R C _ _ hI
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hacc hIR <;> dsimp only at hacc hIR <;> simp only [strongRound, hR]
    · rw [hacc]; exact hp
    · rw [ih (j + 1) A'' lv (Fin.tail d) hIR, hacc]; exact hp

end OrthoDFA
