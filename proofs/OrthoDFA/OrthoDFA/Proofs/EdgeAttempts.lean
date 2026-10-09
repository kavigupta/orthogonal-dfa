import OrthoDFA.Proofs.RoundStrong

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
    {T : Tested α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (h : probeOutcome R t edges k x = .edge ps fd) (b : FreeMonoid α) :
    seedStep K R t pool edges T k x ps fd ≠ .stopped b := by
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

theorem round_strong_no_stop : RoundStrongNoStop := by
  intro α _ _ C R seed Rmax d x ps fd s ho
  have hl := (strongRound_preserves C R (fun _ => True) (fun _ _ _ _ _ _ => trivial)
    (fun _ _ _ _ => trivial) Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) trivial).1.1
  have hnd := seedStep_ne_dropped C.K R (pool := s.pool) (T := s.tested) hl ho
  have hns := seedStep_ne_stopped C.K R (pool := s.pool) (T := s.tested) hl ho
  rcases hs : seedStep C.K R s.tree s.pool s.edges s.tested C.k x ps fd with
    ⟨dd, s1, y, sp⟩ | ⟨s1, sp, dd⟩ | b | _
  · exact .inl ⟨dd, s1, y, sp, rfl⟩
  · exact .inr ⟨s1, sp, dd, rfl⟩
  · exact absurd hs (hns b)
  · exact absurd hs hnd

end OrthoDFA
