import OrthoDFA.Proofs.LoopCheckFirst
import OrthoDFA.Proofs.ProbeAgree
import OrthoDFA.Proofs.Level3CheckSound

/-!
# The checked round's DFA, from its check and its family's cut

The checked round's tree reads `ε` at its root, so a string whose probe from `0` agrees is
classified as read by the DFA started where `ε` sifts (`ProbeAgreeCorrect`), and a decided read on
the wrong side is a seeded miscut or a seeded indecision (`misread_cut`). So the DFA misjudges at
most the strings failing the check from `0` and those the family misreads (`check_dfa_of_reads`).
The check certifies its rate over its own strings but for chance `δ/8`, whatever the reads
(`Level3CheckSound`); in the loop, given the noise and the clustering, so too
(`acc_round0_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The tree's root reads the string itself. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

omit [DecidableEq α] in
theorem rootOne_splitAt {T : DTree α} (h : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := h
  rcases p with _ | ⟨_ | _, p⟩ <;> exact ⟨_, _, rfl⟩

omit [DecidableEq α] in
theorem rootOne_of_classSet : ∀ (n : ℕ) (T : DTree α), T ∈ Random.classSet n → RootOne T
  | 0, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_singleton] at h
    exact ⟨_, _, h⟩
  | n + 1, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T₀, hT₀, _, _, rfl⟩
    · exact rootOne_of_classSet n T h
    · exact rootOne_splitAt (rootOne_of_classSet n T₀ hT₀) _ _

theorem rootOne_check (read : FreeMonoid α → ARU) (C : Level3Settings) (K : CheckSettings)
    (ds : Fin C.probeBudget → CheckDraw α K) :
    RootOne (indecisiveLStarCheck C K read ds).hypothesis.tree := by
  obtain ⟨h1, -⟩ := runChecked_tree read C K (X := (List.ofFn ds).map Prod.fst)
    (Y := (List.ofFn ds).flatMap fun d => List.ofFn d.2) (List.ofFn ds) initialState
    (fun d hd => ⟨List.mem_map_of_mem hd, fun i => List.mem_flatMap.2 ⟨d, hd, by simp⟩⟩)
    (checkRunInv_start read C _ _)
  exact rootOne_of_classSet _ _ h1.inv.cls

omit [Fintype α] in
/-- A string passing the check from `0` neither searches nor stops undecided at its ends. -/
theorem not_dis_of_startFails {read : FreeMonoid α → ARU} {H : HypothesisDFA α} {x : FreeMonoid α}
    (h : startFails read H.tree H.edges x = false) :
    x ∉ searchSet read 0 H ∪ endsSet read 0 H := by
  have hp : probeOrAgree read H.tree H.edges 0 x = probe read H.tree H.edges 0 x := by
    simp [probeOrAgree]
  unfold startFails at h
  simp only [Set.mem_union, searchSet, endsSet, Set.mem_ofPred_eq, hp, not_or]
  generalize probe read H.tree H.edges 0 x = o at h
  rcases o with _ | _ | _ | _ | ⟨zs, b⟩ <;> simp_all [endsString]
  rcases zs with _ | _ <;> rcases b <;> simp_all

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

omit [DecidableEq α] in
open scoped Classical in
/-- A read on the wrong side is, with the seed's vote, a miscut or undecided. -/
lemma misread_cut (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (p : FreeMonoid α) (ω : Ω)
    (hω : familyRead O.mq F lo (hi + 1) p ω = if p ∈ O.L then .reject else .accept) :
    ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω ∨ ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω := by
  classical
  simp only [familyRead] at hω
  simp only [cutCorrect, decided]
  have hb := voteCount_insert O.mq hF p ω
  have hb01 : (if O.mq (p * 1) ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  set v := voteCount O.mq F p ω
  set vf := voteCount O.mq (insert 1 F) p ω
  by_cases hp : p ∈ O.L
  · have hl : O.label p = 1 := by simp [Oracle.label, hp]
    rw [if_pos hp] at hω
    have hv : v ≤ lo := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h2
    by_cases hlo : vf ≤ lo
    · left
      intro hcc
      have := hcc.2 hlo
      rw [hl] at this
      norm_num at this
    · right
      omega
  · have hl : O.label p = 0 := by simp [Oracle.label, hp]
    rw [if_neg hp] at hω
    have hv : hi + 1 ≤ v := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h1
    by_cases hhi : hi + 1 < vf
    · left
      intro hcc
      have := hcc.1 hhi
      rw [hl] at this
      norm_num at this
    · right
      omega

/-- Where some string passes the check from `0`, the DFA started where `ε` sifts misjudges only
strings failing the check or misread by the family with its seed. -/
theorem check_dfa_of_reads {σ : Type*} (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (ω : Ω) (H : HypothesisDFA α) (hroot : RootOne H.tree)
    (hx₀ : ∃ x, startFails (fun z => familyRead O.mq F lo (hi + 1) z ω) H.tree H.edges x = false) :
    ∃ s, H.tree.sift (fun z => familyRead O.mq F lo (hi + 1) z ω) 1 = .inl s
      ∧ {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)}
        ⊆ {x | startFails (fun z => familyRead O.mq F lo (hi + 1) z ω) H.tree H.edges x = true}
          ∪ ({p | ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω}
            ∪ {p | ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω}) := by
  classical
  set read := fun z => familyRead O.mq F lo (hi + 1) z ω
  have hpa := probeAgreeCorrect (α := α)
  obtain ⟨x₀, hx₀⟩ := hx₀
  obtain ⟨-, h0⟩ := hpa read H 0 x₀ (Nat.zero_le _) (not_dis_of_startFails hx₀)
  obtain ⟨s, hs, -⟩ := h0 rfl hroot
  refine ⟨s, hs, fun x hx => ?_⟩
  by_contra hn
  simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, Bool.not_eq_true] at hn
  obtain ⟨hnd, hnc, hnu⟩ := hn
  obtain ⟨-, h1⟩ := hpa read H 0 x (Nat.zero_le _) (not_dis_of_startFails hnd)
  obtain ⟨s', hs', hdec, hcl⟩ := h1 rfl hroot
  rw [hs] at hs'
  cases hs'
  have hright : (read x = .accept ↔ M.eval x.toList ∈ M.accept) := by
    rw [← hL x]
    by_contra hw
    have hwrong : familyRead O.mq F lo (hi + 1) x ω = if x ∈ O.L then .reject else .accept := by
      show read x = _
      rcases hr : read x with _ | _ | _
      · rw [hr] at hw; split_ifs with hx' <;> simp_all
      · rw [hr] at hw; split_ifs with hx' <;> simp_all
      · exact absurd hr hdec
    rcases misread_cut O hF hlh x ω hwrong with h | h
    · exact hnc h
    · exact hnu h
  apply hx
  obtain ⟨q, hq, hqb⟩ := hcl
  exact ⟨q, hq, hqb.trans hright⟩

end OrthoDFA
