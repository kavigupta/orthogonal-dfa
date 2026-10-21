import OrthoDFA.Level5Claim
import OrthoDFA.Proofs.EpsSteps

/-!
# What a probe reads

A sift reads its string followed by the midfixes on its way, up to its first undecided read
(`siftReads`). A probe reads its start sift, the sift of the successor of the unlearned edge its
walk stops at, if any, and the sifts its search queries, with those at the edge it finds or the
undecided position it stops at (`probeReads`). These reads fix the probe and themselves
(`probeReads_congr`). A string among them read undecided is the one the probe stops at, which the
tests count (`probeReads_undecided`), and an edge the probe learns or disagrees at sifts its two
strings with reads among them (`probeReads_edge`).
-/

namespace OrthoDFA

namespace Random

namespace Kappa

variable {α : Type*}

/-! ## Sifts -/

section Sift

open scoped Classical in
/-- The strings sifting `z` reads. -/
noncomputable def siftReads (read : FreeMonoid α → ARU) :
    DTree α → FreeMonoid α → Finset (FreeMonoid α)
  | .leaf, _ => ∅
  | .node m r a, z => insert (z * m) (match read (z * m) with
    | .accept => siftReads read a z
    | .reject => siftReads read r z
    | .undecided => ∅)

theorem siftReads_congr {rd rd' : FreeMonoid α → ARU} : ∀ (T : DTree α) (z : FreeMonoid α),
    (∀ w ∈ siftReads rd T z, rd w = rd' w) →
    T.sift rd' z = T.sift rd z ∧ siftReads rd' T z = siftReads rd T z
  | .leaf, _, _ => ⟨rfl, rfl⟩
  | .node m r a, z, h => by
    classical
    have hm : rd (z * m) = rd' (z * m) := h _ (by simp [siftReads])
    have hsub : ∀ w ∈ (match rd (z * m) with
        | .accept => siftReads rd a z
        | .reject => siftReads rd r z
        | .undecided => ∅), rd w = rd' w := fun w hw => h w (by
      simp only [siftReads]; exact Finset.mem_insert_of_mem hw)
    rcases hc : rd (z * m) with _ | _ | _ <;> rw [hc] at hsub <;> rw [hc] at hm
    · obtain ⟨h1, h2⟩ := siftReads_congr a z hsub
      simp only [DTree.sift, siftReads, ← hm, hc, h1, h2]
      exact ⟨trivial, trivial⟩
    · obtain ⟨h1, h2⟩ := siftReads_congr r z hsub
      simp only [DTree.sift, siftReads, ← hm, hc, h1, h2]
      exact ⟨trivial, trivial⟩
    · simp only [DTree.sift, siftReads, ← hm, hc]
      exact ⟨trivial, trivial⟩

/-- A sift reads a string undecided only where it stops. -/
theorem siftReads_undecided {rd : FreeMonoid α → ARU} : ∀ (T : DTree α) (z w : FreeMonoid α),
    w ∈ siftReads rd T z → rd w = .undecided → T.sift rd z = .inr w
  | .leaf, _, _, h, _ => by simp [siftReads] at h
  | .node m r a, z, w, h, hw => by
    classical
    simp only [siftReads, Finset.mem_insert] at h
    rcases h with rfl | h
    · simp [DTree.sift, hw]
    · rcases hc : rd (z * m) with _ | _ | _ <;> rw [hc] at h
      · simp [DTree.sift, hc, siftReads_undecided a z w h hw]
      · simp [DTree.sift, hc, siftReads_undecided r z w h hw]
      · simp at h

/-- A sift reads its string followed by midfixes. -/
theorem siftReads_mids {rd : FreeMonoid α → ARU} : ∀ (T : DTree α) (z w : FreeMonoid α),
    w ∈ siftReads rd T z → ∃ m ∈ mids T, w = z * m
  | .leaf, _, _, h => by simp [siftReads] at h
  | .node m r a, z, w, h => by
    classical
    simp only [siftReads, Finset.mem_insert] at h
    rcases h with rfl | h
    · exact ⟨m, by simp [mids], rfl⟩
    · rcases hc : rd (z * m) with _ | _ | _ <;> rw [hc] at h
      · obtain ⟨m', hm', rfl⟩ := siftReads_mids a z w h
        exact ⟨m', by simp [mids, hm'], rfl⟩
      · obtain ⟨m', hm', rfl⟩ := siftReads_mids r z w h
        exact ⟨m', by simp [mids, hm'], rfl⟩
      · simp at h

variable {σ : Type*} (M : DFA α σ) (side : σ → Bool)

/-- A sift with no wrong read among its reads that ends at a leaf ends at the true leaf. -/
theorem sift_true_reads (rd : FreeMonoid α → ARU) : ∀ (T : DTree α) (z : FreeMonoid α)
    (p : List Bool), T.sift rd z = .inl p →
    (∀ w ∈ siftReads rd T z, rd w ≠ if side (M.eval w.toList) then .reject else .accept) →
    p = leafOf M side T (M.eval z.toList)
  | .leaf, z, p, h, _ => by simp only [DTree.sift, Sum.inl.injEq] at h; simp [leafOf, ← h]
  | .node m r a, z, p, h, hw => by
    classical
    have hm := hw (z * m) (by simp [siftReads])
    simp only [DTree.sift] at h
    unfold leafOf
    rw [← eval_mul]
    rcases hc : rd (z * m) with _ | _ | _ <;> rw [hc] at h hm
    · have hs : side (M.eval (z * m).toList) = true := by
        cases hsd : side (M.eval (z * m).toList)
        · exact absurd (by rw [hsd]; simp) hm
        · rfl
      rcases hq : a.sift rd z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_pos hs, ← h, sift_true_reads rd a z q hq fun w hw' => hw w (by
        simp only [siftReads, hc]; exact Finset.mem_insert_of_mem hw')]
    · have hs : side (M.eval (z * m).toList) = false := by
        cases hsd : side (M.eval (z * m).toList)
        · rfl
        · exact absurd (by rw [hsd]; simp) hm
      rcases hq : r.sift rd z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_neg (by rw [hs]; exact Bool.false_ne_true), ← h,
        sift_true_reads rd r z q hq fun w hw' => hw w (by
          simp only [siftReads, hc]; exact Finset.mem_insert_of_mem hw')]
    · simp at h

end Sift

/-! ## The search -/

section Search

theorem bisectBoundary_unf (g : ℕ → Option Bool) {lo hi : ℕ} (hlt : ¬ hi ≤ lo + 1) :
    bisectBoundary false g lo hi = match g ((lo + hi) / 2) with
      | some true => ((bisectBoundary false g ((lo + hi) / 2) hi).1,
          (lo + hi) / 2 :: (bisectBoundary false g ((lo + hi) / 2) hi).2)
      | some false => ((bisectBoundary false g lo ((lo + hi) / 2)).1,
          (lo + hi) / 2 :: (bisectBoundary false g lo ((lo + hi) / 2)).2)
      | none => (.undecidedMid ((lo + hi) / 2), [(lo + hi) / 2]) := by
  rw [bisectBoundary.eq_1]
  simp only [hlt, dite_false]
  rcases g ((lo + hi) / 2) with _ | _ | _ <;> rfl

/-- Agreeing on what the bisection queries gives the same bisection. -/
theorem bisectBoundary_congrQ {ag ag' : ℕ → Option Bool} :
    ∀ (n lo hi : ℕ), hi - lo = n → (∀ i ∈ (bisectBoundary false ag lo hi).2, ag i = ag' i) →
      bisectBoundary false ag' lo hi = bisectBoundary false ag lo hi := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn h
  by_cases hlt : hi ≤ lo + 1
  · rw [bisectBoundary.eq_1 false ag lo hi, bisectBoundary.eq_1 false ag' lo hi]
    simp [hlt]
  rw [bisectBoundary_unf ag hlt] at h
  rw [bisectBoundary_unf ag hlt, bisectBoundary_unf ag' hlt]
  have hmid : ag ((lo + hi) / 2) = ag' ((lo + hi) / 2) := by
    apply h
    rcases ag ((lo + hi) / 2) with _ | _ | _ <;> simp
  rw [← hmid]
  rcases hc : ag ((lo + hi) / 2) with _ | _ | _
  · rfl
  · rw [hc] at h
    dsimp only at h ⊢
    rw [ih _ (by omega) lo _ rfl fun i hi => h i (List.mem_cons_of_mem _ hi)]
  · rw [hc] at h
    dsimp only at h ⊢
    rw [ih _ (by omega) _ hi rfl fun i hi => h i (List.mem_cons_of_mem _ hi)]

/-- Agreeing on what the search queries gives the same search. -/
theorem tripartiteSearch_congrQ {ag ag' : ℕ → Option Bool} {lo hi : ℕ}
    (h : ∀ i ∈ (tripartiteSearch false ag lo hi).2, ag i = ag' i) :
    tripartiteSearch false ag' lo hi = tripartiteSearch false ag lo hi := by
  unfold tripartiteSearch at h ⊢
  have hlo : ag lo = ag' lo := by
    apply h
    rcases ag lo with _ | _ | _
    · simp
    · simp
    · dsimp only; rcases ag hi with _ | _ | _ <;> simp
  rw [← hlo]
  rcases hc : ag lo with _ | _ | _
  · rfl
  · rfl
  rw [hc] at h
  dsimp only at h ⊢
  have hhi : ag hi = ag' hi := by
    apply h
    rcases ag hi with _ | _ | _ <;> simp
  rw [← hhi]
  rcases hd : ag hi with _ | _ | _
  · rfl
  · rw [hd] at h
    dsimp only at h ⊢
    rw [bisectBoundary_congrQ _ lo hi rfl fun i hi =>
      h i (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hi))]
  · rfl

/-- A position the bisection queries undecided is where it stops. -/
theorem bisectBoundary_undecided {ag : ℕ → Option Bool} :
    ∀ (n lo hi : ℕ), hi - lo = n → ∀ p ∈ (bisectBoundary false ag lo hi).2, ag p = none →
      (bisectBoundary false ag lo hi).1 = .undecidedMid p := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn p hp hnone
  rw [bisectBoundary.eq_1 false ag lo hi] at hp ⊢
  by_cases hlt : hi ≤ lo + 1
  · simp [hlt] at hp
  simp only [hlt, dite_false] at hp ⊢
  rcases hc : ag ((lo + hi) / 2) with _ | b
  · simp only [hc, Bool.false_eq_true, if_false, List.mem_singleton] at hp ⊢
    rw [hp]
  · cases b
    · simp only [hc, List.mem_cons] at hp ⊢
      rcases hp with rfl | hp
      · rw [hc] at hnone; simp at hnone
      · exact ih _ (by omega) _ _ rfl p hp hnone
    · simp only [hc, List.mem_cons] at hp ⊢
      rcases hp with rfl | hp
      · rw [hc] at hnone; simp at hnone
      · exact ih _ (by omega) _ _ rfl p hp hnone

/-- A position the search queries undecided is where it stops. -/
theorem tripartiteSearch_undecided {ag : ℕ → Option Bool} {lo hi p : ℕ}
    (hp : p ∈ (tripartiteSearch false ag lo hi).2) (hnone : ag p = none) :
    ((tripartiteSearch false ag lo hi).1 = .undecidedLo ∧ p = lo)
      ∨ ((tripartiteSearch false ag lo hi).1 = .undecidedHi ∧ p = hi)
      ∨ (tripartiteSearch false ag lo hi).1 = .undecidedMid p := by
  unfold tripartiteSearch at hp ⊢
  rcases hlo : ag lo with _ | b
  · simp only [hlo, List.mem_singleton] at hp ⊢
    exact Or.inl ⟨trivial, hp⟩
  cases b
  · simp only [hlo, List.mem_singleton] at hp ⊢
    subst hp; rw [hlo] at hnone; simp at hnone
  simp only [hlo] at hp ⊢
  rcases hhi : ag hi with _ | b
  · simp only [hhi, List.mem_cons, List.not_mem_nil, or_false] at hp ⊢
    rcases hp with rfl | rfl
    · rw [hlo] at hnone; simp at hnone
    · exact Or.inr (Or.inl ⟨trivial, rfl⟩)
  cases b
  · simp only [hhi, List.mem_cons] at hp ⊢
    rcases hp with rfl | rfl | hp
    · rw [hlo] at hnone; simp at hnone
    · rw [hhi] at hnone; simp at hnone
    · exact Or.inr (Or.inr (bisectBoundary_undecided _ lo hi rfl p hp hnone))
  · simp only [hhi, List.mem_cons, List.not_mem_nil, or_false] at hp ⊢
    rcases hp with rfl | rfl
    · rw [hlo] at hnone; simp at hnone
    · rw [hhi] at hnone; simp at hnone

end Search

/-! ## The probe -/

section Probe

/-- The positions whose sifts a search reads: those it queries, and those its result names. -/
def searchPos (res : SearchResult × List ℕ) (hi : ℕ) : List ℕ :=
  res.2 ++ match res.1 with
    | .edge p => [p - 1, p]
    | .undecidedMid p => [p]
    | .undecidedHi => [hi]
    | _ => []

open scoped Classical in
/-- The strings the sifts of the prefixes of `x` at the positions `ps` read. -/
noncomputable def posReads (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α)
    (ps : List ℕ) : Finset (FreeMonoid α) :=
  ps.toFinset.biUnion fun p => siftReads read T (prefixOf x p)

open scoped Classical in
/-- The strings a probe reads. -/
noncomputable def probeReads (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  if x.toList.length < k then ∅ else
  siftReads read T (prefixOf x k) ∪
    match T.sift read (prefixOf x k) with
    | .inr _ => ∅
    | .inl a =>
      let hi := k + (walk E a (x.toList.drop k)).length - 1
      let sr := posReads read T x (searchPos (tripartiteSearch false
        (siftAgrees read T x fun p => (walk E a (x.toList.drop k)).getD (p - k) []) k hi) hi)
      match x.toList[hi]? with
      | some _ => siftReads read T (prefixOf x (hi + 1)) ∪
        match T.sift read (prefixOf x (hi + 1)) with
        | .inr _ => ∅
        | .inl _ => sr
      | none => sr

theorem mem_posReads {rd : FreeMonoid α → ARU} {T : DTree α} {x w : FreeMonoid α}
    {ps : List ℕ} : w ∈ posReads rd T x ps ↔ ∃ p ∈ ps, w ∈ siftReads rd T (prefixOf x p) := by
  classical
  simp [posReads]

theorem posReads_congr {rd rd' : FreeMonoid α → ARU} {T : DTree α} {x : FreeMonoid α}
    {ps : List ℕ} (h : ∀ w ∈ posReads rd T x ps, rd w = rd' w) :
    (∀ p ∈ ps, T.sift rd' (prefixOf x p) = T.sift rd (prefixOf x p))
      ∧ posReads rd' T x ps = posReads rd T x ps := by
  classical
  have hp : ∀ p ∈ ps, T.sift rd' (prefixOf x p) = T.sift rd (prefixOf x p)
      ∧ siftReads rd' T (prefixOf x p) = siftReads rd T (prefixOf x p) := fun p hp =>
    siftReads_congr T _ fun w hw => h w (mem_posReads.2 ⟨p, hp, hw⟩)
  refine ⟨fun p hp' => (hp p hp').1, ?_⟩
  unfold posReads
  exact Finset.biUnion_congr rfl fun p hp' => (hp p (List.mem_toFinset.1 hp')).2

variable {rd rd' : FreeMonoid α → ARU} {T : DTree α} {x : FreeMonoid α}

/-- Agreeing on what a search reads gives the same search, outcome and reads. -/
theorem search_congr (st : ℕ → List Bool) (lo hi : ℕ)
    (h : ∀ w ∈ posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) lo hi)
      hi), rd w = rd' w) :
    tripartiteSearch false (siftAgrees rd' T x st) lo hi
        = tripartiteSearch false (siftAgrees rd T x st) lo hi
      ∧ posReads rd' T x (searchPos (tripartiteSearch false (siftAgrees rd' T x st) lo hi) hi)
        = posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) lo hi) hi)
      ∧ ∀ un, searchOutcome rd' T x st un hi
          (tripartiteSearch false (siftAgrees rd' T x st) lo hi).1
        = searchOutcome rd T x st un hi
          (tripartiteSearch false (siftAgrees rd T x st) lo hi).1 := by
  obtain ⟨hs, hp⟩ := posReads_congr h
  have hts : tripartiteSearch false (siftAgrees rd' T x st) lo hi
      = tripartiteSearch false (siftAgrees rd T x st) lo hi :=
    tripartiteSearch_congrQ fun i hi' => by
      unfold siftAgrees
      rw [hs i (List.mem_append_left _ hi')]
  rw [hts]
  refine ⟨rfl, hp, fun un => ?_⟩
  generalize hres : tripartiteSearch false (siftAgrees rd T x st) lo hi = res at hs
  obtain ⟨r1, r2⟩ := res
  have hx : ∀ p ∈ (match r1 with
      | .edge p => [p - 1, p]
      | .undecidedMid p => [p]
      | .undecidedHi => [hi]
      | _ => []), T.sift rd' (prefixOf x p) = T.sift rd (prefixOf x p) :=
    fun p hp' => hs p (List.mem_append_right _ hp')
  cases r1 with
  | edge p =>
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hx
    simp only [searchOutcome, hx.2]
  | undecidedMid p =>
    simp only [List.mem_singleton, forall_eq] at hx
    simp only [searchOutcome, undecidedAt, hx]
  | undecidedHi =>
    simp only [List.mem_singleton, forall_eq] at hx
    simp only [searchOutcome, undecidedAt, hx]
  | _ => rfl

variable (E : Edges α) (k : ℕ)

/-- Agreeing on what a probe reads gives the same probe and reads. -/
theorem probeReads_congr (h : ∀ w ∈ probeReads rd T E k x, rd w = rd' w) :
    probeOrAgree rd' T E k x = probeOrAgree rd T E k x
      ∧ probeReads rd' T E k x = probeReads rd T E k x := by
  classical
  unfold probeOrAgree probeReads at *
  by_cases hk : x.toList.length < k
  · simp [hk]
  simp only [hk, if_false] at h ⊢
  have h0 : ∀ w ∈ siftReads rd T (prefixOf x k), rd w = rd' w :=
    fun w hw => h w (Finset.mem_union_left _ hw)
  obtain ⟨hs0, hr0⟩ := siftReads_congr T _ h0
  unfold probe
  rw [hs0, hr0]
  rcases ha : T.sift rd (prefixOf x k) with a | z
  · rw [ha] at h
    have h1 := fun w hw => h w (Finset.mem_union_right _ hw)
    dsimp only at h1 ⊢
    rcases hc : x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c
    · rw [hc] at h1
      dsimp only at h1 ⊢
      obtain ⟨e1, e2, e3⟩ := search_congr _ _ _ h1
      rw [e2, e3]
      exact ⟨rfl, rfl⟩
    · rw [hc] at h1
      dsimp only at h1 ⊢
      have h2 : ∀ w ∈ siftReads rd T
          (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1)), rd w = rd' w :=
        fun w hw => h1 w (Finset.mem_union_left _ hw)
      obtain ⟨hs1, hr1⟩ := siftReads_congr T _ h2
      rw [hs1, hr1]
      rcases hb : T.sift rd (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1))
        with t | z
      · rw [hb] at h1
        dsimp only at h1 ⊢
        obtain ⟨e1, e2, e3⟩ := search_congr _ _ _ fun w hw => h1 w (Finset.mem_union_right _ hw)
        rw [e2, e3]
        exact ⟨rfl, rfl⟩
      · exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩

/-- The positions a search reads lie between its ends. -/
theorem searchPos_mem (ag : ℕ → Option Bool) {lo hi : ℕ} (hle : lo ≤ hi) :
    ∀ p ∈ searchPos (tripartiteSearch false ag lo hi) hi, lo ≤ p ∧ p ≤ hi := by
  intro p hp
  have hok := tripartiteSearch_ok (recover := false) (agrees := ag) hle
  rcases List.mem_append.1 hp with hp | hp
  · exact tripartiteSearch_mem false ag lo hi hle p hp
  generalize (tripartiteSearch false ag lo hi).1 = r at hp hok
  cases r with
  | edge q =>
    obtain ⟨h1, h2, -⟩ := hok
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    omega
  | undecidedMid q =>
    obtain ⟨-, h1, h2, -⟩ := hok
    simp only [List.mem_singleton] at hp
    omega
  | undecidedHi =>
    simp only [List.mem_singleton] at hp
    omega
  | _ => simp at hp

/-- A string a search reads undecided is the one its outcome carries. -/
theorem search_undecided (st : ℕ → List Bool) (un : Option (α × List Bool)) {lo hi : ℕ}
    (hle : lo ≤ hi) (hlo : T.sift rd (prefixOf x lo) = .inl (st lo)) {w : FreeMonoid α}
    (hw : w ∈ posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) lo hi)
      hi)) (hu : rd w = .undecided) :
    w ∈ endsString (searchOutcome rd T x st un hi
        (tripartiteSearch false (siftAgrees rd T x st) lo hi).1)
      ∨ w ∈ middleString (searchOutcome rd T x st un hi
        (tripartiteSearch false (siftAgrees rd T x st) lo hi).1) := by
  obtain ⟨p, hp, hwp⟩ := mem_posReads.1 hw
  have hsp := siftReads_undecided T _ w hwp hu
  have hag : siftAgrees rd T x st p = none := by simp [siftAgrees, hsp]
  have hag0 : siftAgrees rd T x st lo = some true := by simp [siftAgrees, hlo]
  have hok := tripartiteSearch_ok (recover := false) (agrees := siftAgrees rd T x st) hle
  rcases List.mem_append.1 hp with hq | hx
  · rcases tripartiteSearch_undecided hq hag with ⟨-, rfl⟩ | ⟨h1, rfl⟩ | h1
    · rw [hag0] at hag; simp at hag
    · rw [h1]; left; simp [searchOutcome, undecidedAt, hsp, endsString]
    · rw [h1]; right; simp [searchOutcome, undecidedAt, hsp, middleString]
  generalize (tripartiteSearch false (siftAgrees rd T x st) lo hi).1 = r at hx hok ⊢
  cases r with
  | edge q =>
    obtain ⟨-, -, h3, h4⟩ := hok
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · rw [h3] at hag; simp at hag
    · rw [h4] at hag; simp at hag
  | undecidedMid q =>
    simp only [List.mem_singleton] at hx
    subst hx
    right; simp [searchOutcome, undecidedAt, hsp, middleString]
  | undecidedHi =>
    simp only [List.mem_singleton] at hx
    subst hx
    left; simp [searchOutcome, undecidedAt, hsp, endsString]
  | _ => simp at hx

open scoped Classical in
/-- An edge a search's outcome names sifts its two strings with reads the search makes. -/
theorem search_edge (st : ℕ → List Bool) (un : Option (α × List Bool)) {lo hi : ℕ}
    (hle : lo ≤ hi) {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool} :
    (searchOutcome rd T x st un hi (tripartiteSearch false (siftAgrees rd T x st) lo hi).1
        = .wrongEdge p c u t →
      ∀ w ∈ siftReads rd T u ∪ siftReads rd T (u * FreeMonoid.of c),
        w ∈ posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) lo hi) hi))
    ∧ (searchOutcome rd T x st un hi (tripartiteSearch false (siftAgrees rd T x st) lo hi).1
        = .unlearnedEdge p c u t → un = some (c, t) ∧ u = prefixOf x hi
      ∧ ∀ w ∈ siftReads rd T u,
        w ∈ posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) lo hi)
          hi)) := by
  have hok := tripartiteSearch_ok (recover := false) (agrees := siftAgrees rd T x st) hle
  have hall : (tripartiteSearch false (siftAgrees rd T x st) lo hi).1 = .allAgree →
      hi ∈ (tripartiteSearch false (siftAgrees rd T x st) lo hi).2 := by
    intro h
    rw [h] at hok
    obtain ⟨h1, h2⟩ := hok
    simp [tripartiteSearch, h1, h2]
  unfold searchPos
  generalize (tripartiteSearch false (siftAgrees rd T x st) lo hi) = res at hok hall ⊢
  obtain ⟨r1, r2⟩ := res
  dsimp only at hok hall ⊢
  cases r1 with
  | edge q =>
    obtain ⟨h1, -, -, -⟩ := hok
    refine ⟨fun h w hw => ?_, fun h => ?_⟩
    · simp only [searchOutcome] at h
      split at h
      · rename_i c' t' hc ht
        simp only [Outcome.wrongEdge.injEq] at h
        obtain ⟨-, rfl, rfl, -⟩ := h
        have hq : prefixOf x (q - 1) * FreeMonoid.of c' = prefixOf x q := by
          rw [← prefixOf_succ hc, show q - 1 + 1 = q by omega]
        rw [hq] at hw
        refine mem_posReads.2 ?_
        rcases Finset.mem_union.1 hw with hw | hw
        · exact ⟨q - 1, List.mem_append_right _ (by simp), hw⟩
        · exact ⟨q, List.mem_append_right _ (by simp), hw⟩
      · simp at h
    · simp only [searchOutcome] at h
      split at h <;> simp at h
  | allAgree =>
    refine ⟨fun h => ?_, fun h => ?_⟩
    · simp only [searchOutcome] at h
      split at h <;> simp at h
    · simp only [searchOutcome] at h
      split at h
      · rename_i c' t'
        simp only [Outcome.unlearnedEdge.injEq] at h
        obtain ⟨-, rfl, rfl, rfl⟩ := h
        exact ⟨rfl, rfl, fun w hw => mem_posReads.2 ⟨hi, List.mem_append_left _ (hall rfl), hw⟩⟩
      · simp at h
  | _ => exact ⟨fun h => by simp [searchOutcome] at h, fun h => by simp [searchOutcome] at h⟩

/-- A string a probe reads undecided is the one it stops at, at its start or end or a middle. -/
theorem probeReads_undecided {w : FreeMonoid α} (hw : w ∈ probeReads rd T E k x)
    (hu : rd w = .undecided) :
    w ∈ endsString (probeOrAgree rd T E k x) ∨ w ∈ middleString (probeOrAgree rd T E k x) := by
  classical
  unfold probeReads at hw
  unfold probeOrAgree
  by_cases hk : x.toList.length < k
  · simp [hk] at hw
  rw [if_neg hk] at hw ⊢
  unfold probe
  rcases Finset.mem_union.1 hw with hw | hw
  · rw [siftReads_undecided T _ w hw hu]
    left; simp [endsString]
  rcases ha : T.sift rd (prefixOf x k) with a | z
  · rw [ha] at hw
    dsimp only at hw ⊢
    obtain ⟨-, hhix, hst⟩ := walk_facts E k x (p₀ := a) (by omega)
    have hpos := length_walk_pos E a (x.toList.drop k)
    have hlo : T.sift rd (prefixOf x k) = .inl ((walk E a (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, hst, ha]
    rcases hc : x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c
    · rw [hc] at hw
      dsimp only at hw ⊢
      exact search_undecided _ _ (by omega) hlo hw hu
    · rw [hc] at hw
      dsimp only at hw ⊢
      rcases Finset.mem_union.1 hw with hw | hw
      · rw [siftReads_undecided T _ w hw hu]
        left; simp [endsString]
      rcases hb : T.sift rd (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1))
        with t | z
      · rw [hb] at hw
        dsimp only at hw ⊢
        exact search_undecided _ _ (by omega) hlo hw hu
      · rw [hb] at hw; simp at hw
  · rw [ha] at hw; simp at hw

open scoped Classical in
/-- An edge a probe learns or disagrees at sifts its two strings with reads the probe makes. -/
theorem probeReads_edge {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probeOrAgree rd T E k x = .wrongEdge p c u t
      ∨ probeOrAgree rd T E k x = .unlearnedEdge p c u t) :
    ∀ w ∈ siftReads rd T u ∪ siftReads rd T (u * FreeMonoid.of c), w ∈ probeReads rd T E k x := by
  classical
  unfold probeOrAgree at h
  unfold probeReads
  by_cases hk : x.toList.length < k
  · simp [hk] at h
  rw [if_neg hk] at h ⊢
  unfold probe at h
  intro w hw
  refine Finset.mem_union_right _ ?_
  rcases ha : T.sift rd (prefixOf x k) with a | z
  · rw [ha] at h
    dsimp only at h ⊢
    obtain ⟨-, hhix, hst⟩ := walk_facts E k x (p₀ := a) (by omega)
    have hpos := length_walk_pos E a (x.toList.drop k)
    have hle : k ≤ k + (walk E a (x.toList.drop k)).length - 1 := by omega
    rcases hc : x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c'
    · rw [hc] at h
      dsimp only at h ⊢
      rcases h with h | h
      · exact (search_edge _ _ hle).1 h w hw
      · exact absurd (search_edge _ _ hle).2 (by intro h'; simpa using (h' h).1)
    · rw [hc] at h
      dsimp only at h ⊢
      rcases hb : T.sift rd (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1))
        with t' | z
      · rw [hb] at h
        dsimp only at h ⊢
        rcases h with h | h
        · exact Finset.mem_union_right _ ((search_edge _ _ hle).1 h w hw)
        · obtain ⟨h1, rfl, h3⟩ := (search_edge _ _ hle).2 h
          simp only [Option.some.injEq, Prod.mk.injEq] at h1
          obtain ⟨rfl, -⟩ := h1
          rcases Finset.mem_union.1 hw with hw | hw
          · exact Finset.mem_union_right _ (h3 w hw)
          · rw [← prefixOf_succ hc] at hw
            exact Finset.mem_union_left _ hw
      · rw [hb] at h; simp at h
  · rw [ha] at h; simp at h

/-- A probe reads prefixes at least `k` long followed by midfixes. -/
theorem probeReads_pot : probeReads rd T E k x ⊆ pot k x T := by
  classical
  intro w hw
  unfold probeReads at hw
  by_cases hk : x.toList.length < k
  · simp [hk] at hw
  rw [if_neg hk] at hw
  have hsift : ∀ p, k ≤ p → p ≤ x.toList.length → w ∈ siftReads rd T (prefixOf x p) →
      w ∈ pot k x T := fun p h1 h2 hwp => by
    obtain ⟨m, hm, rfl⟩ := siftReads_mids T _ w hwp
    exact mem_pot h1 h2 hm
  have hpos : ∀ (st : ℕ → List Bool) (hi : ℕ), k ≤ hi → hi ≤ x.toList.length →
      w ∈ posReads rd T x (searchPos (tripartiteSearch false (siftAgrees rd T x st) k hi) hi) →
      w ∈ pot k x T := fun st hi h1 h2 hwp => by
    obtain ⟨p, hp, hwp⟩ := mem_posReads.1 hwp
    obtain ⟨h3, h4⟩ := searchPos_mem _ h1 p hp
    exact hsift p h3 (by omega) hwp
  rcases Finset.mem_union.1 hw with hw | hw
  · exact hsift k le_rfl (by omega) hw
  rcases ha : T.sift rd (prefixOf x k) with a | z
  · rw [ha] at hw
    dsimp only at hw
    obtain ⟨-, hhix, -⟩ := walk_facts E k x (p₀ := a) (by omega)
    have hp := length_walk_pos E a (x.toList.drop k)
    rcases hc : x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c
    · rw [hc] at hw
      exact hpos _ _ (by omega) hhix hw
    · rw [hc] at hw
      dsimp only at hw
      have hlt : k + (walk E a (x.toList.drop k)).length - 1 < x.toList.length := by
        by_contra hge
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      rcases Finset.mem_union.1 hw with hw | hw
      · exact hsift _ (by omega) (by omega) hw
      rcases hb : T.sift rd (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1))
        with t | z
      · rw [hb] at hw
        exact hpos _ _ (by omega) hhix hw
      · rw [hb] at hw; simp at hw
  · rw [ha] at hw; simp at hw


/-- A search reads the sifts only of the positions it queries. -/
theorem searchPos_sub (ag : ℕ → Option Bool) {lo hi : ℕ} (hle : lo ≤ hi) :
    ∀ p ∈ searchPos (tripartiteSearch false ag lo hi) hi,
      p ∈ (tripartiteSearch false ag lo hi).2 := by
  intro p hp
  rcases List.mem_append.1 hp with hp | hp
  · exact hp
  have he := tripartiteSearch_edge_mem ag lo hi hle
  have hm := tripartiteSearch_undecidedMid_mem ag lo hi
  have hh := tripartiteSearch_undecidedHi_mem ag lo hi
  generalize (tripartiteSearch false ag lo hi).1 = r at hp he hm hh
  cases r with
  | edge q =>
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact (he _ rfl).2
    · exact (he _ rfl).1
  | undecidedMid q =>
    simp only [List.mem_singleton] at hp
    subst hp
    exact hm _ rfl
  | undecidedHi =>
    simp only [List.mem_singleton] at hp
    subst hp
    exact hh rfl
  | _ => simp at hp

theorem siftReads_card (rd : FreeMonoid α → ARU) (T : DTree α) (z : FreeMonoid α) :
    (siftReads rd T z).card ≤ (mids T).card := by
  classical
  have : siftReads rd T z ⊆ (mids T).image (z * ·) := fun w hw => by
    obtain ⟨m, hm, rfl⟩ := siftReads_mids T z w hw
    exact Finset.mem_image_of_mem _ hm
  exact (Finset.card_le_card this).trans Finset.card_image_le

theorem posReads_card (rd : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α)
    (ps : List ℕ) : (posReads rd T x ps).card ≤ ps.toFinset.card * (mids T).card := by
  classical
  unfold posReads
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _p ∈ ps.toFinset, (mids T).card :=
        Finset.sum_le_sum fun p _ => siftReads_card rd T _
    _ = _ := by rw [Finset.sum_const, smul_eq_mul]

/-- A probe at most `L` long reads the sifts of at most `3 + ⌈log₂ L⌉` of its prefixes: its
search's queries, the first its start, and the successor of the walk's end. -/
theorem probeReads_sub {L : ℕ} (hx : x.toList.length ≤ L) :
    ∃ ps : List ℕ, ps.toFinset.card ≤ 3 + Nat.clog 2 L
      ∧ probeReads rd T E k x ⊆ posReads rd T x ps := by
  classical
  unfold probeReads
  by_cases hk : x.toList.length < k
  · exact ⟨[], by simp, by simp [hk]⟩
  rw [if_neg hk]
  rcases ha : T.sift rd (prefixOf x k) with a | z
  · dsimp only
    obtain ⟨hpos, hhix, -⟩ := walk_facts E k x (p₀ := a) (by omega)
    generalize walk E a (x.toList.drop k) = ss at hpos hhix ⊢
    refine ⟨(k + ss.length - 1 + 1) :: (tripartiteSearch false
      (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)).2, ?_, ?_⟩
    · have h1 := tripartiteSearch_length_noRecovery
        (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)
      have h2 := Nat.clog_mono_right 2 (show k + ss.length - 1 - k ≤ L by omega)
      rw [List.toFinset_cons]
      have h3 := List.toFinset_card_le (tripartiteSearch false
        (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)).2
      have h4 := Finset.card_insert_le (k + ss.length - 1 + 1) (tripartiteSearch false
        (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)).2.toFinset
      omega
    · have hsr : posReads rd T x (searchPos (tripartiteSearch false
          (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1))
          (k + ss.length - 1)) ⊆ posReads rd T x
          ((k + ss.length - 1 + 1) :: (tripartiteSearch false
            (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)).2) :=
        fun w hw => by
          obtain ⟨p, hp, hwp⟩ := mem_posReads.1 hw
          exact mem_posReads.2 ⟨p, List.mem_cons_of_mem _ (searchPos_sub _ (by omega) p hp),
            hwp⟩
      have hst : ∀ w ∈ siftReads rd T (prefixOf x k), w ∈ posReads rd T x
          ((k + ss.length - 1 + 1) :: (tripartiteSearch false
            (siftAgrees rd T x fun p => ss.getD (p - k) []) k (k + ss.length - 1)).2) :=
        fun w hw => mem_posReads.2 ⟨k, List.mem_cons_of_mem _ (tripartiteSearch_lo_mem _ _ _), hw⟩
      refine Finset.union_subset hst ?_
      split
      · refine Finset.union_subset (fun w hw => mem_posReads.2 ⟨_, List.mem_cons_self, hw⟩) ?_
        split
        · exact Finset.empty_subset _
        · exact hsr
      · exact hsr
  · exact ⟨[k], by simp; omega, Finset.union_subset (fun w hw =>
      mem_posReads.2 ⟨k, List.mem_singleton_self _, hw⟩) (Finset.empty_subset _)⟩

end Probe

end Kappa

end Random

end OrthoDFA
