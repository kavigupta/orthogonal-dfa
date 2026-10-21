import OrthoDFA.Proofs.RandomProbe

/-!
# What a probe reads

A probe of `x` reads only the strings of `pot k x T`, and the edges only through their targets at
the tree's leaves (`probeOrAgree_congr`, `probe_econgr`), so over a tree there are finitely many
probes to bound, one per edge map of `edgeMaps T`. It searches exactly where its start sifts to a
leaf, the successor of the unlearned edge its walk stops at, if any, sifts decided, and the prefix
at the walk's end sifts decided off the walk (`probe_searches_iff`); each is a sift reaching a
leaf, which a string read undecided can only stop. So making one string undecided can only stop a
search (`searches_update`). The string a probe stops at undecided at its start or end is read by
the start sift, or by the sift at the walk's end or just past it (`probe_ends_cases`).
-/

namespace OrthoDFA

namespace Random

variable {α : Type*}

/-! ## Sifts -/

section Sift

theorem sift_congr_mids : ∀ (T : DTree α) (u : FreeMonoid α) {rd rd' : FreeMonoid α → ARU},
    (∀ m ∈ mids T, rd (u * m) = rd' (u * m)) → T.sift rd u = T.sift rd' u
  | .leaf, _, _, _, _ => rfl
  | .node m r a, u, rd, rd', h => by
    classical
    have hm : rd (u * m) = rd' (u * m) := h m (by simp [mids])
    have hr := sift_congr_mids r u fun m' hm' => h m' (by simp [mids, hm'])
    have ha := sift_congr_mids a u fun m' hm' => h m' (by simp [mids, hm'])
    simp only [DTree.sift, hm, hr, ha]

open scoped Classical in
/-- A sift that reaches a leaf with `z` read undecided reaches it whatever `z` reads. -/
theorem sift_update_inl (rd : FreeMonoid α → ARU) (z : FreeMonoid α) :
    ∀ (T : DTree α) (u : FreeMonoid α) (p : List Bool),
      T.sift (Function.update rd z .undecided) u = .inl p → T.sift rd u = .inl p
  | .leaf, _, _, h => h
  | .node m r a, u, p, h => by
    simp only [DTree.sift] at h ⊢
    by_cases hz : u * m = z
    · subst hz
      simp at h
    rw [Function.update_of_ne hz] at h
    rcases hc : rd (u * m) with _ | _ | _ <;> rw [hc] at h <;> simp only [reduceCtorEq] at h
    · rcases hs : a.sift (Function.update rd z .undecided) u with q | q <;> rw [hs] at h <;>
        simp only [Sum.map_inl, Sum.map_inr, reduceCtorEq, Sum.inl.injEq] at h
      rw [sift_update_inl rd z a u q hs, ← h]; rfl
    · rcases hs : r.sift (Function.update rd z .undecided) u with q | q <;> rw [hs] at h <;>
        simp only [Sum.map_inl, Sum.map_inr, reduceCtorEq, Sum.inl.injEq] at h
      rw [sift_update_inl rd z r u q hs, ← h]; rfl

end Sift

/-! ## The search -/

section Search

theorem bisectBoundary_congr (recover : Bool) {ag ag' : ℕ → Option Bool} :
    ∀ (n lo hi : ℕ), hi - lo = n → (∀ i, lo ≤ i → ag i = ag' i) →
      bisectBoundary recover ag lo hi = bisectBoundary recover ag' lo hi := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn h
  rw [bisectBoundary.eq_1 recover ag lo hi, bisectBoundary.eq_1 recover ag' lo hi]
  split
  · rfl
  rename_i hlt
  have e1 := ih _ (by omega) ((lo + hi) / 2) hi rfl fun i hi' => h i (by omega)
  have e2 := ih _ (by omega) lo ((lo + hi) / 2) rfl h
  have e3 := ih _ (by omega) lo ((lo + hi) / 2 - 1) rfl h
  have e4 := ih _ (by omega) ((lo + hi) / 2 + 1) hi rfl fun i hi' => h i (by omega)
  simp only [h ((lo + hi) / 2) (by omega), h ((lo + hi) / 2 - 1) (by omega),
    h ((lo + hi) / 2 + 1) (by omega), e1, e2, e3, e4]

theorem tripartiteSearch_congr (recover : Bool) {ag ag' : ℕ → Option Bool} {lo hi : ℕ}
    (hle : lo ≤ hi) (h : ∀ i, lo ≤ i → ag i = ag' i) :
    tripartiteSearch recover ag lo hi = tripartiteSearch recover ag' lo hi := by
  unfold tripartiteSearch
  rw [h lo le_rfl, h hi hle, bisectBoundary_congr recover _ lo hi rfl h]

end Search

section Structure

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- What a search from `k`, which agrees with the walk `st`, to `hi` comes to. -/
theorem searchOutcome_cases (st : ℕ → List Bool) (un : Option (α × List Bool)) {hi : ℕ}
    (hk : k ≤ hi) (hx : hi ≤ x.toList.length) (hlo : T.sift read (prefixOf x k) = .inl (st k))
    {o : Outcome α} (ho : searchOutcome read T x st un hi
      (tripartiteSearch false (siftAgrees read T x st) k hi).1 = o) :
    (∃ z, T.sift read (prefixOf x hi) = .inr z ∧ o = .undecided [z] true)
    ∨ (T.sift read (prefixOf x hi) = .inl (st hi) ∧ o = searchOutcome read T x st un hi .allAgree)
    ∨ ((∃ l, T.sift read (prefixOf x hi) = .inl l ∧ l ≠ st hi) ∧
      ((∃ p c t, k < p ∧ p ≤ hi ∧ x.toList[p - 1]? = some c
          ∧ T.sift read (prefixOf x (p - 1)) = .inl (st (p - 1))
          ∧ T.sift read (prefixOf x p) = .inl t ∧ t ≠ st p
          ∧ o = .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t)
        ∨ (∃ p z, k < p ∧ p < hi ∧ T.sift read (prefixOf x p) = .inr z
          ∧ o = .undecided [z] false))) := by
  have hag : siftAgrees read T x st k = some true := by
    rw [agrees_inl read T x st hlo]; simp
  have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read T x st) hk
  have hn := tripartiteSearch_noRecovery (agrees := siftAgrees read T x st) (lo := k) (hi := hi)
  rcases hsh : T.sift read (prefixOf x hi) with l | z
  · by_cases hl : l = st hi
    · subst hl
      right; left
      refine ⟨rfl, ?_⟩
      have : siftAgrees read T x st hi = some true := by rw [agrees_inl read T x st hsh]; simp
      rw [← ho]
      simp only [tripartiteSearch, hag, this]
    · right; right
      refine ⟨⟨l, rfl, hl⟩, ?_⟩
      have hhi : siftAgrees read T x st hi = some false := by
        rw [agrees_inl read T x st hsh]; simp [hl]
      generalize hr : (tripartiteSearch false (siftAgrees read T x st) k hi).1 = r at ho hs hn
      cases r with
      | undecidedLo => simp [SearchOK, hag] at hs
      | disagreeLo => simp [SearchOK, hag] at hs
      | undecidedHi => simp [SearchOK, hhi] at hs
      | allAgree => simp [SearchOK, hhi] at hs
      | edge p =>
        obtain ⟨h1, h2, h3, h4⟩ := hs
        obtain ⟨t, ht, htne⟩ := agrees_false read T x st h4
        obtain ⟨c, hc⟩ : ∃ c, x.toList[p - 1]? = some c :=
          ⟨_, List.getElem?_eq_getElem (by omega)⟩
        left
        refine ⟨p, c, t, h1, h2, hc, agrees_true read T x st h3, ht, htne, ?_⟩
        rw [← ho]
        simp only [searchOutcome, hc, ht]
      | undecidedMid p =>
        obtain ⟨-, h1, h2, h3⟩ := hs
        unfold siftAgrees at h3
        rcases hp : T.sift read (prefixOf x p) with q | z' <;> rw [hp] at h3
        · simp at h3
        right
        refine ⟨p, z', h1, h2, hp, ?_⟩
        rw [← ho]
        simp only [searchOutcome, undecidedAt, hp]; rfl
      | pair p => exact absurd rfl (hn p).1
      | triple p => exact absurd rfl (hn p).2
  · left
    refine ⟨z, rfl, ?_⟩
    have : siftAgrees read T x st hi = none := by simp [siftAgrees, hsh]
    rw [← ho]
    simp only [tripartiteSearch, hag, this, searchOutcome, undecidedAt, hsh]; rfl

/-- A probe at least `k` long whose start sifts to `p₀`: the successor of the unlearned edge its
walk stops at, if any, sifts undecided, or its search from `k` to the walk's end. -/
theorem probe_long_cases {p₀ : List Bool} (hk : k ≤ x.toList.length)
    (h₀ : T.sift read (prefixOf x k) = .inl p₀) {o : Outcome α}
    (ho : probeOrAgree read T E k x = o) :
    (∃ c z, x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? = some c
      ∧ T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) = .inr z
      ∧ o = .undecided [z] true)
    ∨ ∃ un : Option (α × List Bool),
      (∀ c t, un = some (c, t) → x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? = some c
        ∧ T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) = .inl t)
      ∧ (un = none → x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? = none)
      ∧ searchOutcome read T x (fun j => (walk E p₀ (x.toList.drop k)).getD (j - k) []) un
        (k + (walk E p₀ (x.toList.drop k)).length - 1)
        (tripartiteSearch false (siftAgrees read T x
          (fun j => (walk E p₀ (x.toList.drop k)).getD (j - k) [])) k
          (k + (walk E p₀ (x.toList.drop k)).length - 1)).1 = o := by
  unfold probeOrAgree probe at ho
  rw [if_neg (by omega), h₀] at ho
  simp only at ho
  rcases hc : x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? with _ | c <;>
    rw [hc] at ho <;> simp only at ho
  · exact .inr ⟨none, by simp, fun _ => rfl, ho⟩
  rcases ht : T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1))
    with t | z <;> rw [ht] at ho <;> simp only at ho
  · exact .inr ⟨some (c, t), fun c' t' h => by cases h; exact ⟨rfl, rfl⟩, by simp, ho⟩
  · exact .inl ⟨c, z, rfl, rfl, ho.symm⟩

/-- The walk from `p₀` along the probe from `k` starts at `p₀` and ends by `x`'s end. -/
theorem walk_facts {p₀ : List Bool} (hk : k ≤ x.toList.length) :
    let ss := walk E p₀ (x.toList.drop k)
    0 < ss.length ∧ k + ss.length - 1 ≤ x.toList.length ∧ ss.getD 0 [] = p₀ := by
  intro ss
  have hpos : 0 < ss.length := length_walk_pos E p₀ _
  have hle : ss.length ≤ (x.toList.drop k).length + 1 := length_walk_le E p₀ _
  simp only [List.length_drop] at hle
  exact ⟨hpos, by omega, walk_getD_zero E p₀ _⟩

/-- A probe searches where its start sifts to a leaf, the successor of the unlearned edge its walk
stops at, if any, sifts decided, and the prefix at the walk's end sifts off the walk. -/
theorem probe_searches_iff :
    searches (probeOrAgree read T E k x) = true ↔ k ≤ x.toList.length
      ∧ ∃ p₀, T.sift read (prefixOf x k) = .inl p₀
      ∧ (∀ c, x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? = some c → ∃ t,
          T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) = .inl t)
      ∧ ∃ l, T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1)) = .inl l
        ∧ l ≠ (walk E p₀ (x.toList.drop k)).getD
          (k + (walk E p₀ (x.toList.drop k)).length - 1 - k) [] := by
  constructor
  · intro h
    generalize ho : probeOrAgree read T E k x = o at h
    by_cases hs : x.toList.length < k
    · simp only [probeOrAgree, hs, if_true] at ho
      subst ho; simp [searches] at h
    rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
    swap
    · simp only [probeOrAgree, hs, if_false, probe, h₀] at ho
      subst ho; simp [searches] at h
    refine ⟨by omega, p₀, rfl, ?_⟩
    obtain ⟨hpos, hhi, hst0⟩ := walk_facts E k x (p₀ := p₀) (by omega)
    have hlo : T.sift read (prefixOf x k)
        = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, hst0, h₀]
    rcases probe_long_cases read T E k x (by omega) h₀ ho with
      ⟨c, z, -, -, rfl⟩ | ⟨un, hu1, hu2, h'⟩
    · simp [searches] at h
    rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
      ⟨z, -, rfl⟩ | ⟨-, rfl⟩ | ⟨hl, -⟩
    · simp [searches] at h
    · rcases un with _ | ⟨c, t⟩ <;> simp [searchOutcome, searches] at h
    · refine ⟨fun c hc => ?_, hl⟩
      rcases un with _ | ⟨c', t⟩
      · rw [hu2 rfl] at hc; cases hc
      · exact ⟨t, (hu1 c' t rfl).2⟩
  · rintro ⟨hk, p₀, h₀, hc, l, hl, hne⟩
    obtain ⟨hpos, hhi, hst0⟩ := walk_facts E k x (p₀ := p₀) hk
    have hlo : T.sift read (prefixOf x k)
        = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, hst0, h₀]
    rcases probe_long_cases read T E k x hk h₀ rfl with ⟨c, z, hxc, hz, -⟩ | ⟨un, -, -, h'⟩
    · obtain ⟨t, ht⟩ := hc c hxc
      rw [ht] at hz; cases hz
    rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
      ⟨z, hz, -⟩ | ⟨hsh, -⟩ | ⟨-, ⟨p, c, t, -, -, -, -, -, -, ho⟩ | ⟨p, z, -, -, -, ho⟩⟩
    · rw [hl] at hz; cases hz
    · rw [hl] at hsh; cases hsh; exact absurd rfl hne
    · rw [ho]; rfl
    · rw [ho]; rfl

/-- The string a probe stops at undecided at its start or end is read by the start sift, or, the
start sifting to `p₀`, by the sift at the walk's end or just past it. -/
theorem probe_ends_cases {z : FreeMonoid α} (h : z ∈ endsString (probeOrAgree read T E k x)) :
    k ≤ x.toList.length ∧ (T.sift read (prefixOf x k) = .inr z
      ∨ ∃ p₀, T.sift read (prefixOf x k) = .inl p₀
        ∧ (T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1)) = .inr z
          ∨ T.sift read (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1))
            = .inr z)) := by
  generalize ho : probeOrAgree read T E k x = o at h
  by_cases hs : x.toList.length < k
  · simp only [probeOrAgree, hs, if_true] at ho
    subst ho; simp [endsString] at h
  refine ⟨by omega, ?_⟩
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z₀
  swap
  · simp only [probeOrAgree, hs, if_false, probe, h₀] at ho
    subst ho
    simp only [endsString, List.mem_singleton] at h
    subst h; exact .inl rfl
  right
  refine ⟨p₀, rfl, ?_⟩
  obtain ⟨hpos, hhi, hst0⟩ := walk_facts E k x (p₀ := p₀) (by omega)
  have hlo : T.sift read (prefixOf x k)
      = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, h₀]
  rcases probe_long_cases read T E k x (by omega) h₀ ho with
    ⟨c, z', -, hz', rfl⟩ | ⟨un, -, -, h'⟩
  · simp only [endsString, List.take_succ_cons, List.take_zero, List.mem_singleton] at h
    subst h; exact .inr hz'
  rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
    ⟨z', hz', rfl⟩ | ⟨-, rfl⟩ | ⟨-, ⟨p, c, t, -, -, -, -, -, -, rfl⟩ | ⟨p, z', -, -, -, rfl⟩⟩
  · simp only [endsString, List.take_succ_cons, List.take_zero, List.mem_singleton] at h
    subst h; exact .inl hz'
  · rcases un with _ | ⟨c, t⟩ <;> simp [searchOutcome, endsString] at h
  · simp [endsString] at h
  · simp [endsString] at h

/-- A probe reads the sifts only of its prefixes from `k` on. -/
theorem probe_congrK {read' : FreeMonoid α → ARU}
    (h : ∀ j, k ≤ j → T.sift read (prefixOf x j) = T.sift read' (prefixOf x j)) :
    probeOrAgree read T E k x = probeOrAgree read' T E k x := by
  unfold probeOrAgree
  split_ifs with hs
  · rfl
  unfold probe
  rw [h k le_rfl]
  rcases h₀ : T.sift read' (prefixOf x k) with p₀ | z
  · simp only
    obtain ⟨hpos, hhi, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
    generalize walk E p₀ (x.toList.drop k) = ss at hpos hhi ⊢
    have hag : ∀ st, ∀ i, k ≤ i → siftAgrees read T x st i = siftAgrees read' T x st i := by
      intro st i hi; simp only [siftAgrees, h i hi]
    have hsearch : ∀ st un, searchOutcome read T x st un (k + ss.length - 1)
        (tripartiteSearch false (siftAgrees read T x st) k (k + ss.length - 1)).1
        = searchOutcome read' T x st un (k + ss.length - 1)
          (tripartiteSearch false (siftAgrees read' T x st) k (k + ss.length - 1)).1 := by
      intro st un
      rw [tripartiteSearch_congr false (by omega) (hag st)]
      have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read' T x st)
        (lo := k) (hi := k + ss.length - 1) (by omega)
      generalize (tripartiteSearch false (siftAgrees read' T x st) k (k + ss.length - 1)).1 = r
        at hs ⊢
      cases r with
      | undecidedHi =>
        simp only [searchOutcome, undecidedAt, h _ (by omega : k ≤ k + ss.length - 1)]
      | edge p =>
        obtain ⟨h1, -⟩ := hs
        simp only [searchOutcome, h p (by omega)]
      | undecidedMid p =>
        obtain ⟨-, h1, -⟩ := hs
        simp only [searchOutcome, undecidedAt, h p (by omega)]
      | _ => rfl
    rcases x.toList[k + ss.length - 1]? with _ | c
    · exact hsearch _ _
    · simp only
      rw [h _ (by omega)]
      rcases T.sift read' (prefixOf x (k + ss.length - 1 + 1)) with t | z
      · exact hsearch _ _
      · rfl
  · rfl

theorem sift_congr_pot {read' : FreeMonoid α → ARU} (h : ∀ z ∈ pot k x T, read z = read' z)
    (hs : k ≤ x.toList.length) (j : ℕ) (hj : k ≤ j) :
    T.sift read (prefixOf x j) = T.sift read' (prefixOf x j) := by
  have hj' : prefixOf x j = prefixOf x (min j x.toList.length) := by
    simp only [prefixOf]
    rcases le_total j x.toList.length with hjl | hjl
    · rw [min_eq_left hjl]
    · rw [min_eq_right hjl, List.take_of_length_le hjl, List.take_of_length_le le_rfl]
  rw [hj']
  exact sift_congr_mids T _ fun m hm => h _ (mem_pot (by omega) (min_le_right _ _) hm)

/-- A probe reads only the strings of `pot k x T`. -/
theorem probeOrAgree_congr {read' : FreeMonoid α → ARU} (h : ∀ z ∈ pot k x T, read z = read' z) :
    probeOrAgree read T E k x = probeOrAgree read' T E k x := by
  by_cases hs : x.toList.length < k
  · simp [probeOrAgree, hs]
  exact probe_congrK read T E k x (sift_congr_pot read T k x h (by omega))

/-- A probe's search queries by the strings of `pot k x T` only. -/
theorem probeQueries_congr {read' : FreeMonoid α → ARU} (h : ∀ z ∈ pot k x T, read z = read' z) :
    probeQueries read T E k x = probeQueries read' T E k x := by
  unfold probeQueries
  split_ifs with hs
  · rfl
  have hj := sift_congr_pot read T k x h (by omega)
  rw [hj k le_rfl]
  rcases T.sift read' (prefixOf x k) with a | w
  · simp only
    obtain ⟨hpos, -, -⟩ := walk_facts E k x (p₀ := a) (by omega)
    exact congrArg Prod.snd (tripartiteSearch_congr false (by omega) fun i hi => by
      simp only [siftAgrees, hj i hi])
  · rfl

theorem probeQueries_length {L : ℕ} (hx : x.toList.length ≤ L) :
    (probeQueries read T E k x).length ≤ 2 + Nat.clog 2 L := by
  unfold probeQueries
  split_ifs with hs
  · simp
  rcases T.sift read (prefixOf x k) with a | w
  · simp only
    obtain ⟨-, hhi, -⟩ := walk_facts E k x (p₀ := a) (by omega)
    refine (tripartiteSearch_length_noRecovery _ _ _).trans ?_
    have := Nat.clog_mono_right 2
      (show k + (walk E a (x.toList.drop k)).length - 1 - k ≤ L by omega)
    omega
  · simp

open scoped Classical in
/-- Making `z` undecided can only stop a search. -/
theorem searches_update (z : FreeMonoid α)
    (h : searches (probeOrAgree (Function.update read z .undecided) T E k x) = true) :
    searches (probeOrAgree read T E k x) = true := by
  obtain ⟨hk, p₀, h₀, hc, l, hl, hne⟩ := (probe_searches_iff _ T E k x).1 h
  refine (probe_searches_iff read T E k x).2
    ⟨hk, p₀, sift_update_inl read z T _ _ h₀, fun c hc' => ?_, l,
      sift_update_inl read z T _ _ hl, hne⟩
  obtain ⟨t, ht⟩ := hc c hc'
  exact ⟨t, sift_update_inl read z T _ _ ht⟩

open scoped Classical in
/-- Making `z` undecided can only drop positions from a search's queries. -/
theorem probeQueries_update (z : FreeMonoid α) {p : ℕ}
    (h : p ∈ probeQueries (Function.update read z .undecided) T E k x) :
    p ∈ probeQueries read T E k x := by
  unfold probeQueries at h ⊢
  split_ifs at h ⊢ with hs
  · simp at h
  rcases h₀ : T.sift (Function.update read z .undecided) (prefixOf x k) with a | w
  · rw [h₀] at h
    rw [sift_update_inl read z T _ _ h₀]
    refine tripartiteSearch_queries_mono _ _ (fun i b hi => ?_) p h
    unfold siftAgrees at hi ⊢
    rcases hs' : T.sift (Function.update read z .undecided) (prefixOf x i) with l | w <;>
      rw [hs'] at hi
    · rw [sift_update_inl read z T _ _ hs']; exact hi
    · simp at hi
  · rw [h₀] at h; simp at h

end Structure

/-! ## Edges -/

section Edges

/-- Every edge out of a leaf points at a leaf. -/
def EdgesInto (T : DTree α) (e : Edges α) : Prop :=
  ∀ p ∈ T.leaves, ∀ c t w, e (p, c) = some (t, w) → t ∈ T.leaves

/-- The two edge maps point every leaf's edges at the same targets. -/
def TgtAgree (T : DTree α) (e e' : Edges α) : Prop :=
  ∀ p ∈ T.leaves, ∀ c, (e (p, c)).map Prod.fst = (e' (p, c)).map Prod.fst

theorem edgesInto_of_ok {read : FreeMonoid α → ARU} {T : DTree α} {e : Edges α}
    (h : EdgesOK read T e) : EdgesInto T e := fun p hp c t w he =>
  DTree.sift_inl_mem read T _ t (h p hp c t w he).2

/-- A walk from a leaf reads the edges only through their targets. -/
theorem walk_congr {L : List (List Bool)} {e e' : Edges α}
    (he : ∀ p ∈ L, ∀ c t w, e (p, c) = some (t, w) → t ∈ L)
    (h : ∀ p ∈ L, ∀ c, (e (p, c)).map Prod.fst = (e' (p, c)).map Prod.fst) :
    ∀ (cs : List α) (p : List Bool), p ∈ L → walk e p cs = walk e' p cs
  | [], _, _ => rfl
  | c :: cs, p, hp => by
    have hc := h p hp c
    simp only [walk]
    rcases h1 : e (p, c) with _ | ⟨q, w⟩ <;> rcases h2 : e' (p, c) with _ | ⟨q', w'⟩ <;>
      rw [h1, h2] at hc <;> simp only [Option.map_none, Option.map_some, reduceCtorEq,
        Option.some.injEq] at hc
    all_goals first
      | rfl
      | (subst hc; simp only; rw [walk_congr he h cs q (he _ hp _ _ _ h1)])

/-- A probe reads the edges only through their targets. -/
theorem probe_econgr (read : FreeMonoid α → ARU) (T : DTree α) (k : ℕ) (x : FreeMonoid α)
    {e e' : Edges α} (he : EdgesInto T e) (h : TgtAgree T e e') :
    probeOrAgree read T e k x = probeOrAgree read T e' k x := by
  unfold probeOrAgree probe
  split_ifs
  · rfl
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
  · simp only
    rw [walk_congr he h _ p₀ (DTree.sift_inl_mem read T _ _ h₀)]
  · rfl

variable [Fintype α]

open scoped Classical in
/-- Edge maps out of the leaves into the leaves, each witness `1`. -/
noncomputable def edgeMaps (T : DTree α) : Finset (Edges α) :=
  (Finset.univ : Finset (↥(T.leaves.toFinset ×ˢ (Finset.univ : Finset α))
    → Option ↥T.leaves.toFinset)).image fun f pc =>
      if h : pc ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
        (f ⟨pc, h⟩).map fun t => (t.1, 1)
      else none

theorem edgeMaps_card (T : DTree α) :
    (edgeMaps T).card ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := by
  classical
  refine Finset.card_image_le.trans ?_
  simp only [Finset.card_univ, Fintype.card_fun, Fintype.card_option, Fintype.card_coe,
    Finset.card_product, Finset.card_univ]
  have h := List.toFinset_card_le T.leaves
  calc _ ≤ (T.leaves.length + 1) ^ (T.leaves.toFinset.card * Fintype.card α) := by gcongr
    _ ≤ _ := by
      apply Nat.pow_le_pow_right (by omega)
      gcongr

theorem exists_edgeMaps {T : DTree α} {e : Edges α} (he : EdgesInto T e) :
    ∃ e' ∈ edgeMaps T, TgtAgree T e e' := by
  classical
  refine ⟨fun pc => if h : pc ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
      (e pc).map fun q => (q.1, 1) else none, ?_, ?_⟩
  · refine Finset.mem_image.2 ⟨fun pc => (e pc.1).bind fun q =>
      if hq : q.1 ∈ T.leaves.toFinset then some ⟨q.1, hq⟩ else none, Finset.mem_univ _, ?_⟩
    funext ⟨p, c⟩
    split_ifs with h
    · rcases hpc : e (p, c) with _ | ⟨q, w⟩
      · simp [hpc]
      · have hq : q ∈ T.leaves :=
          he p (by simpa using (Finset.mem_product.1 h).1) c q w hpc
        simp [hpc, hq]
    · rfl
  · intro p hp c
    have h : (p, c) ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) := by simp [hp]
    simp only [dif_pos h]
    rcases e (p, c) with _ | ⟨q, w⟩ <;> simp

end Edges

end Random

end OrthoDFA
