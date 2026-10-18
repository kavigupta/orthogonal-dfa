import OrthoDFA.Level5
import OrthoDFA.Proofs.Probe

/-!
# Level 5's probe

Level 5 reads through its cut (`ofCut`), and a sift through it reads, in order, the strings
`siftWithPath` lists. A probe is main's (`probeOrAgree`); what each outcome says about the walk
and the sifts (`probe_wrongEdge`, `probe_unlearnedEdge`, `probe_middle`, `probe_isSearch_iff`),
that it reads the sifts only from position `k` on and the edges only through their targets
(`probe_congrK`, `probe_econgr`), and that `probeQuery` is its query tree (`run_probeQuery`), reading
only prefixes from `k` on (`asks_probeQuery`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace QTree

variable {Λ ι V β γ : Type*}

theorem run_bind (rd : ι → V) (f : β → QTree Λ ι V γ) :
    ∀ t : QTree Λ ι V β, (t.bind f).run rd = (f (t.run rd)).run rd
  | done _ => rfl
  | ask _ w next => run_bind rd f (next (rd w))

theorem asks_bind (rd : ι → V) (f : β → QTree Λ ι V γ) :
    ∀ t : QTree Λ ι V β, (t.bind f).asks rd = t.asks rd ++ (f (t.run rd)).asks rd
  | done _ => rfl
  | ask ℓ w next => by
    simp only [bind, asks, run, List.cons_append]
    rw [asks_bind rd f (next (rd w))]

theorem run_map (rd : ι → V) (f : β → γ) (t : QTree Λ ι V β) :
    (t.map f).run rd = f (t.run rd) := run_bind rd _ t

theorem asks_map (rd : ι → V) (f : β → γ) (t : QTree Λ ι V β) :
    (t.map f).asks rd = t.asks rd := by
  simp [map, asks_bind, asks]

end QTree

section Cut

theorem ofCut_cut (cut : FreeMonoid α → Option Bool) (z : FreeMonoid α) :
    (ofCut cut z).cut = cut z := by
  unfold ofCut; rcases cut z with _ | _ | _ <;> rfl

theorem cut_ofCut (cut : FreeMonoid α → Option Bool) : (fun z => (ofCut cut z).cut) = cut :=
  funext (ofCut_cut cut)

theorem ofCut_rd (rd : FreeMonoid α → ARU) : ofCut (fun z => (rd z).cut) = rd := by
  funext z; unfold ofCut; rcases h : rd z <;> simp [h, ARU.cut]

omit [Fintype α] [DecidableEq α] in
theorem map_cons_eq_inl {s : List Bool ⊕ FreeMonoid α} {b : Bool} {q : List Bool} :
    s.map (b :: ·) id = .inl q ↔ ∃ p, q = b :: p ∧ s = .inl p := by
  rcases s with p | z <;> simp [eq_comm]

end Cut

section Sift

variable {cut : FreeMonoid α → Option Bool}

theorem sift_ofCut_node (m : FreeMonoid α) (r a : DTree α) (z : FreeMonoid α) :
    (DTree.node m r a).sift (ofCut cut) z = match cut (z * m) with
      | none => .inr (z * m)
      | some true => (a.sift (ofCut cut) z).map (true :: ·) id
      | some false => (r.sift (ofCut cut) z).map (false :: ·) id := by
  simp only [DTree.sift, ofCut]
  rcases cut (z * m) with _ | _ | _ <;> rfl

/-- The strings a sift reads are its string followed by midfixes. -/
theorem reads_mid (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) (u q : FreeMonoid α), q ∈ (t.siftWithPath cut u).1 →
      ∃ m ∈ t.midfixes, q = u * m
  | .leaf, _, _, h => by simp [DTree.siftWithPath] at h
  | .node m r a, u, q, h => by
    classical
    simp only [DTree.siftWithPath] at h
    split at h
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := reads_mid cut a u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := reads_mid cut r u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩
    · simp only [List.mem_singleton] at h
      exact ⟨m, by simp [DTree.midfixes], h⟩

/-- A sift stopped undecided stops at a string it read undecided. -/
theorem sift_inr_reads (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) (w b : FreeMonoid α), t.sift (ofCut cut) w = .inr b →
      b ∈ (t.siftWithPath cut w).1 ∧ cut b = none
  | .leaf, _, _, h => by simp [DTree.sift] at h
  | .node m r a, w, b, h => by
    rw [sift_ofCut_node] at h
    simp only [DTree.siftWithPath]
    rcases hc : cut (w * m) with _ | _ | _ <;> simp only [hc] at h ⊢
    · simp only [Sum.inr.injEq] at h
      subst h
      exact ⟨List.mem_singleton_self _, hc⟩
    · rcases hr : r.sift (ofCut cut) w with q | b' <;> rw [hr] at h
      · simp at h
      · simp only [Sum.map_inr, id, Sum.inr.injEq] at h
        subst h
        obtain ⟨h1, h2⟩ := sift_inr_reads cut r w b' hr
        exact ⟨List.mem_cons_of_mem _ h1, h2⟩
    · rcases ha : a.sift (ofCut cut) w with q | b' <;> rw [ha] at h
      · simp at h
      · simp only [Sum.map_inr, id, Sum.inr.injEq] at h
        subst h
        obtain ⟨h1, h2⟩ := sift_inr_reads cut a w b' ha
        exact ⟨List.mem_cons_of_mem _ h1, h2⟩

/-- A string read undecided on a sift's way is where the sift stops. -/
theorem sift_of_reads_und :
    ∀ (T : DTree α) (u w : FreeMonoid α), w ∈ (T.siftWithPath cut u).1 → cut w = none →
      T.sift (ofCut cut) u = .inr w
  | .leaf, _, _, h, _ => by simp [DTree.siftWithPath] at h
  | .node m r a, u, w, h, hw => by
    rw [sift_ofCut_node]
    simp only [DTree.siftWithPath] at h
    rcases hc : cut (u * m) with _ | _ | _ <;>
      simp only [hc, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at h ⊢
    · rw [h]
    · rcases h with rfl | h
      · rw [hw] at hc; cases hc
      · rw [sift_of_reads_und r u w h hw]; rfl
    · rcases h with rfl | h
      · rw [hw] at hc; cases hc
      · rw [sift_of_reads_und a u w h hw]; rfl

/-- What a sift reads, and where it lands, depends only on the reads of its string followed by
the tree's midfixes. -/
theorem siftWithPath_congr {cut cut' : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α), (∀ m ∈ T.midfixes, cut (u * m) = cut' (u * m)) →
      T.siftWithPath cut u = T.siftWithPath cut' u
  | .leaf, _, _ => rfl
  | .node m r a, u, h => by
    classical
    have hm := h m (by simp [DTree.midfixes])
    have hr := siftWithPath_congr r u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    have ha := siftWithPath_congr a u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    simp only [DTree.siftWithPath, hm, hr, ha]

theorem sift_congr_mid {cut cut' : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α), (∀ m ∈ T.midfixes, cut (u * m) = cut' (u * m)) →
      T.sift (ofCut cut) u = T.sift (ofCut cut') u
  | .leaf, _, _ => rfl
  | .node m r a, u, h => by
    classical
    have hm := h m (by simp [DTree.midfixes])
    have hr := sift_congr_mid r u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    have ha := sift_congr_mid a u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    rw [sift_ofCut_node, sift_ofCut_node, hm, hr, ha]

/-- A sift stopped undecided stops at its string followed by a midfix, read undecided. -/
theorem sift_inr_mid :
    ∀ (T : DTree α) (u b : FreeMonoid α), T.sift (ofCut cut) u = .inr b →
      ∃ m ∈ T.midfixes, b = u * m ∧ cut (u * m) = none := by
  intro T u b h
  obtain ⟨h1, h2⟩ := sift_inr_reads cut T u b h
  obtain ⟨m, hm, rfl⟩ := reads_mid cut T u b h1
  exact ⟨m, hm, rfl, h2⟩

end Sift

section Search

/-- The bisection reads `agrees` only from `lo` on. -/
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

/-- The walk from `p₀` along the probe from `k`: it starts at `p₀`, ends by `x`'s end, and stops
short only at an unlearned edge, on the letter at its end. -/
theorem walk_facts {p₀ : List Bool} (hk : k ≤ x.toList.length) :
    let ss := walk E p₀ (x.toList.drop k)
    0 < ss.length ∧ k + ss.length - 1 ≤ x.toList.length ∧ ss.getD 0 [] = p₀
      ∧ (∀ c, x.toList[k + ss.length - 1]? = some c → E (ss.getD (ss.length - 1) [], c) = none)
      ∧ (x.toList[k + ss.length - 1]? = none → k + ss.length - 1 = x.toList.length)
      ∧ (∀ p, k ≤ p → p < k + ss.length - 1 → ∃ c w, x.toList[p]? = some c
          ∧ E (ss.getD (p - k) [], c) = some (ss.getD (p + 1 - k) [], w)) := by
  intro ss
  have hpos : 0 < ss.length := length_walk_pos E p₀ _
  have hle : ss.length ≤ (x.toList.drop k).length + 1 := length_walk_le E p₀ _
  simp only [List.length_drop] at hle
  refine ⟨hpos, by omega, walk_getD_zero E p₀ _, fun c hc => ?_, fun hc => ?_, fun p h1 h2 => ?_⟩
  · have hstop : ss.length ≤ (x.toList.drop k).length := by
      by_contra hcon
      simp only [List.length_drop] at hcon
      rw [List.getElem?_eq_none (by omega)] at hc
      cases hc
    obtain ⟨c', hc', hE⟩ := walk_stop E p₀ _ hstop
    rw [List.getElem?_drop, show k + (ss.length - 1) = k + ss.length - 1 by omega, hc] at hc'
    cases hc'
    exact hE
  · by_contra hne
    rw [List.getElem?_eq_getElem (by omega)] at hc
    cases hc
  · have hss : ss = walk E p₀ (x.toList.drop k) := rfl
    obtain ⟨c, w, hc, hw⟩ := walk_succ E p₀ (x.toList.drop k) (p - k) (by rw [← hss]; omega)
    refine ⟨c, w, ?_, by rwa [show p + 1 - k = p - k + 1 by omega]⟩
    rwa [List.getElem?_drop, show k + (p - k) = p by omega] at hc

/-- A wrong edge is one the walk crossed: its leaf the walk's at a position between `k` and `x`'s
end where the prefix sifts there, the next prefix sifting off the edge's target. -/
theorem probe_wrongEdge {s : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probeOrAgree read T E k x = .wrongEdge s c u t) :
    ∃ p₀ fd, T.sift read (prefixOf x k) = .inl p₀ ∧ k < fd ∧ fd ≤ x.toList.length
      ∧ u = prefixOf x (fd - 1) ∧ x.toList[fd - 1]? = some c
      ∧ fd - 1 - k < (walk E p₀ (x.toList.drop k)).length
      ∧ s = (walk E p₀ (x.toList.drop k)).getD (fd - 1 - k) []
      ∧ T.sift read u = .inl s ∧ T.sift read (prefixOf x fd) = .inl t
      ∧ ∃ t₀ w, E (s, c) = some (t₀, w) ∧ t₀ ≠ t := by
  by_cases hs : x.toList.length < k
  · simp [probeOrAgree, hs] at h
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
  swap
  · simp [probeOrAgree, hs, probe, h₀] at h
  obtain ⟨hpos, hhi, hst0, -, -, hstep⟩ := walk_facts E k x (p₀ := p₀) (by omega)
  rcases probe_long_cases read T E k x (by omega) h₀ h with ⟨c', z, -, -, h'⟩ | ⟨un, -, -, h'⟩
  · cases h'
  have hlo : T.sift read (prefixOf x k) = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, h₀]
  rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
    ⟨z, -, h''⟩ | ⟨-, h''⟩ | ⟨-, ⟨p, c', t', h1, h2, h3, h4, h5, h6, h''⟩ | ⟨p, z, -, -, -, h''⟩⟩
  · cases h''
  · rcases un with _ | ⟨c', t'⟩ <;> simp [searchOutcome] at h''
  · simp only [Outcome.wrongEdge.injEq] at h''
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h''
    obtain ⟨c'', w, hc'', hE⟩ := hstep (p - 1) (by omega) (by omega)
    rw [hc''] at h3
    cases h3
    refine ⟨p₀, p, rfl, h1, by omega, rfl, hc'', by omega, rfl, h4, h5, _, w, hE, ?_⟩
    rw [show p - 1 + 1 - k = p - k by omega]
    exact fun he => h6 he.symm
  · cases h''

/-- An unlearned edge is the one the walk stopped at: out of the leaf the walk ends at, where the
prefix sifts, on the next letter, whose prefix sifts to the target. -/
theorem probe_unlearnedEdge {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probeOrAgree read T E k x = .unlearnedEdge p c u t) :
    ∃ p₀ j, T.sift read (prefixOf x k) = .inl p₀ ∧ k ≤ j ∧ j < x.toList.length
      ∧ u = prefixOf x j ∧ x.toList[j]? = some c
      ∧ j - k < (walk E p₀ (x.toList.drop k)).length
      ∧ p = (walk E p₀ (x.toList.drop k)).getD (j - k) []
      ∧ T.sift read u = .inl p ∧ T.sift read (prefixOf x (j + 1)) = .inl t
      ∧ E (p, c) = none := by
  by_cases hs : x.toList.length < k
  · simp [probeOrAgree, hs] at h
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
  swap
  · simp [probeOrAgree, hs, probe, h₀] at h
  obtain ⟨hpos, hhi, hst0, hun, -, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
  rcases probe_long_cases read T E k x (by omega) h₀ h with ⟨c', z, -, -, h'⟩ | ⟨un, hu1, -, h'⟩
  · cases h'
  have hlo : T.sift read (prefixOf x k) = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, h₀]
  rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
    ⟨z, -, h''⟩ | ⟨hsh, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, -, -, -, h''⟩⟩
  · cases h''
  · rcases un with _ | ⟨c', t'⟩
    · simp [searchOutcome] at h''
    simp only [searchOutcome, Outcome.unlearnedEdge.injEq] at h''
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h''
    obtain ⟨hc, ht⟩ := hu1 c t rfl
    have hlt : k + (walk E p₀ (x.toList.drop k)).length - 1 < x.toList.length :=
      (List.getElem?_eq_some_iff.1 hc).1
    refine ⟨p₀, _, rfl, by omega, hlt, rfl, hc, by omega, rfl, hsh, ht, ?_⟩
    rw [show k + (walk E p₀ (x.toList.drop k)).length - 1 - k
      = (walk E p₀ (x.toList.drop k)).length - 1 by omega]
    exact hun c hc
  · cases h''
  · cases h''

/-- A probe that stops undecided past its start without coming out clean stopped at one string,
the undecided read of a prefix strictly between `k` and `x`'s end. -/
theorem probe_middle {zs : List (FreeMonoid α)}
    (h : probeOrAgree read T E k x = .undecided zs false) :
    ∃ j b, k < j ∧ j < x.toList.length ∧ T.sift read (prefixOf x j) = .inr b ∧ zs = [b] := by
  by_cases hs : x.toList.length < k
  · simp [probeOrAgree, hs] at h
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
  swap
  · simp [probeOrAgree, hs, probe, h₀] at h
  obtain ⟨hpos, hhi, hst0, -, -, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
  rcases probe_long_cases read T E k x (by omega) h₀ h with ⟨c', z, -, -, h'⟩ | ⟨un, -, -, h'⟩
  · cases h'
  have hlo : T.sift read (prefixOf x k) = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, h₀]
  rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
    ⟨z, -, h''⟩ | ⟨-, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, h1, h2, h3, h''⟩⟩
  · cases h''
  · rcases un with _ | ⟨c', t'⟩ <;> simp [searchOutcome] at h''
  · cases h''
  · simp only [Outcome.undecided.injEq, and_true] at h''
    exact ⟨p', z, h1, by omega, h3, h''⟩

end Structure

section Search

/-- The outcomes a search ends at: a wrong edge, or an undecided middle. -/
def Outcome.IsSearch : Outcome α → Prop
  | .wrongEdge _ _ _ _ | .undecided _ false => True
  | _ => False

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- A probe searches where its start sifts to a leaf, the successor of the unlearned edge its walk
stops at, if any, sifts decided, and the prefix at the walk's end sifts off the walk. -/
theorem probe_isSearch_iff :
    (probeOrAgree read T E k x).IsSearch ↔ k ≤ x.toList.length
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
      subst ho; exact h.elim
    rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
    swap
    · simp only [probeOrAgree, hs, if_false, probe, h₀] at ho
      subst ho; exact h.elim
    refine ⟨by omega, p₀, rfl, ?_⟩
    obtain ⟨hpos, hhi, hst0, -, -, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
    have hlo : T.sift read (prefixOf x k)
        = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, hst0, h₀]
    rcases probe_long_cases read T E k x (by omega) h₀ ho with ⟨c, z, -, -, rfl⟩ | ⟨un, hu1, hu2, h'⟩
    · exact h.elim
    rcases searchOutcome_cases read T k x _ un (by omega) hhi hlo h' with
      ⟨z, -, rfl⟩ | ⟨-, rfl⟩ | ⟨hl, -⟩
    · exact h.elim
    · rcases un with _ | ⟨c, t⟩ <;> exact h.elim
    · refine ⟨fun c hc => ?_, hl⟩
      rcases un with _ | ⟨c', t⟩
      · rw [hu2 rfl] at hc; cases hc
      · exact ⟨t, (hu1 c' t rfl).2⟩
  · rintro ⟨hk, p₀, h₀, hc, l, hl, hne⟩
    obtain ⟨hpos, hhi, hst0, -, -, -⟩ := walk_facts E k x (p₀ := p₀) hk
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
    · rw [ho]; trivial
    · rw [ho]; trivial

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
    obtain ⟨hpos, hhi, -, -, -, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
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
      | undecidedHi => simp only [searchOutcome, undecidedAt, h _ (by omega : k ≤ k + ss.length - 1)]
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

/-- A walk from a leaf reads the edges only through their targets. -/
theorem walk_congr {L : List (List Bool)} {e e' : Edges α}
    (he : ∀ q c t w, e (q, c) = some (t, w) → t ∈ L)
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
      | (subst hc; simp only; rw [walk_congr he h cs q (he _ _ _ _ h1)])

/-- A probe reads the edges only through their targets. -/
theorem probe_econgr {e e' : Edges α} (he : ∀ q c t w, e (q, c) = some (t, w) → t ∈ T.leaves)
    (h : ∀ p ∈ T.leaves, ∀ c, (e (p, c)).map Prod.fst = (e' (p, c)).map Prod.fst) :
    probeOrAgree read T e k x = probeOrAgree read T e' k x := by
  unfold probeOrAgree probe
  split_ifs
  · rfl
  rcases h₀ : T.sift read (prefixOf x k) with p₀ | z
  · simp only
    rw [walk_congr he h _ p₀ (DTree.sift_inl_mem read T _ _ h₀)]
  · rfl

end Search

section Query

theorem run_siftQuery (ℓ : ReadLabel α) (rd : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (u : FreeMonoid α), (siftQuery ℓ T u).run rd = T.sift rd u
  | .leaf, _ => by simp [siftQuery, QTree.run, DTree.sift]
  | .node m r a, u => by
    simp only [siftQuery, QTree.run, DTree.sift]
    rcases h : rd (u * m) with _ | _ | _ <;> simp only [ARU.cut]
    · rw [QTree.run_map, run_siftQuery ℓ rd a u]
    · rw [QTree.run_map, run_siftQuery ℓ rd r u]
    · rfl

theorem asks_siftQuery (ℓ : ReadLabel α) (rd : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (u : FreeMonoid α),
      (siftQuery ℓ T u).asks rd = (T.siftWithPath (fun z => (rd z).cut) u).1.map (ℓ, ·)
  | .leaf, _ => by simp [siftQuery, QTree.asks, DTree.siftWithPath]
  | .node m r a, u => by
    simp only [siftQuery, QTree.asks, DTree.siftWithPath]
    rcases h : (rd (u * m)).cut with _ | _ | _ <;> simp only
    · simp [QTree.asks]
    · rw [QTree.asks_map, asks_siftQuery ℓ rd r u]; simp
    · rw [QTree.asks_map, asks_siftQuery ℓ rd a u]; simp

theorem mem_asks_siftQuery {rd : FreeMonoid α → ARU} {ℓ : ReadLabel α} {T : DTree α}
    {u : FreeMonoid α} {q : ReadLabel α × FreeMonoid α} (hq : q ∈ (siftQuery ℓ T u).asks rd) :
    q.1 = ℓ ∧ q.2 ∈ (T.siftWithPath (fun z => (rd z).cut) u).1 := by
  rw [asks_siftQuery] at hq
  obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hq
  exact ⟨rfl, hw⟩

variable (rd : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (k : ℕ)
  (lab : ℕ → ReadLabel α) (st : ℕ → List Bool)

theorem run_bracketQuery (un : Option (α × List Bool)) (hi₀ : ℕ) :
    ∀ fuel lo hi t, hi - lo ≤ fuel → T.sift rd (prefixOf x hi) = .inl t →
      (bracketQuery T x lab st fuel lo hi t).run rd
        = searchOutcome rd T x st un hi₀ (bisectBoundary false (siftAgrees rd T x st) lo hi).1
  | 0, lo, hi, t, hf, ht => by
    rw [bisectBoundary, dif_pos (by omega)]
    simp only [bracketQuery, QTree.run, searchOutcome, edgeOutcome, ht]
    rcases x.toList[hi - 1]? with _ | c <;> rfl
  | fuel + 1, lo, hi, t, hf, ht => by
    simp only [bracketQuery]
    by_cases h1 : lo + 1 < hi
    · rw [if_pos h1, bisectBoundary, dif_neg (by omega), QTree.run_bind, run_siftQuery]
      rcases hm : T.sift rd (prefixOf x ((lo + hi) / 2)) with q | z
      · rw [agrees_inl rd T x st hm]
        simp only
        by_cases hq : q = st ((lo + hi) / 2)
        · rw [if_pos hq, decide_eq_true hq]
          exact run_bracketQuery un hi₀ fuel _ hi t (by omega) ht
        · rw [if_neg hq, decide_eq_false hq]
          exact run_bracketQuery un hi₀ fuel lo _ q (by omega) hm
      · have : siftAgrees rd T x st ((lo + hi) / 2) = none := by simp [siftAgrees, hm]
        simp only [this, Bool.false_eq_true, ↓reduceIte, QTree.run, searchOutcome, undecidedAt,
          hm, Sum.elim_inr]
    · rw [if_neg h1, bisectBoundary, dif_pos (by omega)]
      simp only [QTree.run, searchOutcome, edgeOutcome, ht]
      rcases x.toList[hi - 1]? with _ | c <;> rfl

theorem run_searchQuery (un : Option (α × List Bool)) (hi : ℕ) (hk : k ≤ hi)
    (hst : T.sift rd (prefixOf x k) = .inl (st k)) :
    (searchQuery T k x lab st un hi).run rd
      = searchOutcome rd T x st un hi (tripartiteSearch false (siftAgrees rd T x st) k hi).1 := by
  have hlo : siftAgrees rd T x st k = some true := by
    rw [agrees_inl rd T x st hst]; simp
  unfold searchQuery
  rw [QTree.run_bind, run_siftQuery]
  simp only [tripartiteSearch, hlo]
  rcases ha : T.sift rd (prefixOf x hi) with a | z
  · rw [agrees_inl rd T x st ha]
    simp only
    by_cases hq : a = st hi
    · rw [if_pos hq, decide_eq_true hq]
      simp only [QTree.run, searchOutcome]
      rcases un with _ | ⟨c, t⟩ <;> rfl
    · rw [if_neg hq, decide_eq_false hq]
      exact run_bracketQuery rd T x lab st un hi _ k hi a le_rfl ha
  · have : siftAgrees rd T x st hi = none := by simp [siftAgrees, ha]
    simp only [this, QTree.run, searchOutcome, undecidedAt, ha]
    rfl

/-- `probeQuery` is the probe's query tree. -/
theorem run_probeQuery (E : Edges α) :
    (probeQuery T E k x).run rd = probeOrAgree rd T E k x := by
  unfold probeQuery probeOrAgree
  by_cases hshort : x.toList.length < k
  · simp only [if_pos hshort, QTree.run]
  rw [if_neg hshort, if_neg hshort, QTree.run_bind, run_siftQuery]
  unfold probe
  rcases hk : T.sift rd (prefixOf x k) with p₀ | z
  · simp only
    have hpos := length_walk_pos E p₀ (x.toList.drop k)
    have hst : T.sift rd (prefixOf x k)
        = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, walk_getD_zero, hk]
    rcases hc : x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? with _ | c
    · exact run_searchQuery rd T x k _ _ _ _ (by omega) hst
    · simp only
      rw [QTree.run_bind, run_siftQuery]
      rcases T.sift rd (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) with t | z
      · exact run_searchQuery rd T x k _ _ _ _ (by omega) hst
      · rfl
  · rfl

theorem asks_bracketQuery :
    ∀ fuel lo hi t q, q ∈ (bracketQuery T x lab st fuel lo hi t).asks rd →
      ∃ i, lo < i ∧ i < hi ∧ q ∈ (siftQuery (lab i) T (prefixOf x i)).asks rd
  | 0, _, _, _, q, hq => by simp [bracketQuery, QTree.asks] at hq
  | fuel + 1, lo, hi, t, q, hq => by
    simp only [bracketQuery] at hq
    split_ifs at hq with hlh
    · rw [QTree.asks_bind, List.mem_append] at hq
      rcases hq with hq | hq
      · exact ⟨_, by omega, by omega, hq⟩
      rw [run_siftQuery] at hq
      rcases hm : T.sift rd (prefixOf x ((lo + hi) / 2)) with q' | z <;> rw [hm] at hq <;>
        simp only at hq
      · split_ifs at hq
        · obtain ⟨i, h1, h2, h3⟩ := asks_bracketQuery fuel _ hi t q hq
          exact ⟨i, by omega, h2, h3⟩
        · obtain ⟨i, h1, h2, h3⟩ := asks_bracketQuery fuel lo _ q' q hq
          exact ⟨i, h1, by omega, h3⟩
      · simp [QTree.asks] at hq
    · simp [QTree.asks] at hq

theorem asks_searchQuery (un : Option (α × List Bool)) (hi : ℕ) (q : ReadLabel α × FreeMonoid α)
    (hq : q ∈ (searchQuery T k x lab st un hi).asks rd) :
    q ∈ (siftQuery (lab hi) T (prefixOf x hi)).asks rd
      ∨ ∃ i, k < i ∧ i < hi ∧ q ∈ (siftQuery (lab i) T (prefixOf x i)).asks rd := by
  unfold searchQuery at hq
  rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
  rcases hq with hq | hq
  · exact .inl hq
  right
  rcases hs : T.sift rd (prefixOf x hi) with a | z <;> rw [hs] at hq <;> simp only at hq
  · split_ifs at hq
    · simp [QTree.asks] at hq
    · exact asks_bracketQuery rd T x lab st _ _ _ _ q hq
  · simp [QTree.asks] at hq

/-- A probe reads its prefixes from `k` to its end, each under its position's label. -/
theorem asks_probeQuery (E : Edges α) (q : ReadLabel α × FreeMonoid α)
    (hq : q ∈ (probeQuery T E k x).asks rd) :
    q.2 ∈ (T.siftWithPath (fun z => (rd z).cut) (prefixOf x q.1.1)).1 ∧ k ≤ q.1.1
      ∧ q.1.1 ≤ x.toList.length
      ∧ (q.1 = (k, none)
        ∨ ∃ p₀, T.sift rd (prefixOf x k) = .inl p₀ ∧ q.1 = readLabel E k x p₀ q.1.1) := by
  unfold probeQuery at hq
  split_ifs at hq with hshort
  · simp [QTree.asks] at hq
  rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
  rcases hq with hq | hq
  · obtain ⟨h1, h2⟩ := mem_asks_siftQuery hq
    have hk : q.1.1 = k := by rw [h1]
    rw [hk]
    exact ⟨h2, le_rfl, by omega, .inl h1⟩
  obtain ⟨p₀, hk⟩ | ⟨z, hk⟩ : (∃ p₀, T.sift rd (prefixOf x k) = .inl p₀)
      ∨ ∃ z, T.sift rd (prefixOf x k) = .inr z := by
    rcases T.sift rd (prefixOf x k) with p | z
    exacts [.inl ⟨p, rfl⟩, .inr ⟨z, rfl⟩]
  swap
  · rw [hk] at hq; simp [QTree.asks] at hq
  rw [hk] at hq
  simp only at hq
  obtain ⟨hpos, hhi, -, -, -, -⟩ := walk_facts E k x (p₀ := p₀) (by omega)
  have hlab : ∀ i, k ≤ i → i ≤ x.toList.length →
      q ∈ (siftQuery (readLabel E k x p₀ i) T (prefixOf x i)).asks rd →
      q.2 ∈ (T.siftWithPath (fun z => (rd z).cut) (prefixOf x q.1.1)).1 ∧ k ≤ q.1.1
        ∧ q.1.1 ≤ x.toList.length
        ∧ (q.1 = (k, none)
          ∨ ∃ p₀, T.sift rd (prefixOf x k) = .inl p₀ ∧ q.1 = readLabel E k x p₀ q.1.1) := by
    intro i h1 h2 h
    obtain ⟨h3, h4⟩ := mem_asks_siftQuery h
    have hi : q.1.1 = i := by rw [h3]; rfl
    rw [hi]
    exact ⟨h4, h1, h2, .inr ⟨p₀, hk, h3⟩⟩
  have hsearch : ∀ un, q ∈ (searchQuery T k x (readLabel E k x p₀)
      (fun j => (walk E p₀ (x.toList.drop k)).getD (j - k) []) un
      (k + (walk E p₀ (x.toList.drop k)).length - 1)).asks rd →
      q.2 ∈ (T.siftWithPath (fun z => (rd z).cut) (prefixOf x q.1.1)).1 ∧ k ≤ q.1.1
        ∧ q.1.1 ≤ x.toList.length
        ∧ (q.1 = (k, none)
          ∨ ∃ p₀, T.sift rd (prefixOf x k) = .inl p₀ ∧ q.1 = readLabel E k x p₀ q.1.1) := by
    intro un h
    rcases asks_searchQuery rd T x k _ _ un _ q h with h | ⟨i, h1, h2, h⟩
    · exact hlab _ (by omega) hhi h
    · exact hlab i (by omega) (by omega) h
  rcases hc : x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? with _ | c <;>
    rw [hc] at hq <;> simp only at hq
  · exact hsearch none hq
  have hlt := (List.getElem?_eq_some_iff.1 hc).1
  rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
  rcases hq with hq | hq
  · exact hlab _ (by omega) (by omega) hq
  rcases ht : T.sift rd (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) with t | z <;>
    rw [ht] at hq <;> simp only at hq
  · exact hsearch _ hq
  · simp [QTree.asks] at hq

end Query

end OrthoDFA
