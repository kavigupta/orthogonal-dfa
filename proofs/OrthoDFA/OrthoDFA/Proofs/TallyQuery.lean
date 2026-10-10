import OrthoDFA.Proofs.QueryTree
import OrthoDFA.Proofs.TallyEdgeNoise

/-!
# A probe as a query tree

One probe of `x` at a fixed hypothesis reads strings adaptively: the sift of its start, then the
sifts its walk check and its bisection call for, each sift reading a node's string and going on by
its read. `probeQ` is that computation as a query tree, each read labelled by the position sifted
and the edge the position is charged to, which the start's sift fixes before; its run is `probeBy`
(`run_probeQ`). `recordQ` then sifts the record's two positions again, so a record that is not true
has a read on its state's rarer side among its reads (`untrue_minority`), and a read undecided
stops the sift of a position the tests charge it to (`und_twins`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

/-- A read's label: the position sifted, and the edge it is charged to. -/
abbrev PLabel (α : Type*) := ℕ × Option (List Bool × α)

/-- A probe's query trees. -/
abbrev PQ (α β : Type*) := QTree (PLabel α) (FreeMonoid α) ARU β

section Sift

/-- The sift of `u`, each node's string read under the label `ℓ`. -/
def siftQ (ℓ : PLabel α) : DTree α → FreeMonoid α → PQ α (List Bool ⊕ FreeMonoid α)
  | .leaf, _ => .done (.inl [])
  | .node m r a, u => .ask ℓ (u * m) fun v => match v.cut with
    | none => .done (.inr (u * m))
    | some true => (siftQ ℓ a u).map fun s => s.map (true :: ·) id
    | some false => (siftQ ℓ r u).map fun s => s.map (false :: ·) id

theorem run_siftQ (ℓ : PLabel α) (rd : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (u : FreeMonoid α),
      (siftQ ℓ T u).run rd = T.sift (fun z => (rd z).cut) u
  | .leaf, _ => by simp [siftQ, QTree.run, DTree.sift, DTree.route]
  | .node m r a, u => by
    simp only [siftQ, QTree.run, DTree.sift, DTree.route]
    rcases h : (rd (u * m)).cut with _ | _ | _ <;> simp only []
    · rfl
    · rw [QTree.run_map, run_siftQ ℓ rd r u]; rfl
    · rw [QTree.run_map, run_siftQ ℓ rd a u]; rfl

theorem asks_siftQ (ℓ : PLabel α) (rd : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (u : FreeMonoid α),
      (siftQ ℓ T u).asks rd = (T.route (fun z => (rd z).cut) u).1.map (ℓ, ·)
  | .leaf, _ => by simp [siftQ, QTree.asks, DTree.route]
  | .node m r a, u => by
    simp only [siftQ, QTree.asks, DTree.route]
    rcases h : (rd (u * m)).cut with _ | _ | _ <;> simp only []
    · simp [QTree.asks]
    · rw [QTree.asks_map, asks_siftQ ℓ rd r u]; simp
    · rw [QTree.asks_map, asks_siftQ ℓ rd a u]; simp

/-- A string read undecided on a sift's route is where the sift stops. -/
theorem sift_of_route_und {cut : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u w : FreeMonoid α), w ∈ (T.route cut u).1 → cut w = none →
      T.sift cut u = .inr w
  | .leaf, _, _, h, _ => by simp [DTree.route] at h
  | .node m r a, u, w, h, hw => by
    simp only [DTree.sift, DTree.route] at h ⊢
    rcases hc : cut (u * m) with _ | _ | _ <;> simp only [hc, List.mem_cons, List.mem_singleton]
      at h ⊢
    · rcases h with rfl | h
      · rfl
      · simp at h
    · rcases h with rfl | h
      · rw [hw] at hc; cases hc
      · have := sift_of_route_und r u w h hw
        simp only [DTree.sift] at this
        rw [this]; rfl
    · rcases h with rfl | h
      · rw [hw] at hc; cases hc
      · have := sift_of_route_und a u w h hw
        simp only [DTree.sift] at this
        rw [this]; rfl

variable {σ : Type*} (G : ReadModel α σ)

/-- A decided sift off the true leaf read a string on its state's rarer side. -/
theorem minority_of_sift {cut : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α) (p : List Bool), T.sift cut u = .inl p →
      G.leafOf T (G.M.eval u.toList) ≠ p →
      ∃ w ∈ (T.route cut u).1, cut w = some (!G.side (G.M.eval w.toList))
  | .leaf, u, p, h, hne => by
    simp only [DTree.sift, DTree.route, Sum.inl.injEq] at h
    subst h; exact absurd rfl hne
  | .node m r a, u, p, h, hne => by
    have hst : G.M.eval (u * m).toList = G.at' (G.M.eval u.toList) m := by
      simp [ReadModel.at', FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
    simp only [DTree.sift, DTree.route] at h ⊢
    simp only [ReadModel.leafOf] at hne
    rcases hc : cut (u * m) with _ | _ | _ <;> simp only [hc] at h ⊢
    · cases h
    · simp only [map_cons_eq_inl] at h
      obtain ⟨p', rfl, hp'⟩ := h
      by_cases hs : G.side (G.at' (G.M.eval u.toList) m)
      · exact ⟨u * m, List.mem_cons_self, by rw [hst, hs, hc]; rfl⟩
      · simp only [hs, Bool.false_eq_true, if_false, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨w, hw, hcw⟩ := minority_of_sift r u p' hp' hne
        exact ⟨w, List.mem_cons_of_mem _ hw, hcw⟩
    · simp only [map_cons_eq_inl] at h
      obtain ⟨p', rfl, hp'⟩ := h
      by_cases hs : G.side (G.at' (G.M.eval u.toList) m)
      · simp only [hs, if_true, ne_eq, List.cons.injEq, true_and] at hne
        obtain ⟨w, hw, hcw⟩ := minority_of_sift a u p' hp' hne
        exact ⟨w, List.mem_cons_of_mem _ hw, hcw⟩
      · exact ⟨u * m, List.mem_cons_self, by rw [hst, hc]; simp [hs]⟩

end Sift

section Bracket

/-- The agreement at `p`, the range's ends known. -/
def agAt (agQ : ℕ → PQ α (Option Bool)) (lo hi p : ℕ) : PQ α (Option Bool) :=
  if p = lo then .done (some true) else if p = hi then .done (some false) else agQ p

/-- `bracketAt`, each position's agreement read through `agQ`. -/
def bracketQ (agQ : ℕ → PQ α (Option Bool)) (ps : List (List Bool)) :
    ℕ → ℕ → ℕ → PQ α (Outcome α)
  | 0, _, hi => .done (.edge ps hi)
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      (agAt agQ lo hi ((lo + hi) / 2)).bind fun am => match am with
      | some true => bracketQ agQ ps fuel ((lo + hi) / 2) hi
      | some false => bracketQ agQ ps fuel lo ((lo + hi) / 2)
      | none => (agAt agQ lo hi ((lo + hi) / 2 - 1)).bind fun al => match al with
        | none => .done (.pair ((lo + hi) / 2 - 1))
        | some bl => (agAt agQ lo hi ((lo + hi) / 2 + 1)).bind fun ar => match bl, ar with
          | _, none => .done (.pair ((lo + hi) / 2))
          | true, some false => .done (.triple ((lo + hi) / 2))
          | true, some true => bracketQ agQ ps fuel ((lo + hi) / 2 + 1) hi
          | false, some _ => bracketQ agQ ps fuel lo ((lo + hi) / 2 - 1)
    else .done (.edge ps hi)

theorem run_agAt (agQ : ℕ → PQ α (Option Bool)) (rd : FreeMonoid α → ARU)
    (agrees : ℕ → Option Bool) (h : ∀ i, (agQ i).run rd = agrees i) (lo hi p : ℕ) :
    (agAt agQ lo hi p).run rd
      = if p = lo then some true else if p = hi then some false else agrees p := by
  unfold agAt; split_ifs <;> simp [QTree.run, h]

theorem run_bracketQ (agQ : ℕ → PQ α (Option Bool)) (ps : List (List Bool))
    (rd : FreeMonoid α → ARU) (agrees : ℕ → Option Bool) (h : ∀ i, (agQ i).run rd = agrees i) :
    ∀ fuel lo hi, (bracketQ agQ ps fuel lo hi).run rd = bracketAt agrees ps fuel lo hi
  | 0, _, _ => rfl
  | fuel + 1, lo, hi => by
    have hag := run_agAt agQ rd agrees h lo hi
    simp only [bracketQ, bracketAt]
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh, if_neg hlh]; rfl
    rw [if_pos hlh, if_pos hlh, QTree.run_bind, hag]
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = vm
    rcases vm with _ | _ | _ <;> simp only []
    · rw [QTree.run_bind, hag]
      generalize (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = vl
      rcases vl with _ | bl <;> simp only [QTree.run]
      rw [QTree.run_bind, hag]
      generalize (if (lo + hi) / 2 + 1 = lo then some true
        else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = vr
      rcases vr with _ | br <;> cases bl <;> (try cases br) <;> simp only [QTree.run] <;>
        exact run_bracketQ agQ ps rd agrees h fuel _ _
    · exact run_bracketQ agQ ps rd agrees h fuel _ _
    · exact run_bracketQ agQ ps rd agrees h fuel _ _

theorem asks_agAt (agQ : ℕ → PQ α (Option Bool)) (rd : FreeMonoid α → ARU) (lo hi p : ℕ)
    (q : PLabel α × FreeMonoid α) (hq : q ∈ (agAt agQ lo hi p).asks rd) :
    p ≠ lo ∧ p ≠ hi ∧ q ∈ (agQ p).asks rd := by
  unfold agAt at hq
  split_ifs at hq with h1 h2
  · simp [QTree.asks] at hq
  · simp [QTree.asks] at hq
  · exact ⟨h1, h2, hq⟩

/-- The bisection reads only through `agQ`, at positions strictly inside its range that
`bracketSifts` lists. -/
theorem asks_bracketQ (agQ : ℕ → PQ α (Option Bool)) (ps : List (List Bool))
    (rd : FreeMonoid α → ARU) (agrees : ℕ → Option Bool) (h : ∀ i, (agQ i).run rd = agrees i) :
    ∀ fuel lo hi q, q ∈ (bracketQ agQ ps fuel lo hi).asks rd →
      ∃ i, i ∈ bracketSifts agrees fuel lo hi ∧ lo < i ∧ i < hi ∧ q ∈ (agQ i).asks rd
  | 0, _, _, q, hq => by simp [bracketQ, QTree.asks] at hq
  | fuel + 1, lo, hi, q, hq => by
    have hag := run_agAt agQ rd agrees h lo hi
    simp only [bracketQ] at hq
    simp only [bracketSifts]
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at hq; simp [QTree.asks] at hq
    rw [if_pos hlh] at hq
    rw [if_pos hlh]
    have hmid : lo < (lo + hi) / 2 ∧ (lo + hi) / 2 < hi := by omega
    have hrec : ∀ lo' hi', lo ≤ lo' → hi' ≤ hi → q ∈ (bracketQ agQ ps fuel lo' hi').asks rd →
        ∃ i, i ∈ bracketSifts agrees fuel lo' hi' ∧ lo < i ∧ i < hi ∧ q ∈ (agQ i).asks rd := by
      intro lo' hi' h1 h2 hq'
      obtain ⟨i, hi', h3, h4, hq'⟩ := asks_bracketQ agQ ps rd agrees h fuel lo' hi' q hq'
      exact ⟨i, hi', by omega, by omega, hq'⟩
    rw [QTree.asks_bind, hag, List.mem_append] at hq
    generalize hvm : (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then
      some false else agrees ((lo + hi) / 2)) = vm at hq ⊢
    rcases hq with hq | hq
    · obtain ⟨-, -, hq⟩ := asks_agAt agQ rd lo hi _ q hq
      refine ⟨(lo + hi) / 2, ?_, hmid.1, hmid.2, hq⟩
      rcases vm with _ | _ | _ <;> simp only [List.mem_cons, true_or]
      split <;> simp
    rcases vm with _ | _ | _ <;> simp only [] at hq ⊢
    · rw [QTree.asks_bind, hag, List.mem_append] at hq
      generalize hvl : (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = vl
        at hq ⊢
      rcases hq with hq | hq
      · obtain ⟨h1, -, hq⟩ := asks_agAt agQ rd lo hi _ q hq
        refine ⟨(lo + hi) / 2 - 1, ?_, by omega, by omega, hq⟩
        split <;> simp
      rcases vl with _ | bl <;> simp only [QTree.asks, List.not_mem_nil] at hq ⊢
      rw [QTree.asks_bind, hag, List.mem_append] at hq
      generalize hvr : (if (lo + hi) / 2 + 1 = lo then some true
        else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = vr
        at hq ⊢
      rcases hq with hq | hq
      · obtain ⟨-, h2, hq⟩ := asks_agAt agQ rd lo hi _ q hq
        refine ⟨(lo + hi) / 2 + 1, ?_, by omega, by omega, hq⟩
        cases bl <;> rcases vr with _ | _ | _ <;> simp
      cases bl <;> rcases vr with _ | _ | _ <;> simp only [QTree.asks, List.not_mem_nil] at hq ⊢
      · obtain ⟨i, hi', h3⟩ := hrec _ _ le_rfl (by omega) hq
        exact ⟨i, by simp [hi'], h3⟩
      · obtain ⟨i, hi', h3⟩ := hrec _ _ le_rfl (by omega) hq
        exact ⟨i, by simp [hi'], h3⟩
      · obtain ⟨i, hi', h3⟩ := hrec _ _ (by omega) le_rfl hq
        exact ⟨i, by simp [hi'], h3⟩
    · obtain ⟨i, hi', h3⟩ := hrec _ _ le_rfl (by omega) hq
      exact ⟨i, by simp [hi'], h3⟩
    · obtain ⟨i, hi', h3⟩ := hrec _ _ (by omega) le_rfl hq
      exact ⟨i, by simp [hi'], h3⟩

end Bracket

section Walk

omit [Fintype α] [DecidableEq α] in
theorem follow_take_inl {edges : Edges α} :
    ∀ (l : List α) (p : List Bool) (ps : List (List Bool)), follow edges p l = .inl ps →
      ∀ n, ∃ ps', follow edges p (l.take n) = .inl ps' ∧ ps' ≠ []
  | [], p, ps, _, n => ⟨[p], by simp [follow], by simp⟩
  | c :: l, p, ps, h, n => by
    rcases n with _ | n
    · exact ⟨[p], by simp [follow], by simp⟩
    simp only [follow] at h
    rcases he : edges p c with _ | ⟨q, y⟩ <;> rw [he] at h
    · cases h
    simp only [] at h
    rcases hf : follow edges q l with ps₁ | r <;> rw [hf] at h
    · obtain ⟨ps', h', -⟩ := follow_take_inl l q ps₁ hf n
      exact ⟨p :: ps', by simp [follow, he, h'], by simp⟩
    · cases h

theorem posEdgeFrom_isSome {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {p₀ : List Bool}
    {i i' n : ℕ} (hi : i' = i ∨ i' + 1 = i) (hlt : i' < x.toList.length) (hn : i' - k ≤ n)
    {ps : List (List Bool)} (hf : follow edges p₀ ((x.toList.drop k).take n) = .inl ps) :
    (posEdgeFrom edges k x p₀ i).isSome := by
  obtain ⟨ps', hps', hne⟩ := follow_take_inl _ _ _ hf (i' - k)
  rw [List.take_take, min_eq_left hn] at hps'
  have hw : walkFrom edges k x p₀ i' = ps' := by simp [walkFrom, hps']
  have hl : (walkFrom edges k x p₀ i').getLast? = some (ps'.getLast hne) := by
    rw [hw, List.getLast?_eq_getLast hne]
  have hx : ∃ c, x.toList[i']? = some c := ⟨_, List.getElem?_eq_getElem hlt⟩
  obtain ⟨c, hc⟩ := hx
  unfold posEdgeFrom
  rcases hi with rfl | rfl
  · simp [hl, hc]
  · simp only [Nat.add_sub_cancel]
    simp [hl, hc]

end Walk

section Probe

variable (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- The label of position `i`'s reads, the start at the leaf `p₀`. -/
def plab (p₀ : List Bool) (i : ℕ) : PLabel α := (i, posEdgeFrom edges k x p₀ i)

/-- The agreement of position `i`'s sift with the walk. -/
def agQ (p₀ : List Bool) (walkAt : ℕ → List Bool) (i : ℕ) : PQ α (Option Bool) :=
  (siftQ (plab edges k x p₀ i) T (prefixOf x i)).map fun s =>
    s.elim (fun q => some (decide (q = walkAt i))) fun _ => none

/-- One probe of `x` from `k`, as a query tree. -/
def probeQ : PQ α (Outcome α) :=
  (siftQ (k, none) T (prefixOf x k)).bind fun a => match a with
  | .inr _ => .done (.startUndecided (prefixOf x k))
  | .inl p₀ => match follow edges p₀ (x.toList.drop k) with
    | .inl ps =>
      (siftQ (plab edges k x p₀ x.toList.length) T (prefixOf x x.toList.length)).bind fun r =>
        match r with
        | .inr _ => .done (.endUndecided x)
        | .inl a' => if some a' = ps.getLast? then .done .agree
            else bracketQ (agQ T edges k x p₀ fun j => ps.getD (j - k) []) ps
              (x.toList.length - k) k x.toList.length
    | .inr (s, _, i) =>
      (siftQ (plab edges k x p₀ (k + i + 1)) T (prefixOf x (k + i + 1))).bind fun r1 =>
        match r1 with
        | .inr _ => .done (.endUndecided (prefixOf x (k + i + 1)))
        | .inl _ => (siftQ (plab edges k x p₀ (k + i)) T (prefixOf x (k + i))).bind fun r2 =>
          match r2 with
          | .inr _ => .done (.endUndecided (prefixOf x (k + i)))
          | .inl p => if p = s then .done (.member (prefixOf x (k + i)))
              else bracketQ
                (agQ T edges k x p₀ fun j => (walkFrom edges k x p₀ (k + i)).getD (j - k) [])
                (walkFrom edges k x p₀ (k + i)) (k + i - k) k (k + i)

/-- The probe, and then the sifts of its record's two positions again. -/
def recordQ : PQ α (Outcome α) :=
  (probeQ T edges k x).bind fun o => match o with
  | .edge _ fd => (siftQ (fd - 1, none) T (prefixOf x (fd - 1))).bind fun _ =>
      (siftQ (fd, none) T (prefixOf x fd)).map fun _ => o
  | _ => .done o

variable (rd : FreeMonoid α → ARU)

theorem run_agQ (p₀ : List Bool) (walkAt : ℕ → List Bool) (i : ℕ) :
    (agQ T edges k x p₀ walkAt i).run rd = agreesAtBy (fun z => (rd z).cut) T x walkAt i := by
  simp only [agQ, QTree.run_map, run_siftQ, agreesAtBy]

theorem walkToBy_of_sift {p₀ : List Bool}
    (hk : T.sift (fun z => (rd z).cut) (prefixOf x k) = .inl p₀) (j : ℕ) :
    walkToBy (fun z => (rd z).cut) T edges k x j = walkFrom edges k x p₀ j := by
  simp only [walkToBy, hk, walkFrom]

theorem run_probeQ : (probeQ T edges k x).run rd = probeBy (fun z => (rd z).cut) T edges k x := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  unfold probeQ probeBy walkCheckBy kWalkBy
  rw [QTree.run_bind, run_siftQ]
  rcases hk : T.sift cut (prefixOf x k) with p₀ | b
  · simp only []
    rcases hf : follow edges p₀ (x.toList.drop k) with ps | ⟨s, c, i⟩
    · simp only []
      rw [QTree.run_bind, run_siftQ, prefixOf_length]
      rcases hx : T.sift cut x with a' | b'
      · simp only []
        split_ifs
        · rfl
        · simp only [Sum.elim_inr]
          exact run_bracketQ _ _ rd _ (run_agQ T edges k x rd p₀ _) _ _ _
      · rfl
    · simp only []
      rw [QTree.run_bind, run_siftQ]
      rcases h1 : T.sift cut (prefixOf x (k + i + 1)) with q₁ | b₁
      · simp only []
        rw [QTree.run_bind, run_siftQ]
        rcases h2 : T.sift cut (prefixOf x (k + i)) with q₂ | b₂
        · simp only []
          split_ifs
          · rfl
          · simp only [Sum.elim_inr]
            rw [walkToBy_of_sift T edges k x rd hk]
            exact run_bracketQ _ _ rd _ (run_agQ T edges k x rd p₀ _) _ _ _
        · rfl
      · rfl
  · rfl

theorem mem_asks_siftQ {ℓ : PLabel α} {u : FreeMonoid α} {q : PLabel α × FreeMonoid α}
    (hq : q ∈ (siftQ ℓ T u).asks rd) : q.1 = ℓ ∧ q.2 ∈ (T.route (fun z => (rd z).cut) u).1 := by
  rw [asks_siftQ] at hq
  obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hq
  exact ⟨rfl, hw⟩

/-- A read of the probe is on the route of the position its label names: the start's prefix, or a
position of `siftsBy` charged to the edge the label names. -/
theorem asks_probeQ (q : PLabel α × FreeMonoid α) (hq : q ∈ (probeQ T edges k x).asks rd) :
    q.2 ∈ (T.route (fun z => (rd z).cut) (prefixOf x q.1.1)).1
      ∧ (prefixOf x q.1.1 = prefixOf x k
        ∨ (q.1.1 ∈ siftsBy (fun z => (rd z).cut) T edges k x
          ∧ ∃ e, posEdgeBy (fun z => (rd z).cut) T edges k x q.1.1 = some e
            ∧ q.1.2 = some e)) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  unfold probeQ at hq
  rw [QTree.asks_bind, List.mem_append, run_siftQ] at hq
  rcases hq with hq | hq
  · obtain ⟨h1, h2⟩ := mem_asks_siftQ T rd hq
    have e1 : q.1.1 = k := by rw [h1]
    rw [e1]
    exact ⟨h2, .inl rfl⟩
  rcases hk : T.sift cut (prefixOf x k) with p₀ | b <;> rw [hk] at hq
  swap
  · simp [QTree.asks] at hq
  simp only [] at hq
  have hlab : ∀ i, q.1 = plab edges k x p₀ i → k < i → i ∈ siftsBy cut T edges k x →
      (posEdgeFrom edges k x p₀ i).isSome →
      (q.1.1 ∈ siftsBy cut T edges k x ∧ ∃ e, posEdgeBy cut T edges k x q.1.1 = some e
        ∧ q.1.2 = some e) := by
    intro i hl _ hs hsome
    rw [hl]
    simp only [plab]
    refine ⟨hs, (posEdgeFrom edges k x p₀ i).get hsome, ?_, by simp⟩
    rw [posEdgeBy_of_inl hk]; simp
  rcases hf : follow edges p₀ (x.toList.drop k) with ps | ⟨s, c, i₀⟩ <;> rw [hf] at hq <;>
    simp only [] at hq
  · have hfull : follow edges p₀ ((x.toList.drop k).take (x.toList.drop k).length) = .inl ps := by
      rw [List.take_length]; exact hf
    have hkw : kWalkBy cut T edges k x = .reached ps := by simp [kWalkBy, hk, hf]
    rw [QTree.asks_bind, List.mem_append, run_siftQ] at hq
    rcases hq with hq | hq
    · obtain ⟨h1, h2⟩ := mem_asks_siftQ T rd hq
      have e1 : q.1.1 = x.toList.length := by rw [h1]; rfl
      refine ⟨by rw [e1]; exact h2, ?_⟩
      by_cases hlx : x.toList.length ≤ k
      · left
        rw [e1]
        simp only [prefixOf, List.take_of_length_le hlx, List.take_of_length_le le_rfl]
      push_neg at hlx
      right
      refine hlab _ h1 hlx ?_ (posEdgeFrom_isSome (i' := x.toList.length - 1) (.inr (by omega))
        (by omega) (by simp; omega) hfull)
      simp only [siftsBy, hkw, prefixOf_length, List.mem_dedup, List.mem_filter,
        decide_eq_true_eq]
      refine ⟨?_, hlx⟩
      split <;> (try split_ifs) <;> simp
    rcases hx : T.sift cut (prefixOf x x.toList.length) with a' | b' <;> rw [hx] at hq <;>
      simp only [] at hq
    · split_ifs at hq with hag
      · simp [QTree.asks] at hq
      obtain ⟨i, hi, hki, hix, hq'⟩ := asks_bracketQ _ ps rd _ (run_agQ T edges k x rd p₀ _) _ _ _ q hq
      rw [agQ, QTree.asks_map] at hq'
      obtain ⟨h1, h2⟩ := mem_asks_siftQ T rd hq'
      have e1 : q.1.1 = i := by rw [h1]; rfl
      refine ⟨by rw [e1]; exact h2, .inr (hlab _ h1 hki ?_ (posEdgeFrom_isSome (i' := i) (.inl rfl) hix
        (by simp; omega) hfull))⟩
      rw [prefixOf_length] at hx
      simp only [siftsBy, hkw, hx, List.mem_dedup, List.mem_filter, decide_eq_true_eq]
      refine ⟨?_, hki⟩
      rw [if_neg hag]
      simp only [List.mem_cons]; exact .inr (by simpa using hi)
    · simp [QTree.asks] at hq
  · obtain ⟨hi₀, -, -, ps', hps', hlast⟩ := follow_inr _ _ _ _ _ hf
    simp only [List.length_drop] at hi₀
    have hkw : kWalkBy cut T edges k x = .edge s c (k + i₀) := by simp [kWalkBy, hk, hf]
    rw [QTree.asks_bind, List.mem_append, run_siftQ] at hq
    rcases hq with hq | hq
    · obtain ⟨h1, h2⟩ := mem_asks_siftQ T rd hq
      have e1 : q.1.1 = k + i₀ + 1 := by rw [h1]; rfl
      refine ⟨by rw [e1]; exact h2, .inr (hlab _ h1 (by omega) ?_ (posEdgeFrom_isSome (i' := k + i₀) (.inr rfl)
        (by omega) (by omega) hps'))⟩
      simp only [siftsBy, hkw, List.mem_dedup, List.mem_filter, decide_eq_true_eq]
      refine ⟨?_, by omega⟩
      split
      · simp
      · split
        · simp
        · split_ifs <;> simp
    rcases h1 : T.sift cut (prefixOf x (k + i₀ + 1)) with q₁ | b₁ <;> rw [h1] at hq <;>
      simp only [] at hq
    swap
    · simp [QTree.asks] at hq
    rw [QTree.asks_bind, List.mem_append, run_siftQ] at hq
    rcases hq with hq | hq
    · obtain ⟨h1', h2⟩ := mem_asks_siftQ T rd hq
      have e1 : q.1.1 = k + i₀ := by rw [h1']; rfl
      refine ⟨by rw [e1]; exact h2, ?_⟩
      rcases Nat.eq_zero_or_pos i₀ with rfl | hpos
      · left; rw [e1]; rfl
      right
      refine hlab _ h1' (by omega) ?_ (posEdgeFrom_isSome (i' := k + i₀) (.inl rfl)
        (by omega) (by omega) hps')
      simp only [siftsBy, hkw, h1, List.mem_dedup, List.mem_filter, decide_eq_true_eq]
      refine ⟨?_, by omega⟩
      split
      · simp
      · split_ifs <;> simp
    rcases h2 : T.sift cut (prefixOf x (k + i₀)) with q₂ | b₂ <;> rw [h2] at hq <;>
      simp only [] at hq
    swap
    · simp [QTree.asks] at hq
    split_ifs at hq with hqs
    · simp [QTree.asks] at hq
    obtain ⟨i, hi, hki, hij, hq'⟩ := asks_bracketQ _ _ rd _ (run_agQ T edges k x rd p₀ _) _ _ _ q hq
    rw [agQ, QTree.asks_map] at hq'
    obtain ⟨h1', h2'⟩ := mem_asks_siftQ T rd hq'
    have e1 : q.1.1 = i := by rw [h1']; rfl
    refine ⟨by rw [e1]; exact h2', .inr (hlab _ h1' hki ?_ (posEdgeFrom_isSome (i' := i) (.inl rfl) (by omega)
      (by omega) hps'))⟩
    simp only [siftsBy, hkw, h1, h2, List.mem_dedup, List.mem_filter, decide_eq_true_eq]
    refine ⟨?_, hki⟩
    rw [if_neg hqs, walkToBy_of_sift T edges k x rd hk]
    simp only [Nat.add_sub_cancel_left] at hi
    simp only [List.mem_cons]; exact .inr (.inr (by simpa using hi))

end Probe

section Record

variable (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (rd : FreeMonoid α → ARU)

theorem sift_route_decided {cut : FreeMonoid α → Option Bool} {u : FreeMonoid α} {p : List Bool}
    (h : T.sift cut u = .inl p) {w : FreeMonoid α} (hw : w ∈ (T.route cut u).1) : cut w ≠ none :=
  fun hc => by rw [sift_of_route_und T u w hw hc] at h; exact Sum.inr_ne_inl h

/-- A probe at an edge sifts the edge's two positions decided, the first to the walk. -/
theorem probeBy_edge {ps : List (List Bool)} {fd : ℕ}
    (h : probeBy (fun z => (rd z).cut) T edges k x = .edge ps fd) :
    k < fd ∧ T.sift (fun z => (rd z).cut) (prefixOf x (fd - 1)) = .inl (ps.getD (fd - 1 - k) [])
      ∧ ∃ t, T.sift (fun z => (rd z).cut) (prefixOf x fd) = .inl t := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  obtain ⟨ps₀, hi, hw, hb⟩ := probeBy_search cut h trivial
  obtain ⟨p₀, hk, hf, hkh, hhx, hag⟩ := walkCheckBy_inr cut hw
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hlo : agreesAtBy cut T x (fun j => ps₀.getD (j - k) []) k = some true := by
    simp [agreesAtBy, hk, ← hhead, List.getD_eq_getElem?_getD]
  obtain ⟨rfl, hfd, -, h1, h2⟩ := bracketAt_edge _ ps₀ (hi - k) k hi ps fd hkh le_rfl hlo hag hb
  unfold agreesAtBy at h1 h2
  refine ⟨hfd, ?_, ?_⟩
  · rcases hs : T.sift cut (prefixOf x (fd - 1)) with p | b <;> rw [hs] at h1 <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, decide_eq_true_eq,
        reduceCtorEq] at h1
    rw [h1]
  · rcases hs : T.sift cut (prefixOf x fd) with p | b <;> rw [hs] at h2 <;>
      simp only [Sum.elim_inl, Sum.elim_inr, reduceCtorEq] at h2
    exact ⟨p, rfl⟩

theorem asks_recordQ : (recordQ T edges k x).asks rd = (probeQ T edges k x).asks rd
    ++ match probeBy (fun z => (rd z).cut) T edges k x with
      | .edge _ fd => (T.route (fun z => (rd z).cut) (prefixOf x (fd - 1))).1.map ((fd - 1, none), ·)
          ++ (T.route (fun z => (rd z).cut) (prefixOf x fd)).1.map ((fd, none), ·)
      | _ => [] := by
  unfold recordQ
  rw [QTree.asks_bind, run_probeQ]
  congr 1
  rcases probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;>
    simp only [QTree.asks]
  rw [QTree.asks_bind, asks_siftQ, QTree.asks_map, asks_siftQ]

/-- A record that is not true read a string on its state's rarer side. -/
theorem untrue_minority {σ : Type*} (G : ReadModel α σ) (hx : x ∈ G.untrueAt rd k T edges) :
    ∃ q ∈ (recordQ T edges k x).asks rd, (rd q.2).cut = some (!G.side (G.M.eval q.2.toList)) := by
  obtain ⟨⟨p, c, t⟩, sp, hrec, hnt⟩ := hx
  unfold recordBy at hrec
  rw [asks_recordQ]
  rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;>
    rw [ho] at hrec <;> simp only [reduceCtorEq] at hrec
  obtain ⟨hkf, hs1, -⟩ := probeBy_edge T edges k x rd ho
  rcases hc : x.toList[fd - 1]? with _ | c₁ <;> rw [hc] at hrec
  · simp at hrec
  rcases ht : T.sift (fun z => (rd z).cut) (prefixOf x fd) with t' | b <;> rw [ht] at hrec
  swap
  · simp at hrec
  simp only [Option.some.injEq, Prod.mk.injEq] at hrec
  obtain ⟨⟨rfl, rfl, rfl⟩, rfl⟩ := hrec
  have hsc : prefixOf x (fd - 1) * FreeMonoid.of c₁ = prefixOf x fd := by
    apply FreeMonoid.toList.injective
    rw [List.getElem?_eq_some_iff] at hc
    obtain ⟨hlt, hc⟩ := hc
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList, FreeMonoid.toList_of]
    rw [show fd = fd - 1 + 1 by omega, List.take_add_one, List.getElem?_eq_getElem hlt, hc]
    simp
  simp only [ReadModel.TrueRec, not_and_or] at hnt
  rcases hnt with hn | hn
  · obtain ⟨w, hw, hcw⟩ := minority_of_sift G T _ _ hs1 hn
    exact ⟨((fd - 1, none), w), by simp [hw], hcw⟩
  · rw [hsc] at hn
    obtain ⟨w, hw, hcw⟩ := minority_of_sift G T _ _ ht hn
    exact ⟨((fd, none), w), by simp [hw], hcw⟩

/-- An undecided read stops a sift the tests charge it to: the start's, or a position's charged
to an edge out of a leaf. -/
theorem und_mem_twins (he : EdgesInto T edges) (q : PLabel α × FreeMonoid α)
    (hq : q ∈ (recordQ T edges k x).asks rd) (hu : (rd q.2).cut = none) :
    q.2 ∈ startHarvBy (fun z => (rd z).cut) T k x
      ++ (T.paths.toFinset ×ˢ (Finset.univ : Finset α)).toList.flatMap
        (edgeHarvBy (fun z => (rd z).cut) T edges k x) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  rw [asks_recordQ, List.mem_append] at hq
  rcases hq with hq | hq
  · obtain ⟨hr, hpos⟩ := asks_probeQ T edges k x rd q hq
    have hs := sift_of_route_und T _ _ hr hu
    rcases hpos with hpos | ⟨hi, e, he', -⟩
    · rw [hpos] at hs
      refine List.mem_append_left _ ?_
      simp [startHarvBy, cut, hs]
    · refine List.mem_append_right _ (List.mem_flatMap.2 ⟨e, ?_, ?_⟩)
      · simp only [Finset.mem_toList, Finset.mem_product, List.mem_toFinset, Finset.mem_univ,
          and_true]
        exact posEdgeBy_mem he he'
      · simp only [edgeHarvBy, List.mem_filterMap]
        exact ⟨q.1.1, hi, by simp [he', cut, hs]⟩
  · exfalso
    rcases ho : probeBy cut T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;>
      rw [ho] at hq <;> simp only [List.not_mem_nil] at hq
    obtain ⟨-, hs1, t, hs2⟩ := probeBy_edge T edges k x rd ho
    simp only [List.mem_append, List.mem_map] at hq
    rcases hq with ⟨w, hw, rfl⟩ | ⟨w, hw, rfl⟩
    · exact sift_route_decided T hs1 hw hu
    · exact sift_route_decided T hs2 hw hu

/-- The undecided first reads are at most the probe's undecided strings the tests count. -/
theorem und_twins (he : EdgesInto T edges) :
    (((recordQ T edges k x).firsts rd ∅).filter fun q => (rd q.2).cut = none).length
      ≤ twinsBy (fun z => (rd z).cut) T edges k x := by
  set L := startHarvBy (fun z => (rd z).cut) T k x
      ++ (T.paths.toFinset ×ˢ (Finset.univ : Finset α)).toList.flatMap
        (edgeHarvBy (fun z => (rd z).cut) T edges k x)
  have hL : L.length = twinsBy (fun z => (rd z).cut) T edges k x := by
    simp only [L, twinsBy, List.length_append, List.length_flatMap, Finset.sum_map_toList]
  set F := ((recordQ T edges k x).firsts rd ∅).filter fun q => (rd q.2).cut = none
  obtain ⟨hnd, hkeys⟩ := QTree.firsts_keys_eq rd (recordQ T edges k x) ∅
  have hnd' : (F.map Prod.snd).Nodup := hnd.sublist (List.filter_sublist.map _)
  have hsub : F.map Prod.snd ⊆ L := by
    intro w hw
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hw
    rw [List.mem_filter, decide_eq_true_eq] at hq
    have hk : q.2 ∈ ((recordQ T edges k x).asks rd).map Prod.snd := by
      have : q.2 ∈ (((recordQ T edges k x).firsts rd ∅).map Prod.snd).toFinset :=
        List.mem_toFinset.2 (List.mem_map_of_mem hq.1)
      rw [hkeys, Finset.sdiff_empty, List.mem_toFinset] at this
      exact this
    obtain ⟨q', hq', heq⟩ := List.mem_map.1 hk
    rw [← heq] at hq ⊢
    exact und_mem_twins T edges k x rd he q' hq' hq.2
  rw [← hL, ← List.length_map (f := Prod.snd)]
  exact (hnd'.subperm hsub).length_le

end Record

section Mean

attribute [local instance] ARU.fintype

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- The read on a read-state's rarer side. -/
noncomputable def ReadModel.minVal (r : σ) : ARU := if G.side r then .reject else .accept

theorem cut_minVal (r : σ) (v : ARU) : v.cut = some (!G.side r) ↔ v = G.minVal r := by
  unfold ReadModel.minVal; cases v <;> cases G.side r <;> simp [ARU.cut]

theorem dist_minVal (r : σ) :
    G.dist r (G.minVal r) = min (G.dist r .accept) (G.dist r .reject) := by
  unfold ReadModel.minVal ReadModel.side
  by_cases h : G.dist r .reject < G.dist r .accept
  · simp [h, min_eq_right h.le]
  · simp [h, min_eq_left (not_lt.1 h)]

theorem lintegral_ite_read (hmeas : ∀ w, Measurable (read w)) (w : FreeMonoid α) (v : ARU)
    (c : ENNReal) : ∫⁻ ω, (if read w ω = v then c else 0) ∂μ = c * μ {ω | read w ω = v} := by
  have : (fun ω => if read w ω = v then c else 0) = {ω | read w ω = v}.indicator fun _ => c := by
    ext ω; simp [Set.indicator_apply]
  have hS : MeasurableSet {ω | read w ω = v} := hmeas w (measurableSet_singleton v)
  rw [this, lintegral_indicator_const hS]

variable {read}

theorem measure_read (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r)
    (w : FreeMonoid α) (r : ARU) :
    μ {ω | read w ω = r} = ENNReal.ofReal (G.dist (G.M.eval w.toList) r) := by
  rw [← hlaw, measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]

theorem fsum_one {β : Type*} (rd : FreeMonoid α → ARU) (t : PQ α β) :
    QTree.fsum (fun _ _ _ => 1) rd ∅ t = (t.firsts rd ∅).length := by
  simp [QTree.fsum]

/-- Over independent reads with laws `G.dist`, a query tree's first reads on their state's rarer
side average at most `κ` times its undecided first reads plus `ε` times its first reads. -/
theorem minority_mean {β : Type*} (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (t : PQ α β) :
    ∫⁻ ω, QTree.fsum (fun _ w v => if v = G.minVal (G.M.eval w.toList) then 1 else 0)
        (fun y => read y ω) ∅ t ∂μ
      ≤ ∫⁻ ω, QTree.fsum (fun _ _ v => (if v = .undecided then ENNReal.ofReal G.κ else 0)
        + ENNReal.ofReal G.ε) (fun y => read y ω) ∅ t ∂μ := by
  refine QTree.firsts_lintegral_le _ _ t read ∅ hmeas hind fun ℓ w _ => ?_
  have hm : Measurable fun ω => if read w ω = .undecided then ENNReal.ofReal G.κ else 0 :=
    Measurable.ite (hmeas w (measurableSet_singleton _)) measurable_const measurable_const
  rw [lintegral_ite_read read hmeas, lintegral_add_left hm, lintegral_ite_read read hmeas,
    lintegral_const, measure_univ, mul_one, one_mul, measure_read G hlaw, measure_read G hlaw,
    dist_minVal, ← ENNReal.ofReal_mul hκ]
  refine (ENNReal.ofReal_le_ofReal (G.rare _)).trans ?_
  rcases le_total G.ε (G.κ * G.dist (G.M.eval w.toList) .undecided) with h | h
  · rw [max_eq_right h]; exact le_add_right le_rfl
  · rw [max_eq_left h]; exact le_add_left le_rfl

/-- The coupled `spurious` field in mean, at one probe: its chance of a record that is not true
is at most `κ` times its mean undecided strings plus `ε` times its mean first reads. -/
theorem spurious_mean (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    (he : EdgesInto T edges) (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ) :
    ∫⁻ ω, (G.untrueAt (fun y => read y ω) k T edges).indicator 1 x ∂μ
      ≤ ENNReal.ofReal G.κ * ∫⁻ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ENNReal) ∂μ
        + ENNReal.ofReal G.ε
          * ∫⁻ ω, (((recordQ T edges k x).firsts (fun y => read y ω) ∅).length : ENNReal) ∂μ := by
  set t := recordQ T edges k x
  have hfs : ∀ rd : FreeMonoid α → ARU,
      QTree.fsum (fun _ _ v => (if v = .undecided then ENNReal.ofReal G.κ else 0)
        + ENNReal.ofReal G.ε) rd ∅ t
      = ENNReal.ofReal G.κ * ((t.firsts rd ∅).filter fun q => (rd q.2).cut = none).length
        + ENNReal.ofReal G.ε * (t.firsts rd ∅).length := by
    intro rd
    unfold QTree.fsum
    induction t.firsts rd ∅ with
    | nil => simp
    | cons q l ih =>
      simp only [List.map_cons, List.sum_cons, ih, List.filter_cons, List.length_cons]
      rcases h : rd q.2 <;> simp [ARU.cut, h] <;> push_cast <;> ring
  calc ∫⁻ ω, (G.untrueAt (fun y => read y ω) k T edges).indicator 1 x ∂μ
      ≤ ∫⁻ ω, QTree.fsum (fun _ w v => if v = G.minVal (G.M.eval w.toList) then 1 else 0)
          (fun y => read y ω) ∅ t ∂μ := by
        refine lintegral_mono fun ω => ?_
        by_cases hx : x ∈ G.untrueAt (fun y => read y ω) k T edges
        · rw [Set.indicator_of_mem hx, Pi.one_apply]
          obtain ⟨q, hq, hc⟩ := untrue_minority T edges k x _ G hx
          have hk : q.2 ∈ ((t.firsts (fun y => read y ω) ∅).map Prod.snd).toFinset := by
            rw [(QTree.firsts_keys_eq _ t ∅).2, Finset.sdiff_empty, List.mem_toFinset]
            exact List.mem_map_of_mem hq
          obtain ⟨q', hq', heq⟩ := List.mem_map.1 (List.mem_toFinset.1 hk)
          unfold QTree.fsum
          refine le_trans ?_ (List.single_le_sum (fun _ _ => zero_le) _
            (List.mem_map_of_mem hq'))
          have hv := (cut_minVal G _ _).1 hc
          simp only [heq, hv, if_true, le_refl]
        · rw [Set.indicator_of_notMem hx]; exact zero_le
    _ ≤ _ := minority_mean G hmeas hind hlaw hκ t
    _ ≤ ∫⁻ ω, ENNReal.ofReal G.κ * (twinsBy (fun z => (read z ω).cut) T edges k x : ENNReal)
          + ENNReal.ofReal G.ε * ((t.firsts (fun y => read y ω) ∅).length : ENNReal) ∂μ := by
        refine lintegral_mono fun ω => ?_
        rw [hfs]
        gcongr
        exact_mod_cast und_twins T edges k x (fun y => read y ω) he
    _ = _ := by
        have hm : Measurable fun ω =>
            ENNReal.ofReal G.ε * ((t.firsts (fun y => read y ω) ∅).length : ENNReal) := by
          refine Measurable.const_mul ?_ _
          simp_rw [← fsum_one]
          exact QTree.fsum_measurable _ read hmeas ∅ t
        rw [lintegral_add_right _ hm, lintegral_const_mul' _ _ ENNReal.ofReal_ne_top,
          lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]

/-- The per-read edge field in mean, each string counted at its first read in the probe: a query
tree's undecided first reads labelled `e` at read-states where undecided reads are at most `θ'`
average at most `θ'` times its first reads labelled `e`. -/
theorem edge_first_mean {β : Type*} (hmeas : ∀ w, Measurable (read w))
    (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r)
    (P : σ → Prop) [DecidablePred P] (θ' : ℝ) (hP : ∀ r, P r → G.dist r .undecided ≤ θ')
    (e : List Bool × α) (t : PQ α β) :
    ∫⁻ ω, (((t.firsts (fun y => read y ω) ∅).filter fun q =>
        q.1.2 = some e ∧ read q.2 ω = .undecided ∧ P (G.M.eval q.2.toList)).length : ENNReal) ∂μ
      ≤ ENNReal.ofReal θ' * ∫⁻ ω,
        (((t.firsts (fun y => read y ω) ∅).filter fun q => q.1.2 = some e).length : ENNReal) ∂μ := by
  have hcount : ∀ (rd : FreeMonoid α → ARU) (f : PLabel α → FreeMonoid α → ARU → Prop)
      [∀ ℓ w v, Decidable (f ℓ w v)] (c : ENNReal),
      QTree.fsum (fun ℓ w v => if f ℓ w v then c else 0) rd ∅ t
        = c * ((t.firsts rd ∅).filter fun q => f q.1 q.2 (rd q.2)).length := by
    intro rd f _ c
    unfold QTree.fsum
    induction t.firsts rd ∅ with
    | nil => simp
    | cons q l ih =>
      simp only [List.map_cons, List.sum_cons, ih, List.filter_cons]
      by_cases h : f q.1 q.2 (rd q.2)
      · simp [h, mul_add, add_comm]
      · simp [h]
  have h1 := QTree.firsts_lintegral_le (μ := μ)
    (fun ℓ w v => if ℓ.2 = some e ∧ v = .undecided ∧ P (G.M.eval w.toList) then 1 else 0)
    (fun ℓ _ _ => if ℓ.2 = some e then ENNReal.ofReal θ' else 0) t read ∅ hmeas hind
    (fun ℓ w _ => by
      by_cases hl : ℓ.2 = some e
      · by_cases hp : P (G.M.eval w.toList)
        · simp only [hl, hp, true_and, and_true, if_true]
          rw [lintegral_ite_read read hmeas, one_mul, measure_read G hlaw, lintegral_const,
            measure_univ, mul_one]
          exact ENNReal.ofReal_le_ofReal (hP _ hp)
        · simp [hp]
      · simp [hl])
  simp only [hcount, one_mul] at h1
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  convert h1 using 3 <;> simp

/-- Counted with multiplicity, a query tree's undecided reads at read-states where undecided
reads are at most `θ'` average at most `M θ'` times its reads, when no run reads a string more
than `M` times. -/
theorem und_asks_mean {β : Type*} (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r)
    (P : σ → Prop) [DecidablePred P] (θ' : ℝ) (hP : ∀ r, P r → G.dist r .undecided ≤ θ')
    (t : PQ α β) (M : ℕ)
    (hM : ∀ ω w, ((t.asks fun y => read y ω).map Prod.snd).count w ≤ M) :
    ∫⁻ ω, (((t.asks fun y => read y ω).filter fun q =>
        read q.2 ω = .undecided ∧ P (G.M.eval q.2.toList)).length : ENNReal) ∂μ
      ≤ M * ENNReal.ofReal θ' * ∫⁻ ω, ((t.asks fun y => read y ω).length : ENNReal) ∂μ := by
  set a : FreeMonoid α → ARU → ENNReal :=
    fun w v => if v = .undecided ∧ P (G.M.eval w.toList) then 1 else 0
  have hsum : ∀ (l : List (PLabel α × FreeMonoid α)) (rd : FreeMonoid α → ARU),
      (l.map fun q => a q.2 (rd q.2)).sum
        = ((l.filter fun q => rd q.2 = .undecided ∧ P (G.M.eval q.2.toList)).length : ENNReal) := by
    intro l rd
    induction l with
    | nil => simp
    | cons q l ih =>
      simp only [List.map_cons, List.sum_cons, ih, List.filter_cons, a]
      by_cases h : rd q.2 = .undecided ∧ P (G.M.eval q.2.toList)
      · simp [h, add_comm]
      · simp [h]
  have hconst : ∀ (l : List (PLabel α × FreeMonoid α)) (rd : FreeMonoid α → ARU),
      (l.map fun q => (fun (_ : FreeMonoid α) (_ : ARU) => ENNReal.ofReal θ') q.2 (rd q.2)).sum
        = ENNReal.ofReal θ' * l.length := by
    intro l rd; simp [mul_comm]
  have h1 := QTree.firsts_lintegral_le (μ := μ) (fun _ w v => a w v)
    (fun _ _ _ => ENNReal.ofReal θ') t read ∅ hmeas hind (fun _ w _ => by
      by_cases hp : P (G.M.eval w.toList)
      · have : (fun ω => a w (read w ω)) = fun ω => if read w ω = .undecided then 1 else 0 := by
          ext ω; simp [a, hp]
        rw [this, lintegral_ite_read read hmeas, one_mul, measure_read G hlaw, lintegral_const,
          measure_univ, mul_one]
        exact ENNReal.ofReal_le_ofReal (hP _ hp)
      · simp [a, hp])
  unfold QTree.fsum at h1
  calc ∫⁻ ω, (((t.asks fun y => read y ω).filter fun q =>
          read q.2 ω = .undecided ∧ P (G.M.eval q.2.toList)).length : ENNReal) ∂μ
      ≤ ∫⁻ ω, M * ((t.firsts (fun y => read y ω) ∅).map fun q => a q.2 (read q.2 ω)).sum ∂μ := by
        refine lintegral_mono fun ω => ?_
        rw [← hsum _ (fun y => read y ω)]
        exact QTree.asks_sum_le a _ t M (hM ω)
    _ = M * ∫⁻ ω, ((t.firsts (fun y => read y ω) ∅).map fun q => a q.2 (read q.2 ω)).sum ∂μ :=
        lintegral_const_mul' _ _ (ENNReal.natCast_ne_top M)
    _ ≤ M * ∫⁻ ω, ENNReal.ofReal θ' * (t.firsts (fun y => read y ω) ∅).length ∂μ := by
        refine mul_le_mul_of_nonneg_left (h1.trans_eq (lintegral_congr fun ω => ?_)) zero_le
        simp [mul_comm]
    _ ≤ M * ∫⁻ ω, ENNReal.ofReal θ' * (t.asks fun y => read y ω).length ∂μ := by
        gcongr with ω
        have := QTree.firsts_sum_le (fun _ _ => (1 : ENNReal)) (fun y => read y ω) t
        simpa using this
    _ = _ := by rw [lintegral_const_mul' _ _ ENNReal.ofReal_ne_top, mul_assoc]

end Mean

end OrthoDFA
