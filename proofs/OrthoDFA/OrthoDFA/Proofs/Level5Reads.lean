import OrthoDFA.Proofs.Level5Clean
import OrthoDFA.Proofs.AdaptiveReads

/-!
# Level 5's reads on the rarer side

Reads agreeing on what a step reads, its probe's prefixes followed by midfixes and what its split
re-sifts, give the same step (`applyProbe_congr`). Given the probes, the round's reads come in
rounds (`rounds`): round `2j` is what the `j`-th probe's step reads of its prefixes, round
`2j + 1` what its split re-sifts. While nothing read so far is on the rarer side, the hypothesis
stays true, so each round is fixed by what the earlier ones read and has at most
`(L + 1) maxLeaves`, or `maxLeaves² |Σ|`, strings, and `AdaptiveReadsBound` bounds the chance
some step reads a string on the rarer side. Over the probes, a sum over their countably many
values, the bound stays (`pi_wrong_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## Reads that agree on what a step reads -/

namespace QTree

variable {Λ ι V β : Type*} [DecidableEq ι]

theorem asks_congr {rd rd' : ι → V} :
    ∀ (t : QTree Λ ι V β) (P : Finset ι), (∀ q ∈ t.asks rd, rd q.2 = rd' q.2) →
      t.firsts rd P = t.firsts rd' P
  | done _, _, _ => rfl
  | ask ℓ w next, P, h => by
    have hw : rd w = rd' w := h (ℓ, w) List.mem_cons_self
    have hn : ∀ q ∈ (next (rd w)).asks rd, rd q.2 = rd' q.2 := fun q hq =>
      h q (List.mem_cons_of_mem _ hq)
    simp only [firsts]
    rw [← hw, asks_congr (next (rd w)) P hn, asks_congr (next (rd w)) (insert w P) hn]

end QTree

theorem route_congr {cut cut' : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α), (∀ m ∈ T.midfixes, cut (u * m) = cut' (u * m)) →
      T.route cut u = T.route cut' u
  | .leaf, _, _ => rfl
  | .node m r a, u, h => by
    have hm := h m (by simp [DTree.midfixes])
    have hr := route_congr r u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    have ha := route_congr a u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    simp only [DTree.route, hm, hr, ha]

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_min (x : FreeMonoid α) (i : ℕ) :
    prefixOf x i = prefixOf x (min i x.toList.length) := by
  simp only [prefixOf]
  rcases le_total i x.toList.length with hl | hl
  · rw [min_eq_left hl]
  · rw [min_eq_right hl, List.take_of_length_le hl, List.take_length]

section Congr

variable {cut cut' : FreeMonoid α → Option Bool} {T : DTree α} {x : FreeMonoid α}

theorem siftCut_prefix_congr (h : ∀ z ∈ potMid x T, cut z = cut' z) (i : ℕ) :
    T.siftCut cut (prefixOf x i) = T.siftCut cut' (prefixOf x i) := by
  rw [prefixOf_min x i]
  unfold DTree.siftCut
  rw [route_congr T _ fun m hm => h _ (mem_potMid (min_le_right _ _) hm)]

theorem asks_potMid {E : EdgeTable α} {k : ℕ} {rd : FreeMonoid α → ARU}
    {q : ReadLabel α × FreeMonoid α} (hq : q ∈ (probeQuery T E k x).asks rd) :
    q.2 ∈ potMid x T := by
  obtain ⟨h1, -⟩ := asks_probeQuery T E k x rd q hq
  obtain ⟨m, hm, he⟩ := route_mid _ T _ q.2 h1
  rw [he, prefixOf_min x q.1.1]
  exact mem_potMid (min_le_right _ _) hm

theorem ofCut_congr (h : ∀ z ∈ potMid x T, cut z = cut' z) {z : FreeMonoid α}
    (hz : z ∈ potMid x T) : ofCut cut z = ofCut cut' z := by
  simp only [ofCut, h z hz]

theorem firstsBy_congr {E : EdgeTable α} {k : ℕ} (h : ∀ z ∈ potMid x T, cut z = cut' z) :
    firstsBy cut T E k x = firstsBy cut' T E k x :=
  QTree.asks_congr _ _ fun _ hq => ofCut_congr h (asks_potMid hq)

theorem edgeHarvestBy_congr {E : EdgeTable α} {k : ℕ} (h : ∀ z ∈ potMid x T, cut z = cut' z)
    (e : List Bool × α) : edgeHarvestBy cut T E k x e = edgeHarvestBy cut' T E k x e := by
  unfold edgeHarvestBy
  rw [firstsBy_congr h]
  congr 1
  refine List.filter_congr fun q hq => ?_
  rw [h q.2 (asks_potMid (QTree.firsts_sub_asks _ _ _ q hq))]

theorem prefAgreeK_of (h : ∀ z ∈ potMid x T, cut z = cut' z) (k : ℕ) : PrefAgreeK cut cut' T x k :=
  fun j _ => siftCut_prefix_congr h j

variable (C : Level5Settings)

theorem countProbe_congr {s : Level5State α} (h : ∀ z ∈ potMid x s.tree, cut z = cut' z) :
    s.countProbe cut C.walkStart x = s.countProbe cut' C.walkStart x := by
  have hs := siftCut_prefix_congr h
  simp only [Level5State.countProbe, startHarvestBy, edgeReadsBy, firstsBy_congr h,
    edgeHarvestBy_congr h, middleHarvestBy_congrK (edges := s.edges) (prefAgreeK_of h C.walkStart),
    hs]

theorem recordProbe_congr {s : Level5State α} (h : ∀ z ∈ potMid x s.tree, cut z = cut' z) :
    recordProbe C cut s x = recordProbe C cut' s x := by
  have hs := siftCut_prefix_congr h
  have hP := probeBy_congrK (edges := s.edges) (prefAgreeK_of h C.walkStart)
  unfold recordProbe
  rw [countProbe_congr C h, hP]
  rcases hpo : probeBy cut' s.tree s.edges C.walkStart x with _ | _ | _ | _ | _ | u <;>
    simp only
  obtain ⟨j, -, hjx, rfl⟩ := probeBy_member cut' hpo
  have hlen : (prefixOf x j).toList.length = j := by simp [prefixOf]; omega
  rw [hlen, hs j]
  rcases hc : x.toList[j]? with _ | c
  · rfl
  rcases hs' : DTree.siftCut cut' s.tree (prefixOf x j) with p | b
  · simp only
    rw [← prefixOf_succ hc, hs (j + 1)]
  · rfl

theorem recordBy_congr {s : Level5State α} (h : ∀ z ∈ potMid x s.tree, cut z = cut' z) :
    recordBy cut C.walkStart (s.tree, s.edges) x
      = recordBy cut' C.walkStart (s.tree, s.edges) x := by
  have hs := siftCut_prefix_congr h
  simp only [recordBy, probeBy_congrK (edges := s.edges) (prefAgreeK_of h C.walkStart), hs]

theorem splitReads_congr {s : Level5State α} (h : ∀ z ∈ potMid x s.tree, cut z = cut' z) :
    splitReads C cut s x = splitReads C cut' s x := by
  simp only [splitReads, recordBy_congr C h]

/-- Reads agreeing on what a step reads give the same step, its learned edges leaving leaves. -/
theorem applyProbe_congr {s : Level5State α}
    (hlv : ∀ q e t w, s.edges q e = some (t, w) → q ∈ s.tree.leaves)
    (h : ∀ z ∈ potMid x s.tree ∪ splitReads C cut s x, cut z = cut' z) :
    applyProbe C cut s x = applyProbe C cut' s x := by
  have hpot : ∀ z ∈ potMid x s.tree, cut z = cut' z := fun z hz => h z (Finset.mem_union_left _ hz)
  have hrb := recordBy_congr C hpot
  unfold applyProbe
  rw [recordProbe_congr C hpot, ← hrb]
  rcases hr : recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩
  · rfl
  simp only
  have hr' := hr
  rw [hrb] at hr'
  rw [recordProbe_of_record C cut' s x hr']
  have hss : splitState cut
        { s.countProbe cut' C.walkStart x with searches := s.searches + 1 } p c t
      = splitState cut'
        { s.countProbe cut' C.walkStart x with searches := s.searches + 1 } p c t := by
    unfold splitState
    simp only [Level5State.countProbe]
    rcases hpe : s.edges p c with _ | ⟨t₀, w₀⟩
    · rfl
    simp only
    congr 1
    funext q e
    split_ifs
    · rfl
    unfold retargetBy
    rcases hqe : s.edges q e with _ | ⟨s', w⟩
    · rfl
    simp only
    split_ifs
    · unfold DTree.siftCut
      rw [route_congr _ _ fun m hm => h _ (Finset.mem_union_right _ ?_)]
      unfold splitReads
      rw [hr]
      simp only [hpe]
      refine Finset.mem_biUnion.2 ⟨(q, e), by simp [hlv q e s' w hqe], ?_⟩
      simp only [hqe]
      exact Finset.mem_image.2 ⟨m, hm, rfl⟩
    · rfl
  rw [hss]

end Congr

/-! ## The round's reads, in rounds -/

section Rounds

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)

/-- The state before the `j`-th probe of `xs`, from `s`, while the round runs. -/
noncomputable def stateAt (read : FreeMonoid α → ARU) :
    Level5State α → List (FreeMonoid α) → ℕ → Option (Level5State α)
  | s, _, 0 => some s
  | _, [], _ + 1 => none
  | s, x :: xs, j + 1 =>
    match applyProbe C (fun z => (read z).cut) s x with
    | .inl s' => stateAt read s' xs j
    | .inr _ => none

omit [Fintype σ] in
theorem stateAt_succ (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s : Level5State α) (j : ℕ),
      stateAt C read s xs (j + 1) = (stateAt C read s xs j).bind fun s' =>
        xs[j]?.bind fun x => (applyProbe C (fun z => (read z).cut) s' x).elim some fun _ => none
  | [], s, j => by cases j <;> simp [stateAt]
  | x :: xs, s, 0 => by
    simp only [stateAt, Option.bind_some, List.getElem?_cons_zero]
    rcases applyProbe C (fun z => (read z).cut) s x with s' | e <;> rfl
  | x :: xs, s, j + 1 => by
    simp only [stateAt, List.getElem?_cons_succ]
    rcases applyProbe C (fun z => (read z).cut) s x with s' | e
    · exact stateAt_succ read xs s' j
    · rfl

/-- Round `2j` reads the `j`-th probe's prefixes followed by midfixes, round `2j + 1` what its
split re-sifts. -/
noncomputable def rounds (xs : List (FreeMonoid α)) (i : ℕ) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  match stateAt C read initialState xs (i / 2), xs[i / 2]? with
  | some s, some x =>
    if i % 2 = 0 then potMid x s.tree else splitReads C (fun z => (read z).cut) s x
  | _, _ => ∅

omit [Fintype σ] in
theorem rounds_even (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level5State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) : rounds C xs (2 * j) read = potMid x s.tree := by
  have h1 : 2 * j / 2 = j := by omega
  simp only [rounds, h1, hs, hx]
  simp

omit [Fintype σ] in
theorem rounds_odd (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level5State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) :
    rounds C xs (2 * j + 1) read = splitReads C (fun z => (read z).cut) s x := by
  have h1 : (2 * j + 1) / 2 = j := by omega
  simp only [rounds, h1, hs, hx]
  simp

/-- No string the rounds before `n` read is read on its state's rarer side. -/
def Good (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) (n : ℕ) : Prop :=
  ∀ i < n, ∀ z ∈ rounds C xs i read, ¬ G.WrongRead read z

variable (hσ : Fintype.card σ + 2 ≤ C.maxLeaves)
include hσ

/-- While nothing read is on the rarer side, the hypothesis stays true. -/
theorem inv_of_good (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) :
    ∀ (j : ℕ) (s : Level5State α), stateAt C read initialState xs j = some s →
      Good G C xs read (2 * j) → TruthInv G C.maxLeaves s
  | 0, s, h, _ => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    exact truthInv_start G (by omega)
  | j + 1, s, h, hg => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C read initialState xs j with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[j]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe C (fun z => (read z).cut) s₁ x with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    have hI := inv_of_good xs read j s₁ hs₁ fun i hi => hg i (by omega)
    have hx' : ¬ WrongStep G read C s₁ x := by
      rintro ⟨z, hz, hw⟩
      rcases Finset.mem_union.1 hz with hz | hz
      · exact hg (2 * j) (by omega) z (by rw [rounds_even C xs read hs₁ hx]; exact hz) hw
      · exact hg (2 * j + 1) (by omega) z (by rw [rounds_odd C xs read hs₁ hx]; exact hz) hw
    exact (truthInv_step G read C hσ hI hx').1 _ ha

/-- Reads agreeing with good ones on what the rounds before `2j` read reach the same state. -/
theorem stateAt_congr (xs : List (FreeMonoid α)) {r r' : FreeMonoid α → ARU} :
    ∀ j, Good G C xs r (2 * j) → (∀ i < 2 * j, ∀ z ∈ rounds C xs i r, r z = r' z) →
      stateAt C r initialState xs j = stateAt C r' initialState xs j
  | 0, _, _ => by simp only [stateAt]
  | j + 1, hg, ha => by
    rw [stateAt_succ, stateAt_succ, ← stateAt_congr xs j (fun i hi => hg i (by omega))
      fun i hi => ha i (by omega)]
    rcases hs₁ : stateAt C r initialState xs j with _ | s₁
    · rfl
    rcases hx : xs[j]? with _ | x
    · rfl
    simp only [Option.bind_some]
    obtain ⟨-, -, ht, -⟩ := inv_of_good G C hσ xs r j s₁ hs₁ fun i hi => hg i (by omega)
    rw [applyProbe_congr C (fun q e t w h => edge_leaf G ht h) fun z hz => ?_]
    have hz' : (fun z => (r z).cut) z = (fun z => (r' z).cut) z := by
      rcases Finset.mem_union.1 hz with hz | hz
      · simp only [ha (2 * j) (by omega) z (by rw [rounds_even C xs r hs₁ hx]; exact hz)]
      · simp only [ha (2 * j + 1) (by omega) z (by rw [rounds_odd C xs r hs₁ hx]; exact hz)]
    exact hz'

/-- While nothing read is on the rarer side, each round is fixed by what the earlier ones
read. -/
theorem rounds_local (xs : List (FreeMonoid α)) (i : ℕ) (r r' : FreeMonoid α → ARU)
    (hg : Good G C xs r i) (ha : ∀ l < i, ∀ z ∈ rounds C xs l r, r z = r' z) :
    rounds C xs i r = rounds C xs i r' := by
  have hj : 2 * (i / 2) ≤ i := Nat.mul_div_le i 2
  have hst := stateAt_congr G C hσ xs (i / 2) (fun l hl => hg l (by omega))
    fun l hl => ha l (by omega)
  unfold rounds
  rw [← hst]
  rcases hs : stateAt C r initialState xs (i / 2) with _ | s
  · rfl
  rcases hx : xs[i / 2]? with _ | x
  · rfl
  simp only
  split_ifs with hi
  · rfl
  refine splitReads_congr C fun z hz => ?_
  simp only [ha (2 * (i / 2)) (by omega) z (by rw [rounds_even C xs r hs hx]; exact hz)]

omit hσ in
omit [Fintype σ] in
theorem resifts_card (s : Level5State α) (T' : DTree α) :
    (resifts s T').card ≤ s.tree.leaves.length * Fintype.card α * T'.midfixes.card := by
  classical
  unfold resifts
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α), T'.midfixes.card :=
        Finset.sum_le_sum fun e _ => by
          split
          · exact Finset.card_image_le
          · simp
    _ ≤ _ := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_product, Finset.card_univ]
        gcongr
        exact List.toFinset_card_le _

/-- While nothing read is on the rarer side, a probe's round has at most `(L + 1) maxLeaves`
strings and a split's `maxLeaves² |Σ|`. -/
theorem rounds_card (xs : List (FreeMonoid α)) (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L)
    (i : ℕ) (r : FreeMonoid α → ARU) (hg : Good G C xs r i) :
    (rounds C xs i r).card ≤ if i % 2 = 0 then (L + 1) * C.maxLeaves
      else C.maxLeaves * Fintype.card α * C.maxLeaves := by
  have hj : 2 * (i / 2) ≤ i := Nat.mul_div_le i 2
  unfold rounds
  rcases hs : stateAt C r initialState xs (i / 2) with _ | s
  · simp
  rcases hx : xs[i / 2]? with _ | x
  · simp
  simp only
  obtain ⟨-, -, -, hlv⟩ := inv_of_good G C hσ xs r (i / 2) s hs fun l hl => hg l (by omega)
  have hmid : s.tree.midfixes.card ≤ C.maxLeaves := by
    have := midfixes_card s.tree; omega
  have hxL : x.toList.length ≤ L := hL x (List.mem_of_getElem? hx)
  split_ifs with hi
  · calc (potMid x s.tree).card ≤ (x.toList.length + 1) * s.tree.midfixes.card := by
          unfold potMid
          refine Finset.card_biUnion_le.trans ?_
          calc _ ≤ ∑ _i ∈ Finset.range (x.toList.length + 1), s.tree.midfixes.card :=
                Finset.sum_le_sum fun _ _ => Finset.card_image_le
            _ = _ := by simp
      _ ≤ _ := Nat.mul_le_mul (by omega) hmid
  · unfold splitReads
    rcases recordBy (fun z => (r z).cut) C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩
    · simp
    simp only
    rcases s.edges p c with _ | ⟨t₀, w₀⟩
    · simp
    simp only
    refine (resifts_card s _).trans ?_
    have hm := Finset.card_le_card
      (midfixes_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p)
    have := Finset.card_insert_le (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀))
      s.tree.midfixes
    have := midfixes_card s.tree
    gcongr
    omega

end Rounds

section Wrong

variable {S X E : Type*}

/-- Some step of the run, before it ends, is `W`. -/
def AnyStep (step : S → X → S ⊕ E) (W : S → X → Prop) : S → List X → Prop
  | _, [] => False
  | s, x :: xs => W s x ∨ ∃ s', step s x = .inl s' ∧ AnyStep step W s' xs

end Wrong

section Adaptive

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)

omit [Fintype σ] in
theorem anyStep_rounds (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level5State α),
      AnyStep (applyProbe C fun z => (read z).cut) (WrongStep G read C) s₀ xs →
      ∃ j < xs.length, ∃ s x, stateAt C read s₀ xs j = some s ∧ xs[j]? = some x
        ∧ WrongStep G read C s x
  | [], _, h => h.elim
  | x :: xs, s₀, h => by
    rcases h with h | ⟨s', hs', h⟩
    · exact ⟨0, by simp, s₀, x, rfl, rfl, h⟩
    · obtain ⟨j, hj, s, y, h1, h2, h3⟩ := anyStep_rounds read xs s' h
      refine ⟨j + 1, by simp; omega, s, y, ?_, by simpa using h2, h3⟩
      simp only [stateAt, hs']
      exact h1

theorem sum_alt (n a b : ℕ) :
    ∑ i ∈ Finset.range (2 * n), ((if i % 2 = 0 then a else b : ℕ) : ℝ) = n * (a + b) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      ih]
    have h1 : (2 * n + 1) % 2 = 1 := by omega
    have h0 : 2 * n % 2 = 0 := by omega
    simp only [h0, h1, if_true, one_ne_zero, if_false]
    push_cast
    ring


instance : MeasurableSingletonClass ARU := ⟨fun _ => trivial⟩

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

/-- Given probes no longer than `L`, some step reads a string on its state's rarer side with
chance at most `(ε + κ) |xs| ((L + 1) maxLeaves + maxLeaves² |Σ|)`. -/
theorem wrong_le (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (hind : iIndepFun read μ) (hκ : 0 ≤ G.κ) (hε : 0 ≤ G.ε) (xs : List (FreeMonoid α)) (L : ℕ)
    (hL : ∀ x ∈ xs, x.toList.length ≤ L) :
    μ {ω | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C)
        initialState xs}
      ≤ ENNReal.ofReal ((G.ε + G.κ) * (xs.length
        * ((L + 1) * C.maxLeaves + C.maxLeaves * Fintype.card α * C.maxLeaves))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {G.minVal (G.M.eval z.toList)}
  have hWB : ∀ (rd : FreeMonoid α → ARU) z, G.WrongRead rd z ↔ rd z ∈ B z := fun rd z =>
    cut_minVal G _ _
  have hB : ∀ z, μ.real {ω | read z ω ∈ B z} ≤ G.ε + G.κ := fun z => by
    have h := wrongRead_le G hmeas hlaw hκ hε z
    have he : {ω | G.WrongRead (read · ω) z} = {ω | read z ω ∈ B z} := by
      ext ω; simp only [Set.mem_ofPred_eq]; exact hWB _ z
    rw [he] at h
    exact ENNReal.toReal_le_of_le_ofReal (by linarith) h
  have hmain := adaptiveReadsBound_holds μ read B (G.ε + G.κ) (fun i r => rounds C xs i r)
    (fun i => if i % 2 = 0 then (L + 1) * C.maxLeaves
      else C.maxLeaves * Fintype.card α * C.maxLeaves) (2 * xs.length)
    hmeas hind hB
    (fun i r r' hg ha => rounds_local G C hσ xs i r r'
      (fun l hl z hz hw => hg l hl z hz ((hWB r z).1 hw)) ha)
    (fun i r hg => rounds_card G C hσ xs L hL i r
      fun l hl z hz hw => hg l hl z hz ((hWB r z).1 hw))
  rw [sum_alt] at hmain
  refine (measure_mono fun ω hω => ?_).trans (hmain.trans (le_of_eq ?_))
  · obtain ⟨j, hj, s, x, h1, h2, z, hz, hw⟩ := anyStep_rounds G C (read · ω) xs _ hω
    rcases Finset.mem_union.1 hz with hz | hz
    · exact ⟨2 * j, by omega, z, by rw [rounds_even C xs _ h1 h2]; exact hz, (hWB _ z).1 hw⟩
    · exact ⟨2 * j + 1, by omega, z, by rw [rounds_odd C xs _ h1 h2]; exact hz, (hWB _ z).1 hw⟩
  · push_cast
    ring_nf

/-- A measure on a countable type is the sum of its points'. -/
theorem measure_eq_tsum_points {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (ν : Measure X) (A : Set X) :
    ν A = ∑' d, ν {d} * A.indicator 1 d := by
  have hA : MeasurableSet A := (Set.to_countable A).measurableSet
  conv_lhs => rw [← Measure.sum_smul_dirac ν]
  rw [Measure.sum_apply _ hA]
  congr 1
  funext d
  rw [Measure.smul_apply, Measure.dirac_apply' _ hA, smul_eq_mul]

omit [IsProbabilityMeasure μ] in
/-- Over fresh draws `d`, each drawn point bounding the chance of `{ω | d ∈ S ω}` by `c`, the mean
mass of `S` is at most `c`. -/
theorem lintegral_le_of_points {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (ν : Measure X) [IsProbabilityMeasure ν] (S : Ω → Set X)
    (c : ENNReal) (h : ∀ d, ν {d} ≠ 0 → μ {ω | d ∈ S ω} ≤ c) :
    ∫⁻ ω, ν (S ω) ∂μ ≤ c := by
  set E : X → Set Ω := fun d => toMeasurable μ {ω | d ∈ S ω}
  have hpt : ∀ ω, ν (S ω) ≤ ∑' d, ν {d} * (E d).indicator 1 ω := fun ω => by
    rw [measure_eq_tsum_points ν]
    refine ENNReal.tsum_le_tsum fun d => ?_
    gcongr
    by_cases hd : d ∈ S ω
    · rw [Set.indicator_of_mem hd,
        Set.indicator_of_mem (show ω ∈ E d from subset_toMeasurable _ _ hd)]
      simp
    · rw [Set.indicator_of_notMem hd]
      exact bot_le
  calc _ ≤ ∫⁻ ω, ∑' d, ν {d} * (E d).indicator 1 ω ∂μ := lintegral_mono hpt
    _ = ∑' d, ν {d} * μ (E d) := by
        rw [lintegral_tsum fun d => ((measurable_one.indicator
          (measurableSet_toMeasurable _ _)).const_mul _).aemeasurable]
        congr 1
        funext d
        rw [lintegral_const_mul _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ ∑' d, ν {d} * c := ENNReal.tsum_le_tsum fun d => by
        by_cases hd : ν {d} = 0
        · rw [hd, zero_mul, zero_mul]
        · rw [measure_toMeasurable]
          gcongr
          exact h d hd
    _ = c := by
        have h1 : ∑' d, ν {d} = 1 := by
          rw [← measure_univ (μ := ν), measure_eq_tsum_points ν Set.univ]
          simp
        rw [ENNReal.tsum_mul_right, h1, one_mul]

/-- Over the probes and the reads, some step reads a string on its state's rarer side with
chance at most `(ε + κ) probeBudget ((L + 1) maxLeaves + maxLeaves² |Σ|)`. -/
theorem pi_wrong_le (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r)
    (hind : iIndepFun read μ) (hκ : 0 ≤ G.κ) (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => D)
        {xs | AnyStep (applyProbe C fun z => (read z ω).cut) (WrongStep G (read · ω) C)
          initialState (List.ofFn xs)} ∂μ
      ≤ ENNReal.ofReal ((G.ε + G.κ) * (C.probeBudget
        * ((L + 1) * C.maxLeaves + C.maxLeaves * Fintype.card α * C.maxLeaves))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points _ _ _ fun d hd => ?_
  have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ L := by
    intro x hx
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    by_contra hlt
    apply hd
    have hpi : (Measure.pi fun _ : Fin C.probeBudget => D) {d} = ∏ j, D {d j} := by
      rw [← Measure.pi_pi]
      congr 1
      ext d'
      simp [funext_iff]
    rw [hpi]
    refine Finset.prod_eq_zero (Finset.mem_univ i) (measure_mono_null ?_ (ae_iff.1 hlen))
    intro x hx
    rw [Set.mem_singleton_iff] at hx
    subst hx
    exact hlt
  have := wrong_le G C hσ hmeas hlaw hind hκ hε (List.ofFn d) L hL
  rw [List.length_ofFn] at this
  exact this

end Adaptive

end OrthoDFA
