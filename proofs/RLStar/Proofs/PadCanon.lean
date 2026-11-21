import RLStar.Proofs.LoopRuns
import RLStar.Code.LazyRound
import RLStar.Proofs.TrieTable
import RLStar.Proofs.TrieAgree
import Mathlib.Data.List.ReduceOption

/-!
# Level 3 on padded probes reads only canonical strings

A split's new midfix is a probe's letter past its first `walkStart` letters, then a midfix. So
where every probe's letters past `walkStart` satisfy `P`, every midfix of every tree level 3 builds
has only letters satisfying `P` (`midsIn_runProbes`), and the run reads only a probe's prefix past
`walkStart` followed by such letters (`level3_congr_in`).

A probe padded from `X` has only letters of the alphabet past `K`. So, where `X`'s strings are
shorter than `K ≤ |x|`, every string level 3 reads on it is canonical: its pads dropped and padded
again, it is itself (`padIn_unpad_of_read`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type*}

/-! ## Padding -/

section Pad

theorem unpad_mul (a b : FreeMonoid (Option α)) : unpad (a * b) = unpad a * unpad b := by
  simp [unpad, List.reduceOption_append]

theorem unpad_prefixOf_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (p : ℕ) :
    unpad (prefixOf (padIn K X x) p) = prefixOf x (p - (K - walkStartIn X x)) := by
  simp only [unpad, prefixOf, padIn, FreeMonoid.toList_ofList, List.take_append,
    List.length_replicate, List.take_replicate, List.reduceOption_append,
    List.reduceOption_replicate_none, ← List.map_take]
  simp [List.reduceOption]

theorem length_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    (padIn K X x).toList.length = K - walkStartIn X x + x.toList.length := by
  simp [padIn]

/-- Where `X`'s strings are shorter than `K`, no walk start passes `K`. -/
theorem walkStartIn_le_of_short {K : ℕ} {X : Finset (FreeMonoid α)}
    (hX : ∀ c ∈ X, c.toList.length < K) (x : FreeMonoid α) : walkStartIn X x ≤ K := by
  obtain ⟨hle, hin, -⟩ := walkStartIn_spec X x
  by_contra h
  have := hX _ (hin K (by omega))
  rw [length_prefixOf'] at this
  omega

/-- Where `X`'s strings are shorter than `K ≤ |x|`, `x`'s walk start is outside `X`. -/
theorem prefixOf_walkStartIn_notMem {K : ℕ} {X : Finset (FreeMonoid α)}
    (hX : ∀ c ∈ X, c.toList.length < K) {x : FreeMonoid α} (hx : K ≤ x.toList.length) :
    prefixOf x (walkStartIn X x) ∉ X := by
  obtain ⟨hnx, -, hout⟩ := walkStartIn_spec X x
  rcases lt_or_eq_of_le hnx with h | h
  · exact hout h
  · intro hm
    have := hX _ hm
    rw [length_prefixOf'] at this
    omega

/-- A prefix of `x` past its walk start, extended, has `x`'s walk start and first prefix. -/
theorem walkStartIn_prefixOf_mul {K : ℕ} {X : Finset (FreeMonoid α)}
    (hX : ∀ c ∈ X, c.toList.length < K) {x : FreeMonoid α} (hx : K ≤ x.toList.length) {j : ℕ}
    (hj : walkStartIn X x ≤ j) (v : FreeMonoid α) :
    walkStartIn X (prefixOf x j * v) = walkStartIn X x
      ∧ prefixOf (prefixOf x j * v) (walkStartIn X x) = prefixOf x (walkStartIn X x) := by
  obtain ⟨hnx, hin, -⟩ := walkStartIn_spec X x
  have hnot := prefixOf_walkStartIn_notMem hX hx
  generalize walkStartIn X x = n at hnx hin hnot hj
  have hpre : ∀ i ≤ n, prefixOf (prefixOf x j * v) i = prefixOf x i := by
    intro i hi
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
    rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left (by omega)]
  refine ⟨walkStartIn_eq ?_ (fun i hi => ?_) (fun _ => ?_), hpre n le_rfl⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, length_prefixOf']
    omega
  · rw [hpre i hi.le]
    exact hin i hi
  · rw [hpre n le_rfl]
    exact hnot

/-- The probes whose first prefix outside `X` is `c`, where `c ∉ X` and `|c| ≤ K`, hold at most
`max θ p₁` of `D`. -/
theorem real_walkCell_le [Fintype α] (D : Measure (FreeMonoid α)) {K : ℕ}
    {X : Finset (FreeMonoid α)} {θ p₁ : ℝ}
    (hlight : ∀ c : FreeMonoid α, c.toList.length < K → c ∉ X →
      D.real {x | prefixOf x c.toList.length = c} ≤ θ)
    (hp₁ : ∀ u, D.real {x | prefixOf x K = u} ≤ p₁) (c : FreeMonoid α) :
    D.real {x | prefixOf x c.toList.length = c ∧ c ∉ X ∧ c.toList.length ≤ K} ≤ max θ p₁ := by
  by_cases hcS : c ∉ X ∧ c.toList.length ≤ K
  · have hSe : {x | prefixOf x c.toList.length = c ∧ c ∉ X ∧ c.toList.length ≤ K}
        = {x | prefixOf x c.toList.length = c} := by
      ext x
      simp only [Set.mem_ofPred_eq, and_iff_left hcS]
    rw [hSe]
    rcases lt_or_eq_of_le hcS.2 with hlt | heq
    · exact (hlight c hlt hcS.1).trans (le_max_left _ _)
    · rw [heq]
      exact (hp₁ c).trans (le_max_right _ _)
  · have hSe : {x | prefixOf x c.toList.length = c ∧ c ∉ X ∧ c.toList.length ≤ K} = ∅ := by
      ext x
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hcS ⟨h.2.1, h.2.2⟩
    rw [hSe, measureReal_empty]
    exact le_max_of_le_right (measureReal_nonneg.trans (hp₁ 1))

/-- No `K`-letter prefix of a probe padded from `X` holds more than `max θ p₁` of the padded
probes, where `X`'s strings are shorter than `K ≤ |x|`. -/
theorem real_padCell_le [Fintype α] (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] {K : ℕ}
    {X : Finset (FreeMonoid α)} (hX : ∀ c ∈ X, c.toList.length < K)
    (hlen : ∀ᵐ x ∂D, K ≤ x.toList.length) {θ p₁ : ℝ}
    (hlight : ∀ c : FreeMonoid α, c.toList.length < K → c ∉ X →
      D.real {x | prefixOf x c.toList.length = c} ≤ θ)
    (hp₁ : ∀ u, D.real {x | prefixOf x K = u} ≤ p₁) (u : FreeMonoid (Option α)) :
    (D.map (padIn K X)).real {z | prefixOf z K = u} ≤ max θ p₁ := by
  rw [map_measureReal_apply (measurable_of_countable _) (Set.to_countable _).measurableSet]
  refine le_trans ?_ (real_walkCell_le D hlight hp₁ (unpad u))
  refine ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae (hlen.mono fun x hx hxu => ?_))
  have hxu' : prefixOf (padIn K X x) K = u := hxu
  have hnK := walkStartIn_le_of_short hX x
  have hc : unpad u = prefixOf x (walkStartIn X x) := by
    rw [← hxu', unpad_prefixOf_padIn]
    congr 1
    omega
  have hlc : (unpad u).toList.length = walkStartIn X x := by
    rw [hc, length_prefixOf']
    exact min_eq_left (walkStartIn_le X x)
  refine ⟨by rw [hlc, hc], hc ▸ prefixOf_walkStartIn_notMem hX hx, ?_⟩
  rw [hlc]
  exact hnK

end Pad

/-! ## Letters of the midfixes -/

section Run

/-- Every letter of every midfix of `T` satisfies `P`. -/
def MidsIn (P : α → Prop) (T : DTree α) : Prop := ∀ m ∈ mids T, ∀ c ∈ m.toList, P c

/-- Every letter of `x` past its first `k` satisfies `P`. -/
def LettersFrom (P : α → Prop) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∀ i c, k ≤ i → x.toList[i]? = some c → P c

theorem midsIn_split {P : α → Prop} {T : DTree α} (hT : MidsIn P T) {c : α} (hc : P c)
    (q p : List Bool) : MidsIn P (T.splitAt (FreeMonoid.of c * T.midfixAt q) p) := by
  classical
  intro m hm a ha
  rcases Finset.mem_insert.1 (mids_splitAt _ T p hm) with rfl | hm
  · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.singleton_append,
      List.mem_cons] at ha
    rcases ha with rfl | ha
    · exact hc
    · rcases midfixAt_mem T q with h | h
      · rw [h] at ha
        simp at ha
      · exact hT _ h a ha
  · exact hT m hm a ha

theorem midsIn_initialState (P : α → Prop) : MidsIn P (initialState : Level3State α).tree := by
  classical
  intro m hm c hc
  simp only [initialState, newHypothesis, mids, Finset.mem_insert, Finset.notMem_empty,
    Finset.empty_union, or_false] at hm
  subst hm
  simp at hc

variable [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- A step on a probe whose letters past `walkStart` satisfy `P` keeps every midfix's letters
satisfying `P`. -/
theorem midsIn_step {P : α → Prop} {X : List (FreeMonoid α)} {s s' : Level3State α}
    {x : FreeMonoid α} (hxP : LettersFrom P C.walkStart x) (hs : RunInv read C X s)
    (hP : MidsIn P s.tree) (h : applyProbe read C s x = .inl s') : MidsIn P s'.tree := by
  rcases step_cases read C hs.inv h with
    ⟨p, c, u, t, -, -, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, ho, -, -, -, -, -, rfl⟩ | ⟨rfl, -, -⟩
  · exact hP
  · rw [splitLeaf_tree]
    obtain ⟨i, hki, -, -, hc⟩ := probe_prefix read s.tree s.edges C.walkStart x (Or.inl ho)
    exact midsIn_split hP (hxP i c hki hc) _ p
  · exact hP

theorem midsIn_runProbes {P : α → Prop} {X : List (FreeMonoid α)}
    (hXP : ∀ x ∈ X, LettersFrom P C.walkStart x) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv read C X s →
      MidsIn P s.tree → MidsIn P (runProbes read C s xs).1.tree
  | [], s, _, _, hP => by simpa [runProbes_nil] using hP
  | x :: xs, s, hX, hs, hP => by
    simp only [runProbes_cons]
    rcases ha : applyProbe read C s x with s' | e
    · exact midsIn_runProbes hXP xs s' (fun y hy => hX y (by simp [hy]))
        (runInv_step read C (hX x (by simp)) hs ha).1
        (midsIn_step read C (hXP x (hX x (by simp))) hs hP ha)
    · exact hP

/-- Reads agreeing on every prefix of a probe of `X` past its first `walkStart` letters, followed
by letters satisfying `P`, agree on what a step reads, where the probes' letters past `walkStart`
and the midfixes' letters satisfy `P`. -/
theorem step_reads_agree_in {P : α → Prop} {r r' : FreeMonoid α → ARU} {X : List (FreeMonoid α)}
    (hXP : ∀ x ∈ X, LettersFrom P C.walkStart x)
    (h : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      (∀ c ∈ m.toList, P c) → r (prefixOf x j * m) = r' (prefixOf x j * m))
    {s : Level3State α} {x : FreeMonoid α} (hxX : x ∈ X) (hs : RunInv r C X s)
    (hP : MidsIn P s.tree) :
    ∀ z ∈ Kappa.probeReads r s.tree s.edges C.walkStart x ∪ Eps.splitReads r C s x,
      r z = r' z := by
  classical
  intro z hz
  rcases Finset.mem_union.1 hz with hz | hz
  · obtain ⟨p, hp, m, hm, rfl⟩ : ∃ p ∈ Finset.Icc C.walkStart x.toList.length,
        ∃ m ∈ mids s.tree, prefixOf x p * m = z := by
      have := Kappa.probeReads_pot s.edges C.walkStart hz
      simpa [pot] using this
    obtain ⟨hp1, hp2⟩ := Finset.mem_Icc.1 hp
    exact h x hxX p hp1 hp2 m (hP m hm)
  · unfold Eps.splitReads at hz
    rcases ho : probeOrAgree r s.tree s.edges C.walkStart x with
      _ | ⟨p, c, u, t⟩ | _ | _ | _ <;> rw [ho] at hz <;> simp only [Finset.notMem_empty] at hz
    rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩ <;> rw [hE] at hz
    · simp at hz
    simp only [Eps.resifts, Finset.mem_biUnion] at hz
    obtain ⟨e, -, hz⟩ := hz
    rcases he : s.edges e with _ | ⟨t', w⟩ <;> rw [he] at hz
    · simp at hz
    obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
    obtain ⟨x', hx', i, hki, hi, rfl, hc⟩ := hs.wit e t' w he
    rw [prefixOf_mul_of hc]
    obtain ⟨i', hki', -, -, hc'⟩ := probe_prefix r s.tree s.edges C.walkStart x (Or.inl ho)
    exact h x' hx' (i + 1) (by omega) (by omega) m
      (midsIn_split hP (hXP x hxX i' c hki' hc') _ p m hm)

theorem runProbes_congr_in {P : α → Prop} {r r' : FreeMonoid α → ARU} {X : List (FreeMonoid α)}
    (hXP : ∀ x ∈ X, LettersFrom P C.walkStart x)
    (h : ∀ x ∈ X, ∀ j, C.walkStart ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α,
      (∀ c ∈ m.toList, P c) → r (prefixOf x j * m) = r' (prefixOf x j * m)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x ∈ X) → RunInv r C X s →
      MidsIn P s.tree → runProbes r C s xs = runProbes r' C s xs
  | [], s, _, _, _ => rfl
  | x :: xs, s, hX, hs, hP => by
    have hxX := hX x (by simp)
    have hstep : applyProbe r C s x = applyProbe r' C s x :=
      Kappa.applyProbe_congr' C (fun e t w he => hs.keys e t w he)
        (step_reads_agree_in C hXP h hxX hs hP)
    simp only [runProbes_cons]
    rw [← hstep]
    rcases ha : applyProbe r C s x with s' | e
    · exact runProbes_congr_in hXP h xs s' (fun y hy => hX y (by simp [hy]))
        (runInv_step r C hxX hs ha).1 (midsIn_step r C (hXP x hxX) hs hP ha)
    · rfl

/-- Where every probe's letters past `walkStart` satisfy `P`, reads agreeing on each probe's
prefixes past `walkStart` followed by letters satisfying `P` give the same result, and its tree's
midfixes have only such letters. -/
theorem level3_congr_in {P : α → Prop} {r r' : FreeMonoid α → ARU} {T : ℕ}
    (xs : Fin T → FreeMonoid α) (hXP : ∀ i, LettersFrom P C.walkStart (xs i))
    (h : ∀ i j, C.walkStart ≤ j → j ≤ (xs i).toList.length → ∀ m : FreeMonoid α,
      (∀ c ∈ m.toList, P c) → r (prefixOf (xs i) j * m) = r' (prefixOf (xs i) j * m)) :
    indecisiveLStarLevel3 C r xs = indecisiveLStarLevel3 C r' xs
      ∧ MidsIn P (indecisiveLStarLevel3 C r xs).hypothesis.tree := by
  have hXP' : ∀ x ∈ List.ofFn xs, LettersFrom P C.walkStart x := fun x hx => by
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    exact hXP i
  have hrun := runProbes_congr_in C (X := List.ofFn xs) hXP'
    (fun x hx j hj hjx m hm => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact h i j hj hjx m hm)
    (List.ofFn xs) initialState (fun _ h => h) (runInv_start r C _) (midsIn_initialState P)
  refine ⟨by simp only [indecisiveLStarLevel3_eq, hrun], ?_⟩
  rw [indecisiveLStarLevel3_eq]
  exact midsIn_runProbes r C hXP' (List.ofFn xs) initialState (fun _ h => h)
    (runInv_start r C _) (midsIn_initialState P)

omit [DecidableEq α] in
/-- Under a tree whose midfixes' letters satisfy `P`, a probe reads only a prefix past its first
`k` letters followed by letters satisfying `P`. -/
theorem probeOrAgree_congr_in {P : α → Prop} {r r' : FreeMonoid α → ARU} {T : DTree α}
    (hT : MidsIn P T) (E : Edges α) (k : ℕ) {x : FreeMonoid α}
    (h : ∀ j, k ≤ j → j ≤ x.toList.length → ∀ m : FreeMonoid α, (∀ c ∈ m.toList, P c) →
      r (prefixOf x j * m) = r' (prefixOf x j * m)) :
    probeOrAgree r T E k x = probeOrAgree r' T E k x := by
  classical
  exact probeOrAgree_congr_zone E k fun j hj hjx m hm => h j hj hjx m (hT m hm)

end Run

/-! ## Canonical strings -/

section Canon

theorem lettersFrom_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    LettersFrom (fun c : Option α => c.isSome) K (padIn K X x) := by
  intro i c hi hc
  have e : (padIn K X x).toList[i]? = (x.toList.map some)[i - (K - walkStartIn X x)]? := by
    simp only [padIn, FreeMonoid.toList_ofList]
    rw [List.getElem?_append_right (by simp; omega), List.length_replicate]
  rw [e, List.getElem?_map] at hc
  rcases h : x.toList[i - (K - walkStartIn X x)]? with _ | a <;> rw [h] at hc
  · simp at hc
  · simp only [Option.map_some, Option.some.injEq] at hc
    simp [← hc]

theorem map_some_reduceOption {l : List (Option α)} (h : ∀ c ∈ l, c.isSome) :
    l.reduceOption.map some = l := by
  induction l with
  | nil => rfl
  | cons c l ih =>
    rcases c with _ | a
    · simp at h
    · rw [List.reduceOption_cons_of_some, List.map_cons, ih fun c hc => h c (by simp [hc])]

/-- A string a probe padded from `X` reads past its first `K` letters, followed by letters of the
alphabet, is canonical, where `X`'s strings are shorter than `K ≤ |x|`. -/
theorem padIn_unpad_of_read {K : ℕ} {X : Finset (FreeMonoid α)}
    (hX : ∀ c ∈ X, c.toList.length < K) {x : FreeMonoid α} (hx : K ≤ x.toList.length) {j : ℕ}
    (hj : K ≤ j) {m : FreeMonoid (Option α)} (hm : ∀ c ∈ m.toList, c.isSome) :
    padIn K X (unpad (prefixOf (padIn K X x) j * m)) = prefixOf (padIn K X x) j * m := by
  have hnK := walkStartIn_le_of_short hX x
  rw [unpad_mul, unpad_prefixOf_padIn]
  obtain ⟨hws, -⟩ := walkStartIn_prefixOf_mul hX hx (j := j - (K - walkStartIn X x))
    (by omega) (unpad m)
  apply FreeMonoid.toList.injective
  rw [padIn, hws]
  simp only [padIn, prefixOf, unpad, FreeMonoid.toList_ofList, FreeMonoid.toList_mul,
    List.map_append, List.take_append, List.length_replicate, List.take_replicate,
    ← List.map_take, map_some_reduceOption hm, List.append_assoc]
  rw [min_eq_right (by omega)]

/-- Where `X`'s strings are shorter than `K ≤ |x|`, reads agreeing on the canonical strings agree
on what a probe padded from `X` reads under a tree whose midfixes have only letters of the
alphabet. -/
theorem probeOrAgree_padIn_congr [Fintype α] [DecidableEq α] {r r' : FreeMonoid (Option α) → ARU}
    {K : ℕ}
    {X : Finset (FreeMonoid α)} (h : ∀ z, padIn K X (unpad z) = z → r z = r' z)
    (hX : ∀ c ∈ X, c.toList.length < K) {x : FreeMonoid α} (hx : K ≤ x.toList.length)
    {T : DTree (Option α)} (hT : MidsIn (fun c : Option α => c.isSome) T)
    (E : Edges (Option α)) :
    probeOrAgree r T E K (padIn K X x) = probeOrAgree r' T E K (padIn K X x) :=
  probeOrAgree_congr_in hT E K fun _ hj _ _ hm => h _ (padIn_unpad_of_read hX hx hj hm)

/-- Where `X`'s strings are shorter than `K` and the probes at least `K` long, reads agreeing on
the canonical strings give level 3 on the probes padded from `X` the same result, and its tree's
midfixes have only letters of the alphabet. -/
theorem level3_padIn_congr [Fintype α] [DecidableEq α] {r r' : FreeMonoid (Option α) → ARU} {K : ℕ}
    {X : Finset (FreeMonoid α)} (h : ∀ z, padIn K X (unpad z) = z → r z = r' z)
    (hX : ∀ c ∈ X, c.toList.length < K) (C : Level3Settings) (hC : C.walkStart = K) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (hxs : ∀ i, K ≤ (xs i).toList.length) :
    indecisiveLStarLevel3 C r (fun i => padIn K X (xs i))
        = indecisiveLStarLevel3 C r' (fun i => padIn K X (xs i))
      ∧ MidsIn (fun c : Option α => c.isSome)
        (indecisiveLStarLevel3 C r (fun i => padIn K X (xs i))).hypothesis.tree := by
  subst hC
  exact level3_congr_in C _ (fun i => lettersFrom_padIn _ X (xs i))
    fun i _ hj _ _ hm => h _ (padIn_unpad_of_read hX (hxs i) hj hm)

end Canon

end Random

end OrthoDFA

end
