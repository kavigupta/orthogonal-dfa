import OrthoDFA.Round
import OrthoDFA.Proofs.Replay
import Mathlib.Data.List.Basic

/-!
# What the pass reads

Every read the counterexample pass makes asks the family about a string `b·e₁·e₂·m`: `b` a seed
string, a witness or a probe's prefix, `e₁` and `e₂` each empty or a letter, and `m` a midfix of
the tree the pass holds before the probe.  So two oracles that agree there take the pass through
the same probe (`probeStep_congr`), and the state after `k` probes is decided by the oracle on
those strings, against the tree after `k − 1` (`phase_congr`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

/-- The nodes' midfixes. -/
def mids : DTree α → List (FreeMonoid α)
  | .leaf => []
  | .node m r a => m :: (r.mids ++ a.mids)

theorem route_congr_mids {c₁ c₂ : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α), (∀ m ∈ t.mids, c₁ (x * m) = c₂ (x * m)) →
      t.route c₁ x = t.route c₂ x
  | .leaf, _, _ => rfl
  | .node m r a, x, h => by
    have hm := h m (by simp [mids])
    have hr := route_congr_mids r x fun m' hm' => h m' (by simp [mids, hm'])
    have ha := route_congr_mids a x fun m' hm' => h m' (by simp [mids, hm'])
    simp only [route, hm, hr, ha]

theorem mids_splitAt (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool) (m : FreeMonoid α),
      m ∈ (t.splitAt d p).mids ↔ m ∈ t.mids ∨ (m = d ∧ (t.splitAt d p).mids ≠ t.mids)
  | .leaf, [], m => by simp [splitAt, mids]
  | .leaf, _ :: _, m => by simp [splitAt]
  | .node _ _ _, [], m => by simp [splitAt]
  | .node n r a, false :: p, m => by
    have := mids_splitAt d r p m
    by_cases he : (r.splitAt d p).mids = r.mids
    · simp only [splitAt, mids, he]; simp
    · simp only [he, ne_eq, not_false_eq_true, and_true] at this
      simp only [splitAt, mids, List.mem_cons, List.mem_append, this, ne_eq, List.cons.injEq,
        true_and, List.append_cancel_right_eq, he, not_false_eq_true, and_true]
      tauto
  | .node n r a, true :: p, m => by
    have := mids_splitAt d a p m
    by_cases he : (a.splitAt d p).mids = a.mids
    · simp only [splitAt, mids, he]; simp
    · simp only [he, ne_eq, not_false_eq_true, and_true] at this
      simp only [splitAt, mids, List.mem_cons, List.mem_append, this, ne_eq, List.cons.injEq,
        true_and, List.append_cancel_left_eq, he, not_false_eq_true, and_true]
      tauto

theorem mids_splitAt_cases {d m : FreeMonoid α} {t : DTree α} {p : List Bool}
    (h : m ∈ (t.splitAt d p).mids) : m ∈ t.mids ∨ m = d :=
  ((mids_splitAt d t p m).1 h).imp id And.left

end DTree

section Congr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)

/-- The family `F` cut at `B`, read through `f`. -/
abbrev rd (f : FreeMonoid α → ℝ) : CutReads α := ⟨B, F, f⟩

/-- `f₁` and `f₂` agree on what a cut or a training-half read of `y` asks: `y` followed by each
suffix of the family or its training half. -/
def Agree (y : FreeMonoid α) : Prop := ∀ v ∈ F ∪ K.train F, f₁ (y * v) = f₂ (y * v)

/-- They agree on `y` followed by each of `K.suffixes F`, the held-out block too. -/
def AgreeAll (y : FreeMonoid α) : Prop := ∀ v ∈ K.suffixes F, f₁ (y * v) = f₂ (y * v)

/-- `e` as a string: empty, or the letter. -/
def ext : Option α → FreeMonoid α
  | none => 1
  | some c => FreeMonoid.of c

/-- They agree on every read of `b` extended by up to a letter, against `t`. -/
def AgreeOne (t : DTree α) (b : FreeMonoid α) : Prop :=
  ∀ e : Option α, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * ext e * m)

/-- They agree on every read of `b` extended by up to two letters, against `t`, the held-out
block's too. -/
def AgreeDeep (t : DTree α) (b : FreeMonoid α) : Prop :=
  ∀ e₁ e₂ : Option α, ∀ m ∈ t.mids, AgreeAll K F f₁ f₂ (b * ext e₁ * ext e₂ * m)

variable {K B F f₁ f₂}

theorem AgreeAll.agree {y : FreeMonoid α} (h : AgreeAll K F f₁ f₂ y) : Agree K F f₁ f₂ y :=
  fun v hv => h v (Finset.mem_union_left _ hv)

theorem acceptsOn_congr {W : Finset (FreeMonoid α)} (hW : W ⊆ F ∪ K.train F)
    {y : FreeMonoid α} (h : Agree K F f₁ f₂ y) : acceptsOn W f₁ y = acceptsOn W f₂ y := by
  classical
  unfold acceptsOn
  congr 1
  exact Finset.filter_congr fun v hv => by rw [h v (hW hv)]

theorem cut_congr {y : FreeMonoid α} (h : Agree K F f₁ f₂ y) :
    (rd B F f₁).cut y = (rd B F f₂).cut y := by
  simp only [CutReads.cut, rd, acceptsOn_congr Finset.subset_union_left h]

theorem sift_congr {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    t.sift (rd B F f₁).cut x = t.sift (rd B F f₂).cut x := by
  unfold DTree.sift
  rw [DTree.route_congr_mids t x fun m hm => cut_congr (h m hm)]

theorem AgreeOne.tree {t : DTree α} {b : FreeMonoid α} (h : AgreeOne K F f₁ f₂ t b) :
    ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * m) := fun m hm => by
  simpa [ext] using h none m hm

/-- After a split on `ofc'·m'`, a midfix of `t` extended by a letter, the agreement against the
new tree on `b` and `b` extended by a letter. -/
theorem AgreeDeep.split {t : DTree α} {b m' : FreeMonoid α} {c' : α} {p : List Bool}
    (h : AgreeDeep K F f₁ f₂ t b) (hm' : m' ∈ t.mids) :
    AgreeOne K F f₁ f₂ (t.splitAt (FreeMonoid.of c' * m') p) b := fun e m hm => by
  rcases DTree.mids_splitAt_cases hm with hm | rfl
  · simpa [ext] using (h e none m hm).agree
  · simpa [ext, mul_assoc] using (h e (some c') m' hm').agree

theorem prefixOf_length (w : FreeMonoid α) : prefixOf w w.toList.length = w := by
  simp [prefixOf]

end Congr

section PhaseCongr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)

omit K B F f₁ f₂ in
theorem AgreeDeep.mono {K : StageKnobs α} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}
    {t t' : DTree α} {b : FreeMonoid α} (h : AgreeDeep K F f₁ f₂ t' b)
    (ht : ∀ m ∈ t.mids, m ∈ t'.mids) : AgreeDeep K F f₁ f₂ t b :=
  fun e₁ e₂ m hm => h e₁ e₂ m (ht m hm)

end PhaseCongr

end OrthoDFA
