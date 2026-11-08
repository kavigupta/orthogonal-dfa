import RLStar.Code.Query
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Intervals

/-!
# What a query program's run depends on

`eval_congr`: a run against an oracle agreeing with `o` on `reads o p` returns what the run
against `o` returns and asks what it asks. The rest computes `eval` and `reads` of each
combinator; `extendUntil` runs as `growList` (`eval_extendUntil`) and asks only of the steps it
takes (`card_reads_extendUntil_le`).
-/

namespace OrthoDFA

universe u v w w'

section Grow

variable {β : Type*}

/-- `l` extended by `f l.length`, then `f (l.length + 1)`, and so on, until `stop` holds of it or
`n` have been added; `none` at the first `none`. -/
def growList (f : ℕ → Option β) (stop : List β → Prop) [DecidablePred stop] :
    ℕ → List β → Option (List β)
  | 0, l => some l
  | n + 1, l => if stop l then some l else
    match f l.length with
    | some z => growList f stop n (l ++ [z])
    | none => none

variable {f : ℕ → Option β} {stop : List β → Prop} [DecidablePred stop]

/-- What `growList` returns: `l` and at most `n` values of `f` after it, stopped where `stop` first
holds or at `n`. -/
lemma growList_eq_some : ∀ {n : ℕ} {l l' : List β}, growList f stop n l = some l' →
    ∃ e : List β, l' = l ++ e ∧ e.length ≤ n
      ∧ (∀ m (hm : m < e.length), f (l.length + m) = some e[m])
      ∧ (stop l' ∨ e.length = n) ∧ ∀ m < e.length, ¬ stop (l ++ e.take m)
  | 0, l, l', h => by
    simp only [growList, Option.some.injEq] at h
    subst h
    exact ⟨[], by simp, by simp, by simp, Or.inr rfl, by simp⟩
  | n + 1, l, l', h => by
    unfold growList at h
    split_ifs at h with hs
    · simp only [Option.some.injEq] at h
      subst h
      exact ⟨[], by simp, by simp, by simp, Or.inl (by simpa using hs), by simp⟩
    rcases hf : f l.length with _ | z
    · rw [hf] at h
      simp at h
    rw [hf] at h
    obtain ⟨e, rfl, hlen, hfe, hstop, hmin⟩ := growList_eq_some h
    refine ⟨z :: e, by simp, by simp only [List.length_cons]; omega, fun m hm => ?_, ?_,
      fun m hm => ?_⟩
    · rcases m with _ | m
      · simpa using hf
      · have := hfe m (by simpa using hm)
        rw [List.length_append, List.length_singleton] at this
        rw [show l.length + (m + 1) = l.length + 1 + m by omega, this]
        rfl
    · rcases hstop with h1 | h1
      · exact Or.inl (by simpa using h1)
      · exact Or.inr (by simp [h1])
    · rcases m with _ | m
      · simpa using hs
      · simpa using hmin m (by simpa using hm)

/-- Where every value it may take is `some`, `growList` returns. -/
lemma growList_isSome : ∀ {n : ℕ} {l : List β}, (∀ m < n, (f (l.length + m)).isSome) →
    (growList f stop n l).isSome
  | 0, l, _ => rfl
  | n + 1, l, h => by
    unfold growList
    split_ifs with hs
    · rfl
    obtain ⟨z, hz⟩ := Option.isSome_iff_exists.1 (h 0 (by omega))
    rw [Nat.add_zero] at hz
    rw [hz]
    exact growList_isSome fun m hm => by
      rw [show (l ++ [z]).length + m = l.length + (m + 1) by simp; omega]
      exact h (m + 1) (by omega)

end Grow

namespace Query

variable {Q : Type u} {R : Type v} {β : Type w} {γ : Type w'} (o : Q → R)

@[simp] lemma eval_pure (b : β) : (pure b : Query Q R β).eval o = b := rfl

@[simp] lemma eval_ask (q : Q) (k : R → Query Q R β) : (ask q k).eval o = (k (o q)).eval o := rfl

@[simp] lemma eval_bind (p : Query Q R β) (f : β → Query Q R γ) :
    (p.bind f).eval o = (f (p.eval o)).eval o := by
  induction p with
  | pure b => rfl
  | ask q k ih => exact ih (o q)

@[simp] lemma eval_map (f : β → γ) (p : Query Q R β) : (p.map f).eval o = f (p.eval o) := by
  simp [map]

@[simp] lemma eval_query (q : Q) : (query q : Query Q R R).eval o = o q := rfl

variable [instQ : DecidableEq Q]

@[simp] lemma reads_pure (b : β) : (pure b : Query Q R β).reads o = ∅ := rfl

@[simp] lemma reads_ask (q : Q) (k : R → Query Q R β) :
    (ask q k).reads o = insert q ((k (o q)).reads o) := rfl

@[simp] lemma reads_bind (p : Query Q R β) (f : β → Query Q R γ) :
    (p.bind f).reads o = p.reads o ∪ (f (p.eval o)).reads o := by
  induction p with
  | pure b => simp [bind]
  | ask q k ih =>
    simp only [bind, reads_ask, eval_ask, ih (o q), Finset.insert_union]

@[simp] lemma reads_map (f : β → γ) (p : Query Q R β) : (p.map f).reads o = p.reads o := by
  simp [map]

@[simp] lemma reads_query (q : Q) : (query q : Query Q R R).reads o = {q} := rfl

/-- The certificate: a run against `o'` agreeing with `o` where the run against `o` asks returns
the same and asks the same. -/
theorem eval_congr {o' : Q → R} (p : Query Q R β) (h : ∀ q ∈ p.reads o, o' q = o q) :
    p.eval o' = p.eval o ∧ p.reads o' = p.reads o := by
  induction p with
  | pure b => exact ⟨rfl, rfl⟩
  | ask q k ih =>
    simp only [reads_ask, Finset.mem_insert, forall_eq_or_imp] at h
    simp only [eval_ask, reads_ask, h.1]
    obtain ⟨h1, h2⟩ := ih (o q) h.2
    exact ⟨h1, by rw [h2]⟩

section Subst

variable {Q' : Type*} {R' : Type*} [DecidableEq Q'] (o' : Q' → R')

omit instQ [DecidableEq Q'] in
@[simp] lemma eval_subst (h : Q → Query Q' R' R) (p : Query Q R β) :
    (p.subst h).eval o' = p.eval fun q => (h q).eval o' := by
  induction p with
  | pure b => rfl
  | ask q k ih => simp [subst, ih]

lemma reads_subst (h : Q → Query Q' R' R) (p : Query Q R β) :
    (p.subst h).reads o' = (p.reads fun q => (h q).eval o').biUnion fun q => (h q).reads o' := by
  induction p with
  | pure b => rfl
  | ask q k ih => simp [subst, ih, Finset.biUnion_insert]

/-- At most `n` answering questions for each question `p` asks. -/
lemma card_reads_subst_le (h : Q → Query Q' R' R) (p : Query Q R β) {n : ℕ}
    (hn : ∀ q ∈ p.reads fun q => (h q).eval o', ((h q).reads o').card ≤ n) :
    ((p.subst h).reads o').card ≤ (p.reads fun q => (h q).eval o').card * n := by
  rw [reads_subst]
  exact Finset.card_biUnion_le_card_mul _ _ _ hn

end Subst

section Pi

variable {ι : Type*} [DecidableEq ι] [Inhabited γ]

omit instQ in
lemma eval_piList (f : ι → Query Q R γ) (l : List ι) :
    (piList f l).eval o = fun j => if j ∈ l then (f j).eval o else default := by
  induction l with
  | nil => simp [piList]
  | cons i l ih =>
    funext j
    simp only [piList, eval_bind, eval_map, ih, List.mem_cons]
    by_cases hj : j = i
    · subst hj; simp
    · simp [hj]

lemma reads_piList (f : ι → Query Q R γ) (l : List ι) :
    (piList f l).reads o = l.toFinset.biUnion fun i => (f i).reads o := by
  induction l with
  | nil => simp [piList]
  | cons i l ih => simp [piList, ih, Finset.biUnion_insert]

omit instQ in
@[simp] lemma eval_pi [Fintype ι] (f : ι → Query Q R γ) :
    (pi f).eval o = fun j => (f j).eval o := by
  simp [pi, eval_piList]

lemma reads_pi [Fintype ι] (f : ι → Query Q R γ) :
    (pi f).reads o = Finset.univ.biUnion fun i => (f i).reads o := by
  simp [pi, reads_piList]

end Pi

section Table

variable [Inhabited R]

lemma eval_table {i : DecidableEq Q} (l : List Q) :
    (@table Q R i _ l).eval o = fun q => if q ∈ l then o q else default := by
  simp [table, eval_piList]

lemma eval_table_of_mem {i : DecidableEq Q} {l : List Q} {q : Q} (hq : q ∈ l) :
    (@table Q R i _ l).eval o q = o q := by
  simp [eval_table, hq]

@[simp] lemma reads_table {i : DecidableEq Q} (l : List Q) :
    @reads Q R _ instQ o (@table Q R i _ l) = @List.toFinset Q instQ l := by
  ext q
  simp [table, reads_piList]

end Table

section FindSome

variable {δ : Type*} {ε : Type w'} (f : δ → Query Q R (Option ε))

omit instQ in
@[simp] lemma eval_findSome (xs : List δ) :
    (findSome f xs).eval o = xs.findSome? fun x => (f x).eval o := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [findSome, eval_bind, List.findSome?_cons]
    rcases (f x).eval o with _ | y
    · exact ih
    · rfl

/-- A search that finds its value in `l₁` asks nothing of `l₂`. -/
lemma findSome_append_of_eval {l₁ : List δ} (l₂ : List δ) {y : ε}
    (h : (findSome f l₁).eval o = some y) :
    (findSome f (l₁ ++ l₂)).eval o = some y
      ∧ (findSome f (l₁ ++ l₂)).reads o = (findSome f l₁).reads o := by
  induction l₁ with
  | nil => simp [findSome] at h
  | cons x l₁ ih =>
    simp only [List.cons_append, findSome, eval_bind, reads_bind] at h ⊢
    rcases hx : (f x).eval o with _ | y'
    · rw [hx] at h
      obtain ⟨e1, e2⟩ := ih h
      exact ⟨e1, by rw [e2]⟩
    · rw [hx] at h
      exact ⟨h, rfl⟩

lemma card_reads_findSome_le (xs : List δ) {n : ℕ} (hn : ∀ x ∈ xs, ((f x).reads o).card ≤ n) :
    ((findSome f xs).reads o).card ≤ xs.length * n := by
  induction xs with
  | nil => simp [findSome]
  | cons x xs ih =>
    simp only [findSome, reads_bind, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    have h1 := hn x (by simp)
    have := ih fun y hy => hn y (by simp [hy])
    rcases hx : (f x).eval o with _ | y
    · simp only
      rw [Nat.succ_mul]; omega
    · simp only [reads_pure, Finset.card_empty]
      rw [Nat.succ_mul]; omega

end FindSome

section Extend

variable (f : ℕ → Query Q R (Option β)) (stop : List β → Prop) [DecidablePred stop]

omit instQ in
@[simp] lemma eval_extendUntil : ∀ (n : ℕ) (l : List β),
    (extendUntil f stop n l).eval o = growList (fun m => (f m).eval o) stop n l
  | 0, l => rfl
  | n + 1, l => by
    unfold extendUntil growList
    split_ifs with hs
    · rfl
    rw [eval_bind]
    try dsimp only
    rcases (f l.length).eval o with _ | z
    · rfl
    · exact eval_extendUntil n (l ++ [z])

/-- A run returning `l'` asks only of the steps from `l.length` to `l'.length`, at most `t m` at
step `m`. -/
lemma card_reads_extendUntil_le (t : ℕ → ℕ) : ∀ (n : ℕ) (l l' : List β),
    (extendUntil f stop n l).eval o = some l' →
    (∀ m, l.length ≤ m → m < l'.length → ((f m).reads o).card ≤ t m) →
    ((extendUntil f stop n l).reads o).card ≤ ∑ m ∈ Finset.Ico l.length l'.length, t m
  | 0, l, l', _, _ => by simp [extendUntil]
  | n + 1, l, l', h, ht => by
    unfold extendUntil at h ⊢
    split_ifs at h ⊢ with hs
    · simp
    rw [eval_bind] at h
    rw [reads_bind]
    rcases hz : (f l.length).eval o with _ | z
    · rw [hz] at h
      simp at h
    rw [hz] at h
    simp only at h ⊢
    have hgrow := growList_eq_some (by rwa [eval_extendUntil] at h)
    obtain ⟨e, rfl, -, -, -, -⟩ := hgrow
    have hlt : l.length < (l ++ [z] ++ e).length := by simp
    have ih := card_reads_extendUntil_le t n (l ++ [z]) _ h fun m hm1 hm2 =>
      ht m (by simp at hm1; omega) hm2
    rw [Finset.sum_eq_sum_Ico_succ_bot hlt]
    refine (Finset.card_union_le _ _).trans (Nat.add_le_add (ht _ le_rfl hlt) ?_)
    simpa using ih

end Extend

end Query

end OrthoDFA
