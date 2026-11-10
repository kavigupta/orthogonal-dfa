import RLStar.Proofs.AdaptiveReads
import RLStar.Proofs.Query

/-!
# A union bound over the reads of a program

A program's reads, in the order it first asks them, are rounds of one string each, each chosen
from the reads before it (`newAsks_local`). So `AdaptiveReadsBound` applies: if every run with no
bad read asks at most `N` strings, some read is bad with chance at most `ε N`.
-/

namespace OrthoDFA

namespace Query

variable {Z R β : Type*} [DecidableEq Z]

/-- The strings `p` asks against `r` and not in `seen`, in the order it first asks them. -/
def newAsks (r : Z → R) : Finset Z → Query Z R β → List Z
  | _, pure _ => []
  | seen, ask q k =>
    if q ∈ seen then newAsks r seen (k (r q)) else q :: newAsks r (insert q seen) (k (r q))

theorem newAsks_local {r r' : Z → R} :
    ∀ (p : Query Z R β) (seen : Finset Z) (j : ℕ), (∀ z ∈ seen, r z = r' z) →
      (∀ i < j, ∀ z, (newAsks r seen p)[i]? = some z → r z = r' z) →
      (newAsks r seen p)[j]? = (newAsks r' seen p)[j]?
  | pure _, _, _, _, _ => rfl
  | ask q k, seen, j, hs, hi => by
    simp only [newAsks] at hi ⊢
    by_cases hq : q ∈ seen
    · simp only [if_pos hq] at hi ⊢
      rw [← hs q hq]
      exact newAsks_local (k (r q)) seen j hs hi
    · simp only [if_neg hq] at hi ⊢
      rcases j with _ | j
      · rfl
      · have hrq : r q = r' q := hi 0 (by omega) q rfl
        simp only [List.getElem?_cons_succ]
        rw [← hrq]
        refine newAsks_local (k (r q)) (insert q seen) j (fun z hz => ?_) fun i hij z hz => ?_
        · rcases Finset.mem_insert.1 hz with rfl | hz
          · exact hrq
          · exact hs z hz
        · exact hi (i + 1) (by omega) z (by simpa using hz)

theorem newAsks_spec (r : Z → R) :
    ∀ (p : Query Z R β) (seen : Finset Z), (newAsks r seen p).Nodup
      ∧ (∀ z ∈ newAsks r seen p, z ∉ seen)
      ∧ ∀ z, z ∈ p.reads r → z ∈ newAsks r seen p ∨ z ∈ seen
  | pure _, _ => by simp [newAsks]
  | ask q k, seen => by
    simp only [newAsks, reads_ask]
    by_cases hq : q ∈ seen
    · simp only [if_pos hq]
      obtain ⟨h1, h2, h3⟩ := newAsks_spec r (k (r q)) seen
      refine ⟨h1, h2, fun z hz => ?_⟩
      rcases Finset.mem_insert.1 hz with rfl | hz
      · exact Or.inr hq
      · exact h3 z hz
    · simp only [if_neg hq]
      obtain ⟨h1, h2, h3⟩ := newAsks_spec r (k (r q)) (insert q seen)
      refine ⟨List.nodup_cons.2 ⟨fun h => h2 q h (Finset.mem_insert_self _ _), h1⟩,
        fun z hz => ?_, fun z hz => ?_⟩
      · rcases List.mem_cons.1 hz with rfl | hz
        · exact hq
        · exact fun h => h2 z hz (Finset.mem_insert_of_mem h)
      · rcases Finset.mem_insert.1 hz with rfl | hz
        · exact Or.inl (List.mem_cons_self ..)
        · rcases h3 z hz with h | h
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · rcases Finset.mem_insert.1 h with rfl | h
            · exact Or.inl (List.mem_cons_self ..)
            · exact Or.inr h

theorem newAsks_sub_reads (r : Z → R) :
    ∀ (p : Query Z R β) (seen : Finset Z), ∀ z ∈ newAsks r seen p, z ∈ p.reads r
  | pure _, _ => by simp [newAsks]
  | ask q k, seen => by
    intro z hz
    simp only [newAsks] at hz
    simp only [reads_ask, Finset.mem_insert]
    split_ifs at hz
    · exact Or.inr (newAsks_sub_reads r (k (r q)) seen z hz)
    · rcases List.mem_cons.1 hz with rfl | hz
      · exact Or.inl rfl
      · exact Or.inr (newAsks_sub_reads r (k (r q)) _ z hz)

theorem length_newAsks (r : Z → R) (p : Query Z R β) :
    (newAsks r ∅ p).length = (p.reads r).card := by
  obtain ⟨h1, -, h3⟩ := newAsks_spec r p ∅
  rw [← List.toFinset_card_of_nodup h1]
  congr 1
  ext z
  simp only [List.mem_toFinset]
  exact ⟨newAsks_sub_reads r p ∅ z, fun h => (h3 z h).resolve_right (by simp)⟩

end Query

open MeasureTheory ProbabilityTheory Query

/-- Reads independent across strings, each bad with chance at most `ε` and each with a value that
is not bad. If every run with no bad read asks at most `N` strings, some read of `p` is bad with
chance at most `ε N`. -/
theorem query_reads_bad_le {Z Ω R β : Type*} [Countable Z] [DecidableEq Z] [MeasurableSpace Ω]
    [MeasurableSpace R] [Countable R] [MeasurableSingletonClass R] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : Z → Ω → R) (hmeas : ∀ z, Measurable (read z))
    (hind : iIndepFun read μ) (B : Z → Set R) (ε : ℝ)
    (hB : ∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε) (hg : ∀ z, ∃ v, v ∉ B z)
    (p : Query Z R β) (N : ℕ)
    (hN : ∀ o : Z → R, (∀ z ∈ p.reads o, o z ∉ B z) → (p.reads o).card ≤ N) :
    μ {ω | ∃ z ∈ p.reads (read · ω), read z ω ∈ B z} ≤ ENNReal.ofReal (ε * N) := by
  classical
  choose g hg using hg
  set Qr : ℕ → (Z → R) → Finset Z := fun j r => ((newAsks r ∅ p)[j]?).toFinset
  have hmemQ : ∀ j r z, z ∈ Qr j r ↔ (newAsks r ∅ p)[j]? = some z := by
    intro j r z; simp [Qr, eq_comm]
  have hloc : AdaptiveReads.Local Qr B := by
    intro j r r' _ hag
    simp only [Qr]
    rw [newAsks_local p ∅ j (by simp) fun i hi z hz => hag i hi z ((hmemQ i r z).2 hz)]
  -- a run whose first `N` new reads are good asks at most `N` strings
  have hshort : ∀ r : Z → R, (∀ i < N, ∀ z ∈ Qr i r, r z ∉ B z) →
      (newAsks r ∅ p).length ≤ N := by
    intro r hgood
    by_contra hlt
    push Not at hlt
    set F := ((newAsks r ∅ p).take N).toFinset
    set o : Z → R := fun z => if z ∈ F then r z else g z
    have hagree : ∀ i ≤ N, (newAsks r ∅ p)[i]? = (newAsks o ∅ p)[i]? := by
      intro i hi
      refine newAsks_local p ∅ i (by simp) fun l hl z hz => ?_
      have hzF : z ∈ F := by
        simp only [F, List.mem_toFinset]
        exact List.mem_iff_getElem?.2 ⟨l, by rw [List.getElem?_take_of_lt (by omega)]; exact hz⟩
      simp [o, hzF]
    have hlen : N + 1 ≤ (newAsks o ∅ p).length := by
      have h := hagree N le_rfl
      rw [List.getElem?_eq_getElem hlt] at h
      have := (List.getElem?_eq_some_iff.1 h.symm).1
      omega
    have hbad : ∀ z ∈ p.reads o, o z ∉ B z := by
      intro z _
      by_cases hzF : z ∈ F
      · simp only [o, if_pos hzF]
        simp only [F, List.mem_toFinset] at hzF
        obtain ⟨l, hl, hz⟩ := List.getElem_of_mem hzF
        rw [List.length_take] at hl
        refine hgood l (by omega) z ((hmemQ l r z).2 ?_)
        rw [← hz, List.getElem_take, List.getElem?_eq_getElem]
      · simp only [o, if_neg hzF]
        exact hg z
    have := hN o hbad
    rw [← length_newAsks] at this
    omega
  have hb : ∀ j r, (∀ i < j, ∀ z ∈ Qr i r, r z ∉ B z) →
      (Qr j r).card ≤ if j < N then 1 else 0 := by
    intro j r hgood
    split_ifs with hj
    · simp only [Qr]
      cases (newAsks r ∅ p)[j]? <;> simp
    · have hl := hshort r fun i hi => hgood i (by omega)
      simp only [Qr]
      rw [List.getElem?_eq_none (by omega)]
      simp
  have hmain := adaptiveReadsBound_holds μ read B ε Qr (fun j => if j < N then 1 else 0) N hmeas
    hind hB (fun j r r' h1 h2 => hloc j r r' h1 h2) hb
  refine le_trans (measure_mono fun ω hω => ?_) (hmain.trans (le_of_eq ?_))
  · obtain ⟨z, hz, hzB⟩ := hω
    set r : Z → R := (read · ω)
    by_cases hgood : ∀ i < N, ∀ y ∈ Qr i r, r y ∉ B y
    · have hl := hshort r hgood
      obtain ⟨-, -, h3⟩ := newAsks_spec r p ∅
      obtain ⟨i, hi, hiz⟩ := List.getElem_of_mem ((h3 z hz).resolve_right (by simp))
      exact ⟨i, by omega, z, (hmemQ i r z).2 (by rw [List.getElem?_eq_getElem hi, hiz]), hzB⟩
    · push Not at hgood
      obtain ⟨i, hi, y, hy, hyB⟩ := hgood
      exact ⟨i, hi, y, hy, hyB⟩
  · congr 1
    rw [Finset.sum_congr rfl fun j hj => by rw [if_pos (Finset.mem_range.1 hj)]]
    simp

end OrthoDFA
