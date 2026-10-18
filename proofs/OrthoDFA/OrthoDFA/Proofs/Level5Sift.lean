import OrthoDFA.Proofs.Level5Risk
import OrthoDFA.Proofs.SiftSet

/-!
# A consistent round's sifts are decided

A probe whose start or end sift stops undecided reads that string first in the probe: at the
start, or, at its end, charged to its last edge. The start's and the edges' tests count these
reads, so where they stay silent until success, few probes read undecided where they sift.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Route

variable {cut : FreeMonoid α → Option Bool}

/-- A decided sift read every string on its way decided. -/
theorem route_decided_of_inl {T : DTree α} {u : FreeMonoid α} {p : List Bool}
    (h : T.sift (ofCut cut) u = .inl p) {w : FreeMonoid α} (hw : w ∈ (T.siftWithPath cut u).1) :
    cut w ≠ none := fun hc => by
  rw [sift_of_reads_und T u w hw hc] at h
  cases h

end Route

section Firsts

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : Edges α) (k : ℕ)
  (x : FreeMonoid α)

/-- A sift whose reads meet none of `P` and stops undecided has a first read, labelled as the
sift is, that reads undecided. -/
theorem firsts_siftQuery_und {ℓ : ReadLabel α} {u b : FreeMonoid α} (P : Finset (FreeMonoid α))
    (hb : T.sift (ofCut cut) u = .inr b) (hP : b ∉ P) :
    (ℓ, b) ∈ (siftQuery ℓ T u).firsts (ofCut cut) P := by
  have hr : b ∈ (T.siftWithPath (fun z => (ofCut cut z).cut) u).1 := by
    rw [cut_ofCut]; exact (sift_inr_reads cut T u b hb).1
  have hk := QTree.firsts_keys_eq (ofCut cut) (siftQuery ℓ T u) P
  have hmem : b ∈ (((siftQuery ℓ T u).firsts (ofCut cut) P).map Prod.snd).toFinset := by
    rw [hk.2, Finset.mem_sdiff, List.mem_toFinset, asks_siftQuery]
    refine ⟨?_, hP⟩
    simp only [List.map_map]
    exact List.mem_map.2 ⟨b, hr, rfl⟩
  obtain ⟨q, hq, hqb⟩ := List.mem_map.1 (List.mem_toFinset.1 hmem)
  have h1 := (mem_asks_siftQuery (QTree.firsts_sub_asks _ _ _ q hq)).1
  have : q = (ℓ, b) := Prod.ext h1 hqb
  rw [← this]; exact hq

/-- A position of the walk, on a letter of `x`, is charged to an edge. -/
theorem posEdgeFrom_some {p₀ : List Bool} {i : ℕ} (hk : k ≤ i)
    (hw : i - k < (walk edges p₀ (x.toList.drop k)).length) {c : α} (hc : x.toList[i]? = some c) :
    ∃ e, (readLabel edges k x p₀ i).2 = some e := by
  simp only [readLabel, posEdgeFrom]
  rw [List.getElem?_eq_getElem hw, hc]
  exact ⟨_, rfl⟩

/-- The position after the walk's last, on a letter of `x`, is charged to an edge. -/
theorem posEdgeFrom_some_pred {p₀ : List Bool} {i : ℕ} (hk : k + 1 ≤ i)
    (hw : i - 1 - k < (walk edges p₀ (x.toList.drop k)).length) {c : α}
    (hc : x.toList[i - 1]? = some c) : ∃ e, (readLabel edges k x p₀ i).2 = some e := by
  simp only [readLabel, posEdgeFrom]
  rcases (walk edges p₀ (x.toList.drop k))[i - k]? with _ | q <;>
    rcases x.toList[i]? with _ | c' <;>
    simp only [Option.orElse_some, Option.orElse_none] <;>
    first | exact ⟨_, rfl⟩ | (rw [List.getElem?_eq_getElem hw, hc]; exact ⟨_, rfl⟩)

/-- A probe that stops undecided at its end, with its search not stopping in the middle, read that
string first, charged to an edge. -/
theorem firsts_of_endUndecided {zs : List (FreeMonoid α)}
    (hx : probeOrAgree (ofCut cut) T edges k x = .undecided zs true) :
    ∃ q ∈ firstsBy cut T edges k x, cut q.2 = none ∧ ∃ e, q.1.2 = some e := by
  have hkx : ¬ x.toList.length < k := fun h => by
    rw [probeOrAgree_short cut T edges h] at hx; cases hx
  have hc : (fun z => (ofCut cut z).cut) = cut := cut_ofCut cut
  rcases hk : T.sift (ofCut cut) (prefixOf x k) with p₀ | b
  swap
  · simp [probeOrAgree, probe, hk, hkx] at hx
  obtain ⟨hpos, hhi, hst0, -, hnone, -⟩ := walk_facts edges k x (p₀ := p₀) (by omega)
  have hlo : T.sift (ofCut cut) (prefixOf x k)
      = .inl ((walk edges p₀ (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, hk]
  set P₀ := ((siftQuery ((k, none) : ReadLabel α) T (prefixOf x k)).asks (ofCut cut)).map
    Prod.snd |>.toFinset with hP₀
  have hnP : ∀ b, cut b = none → b ∉ (∅ : Finset (FreeMonoid α)) ∪ P₀ := fun b hb hm => by
    rw [Finset.empty_union, hP₀, List.mem_toFinset, asks_siftQuery, hc, List.map_map] at hm
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hm
    exact route_decided_of_inl hk hy hb
  unfold firstsBy probeQuery
  rw [if_neg hkx, QTree.firsts_bind, run_siftQuery, hk]
  simp only []
  rcases probe_long_cases (ofCut cut) T edges k x (by omega) hk hx with
    ⟨c, z, hxc, hz, -⟩ | ⟨un, hu1, hu2, h'⟩
  · rw [hxc]
    simp only []
    rw [QTree.firsts_bind]
    have hb := (sift_inr_reads cut T _ z hz).2
    refine ⟨(readLabel edges k x p₀ (k + (walk edges p₀ (x.toList.drop k)).length - 1 + 1), z),
      List.mem_append_right _ (List.mem_append_left _
        (firsts_siftQuery_und cut T _ hz (hnP z hb))), hb, ?_⟩
    exact posEdgeFrom_some_pred edges k x (by omega) (by omega) (c := c) (by simpa using hxc)
  rcases searchOutcome_cases (ofCut cut) T k x _ un (by omega) hhi hlo h' with
    ⟨z, hz, -⟩ | ⟨-, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, -, -, -, h''⟩⟩
  rotate_left
  · rcases un with _ | ⟨c, t⟩ <;> simp [searchOutcome] at h''
  · cases h''
  · cases h''
  have hb := (sift_inr_reads cut T _ z hz).2
  rcases un with _ | ⟨c, t⟩
  · have hxn := hu2 rfl
    rw [hxn]
    simp only []
    unfold searchQuery
    rw [QTree.firsts_bind]
    refine ⟨(readLabel edges k x p₀ (k + (walk edges p₀ (x.toList.drop k)).length - 1), z),
      List.mem_append_right _ (List.mem_append_left _
        (firsts_siftQuery_und cut T _ hz (hnP z hb))), hb, ?_⟩
    have hend := hnone hxn
    have hkh : k < k + (walk edges p₀ (x.toList.drop k)).length - 1 := by
      by_contra hc'
      have : k + (walk edges p₀ (x.toList.drop k)).length - 1 = k := by omega
      rw [this, hk] at hz; cases hz
    obtain ⟨c, hc'⟩ : ∃ c, x.toList[k + (walk edges p₀ (x.toList.drop k)).length - 1 - 1]? = some c :=
      ⟨_, List.getElem?_eq_getElem (by omega)⟩
    exact posEdgeFrom_some_pred edges k x (by omega) (by omega) hc'
  · obtain ⟨hxc, ht⟩ := hu1 c t rfl
    rw [hxc]
    simp only []
    rw [QTree.firsts_bind, run_siftQuery, ht]
    simp only []
    unfold searchQuery
    rw [QTree.firsts_bind]
    have hn : z ∉ ((∅ : Finset (FreeMonoid α)) ∪ P₀) ∪ (((siftQuery (readLabel edges k x p₀
        (k + (walk edges p₀ (x.toList.drop k)).length - 1 + 1)) T (prefixOf x
        (k + (walk edges p₀ (x.toList.drop k)).length - 1 + 1))).asks (ofCut cut)).map
          Prod.snd).toFinset := fun hm => by
      rcases Finset.mem_union.1 hm with hm | hm
      · exact hnP z hb hm
      · rw [List.mem_toFinset, asks_siftQuery, hc, List.map_map] at hm
        obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hm
        exact route_decided_of_inl ht hy hb
    refine ⟨(readLabel edges k x p₀ (k + (walk edges p₀ (x.toList.drop k)).length - 1), z),
      List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _
        (firsts_siftQuery_und cut T _ hz hn))), hb, ?_⟩
    exact posEdgeFrom_some edges k x (by omega) (by omega) hxc

end Firsts


section Count

variable (C : Level5Settings) (rd : FreeMonoid α → ARU)

/-- The edges out of a tree's leaves. -/
noncomputable def leafEdges (T : DTree α) : Finset (List Bool × α) :=
  T.leaves.toFinset ×ˢ Finset.univ

/-- The undecided strings counted since the hypothesis changed, at the start and at its edges. -/
noncomputable def undCount (s : Level5State α) : ℕ :=
  s.startUndecided.length + ∑ e ∈ leafEdges s.tree, (s.edgeUndecided e.1 e.2).length

/-- The reads counted at its edges since the hypothesis changed. -/
noncomputable def readSum (s : Level5State α) : ℕ :=
  ∑ e ∈ leafEdges s.tree, s.edgeReads e.1 e.2

/-- A probe's undecided strings at the start and at the edges. -/
noncomputable def undInc (T : DTree α) (E : Edges α) (x : FreeMonoid α) : ℕ :=
  (startHarvestBy (fun z => (rd z).cut) T C.walkStart x).length
    + ∑ e ∈ leafEdges T, (edgeHarvestBy (fun z => (rd z).cut) T E C.walkStart x e).length

/-- A probe's first reads at the edges. -/
noncomputable def readInc (T : DTree α) (E : Edges α) (x : FreeMonoid α) : ℕ :=
  ∑ e ∈ leafEdges T, edgeReadsBy (fun z => (rd z).cut) T E C.walkStart x e

theorem undCount_keep (s : Level5State α) (x : FreeMonoid α) :
    undCount (keepProbe C (fun z => (rd z).cut) s x) = undCount s + undInc C rd s.tree s.edges x := by
  simp only [undCount, undInc, keepProbe, Level5State.countProbe, List.length_append,
    Finset.sum_add_distrib]
  ring

theorem readSum_keep (s : Level5State α) (x : FreeMonoid α) :
    readSum (keepProbe C (fun z => (rd z).cut) s x) = readSum s + readInc C rd s.tree s.edges x := by
  simp only [readSum, readInc, keepProbe, Level5State.countProbe, Finset.sum_add_distrib]

theorem one_le_undInc {T : DTree α} {E : Edges α} (he : EdgesInto T E) {x : FreeMonoid α}
    (hx : x ∈ undSiftSet (fun z => (rd z).cut) T E C.walkStart) : 1 ≤ undInc C rd T E x := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  simp only [undSiftSet, Set.mem_setOf_eq] at hx
  rcases ho : probeOrAgree (ofCut cut) T E C.walkStart x with _ | _ | _ | b | ⟨zs, _ | _⟩ <;>
    rw [ho] at hx <;> simp only at hx
  · have hkx : ¬ x.toList.length < C.walkStart := fun h => by
      rw [probeOrAgree_short cut T E h] at ho; cases ho
    have hs : startHarvestBy cut T C.walkStart x ≠ [] := by
      unfold startHarvestBy
      rw [if_neg hkx]
      rcases hk : T.sift (ofCut cut) (prefixOf x C.walkStart) with p₀ | b'
      · exfalso
        rcases probe_long_cases (ofCut cut) T E C.walkStart x (by omega) hk ho with
          ⟨c, z, -, -, h⟩ | ⟨un, -, -, h'⟩
        · cases h
        obtain ⟨hpos, hhi, hst0, -, -, -⟩ := walk_facts E C.walkStart x (p₀ := p₀) (by omega)
        have hlo : T.sift (ofCut cut) (prefixOf x C.walkStart)
            = .inl ((walk E p₀ (x.toList.drop C.walkStart)).getD (C.walkStart - C.walkStart) []) := by
          rw [Nat.sub_self, hst0, hk]
        rcases searchOutcome_cases (ofCut cut) T C.walkStart x _ un (by omega) hhi hlo h' with
          ⟨z, -, h''⟩ | ⟨-, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, -, -, -, h''⟩⟩
        · cases h''
        · rcases un with _ | ⟨c, t⟩ <;> simp [searchOutcome] at h''
        · cases h''
        · cases h''
      · simp
    have := List.length_pos_of_ne_nil hs
    show 1 ≤ (startHarvestBy cut T C.walkStart x).length
      + ∑ e ∈ leafEdges T, (edgeHarvestBy cut T E C.walkStart x e).length
    omega
  · obtain ⟨q, hq, hcu, e, hqe⟩ := firsts_of_endUndecided cut T E C.walkStart x ho
    have hm : q.2 ∈ edgeHarvestBy cut T E C.walkStart x e := by
      unfold edgeHarvestBy
      exact List.mem_map.2 ⟨q, List.mem_filter.2 ⟨hq, by simp [hqe, hcu]⟩, rfl⟩
    have hne : edgeHarvestBy cut T E C.walkStart x e ≠ [] := List.ne_nil_of_mem hm
    have hK : e ∈ leafEdges T := Finset.mem_product.2
      ⟨List.mem_toFinset.2 (edgeHarvestBy_mem cut T C.walkStart x he hne), Finset.mem_univ _⟩
    have h1 := Finset.single_le_sum (f := fun e => (edgeHarvestBy cut T E C.walkStart x e).length)
      (fun _ _ => Nat.zero_le _) hK
    have h2 := List.length_pos_of_ne_nil hne
    show 1 ≤ (startHarvestBy cut T C.walkStart x).length
      + ∑ e ∈ leafEdges T, (edgeHarvestBy cut T E C.walkStart x e).length
    omega

theorem undInc_le (T : DTree α) (E : Edges α) (x : FreeMonoid α) :
    undInc C rd T E x ≤ x.toList.length + 2 := by
  have h1 : (startHarvestBy (fun z => (rd z).cut) T C.walkStart x).length ≤ 1 := by
    unfold startHarvestBy
    split_ifs
    · simp
    · rcases T.sift (ofCut fun z => (rd z).cut) (prefixOf x C.walkStart) with _ | _ <;> simp
  have h2 := sum_edgeHarvestBy_le (fun z => (rd z).cut) T E C.walkStart x (leafEdges T)
  unfold undInc
  omega

theorem readInc_le (T : DTree α) (E : Edges α) (x : FreeMonoid α) :
    readInc C rd T E x + 1 ≤ (x.toList.length + 1) * T.leaves.length + 1 := by
  have h1 : readInc C rd T E x ≤ (firstsBy (fun z => (rd z).cut) T E C.walkStart x).length :=
    sum_filter_some_le (fun q : ReadLabel α × FreeMonoid α => q.1.2) (leafEdges T) _
  have h2 := firstsBy_length_le (fun z => (rd z).cut) T E C.walkStart x
  have h3 := midfixes_card T
  have h4 : (x.toList.length + 1) * T.midfixes.card ≤ (x.toList.length + 1) * T.leaves.length :=
    Nat.mul_le_mul_left _ (by omega)
  omega

open scoped Classical in
/-- At success the start's test, and every edge's, stayed silent, and success's fired. -/
theorem runTests_success_quiet {s : Level5State α} (h : runTests C s = some .success) :
    ¬ seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes s.startUndecided.length
      ∧ (∀ e : List Bool × α, e.1 ∈ s.tree.leaves → ¬ (C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.edgeReads e.1 e.2)
          ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2))
      ∧ seqRate C.disagreementRate C.disagreementAltRate C.testLevel s.probes s.searches := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  · simp at h
  · simp at h
  · simp at h
  · push Not at h2
    exact ⟨h1, fun e he hc => by linarith [h2 e he hc.1, hc.2], h4⟩

/-- A start test of `θ` against `2θ` that has not fired has seen at most `2 log (1/a) + 3θn`
hits. -/
theorem start_quiet_le {θ a : ℝ} (hθ : 0 < θ) (hθ4 : θ ≤ 1 / 4) (ha : 0 < a) (ha1 : a ≤ 1)
    {n h : ℕ}
    (hh : h ≤ n) (hq : ¬ seqRate θ (2 * θ) a n h) :
    (h : ℝ) ≤ 2 * Real.log (1 / a) + 3 * θ * n := by
  rw [seqRate_iff_log hθ (by linarith) (by linarith) (by linarith) ha hh, not_le] at hq
  have e2 : 2 * θ / θ = 2 := by field_simp
  rw [e2] at hq
  have hl2 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hB : -(2 * θ) ≤ Real.log ((1 - 2 * θ) / (1 - θ)) := by
    have hpos : 0 < (1 - 2 * θ) / (1 - θ) := div_pos (by linarith) (by linarith)
    have h1 := Real.one_sub_inv_le_log_of_pos hpos
    rw [inv_div] at h1
    have h3 : (1 - θ) / (1 - 2 * θ) ≤ 1 + 2 * θ := by
      rw [div_le_iff₀ (by linarith)]; nlinarith
    linarith
  have hhn : (h : ℝ) ≤ n := by exact_mod_cast hh
  have hn0 : (0 : ℝ) ≤ n - h := by linarith
  have h4 : (n - h : ℝ) * Real.log ((1 - 2 * θ) / (1 - θ)) ≥ -(2 * θ) * n := by
    nlinarith [mul_le_mul_of_nonneg_left hB hn0]
  have hlg : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha]; linarith)
  have hh0 : (0 : ℝ) ≤ h := Nat.cast_nonneg h
  have h5 := mul_le_mul_of_nonneg_left hl2.le hh0
  nlinarith

/-- A success test of `p` against `7p/8` that fired has seen at least `7 log (1/a) / p` draws. -/
theorem success_long {p a : ℝ} (hp : 0 < p) (hp8 : p ≤ 1 / 8) (ha : 0 < a) {n h : ℕ}
    (hh : h ≤ n) (hs : seqRate p (7 / 8 * p) a n h) : 7 * Real.log (1 / a) ≤ p * n := by
  rw [seqRate_iff_log hp (by linarith) (by linarith) (by linarith) ha hh] at hs
  have hA : Real.log (7 / 8 * p / p) ≤ 0 := by
    rw [show 7 / 8 * p / p = 7 / 8 by field_simp]
    exact (Real.log_neg (by norm_num) (by norm_num)).le
  have hB : Real.log ((1 - 7 / 8 * p) / (1 - p)) ≤ p / 7 := by
    have := Real.log_le_sub_one_of_pos (div_pos (show 0 < 1 - 7 / 8 * p by linarith)
      (show 0 < 1 - p by linarith))
    have e : (1 - 7 / 8 * p) / (1 - p) - 1 = p / (8 * (1 - p)) := by
      field_simp [show (1 - p) ≠ 0 by linarith]; ring
    have h2 : p / (8 * (1 - p)) ≤ p / 7 := by
      rw [div_le_div_iff₀ (by linarith) (by norm_num)]; nlinarith
    linarith
  have hhn : (h : ℝ) ≤ n := by exact_mod_cast hh
  have hh0 : (0 : ℝ) ≤ h := Nat.cast_nonneg h
  have h1 : (h : ℝ) * Real.log (7 / 8 * p / p) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hh0 hA
  have h2 : (n - h : ℝ) * Real.log ((1 - 7 / 8 * p) / (1 - p)) ≤ (n - h) * (p / 7) :=
    mul_le_mul_of_nonneg_left hB (by linarith)
  nlinarith

/-- At success, the undecided strings counted are at most what the silent start's and edges'
tests allow. -/
theorem undCount_success_le {s : Level5State α} (h : runTests C s = some .success)
    (hci : CountInv s) (hθs : 0 < C.startUndecidedRate) (hθs4 : C.startUndecidedRate ≤ 1 / 4)
    (halt : C.startAltRate = 2 * C.startUndecidedRate) (ha : 0 < C.testLevel)
    (ha1 : C.testLevel ≤ 1) (hθe : 0 ≤ C.edgeUndecidedRate) (hφe : 0 ≤ C.edgeTraffic) {X : ℝ} (hX : ∀ j, C.edgeExcess j ≤ X) (hX0 : 0 ≤ X) :
    (undCount s : ℝ) ≤ 2 * Real.log (1 / C.testLevel) + 3 * C.startUndecidedRate * s.probes
      + (leafEdges s.tree).card * (C.edgeTraffic * s.probes + X)
      + C.edgeUndecidedRate * readSum s := by
  obtain ⟨hst, hed, -⟩ := runTests_success_quiet C h
  rw [halt] at hst
  have h1 := start_quiet_le hθs hθs4 ha ha1 hci.1 hst
  have h2 : ∀ e ∈ leafEdges s.tree, ((s.edgeUndecided e.1 e.2).length : ℝ)
      ≤ C.edgeTraffic * s.probes + X + C.edgeUndecidedRate * s.edgeReads e.1 e.2 := by
    intro e he
    have hl : e.1 ∈ s.tree.leaves := List.mem_toFinset.1 (Finset.mem_product.1 he).1
    have hq := hed e hl
    have hUR : ((s.edgeUndecided e.1 e.2).length : ℝ) ≤ s.edgeReads e.1 e.2 := by
      exact_mod_cast hci.2.2.2 e.1 e.2
    have hR0 : (0 : ℝ) ≤ C.edgeUndecidedRate * s.edgeReads e.1 e.2 := by positivity
    by_cases hφ : C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
    · have : ¬ (C.edgeExcess (s.edgeReads e.1 e.2)
          ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2) :=
        fun hc => hq ⟨hφ, hc⟩
      push Not at this
      have := hX (s.edgeReads e.1 e.2)
      have : 0 ≤ C.edgeTraffic * s.probes := by positivity
      linarith
    · push Not at hφ
      linarith
  have h3 := Finset.sum_le_sum h2
  rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum] at h3
  unfold undCount readSum
  push_cast
  linarith

end Count


theorem runEnds_of_consistent (C : Level5Settings) (cut : FreeMonoid α → Option Bool)
    (P : Level5State α → Prop) :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)), (runProbes C cut s xs).2 = .consistent →
      P (runProbes C cut s xs).1 → RunEnds (applyProbe C cut) (fun e s' => e = .success ∧ P s') s xs
  | s, [] => by simp [runProbes]
  | s, x :: xs => by
    intro h1 h2
    simp only [runProbes] at h1 h2
    simp only [RunEnds]
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at h1 h2
    · exact runEnds_of_consistent C cut P s' xs h1 h2
    · simp only [] at h1 h2 ⊢
      refine ⟨?_, h2⟩
      cases e <;> simp_all [toRoundEnd]

section Site

variable (C : Level5Settings) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (rd : FreeMonoid α → ARU)

/-- The hypothesis's start or end sifts stop undecided on at least `u` of the probes. -/
def SiftOK (u : ℝ) (s : Level5State α) : Prop :=
  u ≤ D.real (undSiftSet (fun z => (rd z).cut) s.tree s.edges C.walkStart)
    ∧ EdgesInto s.tree s.edges ∧ s.tree.leaves.length ≤ C.maxLeaves

/-- Success at such a hypothesis. -/
def SiftBad (u : ℝ) (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .success ∧ SiftOK C D rd u s ∧ runTests C s = some .success ∧ CountInv s

/-- A probe's mean count of undecided strings at the start and the edges. -/
noncomputable def siftMean (T : DTree α) (E : Edges α) : ℝ :=
  ∫ x, (undInc C rd T E x : ℝ) ∂D

/-- The undecided strings counted: half their running mean less them, exponentiated. -/
noncomputable def undLowW (L : ℕ) (s : Level5State α) : ℝ :=
  Real.exp ((s.probes * siftMean C D rd s.tree s.edges / 2 - undCount s) / (2 * ((L : ℝ) + 2)))

/-- The edges' reads counted less twice `(L + 1) maxLeaves` a probe, exponentiated. -/
noncomputable def readHighW (L : ℕ) (s : Level5State α) : ℝ :=
  Real.exp ((readSum s - 2 * s.probes * (((L : ℝ) + 1) * C.maxLeaves))
    / (2 * (((L : ℝ) + 1) * C.maxLeaves)))

def SiftLowBad (u : ℝ) (L : ℕ) (e : Level5Ending α) (s : Level5State α) : Prop :=
  SiftBad C D rd u e s ∧ 1 ≤ C.testLevel * undLowW C D rd L s

def SiftReadBad (u : ℝ) (L : ℕ) (e : Level5Ending α) (s : Level5State α) : Prop :=
  SiftBad C D rd u e s ∧ 1 ≤ C.testLevel * readHighW C L s

theorem siftLow_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (u : ℝ)
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (SiftLowBad C D rd u L) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  refine site_le C cut D hL hSq.a_pos hSq.a_lt hn₀ hexc (undLowW C D rd L)
    (fun s x => Real.exp ((siftMean C D rd s.tree s.edges / 2 - undInc C rd s.tree s.edges x)
      / (2 * ((L : ℝ) + 2))))
    (SiftOK C D rd u) (SiftLowBad C D rd u L) (fun s x => (Real.exp_pos _).le) (fun s _ => ?_)
    (fun T E v => by simp [undLowW, undCount, freshState]) (fun s x hs _ => ⟨?_, Iff.rfl⟩)
    (fun s h => by simp [SiftLowBad, SiftBad] at h) (fun s x e hs _ hb hr => ⟨hb.1.2.1, hb.2⟩) T
  · exact mgf_lower D (undInc C rd s.tree s.edges) (by positivity) (by
      filter_upwards [hlen] with x hx
      have := undInc_le C rd s.tree s.edges x
      exact_mod_cast (by omega : undInc C rd s.tree s.edges x ≤ L + 2))
  · have hp : (keepProbe C cut s x).probes = s.probes + 1 := rfl
    have ht : (keepProbe C cut s x).tree = s.tree := rfl
    have he : (keepProbe C cut s x).edges = s.edges := rfl
    have hu : undCount (keepProbe C cut s x) = undCount s + undInc C rd s.tree s.edges x :=
      undCount_keep C rd s x
    simp only [undLowW, hu, hp, ht, he, Nat.cast_add, Nat.cast_one]
    rw [← Real.exp_add]
    congr 1
    ring

theorem siftRead_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (u : ℝ)
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (SiftReadBad C D rd u L) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  set η : ℝ := ((L : ℝ) + 1) * C.maxLeaves with hη
  have hη0 : 0 < η := by
    have : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
    positivity
  refine site_le C cut D hL hSq.a_pos hSq.a_lt hn₀ hexc (readHighW C L)
    (fun s x => Real.exp ((readInc C rd s.tree s.edges x - 2 * η) / (2 * η)))
    (SiftOK C D rd u) (SiftReadBad C D rd u L) (fun s x => (Real.exp_pos _).le) (fun s hs => ?_)
    (fun T E v => by simp [readHighW, readSum, freshState]) (fun s x hs _ => ⟨?_, Iff.rfl⟩)
    (fun s h => by simp [SiftReadBad, SiftBad] at h) (fun s x e hs _ hb hr => ⟨hb.1.2.1, hb.2⟩) T
  · have hb : ∀ᵐ x ∂D, (readInc C rd s.tree s.edges x : ℝ) ≤ η ∧ ((0 : ℕ) : ℝ) ≤ η * 0 := by
      filter_upwards [hlen] with x hx
      refine ⟨?_, by simp⟩
      have h1 := readInc_le C rd s.tree s.edges x
      have h2 : (x.toList.length + 1) * s.tree.leaves.length ≤ (L + 1) * C.maxLeaves :=
        Nat.mul_le_mul (by omega) hs.2.2
      have : readInc C rd s.tree s.edges x ≤ (L + 1) * C.maxLeaves := by omega
      rw [hη]
      exact_mod_cast this
    have hmean : ∫ x, (readInc C rd s.tree s.edges x : ℝ) ∂D
        ≤ 0 * ∫ x, (((fun _ => 0) : FreeMonoid α → ℕ) x : ℝ) ∂D + η := by
      rw [zero_mul, zero_add]
      have hi : Integrable (fun x => (readInc C rd s.tree s.edges x : ℝ)) D :=
        Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable η (by
          filter_upwards [hb] with x hx
          rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.1)
      calc ∫ x, (readInc C rd s.tree s.edges x : ℝ) ∂D ≤ ∫ _x, η ∂D :=
            integral_mono_ae hi (integrable_const _) (by filter_upwards [hb] with x hx; exact hx.1)
        _ = η := by simp
    have := mgf_step D (readInc C rd s.tree s.edges) (fun _ => 0) (θ := 0) (L := η) (R := 0)
      le_rfl hη0 le_rfl hb hη0.le hmean
    refine le_trans (le_of_eq (lintegral_congr fun x => ?_)) this
    congr 2
    push_cast
    ring
  · have hp : (keepProbe C cut s x).probes = s.probes + 1 := rfl
    have ht : (keepProbe C cut s x).tree = s.tree := rfl
    have he : (keepProbe C cut s x).edges = s.edges := rfl
    have hr : readSum (keepProbe C cut s x) = readSum s + readInc C rd s.tree s.edges x :=
      readSum_keep C rd s x
    simp only [readHighW, hr, hp, ht, he, Nat.cast_add, Nat.cast_one]
    rw [← Real.exp_add]
    congr 1
    rw [← hη]
    field_simp
    ring

theorem leafEdges_card (T : DTree α) :
    ((leafEdges T).card : ℝ) ≤ T.leaves.length * Fintype.card α := by
  unfold leafEdges
  rw [Finset.card_product, Finset.card_univ]
  exact_mod_cast Nat.mul_le_mul_right _ (List.toFinset_card_le _)

/-- The probes whose sifts stop undecided are at most the mean count of undecided strings. -/
theorem undSift_le_siftMean {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {T : DTree α}
    {E : Edges α} (he : EdgesInto T E) :
    D.real (undSiftSet (fun z => (rd z).cut) T E C.walkStart) ≤ siftMean C D rd T E := by
  have hi : Integrable (fun x => (undInc C rd T E x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable ((L : ℝ) + 2) (by
      filter_upwards [hlen] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
      have := undInc_le C rd T E x
      exact_mod_cast (by omega : undInc C rd T E x ≤ L + 2))
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  refine integral_mono_of_nonneg (ae_of_all _ fun x => ?_) hi (ae_of_all _ fun x => ?_)
  · by_cases hx : x ∈ undSiftSet (fun z => (rd z).cut) T E C.walkStart <;> simp [hx]
  · by_cases hx : x ∈ undSiftSet (fun z => (rd z).cut) T E C.walkStart
    · rw [Set.indicator_of_mem hx, Pi.one_apply]
      show (1 : ℝ) ≤ (undInc C rd T E x : ℝ)
      exact_mod_cast one_le_undInc C rd he hx
    · rw [Set.indicator_of_notMem hx]
      exact Nat.cast_nonneg _

/-- At success with neither weight past `1/a`, the sifts stop undecided on less than `u` of the
probes. -/
theorem siftBad_split (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hp8 : C.disagreementRate ≤ 1 / 8)
    (hdalt : C.disagreementAltRate = 7 / 8 * C.disagreementRate)
    (hθs4 : C.startUndecidedRate ≤ 1 / 4) (halt : C.startAltRate = 2 * C.startUndecidedRate)
    (hθe : 0 ≤ C.edgeUndecidedRate) (hφe : 0 ≤ C.edgeTraffic) {Xc : ℝ} (hXc : 0 ≤ Xc)
    (hX : ∀ j, C.edgeExcess j ≤ Xc * Real.log (1 / C.testLevel)) {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {u : ℝ}
    (hu : 2 * (3 * C.startUndecidedRate + C.maxLeaves * Fintype.card α * C.edgeTraffic
        + 2 * C.edgeUndecidedRate * ((L + 1) * C.maxLeaves))
      + 2 * C.disagreementRate / 7 * (2 * (L + 2) + 2 + C.maxLeaves * Fintype.card α * Xc
        + 2 * C.edgeUndecidedRate * ((L + 1) * C.maxLeaves)) ≤ u)
    {e : Level5Ending α} {s : Level5State α} (hb : SiftBad C D rd u e s) :
    SiftLowBad C D rd u L e s ∨ SiftReadBad C D rd u L e s := by
  by_contra hno
  simp only [not_or, SiftLowBad, SiftReadBad, hb, true_and, not_le] at hno
  obtain ⟨h1, h2⟩ := hno
  obtain ⟨-, ⟨hu0, he, hlv⟩, hr, hci⟩ := hb
  set a := C.testLevel
  set lg := Real.log (1 / a) with hlg
  have ha := hSq.a_pos
  have ha1 := hSq.a_lt
  have hlg0 : 0 < lg := Real.log_pos (by rw [lt_div_iff₀ ha]; linarith)
  have hlt : ∀ z : ℝ, a * Real.exp z < 1 → z < lg := by
    intro z hz
    rw [hlg, ← Real.exp_lt_exp, Real.exp_log (by positivity), lt_div_iff₀ ha]
    linarith
  set n : ℝ := (s.probes : ℝ)
  set μ := siftMean C D rd s.tree s.edges
  set U : ℝ := (undCount s : ℝ)
  set R : ℝ := (readSum s : ℝ)
  set η : ℝ := ((L : ℝ) + 1) * C.maxLeaves
  set MA : ℝ := (C.maxLeaves : ℝ) * Fintype.card α
  have hMA : 0 ≤ MA := by positivity
  have hη0 : 0 ≤ η := by positivity
  have hn0 : 0 ≤ n := Nat.cast_nonneg _
  -- the two weights below `1/a`
  have hηp : 0 < η := by
    have : (2 : ℝ) ≤ C.maxLeaves := by exact_mod_cast hL
    positivity
  have hW1 : (n * μ / 2 - U) / (2 * ((L : ℝ) + 2)) < lg :=
    hlt ((n * μ / 2 - U) / (2 * ((L : ℝ) + 2))) h1
  have hW2 : (R - 2 * n * η) / (2 * η) < lg := hlt ((R - 2 * n * η) / (2 * η)) h2
  rw [div_lt_iff₀ (by positivity)] at hW1 hW2
  -- the undecided count at success
  have hU := undCount_success_le C hr hci hSq.θs_pos hθs4 halt ha ha1.le hθe hφe hX
    (mul_nonneg hXc hlg0.le)
  have hK := leafEdges_card s.tree
  have hK' : ((leafEdges s.tree).card : ℝ) ≤ MA := by
    refine hK.trans ?_
    have : (s.tree.leaves.length : ℝ) ≤ C.maxLeaves := by exact_mod_cast hlv
    exact mul_le_mul_of_nonneg_right this (Nat.cast_nonneg _)
  -- success saw many probes
  obtain ⟨-, -, hsucc⟩ := runTests_success_quiet C hr
  rw [hdalt] at hsucc
  have hlong := success_long hSq.εd_pos hp8 ha hci.2.2.1 hsucc
  -- the mean count is at least `u`
  have hμ : u ≤ μ := hu0.trans (undSift_le_siftMean C D rd hlen he)
  set εd := C.disagreementRate
  set θs := C.startUndecidedRate
  set θe := C.edgeUndecidedRate
  set φe := C.edgeTraffic
  set Z : ℝ := 2 * ((L : ℝ) + 2) + 2 + MA * Xc + 2 * θe * η with hZ
  have hZ0 : 0 ≤ Z := by positivity
  have hlgZ : lg * Z ≤ εd * n / 7 * Z := mul_le_mul_of_nonneg_right (by linarith) hZ0
  have hKφ : ((leafEdges s.tree).card : ℝ) * (φe * n + Xc * lg) ≤ MA * (φe * n + Xc * lg) :=
    mul_le_mul_of_nonneg_right hK' (by positivity)
  have hθR : θe * R ≤ θe * (2 * n * η + 2 * η * lg) :=
    mul_le_mul_of_nonneg_left (by linarith) hθe
  have hnu : n * u ≤ n * μ := mul_le_mul_of_nonneg_left hμ hn0
  have hn : 0 < n := by
    have : 0 < εd * n := by linarith
    by_contra hc
    push Not at hc
    nlinarith [hSq.εd_pos]
  have key : n * u / 2 < n * (3 * θs + MA * φe + 2 * θe * η) + εd * n / 7 * Z := by
    have : n * u / 2 < n * (3 * θs + MA * φe + 2 * θe * η) + lg * Z := by
      rw [hZ]; nlinarith
    linarith
  have hu' := mul_le_mul_of_nonneg_left hu hn0
  have e1 : n * (2 * (3 * θs + MA * φe + 2 * θe * η) + 2 * εd / 7 * Z) / 2
      = n * (3 * θs + MA * φe + 2 * θe * η) + εd * n / 7 * Z := by ring
  have : n * (2 * (3 * θs + MA * φe + 2 * θe * η) + 2 * εd / 7 * Z) ≤ n * u := by
    convert hu' using 2
  linarith

theorem runEnds_sift (hSq : C.SeqOK) (hn₀ : 1 ≤ C.firstLook) (hexc : ∀ j, 0 < C.edgeExcess j)
    (u : ℝ) : ∀ (l : List (FreeMonoid α)) (s : Level5State α), RunInv C s →
      RunEnds (applyProbe C fun z => (rd z).cut) (fun e s' => e = .success
        ∧ u ≤ D.real (undSiftSet (fun z => (rd z).cut) s'.tree s'.edges C.walkStart)) s l →
      RunEnds (applyProbe C fun z => (rd z).cut) (SiftBad C D rd u) s l
  | [], _, _, h => h.elim
  | x :: xs, s, hs, h => by
    simp only [RunEnds] at h ⊢
    rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | ⟨e, s'⟩ <;> rw [hst] at h
    · exact runEnds_sift hSq hn₀ hexc u xs s' (runInv_step C _ hs hst) h
    · obtain ⟨rfl, hu⟩ := h
      rcases applyProbe_inr_shape C _ hSq.a_pos hSq.a_lt hn₀ hexc hst with h0 | ⟨rfl, hr⟩
      · cases h0
      · exact ⟨rfl, ⟨hu, hs.1.1, hs.1.2.1⟩, hr, countInv_keep C _ hs.2 x⟩

/-- A round ends in success at a hypothesis whose start or end sifts stop undecided on at least
`u` of the probes with chance at most `2 · versionBound · a`. -/
theorem sift_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (hp8 : C.disagreementRate ≤ 1 / 8)
    (hdalt : C.disagreementAltRate = 7 / 8 * C.disagreementRate)
    (hθs4 : C.startUndecidedRate ≤ 1 / 4) (halt : C.startAltRate = 2 * C.startUndecidedRate)
    (hθe : 0 ≤ C.edgeUndecidedRate) (hφe : 0 ≤ C.edgeTraffic) {Xc : ℝ} (hXc : 0 ≤ Xc)
    (hX : ∀ j, C.edgeExcess j ≤ Xc * Real.log (1 / C.testLevel)) {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {u : ℝ}
    (hu : 2 * (3 * C.startUndecidedRate + C.maxLeaves * Fintype.card α * C.edgeTraffic
        + 2 * C.edgeUndecidedRate * ((L + 1) * C.maxLeaves))
      + 2 * C.disagreementRate / 7 * (2 * (L + 2) + 2 + C.maxLeaves * Fintype.card α * Xc
        + 2 * C.edgeUndecidedRate * ((L + 1) * C.maxLeaves)) ≤ u) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (fun e s => e = .success
          ∧ u ≤ D.real (undSiftSet (fun z => (rd z).cut) s.tree s.edges C.walkStart))
          initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (2 * (versionBound C (Fintype.card α) * C.testLevel)) := by
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (applyProbe C fun z => (rd z).cut)
      (fun e s => e = .success
        ∧ u ≤ D.real (undSiftSet (fun z => (rd z).cut) s.tree s.edges C.walkStart))
      initialState (List.ofFn xs)}
      ⊆ {xs | RunEnds (applyProbe C fun z => (rd z).cut) (SiftLowBad C D rd u L) initialState
          (List.ofFn xs)}
        ∪ {xs | RunEnds (applyProbe C fun z => (rd z).cut) (SiftReadBad C D rd u L) initialState
          (List.ofFn xs)} := by
    intro xs hxs
    have h1 := runEnds_sift C D rd hSq hn₀ hexc u _ _ (runInv_start C hL) hxs
    exact runEnds_split _ _ (runEnds_mono (fun e s hb => siftBad_split C D rd hSq hL hp8 hdalt
      hθs4 halt hθe hφe hXc hX hlen hu hb) _ _ h1)
  have hva : 0 ≤ (versionBound C (Fintype.card α) : ℝ) * C.testLevel :=
    mul_nonneg (Nat.cast_nonneg _) hSq.a_pos.le
  refine (measure_mono hsub).trans ((measure_union_le _ _).trans ?_)
  rw [two_mul, ENNReal.ofReal_add hva hva]
  exact add_le_add (siftLow_le C D rd hSq hL hn₀ hexc hlen u T)
    (siftRead_le C D rd hSq hL hn₀ hexc hlen u T)

end Site

end OrthoDFA
