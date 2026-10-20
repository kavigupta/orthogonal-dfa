import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.CoveredStart
import OrthoDFA.Proofs.UniformStrings

/-!
# From a checked round to the learner's DFA

The probes are uniform over the lengths `k` to `L₁`. A probe of length `i` and its extension by a
letter that both agree, read nothing on a rarer side and stop undecided at neither their start's
sift nor their own, walk to the true
leaves of their states, so the edge between those leaves is learned and true (`edge_of_pair`).
Summed over the window, a state visited `c` of the time has every edge out of it crossed by such
pairs unless the disagreeing probes hold `c/((|Σ|+1)N)` of the draws (`edge_of_occupied`). A state the
probes visit is read on its true side once the reads are rarely wrong or undecided
(`sideOf_eq_of_mass`). Where every covered state's edges are true and it is read on its true side,
the hypothesis from the covered start's leaf follows the re-rooted target along every run that
stays covered (`classifies_of_matches`), so it misjudges at most what re-rooting does
(`classify_start_le`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Window

variable [Nonempty α]

/-- The average of `g` over the strings of length `ℓ`. -/
def uavg (ℓ : ℕ) (g : FreeMonoid α → ℝ) : ℝ :=
  (∑ v : List.Vector α ℓ, g (FreeMonoid.ofList v.toList)) / (Fintype.card α : ℝ) ^ ℓ

theorem integrable_uniformStrings (ℓ : ℕ) (g : FreeMonoid α → ℝ) :
    Integrable g (uniformStrings ℓ) :=
  (integrable_finsetSum_measure.2 fun _ _ => integrable_dirac (by simp)).smul_measure (by simp)

theorem integral_uniformStrings (ℓ : ℕ) (g : FreeMonoid α → ℝ) :
    ∫ x, g x ∂(uniformStrings ℓ) = uavg ℓ g := by
  unfold uniformStrings uavg
  rw [integral_smul_measure, integral_finsetSum_measure fun _ _ => integrable_dirac (by simp)]
  simp only [integral_dirac, ENNReal.toReal_inv, ENNReal.toReal_pow, ENNReal.toReal_natCast,
    smul_eq_mul]
  rw [div_eq_inv_mul]

theorem uniformStrings_real_uavg (ℓ : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings ℓ).real S = uavg ℓ (S.indicator 1) := by
  rw [← integral_uniformStrings, integral_indicator_one (MeasurableSpace.measurableSet_top)]

theorem uavg_nonneg (ℓ : ℕ) {g : FreeMonoid α → ℝ} (hg : ∀ x, 0 ≤ g x) : 0 ≤ uavg ℓ g :=
  div_nonneg (Finset.sum_nonneg fun _ _ => hg _) (by positivity)

theorem uavg_sum {ι : Type*} (ℓ : ℕ) (s : Finset ι) (g : ι → FreeMonoid α → ℝ) :
    uavg ℓ (fun x => ∑ i ∈ s, g i x) = ∑ i ∈ s, uavg ℓ (g i) := by
  unfold uavg
  rw [Finset.sum_comm, Finset.sum_div]

/-- The strings of length `j + t`, split after `j` letters. -/
def vecSplit (j t : ℕ) : List.Vector α (j + t) ≃ List.Vector α j × List.Vector α t where
  toFun v := (⟨v.toList.take j, by simp⟩, ⟨v.toList.drop j, by simp⟩)
  invFun w := ⟨w.1.toList ++ w.2.toList, by simp⟩
  left_inv v := by apply List.Vector.toList_injective; simp
  right_inv w := by
    obtain ⟨⟨w₁, h₁⟩, ⟨w₂, h₂⟩⟩ := w
    refine Prod.ext (List.Vector.toList_injective ?_) (List.Vector.toList_injective ?_)
    · simp [List.Vector.toList, ← h₁]
    · simp [List.Vector.toList, ← h₁]

theorem uavg_split (j t : ℕ) (g : FreeMonoid α → ℝ) :
    uavg (j + t) g = (∑ w : List.Vector α j, ∑ z : List.Vector α t,
      g (FreeMonoid.ofList (w.toList ++ z.toList))) / (Fintype.card α : ℝ) ^ (j + t) := by
  unfold uavg
  congr 1
  rw [← Fintype.sum_prod_type', ← (vecSplit j t).sum_comp]
  refine Fintype.sum_congr _ _ fun v => ?_
  rw [show ((vecSplit j t v).1.toList ++ (vecSplit j t v).2.toList) = v.toList from
    congrArg List.Vector.toList ((vecSplit j t).symm_apply_apply v)]

/-- A prefix of a uniform string is uniform, in average. -/
theorem uavg_prefix {j ℓ : ℕ} (h : j ≤ ℓ) (g : FreeMonoid α → ℝ) :
    uavg ℓ (fun x => g (prefixOf x j)) = uavg j g := by
  obtain ⟨t, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [uavg_split]
  have e : ∀ (w : List.Vector α j) (z : List.Vector α t),
      prefixOf (FreeMonoid.ofList (w.toList ++ z.toList)) j = FreeMonoid.ofList w.toList := by
    intro w z
    simp [prefixOf]
  simp only [e, Finset.sum_const, Finset.card_univ, card_vector, nsmul_eq_mul]
  unfold uavg
  rw [← Finset.mul_sum, pow_add]
  have : (0 : ℝ) < Fintype.card α ^ t := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  push_cast
  field_simp

theorem uavg_append (j t : ℕ) (h : FreeMonoid α → ℝ) :
    ∑ z : List.Vector α t, uavg j (fun y => h (y * FreeMonoid.ofList z.toList))
      = (Fintype.card α : ℝ) ^ t * uavg (j + t) h := by
  rw [uavg_split]
  unfold uavg
  rw [← Finset.sum_div, Finset.sum_comm, pow_add]
  have : (0 : ℝ) < Fintype.card α ^ t := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  simp only [← FreeMonoid.ofList_append]
  field_simp

theorem integrable_windowStrings {lo hi : ℕ} (h : lo ≤ hi) (g : FreeMonoid α → ℝ) :
    Integrable g (windowStrings lo hi) :=
  (integrable_finsetSum_measure.2 fun _ _ => integrable_uniformStrings _ g).smul_measure
    (ENNReal.inv_ne_top.2 (Nat.cast_ne_zero.2 (by omega)))

theorem integral_windowStrings {lo hi : ℕ} (h : lo ≤ hi) (g : FreeMonoid α → ℝ) :
    ∫ x, g x ∂(windowStrings lo hi)
      = (∑ n ∈ Finset.Icc lo hi, uavg n g) / ((hi : ℝ) + 1 - lo) := by
  unfold windowStrings
  rw [integral_smul_measure, integral_finsetSum_measure fun _ _ => integrable_uniformStrings _ g]
  simp only [integral_uniformStrings, ENNReal.toReal_inv, ENNReal.toReal_natCast, smul_eq_mul]
  rw [div_eq_inv_mul, Nat.cast_sub (by omega)]
  push_cast
  rfl

theorem windowStrings_isProbability {lo hi : ℕ} (h : lo ≤ hi) :
    IsProbabilityMeasure (windowStrings (α := α) lo hi) := by
  constructor
  unfold windowStrings
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  simp only [measure_univ, Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_one, smul_eq_mul]
  exact ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (by omega)) (ENNReal.natCast_ne_top _)

theorem windowStrings_real {lo hi : ℕ} (h : lo ≤ hi) (S : Set (FreeMonoid α)) :
    (windowStrings lo hi).real S
      = (∑ n ∈ Finset.Icc lo hi, (uniformStrings n).real S) / ((hi : ℝ) + 1 - lo) := by
  rw [← integral_indicator_one (MeasurableSpace.measurableSet_top), integral_windowStrings h]
  simp only [uniformStrings_real_uavg]

theorem windowStrings_length (lo hi : ℕ) :
    ∀ᵐ x ∂(windowStrings (α := α) lo hi), lo ≤ x.toList.length ∧ x.toList.length ≤ hi := by
  rw [ae_iff]
  unfold windowStrings
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  have : ∀ n ∈ Finset.Icc lo hi, uniformStrings (α := α) n
      {x | ¬ (lo ≤ x.toList.length ∧ x.toList.length ≤ hi)} = 0 := by
    intro n hn
    refine measure_mono_null (fun x hx => ?_) (uniformStrings_length n)
    simp only [Finset.mem_Icc] at hn
    simp only [Set.mem_setOf_eq] at hx ⊢
    omega
  rw [Finset.sum_eq_zero this, smul_zero]

/-- A prefix of a uniform string is uniform. -/
theorem uniformStrings_prefix {j n : ℕ} (h : j ≤ n) (S : Set (FreeMonoid α)) :
    (uniformStrings n).real {x | prefixOf x j ∈ S} = (uniformStrings j).real S := by
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, ← uavg_prefix h]
  rfl

theorem prefixMax_windowStrings_le {k lo hi : ℕ} (hk : k ≤ lo) (h : lo ≤ hi) :
    prefixMax (windowStrings (α := α) lo hi) k ≤ 1 / (Fintype.card α : ℝ) ^ k := by
  unfold prefixMax
  refine ciSup_le fun u => ?_
  have hlh : (lo : ℝ) ≤ hi := by exact_mod_cast h
  rw [windowStrings_real h, div_le_iff₀ (by linarith)]
  have hn : ∀ n ∈ Finset.Icc lo hi, (uniformStrings n).real {x : FreeMonoid α | prefixOf x k = u}
      ≤ 1 / (Fintype.card α : ℝ) ^ k := by
    intro n hn
    simp only [Finset.mem_Icc] at hn
    have := uniformStrings_prefix (α := α) (show k ≤ n by omega) {u}
    simp only [Set.mem_singleton_iff] at this
    rw [this]
    exact uniformStrings_singleton_le k u
  refine (Finset.sum_le_sum hn).trans (le_of_eq ?_)
  rw [Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, Nat.cast_sub (by omega)]
  push_cast
  ring

theorem uniformStrings_snoc_le (i : ℕ) (a : α) (S : Set (FreeMonoid α)) :
    (uniformStrings i).real {y | y * FreeMonoid.of a ∈ S}
      ≤ Fintype.card α * (uniformStrings (i + 1)).real S := by
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg]
  have h := uavg_append (α := α) i 1 (S.indicator 1)
  rw [pow_one] at h
  rw [← h]
  have e : uavg i ({y | y * FreeMonoid.of a ∈ S}.indicator 1)
      = uavg i (fun y => S.indicator 1 (y * FreeMonoid.ofList
        (a ::ᵥ List.Vector.nil).toList)) := by
    unfold uavg
    congr 1
  rw [e]
  exact Finset.single_le_sum (f := fun z : List.Vector α 1 =>
    uavg i (fun y => S.indicator 1 (y * FreeMonoid.ofList z.toList)))
    (fun z _ => uavg_nonneg i fun y => Set.indicator_nonneg (fun _ _ => zero_le_one) _)
    (Finset.mem_univ (a ::ᵥ List.Vector.nil))

/-- A covered state is visited `c` of the time, summed over the window. -/
theorem occupancy_of_covered {σ : Type*} (M : _root_.DFA α σ) {L : ℕ} {c : ℝ} {q : σ}
    (hq : q ∈ coveredStates M L c) :
    c ≤ ∑ i ∈ Finset.Icc ((L + 1) / 2) L, (uniformStrings i).real {y | M.eval y.toList = q} := by
  refine hq.trans ?_
  have hsub : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}
      ⊆ ⋃ j ∈ Finset.Icc ((L + 1) / 2) L, {x | prefixOf x j ∈ {y | M.eval y.toList = q}} := by
    rintro x ⟨j, h1, h2, h3⟩
    exact Set.mem_biUnion (Finset.mem_Icc.2 ⟨h1, h2⟩) h3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_)))
  exact uniformStrings_prefix (Finset.mem_Icc.1 hj).2 _

end Window

section Edges

variable {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ) (rd : FreeMonoid α → ARU)

omit [Fintype α] [DecidableEq α] in
theorem walk_snoc (E : Edges α) : ∀ (l : List α) (c : α) (p : List Bool),
    (walk E p (l ++ [c])).length = l.length + 2 →
      ∃ w, E ((walk E p l).getLast?.getD [], c) = some ((walk E p (l ++ [c])).getLast?.getD [], w)
  | [], c, p, h => by
    simp only [List.nil_append, walk] at h ⊢
    rcases he : E (p, c) with _ | ⟨q, y⟩ <;> simp only [he] at h ⊢
    · simp at h
    · exact ⟨y, he⟩
  | d :: l, c, p, h => by
    simp only [List.cons_append, walk] at h ⊢
    rcases he : E (p, d) with _ | ⟨q, y⟩ <;> simp only [he] at h ⊢
    · simp at h
    · obtain ⟨w, hw⟩ := walk_snoc E l c q (by simpa using h)
      obtain ⟨l₁, hl₁⟩ := walk_cons E q l
      obtain ⟨l₂, hl₂⟩ := walk_cons E q (l ++ [c])
      refine ⟨w, ?_⟩
      rw [hl₁, hl₂] at hw ⊢
      simpa [List.getLast?_cons] using hw

theorem cut_none_iff (v : ARU) : v.cut = none ↔ v = .undecided := by
  cases v <;> simp [ARU.cut]

theorem walk_facts_full (E : Edges α) (k : ℕ) (x : FreeMonoid α) {p₀ : List Bool}
    (hk : k ≤ x.toList.length) :
    let ss := walk E p₀ (x.toList.drop k)
    0 < ss.length ∧ k + ss.length - 1 ≤ x.toList.length ∧ ss.getD 0 [] = p₀
      ∧ (∀ c, x.toList[k + ss.length - 1]? = some c → E (ss.getD (ss.length - 1) [], c) = none)
      ∧ (x.toList[k + ss.length - 1]? = none → k + ss.length - 1 = x.toList.length) := by
  intro ss
  have hpos : 0 < ss.length := length_walk_pos E p₀ _
  have hle : ss.length ≤ (x.toList.drop k).length + 1 := length_walk_le E p₀ _
  simp only [List.length_drop] at hle
  refine ⟨hpos, by omega, walk_getD_zero E p₀ _, fun c hc => ?_, fun hc => ?_⟩
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

/-- A probe neither searching nor stopping undecided at its start or end sift walks, learned all
the way, from its start's sift to its own. -/
theorem agree_of_unchecked (H : HypothesisDFA α) {k : ℕ} {x : FreeMonoid α}
    (hk : k ≤ x.toList.length) (hx : x ∉ Random.searchSet rd k H)
    (hu : x ∉ Random.endsSet rd k H) :
    ∃ p e, H.tree.sift rd (prefixOf x k) = .inl p
      ∧ (walk H.edges p (x.toList.drop k)).length = (x.toList.drop k).length + 1
      ∧ H.tree.sift rd x = .inl e ∧ (walk H.edges p (x.toList.drop k)).getLast? = some e := by
  simp only [Random.searchSet, Random.endsSet, Set.mem_ofPred_eq] at hx hu
  push Not at hu
  rcases hp : H.tree.sift rd (prefixOf x k) with p | z
  swap
  · exfalso
    have : probeOrAgree rd H.tree H.edges k x = .undecidedStart z := by
      simp [probeOrAgree, probe, hp, show ¬ x.toList.length < k by omega]
    rw [this] at hu
    simp [Random.endsString] at hu
  obtain ⟨hpos, hhi, hst0, -, hnone⟩ := walk_facts_full H.edges k x (p₀ := p) hk
  have hlo : H.tree.sift rd (prefixOf x k)
      = .inl ((walk H.edges p (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, hp]
  rcases Random.probe_long_cases rd H.tree H.edges k x hk hp rfl with
    ⟨c, z, -, -, h⟩ | ⟨un, -, hu2, h'⟩
  · rw [h] at hu; simp [Random.endsString] at hu
  rcases Random.searchOutcome_cases rd H.tree k x _ un (by omega) hhi hlo h' with
    ⟨z, -, h''⟩ | ⟨hsh, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, -, -, -, h''⟩⟩
  · rw [h''] at hu; simp [Random.endsString] at hu
  swap
  · rw [h''] at hx; simp at hx
  swap
  · rw [h''] at hx; simp at hx
  rcases un with _ | ⟨c, t⟩
  swap
  · simp only [searchOutcome] at h''
    rw [h''] at hx; simp at hx
  have hend := hnone (hu2 rfl)
  have hlen : (walk H.edges p (x.toList.drop k)).length = (x.toList.drop k).length + 1 := by
    simp only [List.length_drop]; omega
  refine ⟨p, (walk H.edges p (x.toList.drop k)).getD
    (k + (walk H.edges p (x.toList.drop k)).length - 1 - k) [], rfl, hlen, ?_, ?_⟩
  · rwa [show prefixOf x (k + (walk H.edges p (x.toList.drop k)).length - 1) = x by
      rw [hend, prefixOf_length]] at hsh
  · rw [List.getLast?_eq_getElem?, List.getD_eq_getElem?_getD,
      show k + (walk H.edges p (x.toList.drop k)).length - 1 - k
        = (walk H.edges p (x.toList.drop k)).length - 1 by omega,
      List.getElem?_eq_getElem (by omega)]
    rfl

/-- A probe and its extension by a letter, neither disagreeing, cross a learned edge between
their states' true leaves. -/
theorem edge_of_pair (H : HypothesisDFA α) {k : ℕ} {y : FreeMonoid α} (a : α)
    (hk : k ≤ y.toList.length) (hy : y ∉ disagreeSet M dist rd k H)
    (hya : y * FreeMonoid.of a ∉ disagreeSet M dist rd k H) :
    ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.eval y.toList), a)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree
        (M.eval (y * FreeMonoid.of a).toList), w) := by
  have hlen : (y * FreeMonoid.of a).toList.length = y.toList.length + 1 := by simp
  simp only [disagreeSet, Set.mem_union, not_or] at hy hya
  obtain ⟨p, e, hp, hf, he, hl⟩ := agree_of_unchecked rd H hk hy.1.1 hy.1.2
  obtain ⟨p', e', hp', hf', he', hl'⟩ :=
    agree_of_unchecked rd H (by omega) hya.1.1 hya.1.2
  have hpre : prefixOf (y * FreeMonoid.of a) k = prefixOf y k := by
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.take_append_of_le_length hk]
  rw [hpre, hp] at hp'
  cases hp'
  have hdrop : (y * FreeMonoid.of a).toList.drop k = y.toList.drop k ++ [a] := by
    simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.drop_append_of_le_length hk]
  rw [hdrop] at hf' hl'
  obtain ⟨w, hw⟩ := walk_snoc H.edges _ a p (by rw [hf']; simp)
  have hclean : ∀ z : FreeMonoid α, z ∉ rarerSet M dist rd k H.tree →
      k ≤ z.toList.length → ∀ m ∈ Random.mids H.tree,
        rd (z * m) ≠ if readSide (dist (M.eval (z * m).toList)) then .reject else .accept :=
    fun z hz hkz m hm hwr => hz ⟨hkz, m, hm, hwr⟩
  have h1 := Random.Eps.sift_true' M (fun q => readSide (dist q)) rd he (hclean y hy.2 hk)
  have h2 := Random.Eps.sift_true' M (fun q => readSide (dist q)) rd he'
    (hclean _ hya.2 (by omega))
  rw [hl, hl'] at hw
  rw [h1, h2]
  exact ⟨w, hw⟩

variable [Nonempty α]

/-- A state visited `c` of the time over the window's lengths below `J` has every edge out of its
true leaf learned and true, unless the disagreeing probes hold `c/((|Σ|+1)N)` of the draws. -/
theorem edge_of_occupied (H : HypothesisDFA α) {k J L₁ : ℕ} (hkJ : k < J) (hJ : J ≤ L₁) (q : σ) (a : α)
    {c β : ℝ}
    (hbad : (windowStrings k L₁).real (disagreeSet M dist rd k H) ≤ β)
    (hocc : c ≤ ∑ i ∈ Finset.Ico k J, (uniformStrings i).real {y | M.eval y.toList = q})
    (hβ : ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * β < c) :
    ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree q, a)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.step q a), w) := by
  set Bad := disagreeSet M dist rd k H
  set A : ℝ := (Fintype.card α : ℝ) with hA
  by_contra hno
  have hkL : k ≤ L₁ := by omega
  have hU0 : ∀ i, 0 ≤ (uniformStrings (α := α) i).real Bad := fun _ => measureReal_nonneg
  have hi : ∀ i ∈ Finset.Ico k J, (uniformStrings i).real {y | M.eval y.toList = q}
      ≤ (uniformStrings i).real Bad + A * (uniformStrings (i + 1)).real Bad := by
    intro i hi
    simp only [Finset.mem_Ico] at hi
    have hsub : {y : FreeMonoid α | M.eval y.toList = q}
        ⊆ (Bad ∪ {y | y * FreeMonoid.of a ∈ Bad}) ∪ {y | y.toList.length ≠ i} := by
      intro y hy
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc hy
      apply hno
      obtain ⟨w, hw⟩ := edge_of_pair M dist rd H a (by omega) hc.1.1 hc.1.2
      rw [FreeMonoid.toList_mul, FreeMonoid.toList_of, DFA.eval_append_singleton, hy] at hw
      exact ⟨w, hw⟩
    have hnull : (uniformStrings (α := α) i).real {y | y.toList.length ≠ i} = 0 := by
      rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
    refine (measureReal_union_le _ _).trans ?_
    rw [hnull, add_zero]
    refine (measureReal_union_le _ _).trans (add_le_add le_rfl ?_)
    exact uniformStrings_snoc_le i a Bad
  have hsum := Finset.sum_le_sum hi
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have h1 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i _ _ => hU0 i
  have h2 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) (i + 1)).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad := by
    have e := Finset.sum_image (f := fun i => (uniformStrings (α := α) i).real Bad)
      (s := Finset.Ico k J) (g := (· + 1)) (fun x _ y _ h => by simpa using h)
    rw [← e]
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_image, Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega)
      fun i _ _ => hU0 i
  have hw := windowStrings_real (α := α) hkL Bad
  have hN : 0 < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    linarith
  rw [eq_div_iff hN.ne'] at hw
  have hA0 : 0 ≤ A := by positivity
  have h3 := mul_le_mul_of_nonneg_left hbad hN.le
  have h4 := mul_le_mul_of_nonneg_left h2 hA0
  nlinarith

end Edges

section Side

open scoped Classical

variable {σ : Type*}

/-- A state the probes visit `b` of the time is read more often on its true side than on the
other, once the reads are wrong at most `εr` and undecided at most `εu` on average and
`2εr + εu < b`. -/
theorem sideOf_eq_of_mass (M : _root_.DFA α σ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (dist : σ → ARU → ℝ) (h0 : ∀ r v, 0 ≤ dist r v)
    (hle1 : ∀ r v, dist r v ≤ 1) (q : σ)
    (hsum : dist q .accept + dist q .reject + dist q .undecided = 1) {b εr εu : ℝ}
    (hmis : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εr)
    (hund : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εu)
    (hocc : b ≤ D.real {p | M.eval p.toList = q}) (hb : 2 * εr + εu < b) :
    readSide (dist q) = decide (q ∈ M.accept) := by
  have hint : ∀ g : FreeMonoid α → ℝ, (∀ p, 0 ≤ g p) → (∀ p, g p ≤ 1) → Integrable g D :=
    fun g g0 g1 => Integrable.of_bound (measurable_of_countable g).aestronglyMeasurable 1
      (ae_of_all _ fun p => by rw [Real.norm_eq_abs, abs_of_nonneg (g0 p)]; exact g1 p)
  have hlow : ∀ g : FreeMonoid α → ℝ, (∀ p, 0 ≤ g p) → (∀ p, g p ≤ 1) → ∀ c : ℝ,
      (∀ p, M.eval p.toList = q → c ≤ g p) → 0 ≤ c →
        D.real {p | M.eval p.toList = q} * c ≤ ∫ p, g p ∂D := by
    intro g g0 g1 c hc c0
    have hS : MeasurableSet {p : FreeMonoid α | M.eval p.toList = q} :=
      MeasurableSpace.measurableSet_top
    rw [← smul_eq_mul, ← integral_indicator_const c hS]
    refine integral_mono (hint _ (fun p => Set.indicator_nonneg (fun _ _ => c0) p)
      (fun p => ?_)) (hint g g0 g1) fun p => ?_
    · by_cases hp : p ∈ {p : FreeMonoid α | M.eval p.toList = q}
      · rw [Set.indicator_of_mem hp]; exact (hc p hp).trans (g1 p)
      · rw [Set.indicator_of_notMem hp]; norm_num
    · by_cases hp : p ∈ {p : FreeMonoid α | M.eval p.toList = q}
      · rw [Set.indicator_of_mem hp]; exact hc p hp
      · rw [Set.indicator_of_notMem hp]; exact g0 p
  have hw0 : ∀ p : FreeMonoid α, 0 ≤ (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) := fun p => by split_ifs <;> exact h0 _ _
  have hw1 : ∀ p : FreeMonoid α, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ≤ 1 := fun p => by split_ifs <;> exact hle1 _ _
  have hD0 : 0 ≤ D.real {p : FreeMonoid α | M.eval p.toList = q} := measureReal_nonneg
  have hu := hlow (fun p => dist (M.eval p.toList) .undecided) (fun p => h0 _ _)
    (fun p => hle1 _ _) (dist q .undecided) (fun p hp => by rw [hp]) (h0 _ _)
  have hb0 : 0 < b := by
    have := (integral_nonneg hw0).trans hmis
    have := (integral_nonneg (fun p : FreeMonoid α => h0 (M.eval p.toList) .undecided)).trans hund
    linarith
  have hsq := hsum
  have hacc := h0 q .accept
  have hrej := h0 q .reject
  have hund0 := h0 q .undecided
  have hbu := mul_le_mul_of_nonneg_right hocc hund0
  by_cases hq : q ∈ M.accept
  · have hm := hlow _ hw0 hw1 (dist q .reject) (fun p hp => by rw [if_pos (hp ▸ hq), hp]) hrej
    have hbr := mul_le_mul_of_nonneg_right hocc hrej
    simp only [readSide, hq, decide_true, decide_eq_true_eq]
    by_contra hlt
    push Not at hlt
    nlinarith
  · have hm := hlow _ hw0 hw1 (dist q .accept) (fun p hp => by rw [if_neg (hp ▸ hq), hp]) hacc
    have hba := mul_le_mul_of_nonneg_right hocc hacc
    simp only [readSide, hq, decide_false, decide_eq_false_iff_not]
    intro hlt
    nlinarith

end Side


end OrthoDFA

end
