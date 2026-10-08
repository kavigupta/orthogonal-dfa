import OrthoDFA.Proofs.EdgeGap
import OrthoDFA.Proofs.PassReads
import OrthoDFA.Proofs.Adaptive

/-!
# The gate fails only through flips

With every state read cleanly and every visited edge read on its edge, a draw the DFA/DT check
counts against the hypothesis has a middle reading off its likelier side on one of its prefixes'
paths (`flip_of_disagree`).  Through a vote family no suffix of which ends another, that reading
either is one the pass made, and the pass's reads flip with chance at most `passReadBound·φ` one
probe at a time (`passFlip_le`), or it is fresh given the pass, and Markov bounds the mass of
draws with such a flip (`freshFlip_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

theorem SuffixFree.eq_of_mul {V : Finset (FreeMonoid α)} (hV : SuffixFree V)
    {z y v v' : FreeMonoid α} (hv : v ∈ V) (hv' : v' ∈ V) (h : z * v = y * v') : z = y := by
  have hl : z.toList ++ v.toList = y.toList ++ v'.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul, h]
  rcases List.append_eq_append_iff.1 hl with ⟨a, hy, hva⟩ | ⟨b, hz, hvb⟩
  · have h1 := hV v hv v' hv' (FreeMonoid.ofList a)
      (FreeMonoid.toList.injective (by simp [hva]))
    have ha : a = [] := by simpa using congrArg FreeMonoid.toList h1
    exact FreeMonoid.toList.injective (by rw [hy, ha, List.append_nil])
  · have h1 := hV v' hv' v hv (FreeMonoid.ofList b)
      (FreeMonoid.toList.injective (by simp [hvb]))
    have hb : b = [] := by simpa using congrArg FreeMonoid.toList h1
    exact FreeMonoid.toList.injective (by rw [hz, hb, List.append_nil])

/-- The oracle's bits a read of each string in `Y` through `V` asks. -/
noncomputable def vBits (V Y : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  Y.biUnion fun y => V.image (y * ·)

theorem disjoint_vBits {V W Y : Finset (FreeMonoid α)} (hV : SuffixFree V) (hW : W ⊆ V)
    {z : FreeMonoid α} (hz : z ∉ Y) :
    Disjoint (vBits V Y : Set (FreeMonoid α)) (W.image (z * ·) : Set (FreeMonoid α)) := by
  rw [Set.disjoint_left]
  intro x hx hx'
  simp only [vBits, Finset.coe_biUnion, Finset.coe_image, Set.mem_iUnion, Set.mem_image,
    Finset.mem_coe] at hx hx'
  obtain ⟨y, hy, v, hv, rfl⟩ := hx
  obtain ⟨w, hw, hwz⟩ := hx'
  exact hz (hV.eq_of_mul (hW hw) hv hwz ▸ hy)

/-- Every bit of the oracle is genuinely `0` or `1`. -/
def cleanAll (O : Oracle μ (FreeMonoid α)) : Set Ω :=
  {ω | ∀ s, O.noise s ω = 0 ∨ O.noise s ω = 1}

theorem measurableSet_cleanAll (O : Oracle μ (FreeMonoid α)) : MeasurableSet (cleanAll O) := by
  have : cleanAll O = ⋂ s, O.noise s ⁻¹' ({0, 1} : Set ℝ) := by
    ext ω; simp [cleanAll]
  rw [this]
  exact MeasurableSet.iInter fun s => O.noise_meas s
    ((measurableSet_singleton 1).insert 0)

theorem measure_cleanAll_compl (O : Oracle μ (FreeMonoid α)) : μ (cleanAll O)ᶜ = 0 := by
  have : (cleanAll O)ᶜ = ⋃ s, {ω | ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)} := by
    ext ω; simp [cleanAll]
  rw [this]
  exact measure_iUnion_null fun s => ae_iff.1 (O.noise_bit s)

/-- A block of the oracle's bits `T ω`, chosen by those bits (`hTZ`), and a set of strings `Z ω`
they decide: the chance some string of `Z` off `T` meets an event of its own fresh bits is at
most `M·φ`, `M` bounding how many there are.  The cells of `T`'s value and its bits' pattern
are disjoint, decide `Z`, and are independent of any string's bits off `T`. -/
theorem cell_bound [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {V : Finset (FreeMonoid α)} (hV : SuffixFree V) (T Z : Ω → Finset (FreeMonoid α))
    (hTZ : ∀ ω ω', (∀ x ∈ vBits V (T ω), O.noise x ω = O.noise x ω') →
      T ω' = T ω ∧ Z ω' = Z ω)
    {M : ℕ} (hM : ∀ ω, (Z ω \ T ω).card ≤ M) {W : Finset (FreeMonoid α)} (hW : W ⊆ V)
    (Fl : FreeMonoid α → Set Ω)
    (hFl : ∀ z, MeasurableSet[noiseAlg O ↑(W.image (z * ·))] (Fl z)) {φ : ℝ≥0∞}
    (hφ : ∀ z, μ (Fl z) ≤ φ) :
    μ {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z} ≤ M * φ := by
  classical
  set PC : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    {ω | noisePattern O (vBits V c.1) ω = c.2} ∩ noiseClean O (vBits V c.1) with hPC
  set C : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    PC c ∩ cleanAll O ∩ {ω | T ω = c.1} with hCdef
  have hagree : ∀ c ω ω', ω ∈ C c → ω' ∈ PC c →
      ∀ x ∈ vBits V (T ω), O.noise x ω = O.noise x ω' := by
    intro c ω ω' hω hω' x hx
    obtain ⟨⟨⟨hp, hc⟩, -⟩, hT⟩ := hω
    rw [show T ω = c.1 from hT] at hx
    exact noise_eq_of_pattern O hc hω'.2 (hp.trans hω'.1.symm) x hx
  have hC_eq : ∀ c, (C c).Nonempty → C c = PC c ∩ cleanAll O := by
    intro c ⟨ω₀, h₀⟩
    refine Set.Subset.antisymm (fun ω hω => hω.1) fun ω hω => ⟨hω, ?_⟩
    have := (hTZ ω₀ ω (hagree c ω₀ ω h₀ hω.1)).1
    exact this.trans h₀.2
  set rep : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Finset (FreeMonoid α) := fun c =>
    if h : (C c).Nonempty then Z h.some else ∅ with hrep
  have hZ_eq : ∀ c ω, ω ∈ C c → Z ω = rep c := by
    intro c ω hω
    have hne : (C c).Nonempty := ⟨ω, hω⟩
    simp only [hrep, dif_pos hne]
    exact (hTZ _ ω (hagree c _ ω hne.some_mem hω.1.1)).2
  have hrep_card : ∀ c, (rep c \ c.1).card ≤ M := by
    intro c
    by_cases hne : (C c).Nonempty
    · simp only [hrep, dif_pos hne]
      have : T hne.some = c.1 := hne.some_mem.2
      rw [← this]
      exact hM _
    · simp [hrep, dif_neg hne]
  have hmeasPC : ∀ c, MeasurableSet (PC c) := fun c =>
    noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _))
  have hmeasC : ∀ c, MeasurableSet (C c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · rw [hC_eq c hne]; exact (hmeasPC c).inter (measurableSet_cleanAll O)
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; exact MeasurableSet.empty
  have hdisj : Pairwise (Function.onFun Disjoint C) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have h1 : c.1 = c'.1 := (show T ω = c.1 from hω.2).symm.trans (show T ω = c'.1 from hω'.2)
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O (vBits V c.1) ω = c.2 := hω.1.1.1
      have b : noisePattern O (vBits V c'.1) ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hcover : {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z}
      ⊆ (cleanAll O)ᶜ ∪ ⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z) := by
    intro ω ⟨z, hz, hzT, hzF⟩
    by_cases hcl : ω ∈ cleanAll O
    · right
      set c := (T ω, noisePattern O (vBits V (T ω)) ω)
      have hωC : ω ∈ C c := ⟨⟨⟨rfl, fun x _ => hcl x⟩, hcl⟩, rfl⟩
      simp only [Set.mem_iUnion]
      refine ⟨c, z, ?_, hωC, hzF⟩
      rw [← hZ_eq c ω hωC]
      exact Finset.mem_sdiff.2 ⟨hz, hzT⟩
    · exact Or.inl hcl
  have hcell : ∀ c, μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) ≤ M * φ * μ (C c) := by
    intro c
    calc μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z))
        ≤ ∑ z ∈ rep c \ c.1, μ (C c ∩ Fl z) := measure_biUnion_finset_le _ _
      _ ≤ ∑ _z ∈ rep c \ c.1, φ * μ (C c) := by
          refine Finset.sum_le_sum fun z hz => ?_
          by_cases hne : (C c).Nonempty
          · have hzT := (Finset.mem_sdiff.1 hz).2
            have hind := indep_noiseAlg O (disjoint_vBits hV hW hzT)
            have hPCm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] (PC c) :=
              (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
            calc μ (C c ∩ Fl z) ≤ μ (PC c ∩ Fl z) :=
                  measure_mono (Set.inter_subset_inter_left _ fun ω hω => hω.1.1)
              _ = μ (PC c) * μ (Fl z) := (Indep_iff _ _ μ).1 hind _ _ hPCm (hFl z)
              _ ≤ μ (PC c) * φ := by gcongr; exact hφ z
              _ = φ * μ (C c) := by
                  rw [mul_comm, hC_eq c hne, measure_inter_conull (measure_cleanAll_compl O)]
          · rw [Set.not_nonempty_iff_eq_empty.1 hne]
            simp
      _ = (rep c \ c.1).card * (φ * μ (C c)) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ M * (φ * μ (C c)) := by
          gcongr
          exact_mod_cast hrep_card c
      _ = M * φ * μ (C c) := by ring
  calc μ {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z}
      ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_mono hcover
    _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_union_le _ _
    _ = μ (⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := by
        rw [measure_cleanAll_compl O, zero_add]
    _ ≤ ∑' c, μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_iUnion_le _
    _ ≤ ∑' c, M * φ * μ (C c) := ENNReal.tsum_le_tsum hcell
    _ = M * φ * ∑' c, μ (C c) := ENNReal.tsum_mul_left
    _ = M * φ * μ (⋃ c, C c) := by rw [measure_iUnion hdisj hmeasC]
    _ ≤ M * φ * 1 := by gcongr; exact prob_le_one
    _ = M * φ := mul_one _

/-- The middle reading of `z` lands off its likelier side. -/
def flipAt (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (z : FreeMonoid α) : Set Ω :=
  {ω | decide (midRead O B F ω z) ≠ majRead O B F z}

theorem measurableSet_midRead_noise [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (z : FreeMonoid α) :
    MeasurableSet[noiseAlg O ↑(F.image (z * ·))] {ω | midRead O B F ω z} := by
  classical
  exact measurableSet_filter_pred_map O (T := ↑(F.image (z * ·))) (A := F) (z * ·)
    (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv))
    (fun U => B.lo + B.hi < 2 * U.card)

theorem measurableSet_flipAt [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (z : FreeMonoid α) :
    MeasurableSet[noiseAlg O ↑(F.image (z * ·))] (flipAt O B F z) := by
  have hm := measurableSet_midRead_noise O B F z
  cases h : majRead O B F z
  · convert hm using 1
    ext ω; simp [flipAt, h]
  · convert hm.compl using 1
    ext ω; simp [flipAt, h]

theorem edgeDisagreeProb_ge_of_ne [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ}
    (hflip : MidFlipPremise A O B F uHi φ) (H : Hypothesis α) (x : FreeMonoid α) (c : α)
    (hclean : ∀ z, stateIndecision A O B F (A.state z) < uHi)
    (hm : majPath O B F H (x * FreeMonoid.of c) ≠ H.step (majPath O B F H x) c) :
    1 - ((majRoute O B F H x).length + (majRoute O B F H (x * FreeMonoid.of c)).length) * φ
      ≤ edgeDisagreeProb O B F H x c := by
  set bad := {ω | midPath (readsAt O B F ω) H x ≠ majPath O B F H x}
    ∪ {ω | midPath (readsAt O B F ω) H (x * FreeMonoid.of c)
      ≠ majPath O B F H (x * FreeMonoid.of c)}
  have hbad : μ.real bad
      ≤ ((majRoute O B F H x).length + (majRoute O B F H (x * FreeMonoid.of c)).length) * φ := by
    refine (measureReal_union_le _ _).trans ?_
    rw [add_mul]
    exact add_le_add (midPath_ne_majPath_le hflip H x fun z _ => hclean z)
      (midPath_ne_majPath_le hflip H _ fun z _ => hclean z)
  have hsub : badᶜ ⊆ {ω | midPath (readsAt O B F ω) H (x * FreeMonoid.of c)
      ≠ H.step (midPath (readsAt O B F ω) H x) c} := fun ω hω => by
    simp only [bad, Set.mem_compl_iff, Set.mem_union, Set.mem_ofPred_eq, not_or, not_not] at hω
    simp only [Set.mem_ofPred_eq]
    rw [hω.1, hω.2]
    exact hm
  have h1 : (1 : ℝ) ≤ μ.real bad + μ.real badᶜ := by
    have hu : μ.real (Set.univ : Set Ω) = 1 := by simp
    rw [← hu, ← Set.union_compl_self bad]
    exact measureReal_union_le _ _
  have h2 := measureReal_mono (μ := μ) hsub (measure_ne_top μ _)
  unfold edgeDisagreeProb
  linarith

/-- With every state read cleanly, an edge read on it but for `η` is the likelier readings'. -/
theorem majPath_step [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ η : ℝ} {d : ℕ}
    (hflip : MidFlipPremise A O B F uHi φ) (H : Hypothesis α)
    (hclean : ∀ z, stateIndecision A O B F (A.state z) < uHi) (hdepth : H.tree.depth ≤ d)
    (hφ : 0 ≤ φ) (hη : η < 1 - 2 * d * φ) {x : FreeMonoid α} {c : α}
    (h : edgeDisagreeProb O B F H x c ≤ η) :
    majPath O B F H (x * FreeMonoid.of c) = H.step (majPath O B F H x) c := by
  by_contra hne
  have hge := edgeDisagreeProb_ge_of_ne hflip H x c hclean hne
  have h1 := (DTree.route_length_le_depth (fun y => some (majRead O B F y)) H.tree x).trans hdepth
  have h2 := (DTree.route_length_le_depth (fun y => some (majRead O B F y)) H.tree
    (x * FreeMonoid.of c)).trans hdepth
  have hl : (((majRoute O B F H x).length
      + (majRoute O B F H (x * FreeMonoid.of c)).length : ℕ) : ℝ) ≤ 2 * d := by
    unfold majRoute
    push_cast
    have : ((H.tree.route (fun y => some (majRead O B F y)) x).1.length : ℝ) ≤ d := by
      exact_mod_cast h1
    have : ((H.tree.route (fun y => some (majRead O B F y))
        (x * FreeMonoid.of c)).1.length : ℝ) ≤ d := by exact_mod_cast h2
    linarith
  have := mul_le_mul_of_nonneg_right hl hφ
  push_cast at this
  linarith

theorem prefixOf_succ {x : FreeMonoid α} {j : ℕ} (hj : j < x.toList.length) :
    prefixOf x (j + 1) = prefixOf x j * FreeMonoid.of (x.toList[j]) := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList, FreeMonoid.toList_of,
    List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some]

/-- A draw the DFA/DT check counts against the hypothesis, every edge of whose walk the likelier
readings follow, has a middle reading off its likelier side on some prefix's path. -/
theorem flip_of_disagree {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)}
    (H : Hypothesis α) (ω : Ω) {x : FreeMonoid α}
    (hstep : ∀ j (hj : j < x.toList.length),
      majPath O B F H (prefixOf x (j + 1)) = H.step (majPath O B F H (prefixOf x j)) x.toList[j])
    (hdis : DFAandDTDisagree (readsAt O B F ω) H x) :
    ∃ j ≤ x.toList.length, ∃ z ∈ majRoute O B F H (prefixOf x j), ω ∈ flipAt O B F z := by
  by_contra hno
  simp only [not_exists, not_and, flipAt, Set.mem_ofPred_eq, ne_eq, not_not] at hno
  have hmid : ∀ j ≤ x.toList.length,
      midPath (readsAt O B F ω) H (prefixOf x j) = majPath O B F H (prefixOf x j) :=
    fun j hj => midPath_eq_majPath fun z hz => hno j hj z hz
  have hfold : ∀ j ≤ x.toList.length,
      (x.toList.take j).foldl H.step (majPath O B F H (prefixOf x 0))
        = majPath O B F H (prefixOf x j) := by
    intro j hj
    induction j with
    | zero => simp
    | succ j ih =>
      rw [List.take_add_one, List.getElem?_eq_getElem (by omega), Option.toList_some,
        List.foldl_append, List.foldl_cons, List.foldl_nil, ih (by omega), hstep j (by omega)]
  apply hdis
  have h0 : midPath (readsAt O B F ω) H 1 = majPath O B F H (prefixOf x 0) := by
    rw [← hmid 0 (Nat.zero_le _), prefixOf_zero]
  rw [h0]
  have := hfold x.toList.length le_rfl
  rw [List.take_length] at this
  rw [this, ← hmid _ le_rfl, prefixOf_length]

theorem flipAt_le [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ}
    (hflip : MidFlipPremise A O B F uHi φ)
    (hclean : ∀ z, stateIndecision A O B F (A.state z) < uHi) (z : FreeMonoid α) :
    μ (flipAt O B F z) ≤ ENNReal.ofReal φ := by
  rw [← ofReal_measureReal (measure_ne_top μ _)]
  exact ENNReal.ofReal_le_ofReal (flip_le hflip (hclean z))

theorem card_sdiff_phase (K : StageKnobs α) (R : CutReads α) (seed ws : List (FreeMonoid α))
    (Bs : Finset (FreeMonoid α)) (k : ℕ) :
    (nodeReads Bs (phase K R seed ws (k + 1)).tree \ nodeReads Bs (phase K R seed ws k).tree).card
      ≤ Bs.card * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1)) := by
  obtain ⟨d, hd⟩ := phase_mids_new K R seed ws k
  refine (Finset.card_le_card fun y hy => ?_).trans (card_nodeReads Bs (.node d .leaf .leaf))
  obtain ⟨hy, hyn⟩ := Finset.mem_sdiff.1 hy
  simp only [nodeReads, Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_image,
    List.mem_toFinset] at hy
  obtain ⟨b, hb, e₁, e₂, m, hm, rfl⟩ := hy
  rcases hd m hm with hm' | rfl
  · exact absurd (mem_nodeReads e₁ e₂ hb hm') hyn
  · exact mem_nodeReads e₁ e₂ hb (by simp [DTree.mids])

section Round

variable [IsProbabilityMeasure μ]

/-- The round's pass reads some string whose middle reading lands off its likelier side with
chance at most `(N + 1)·|bases|·(|α| + 1)²·φ`: probe by probe, the strings the pass first reads
against the tree after `k` probes are chosen by the bits it read before, and are fresh. -/
theorem passFlip_le (S : RoundSetting α μ Q) {uHi φ : ℝ}
    (hV : SuffixFree (S.F ∪ S.K.train S.F)) (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ)
    (hclean : ∀ z, stateIndecision S.A S.O S.B S.F (S.A.state z) < uHi)
    (p : Fin S.N → FreeMonoid α) (hp : ∀ i, (p i).toList.length ≤ S.L) :
    μ {ω | ∃ z ∈ nodeReads (passBases S.seed (List.ofFn p) S.L)
        (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree, ω ∈ flipAt S.O S.B S.F z}
      ≤ (((S.N + 1) * ((passBases S.seed (List.ofFn p) S.L).card
          * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1))) : ℕ) : ℝ≥0∞)
        * ENNReal.ofReal φ := by
  classical
  set Bs := passBases S.seed (List.ofFn p) S.L
  set M := Bs.card * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1))
  set ph : Ω → ℕ → PassState α := fun ω k =>
    phase S.K (rd S.B S.F fun w => S.O.mq w ω) S.seed (List.ofFn p) k with hph
  set T : ℕ → Ω → Finset (FreeMonoid α) := fun k ω =>
    if k = 0 then ∅ else nodeReads Bs (ph ω (k - 1)).tree with hT
  set Z : ℕ → Ω → Finset (FreeMonoid α) := fun k ω => nodeReads Bs (ph ω k).tree with hZ
  have hcover : {ω | ∃ z ∈ nodeReads Bs (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree,
      ω ∈ flipAt S.O S.B S.F z}
      ⊆ ⋃ k ∈ Finset.range (S.N + 1), {ω | ∃ z ∈ Z k ω, z ∉ T k ω ∧ ω ∈ flipAt S.O S.B S.F z} := by
    intro ω ⟨z, hz, hfl⟩
    rw [roundEnd_eq_phase] at hz
    have hex : ∃ k, z ∈ Z k ω := ⟨S.N, hz⟩
    set k := Nat.find hex
    have hk : z ∈ Z k ω := Nat.find_spec hex
    have hkN : k ≤ S.N := Nat.find_min' hex hz
    simp only [Set.mem_iUnion, Finset.mem_range]
    refine ⟨k, by omega, z, hk, ?_, hfl⟩
    simp only [hT]
    split_ifs with hk0
    · simp
    · exact Nat.find_min hex (show k - 1 < k by omega)
  have hTZ : ∀ k ω ω', (∀ x ∈ vBits (S.F ∪ S.K.train S.F) (T k ω), S.O.noise x ω = S.O.noise x ω') →
      T k ω' = T k ω ∧ Z k ω' = Z k ω := by
    intro k ω ω' h
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · refine ⟨by simp [hT], ?_⟩
      simp only [hZ, hph, phase_zero_tree]
    · obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      have h' : ∀ x ∈ vBits (S.F ∪ S.K.train S.F) (nodeReads Bs (ph ω j).tree),
          S.O.noise x ω = S.O.noise x ω' := by simpa [hT] using h
      have hd := phase_determined S.K S.O S.B S.F S.seed p hp ω ω' j (fun x hx => h' x hx)
      refine ⟨?_, ?_⟩
      · simp only [hT, if_neg (Nat.succ_ne_zero j), Nat.add_sub_cancel, hph, hd.1]
      · simp only [hZ, hph, hd.2]
  have hM : ∀ k ω, (Z k ω \ T k ω).card ≤ M := by
    intro k ω
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · simp only [hT, hZ, if_pos rfl, Finset.sdiff_empty, hph, phase_zero_tree]
      exact (card_nodeReads Bs _).trans (by simp [DTree.mids, M])
    · obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      simp only [hT, hZ, if_neg (Nat.succ_ne_zero j), Nat.add_sub_cancel]
      exact card_sdiff_phase S.K _ S.seed (List.ofFn p) Bs j
  calc _ ≤ μ (⋃ k ∈ Finset.range (S.N + 1),
          {ω | ∃ z ∈ Z k ω, z ∉ T k ω ∧ ω ∈ flipAt S.O S.B S.F z}) := measure_mono hcover
    _ ≤ ∑ k ∈ Finset.range (S.N + 1),
          μ {ω | ∃ z ∈ Z k ω, z ∉ T k ω ∧ ω ∈ flipAt S.O S.B S.F z} :=
        measure_biUnion_finset_le _ _
    _ ≤ ∑ _k ∈ Finset.range (S.N + 1), (M : ℝ≥0∞) * ENNReal.ofReal φ :=
        Finset.sum_le_sum fun k _ => cell_bound S.O hV (T k) (Z k) (hTZ k) (hM k)
          Finset.subset_union_left (flipAt S.O S.B S.F) (measurableSet_flipAt S.O S.B S.F)
          (flipAt_le hflip hclean)
    _ = _ := by simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; push_cast; ring

/-- The strings a draw's prefixes' likelier paths read, under the round's hypothesis. -/
noncomputable def drawRoute (S : RoundSetting α μ Q) (ω : Ω) (p : Fin S.N → FreeMonoid α)
    (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1)).biUnion fun j =>
    (majRoute S.O S.B S.F (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp (prefixOf x j)).toFinset

/-- A draw's prefixes' likelier paths have a reading the pass did not make land off its likelier
side with chance at most `(|x| + 1)·(N + 1)·φ`: the pass and the hypothesis are decided by the
bits it read, and the rest are fresh. -/
theorem freshFlip_le (S : RoundSetting α μ Q) {uHi φ : ℝ}
    (hV : SuffixFree (S.F ∪ S.K.train S.F)) (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ)
    (hclean : ∀ z, stateIndecision S.A S.O S.B S.F (S.A.state z) < uHi)
    (p : Fin S.N → FreeMonoid α) (hp : ∀ i, (p i).toList.length ≤ S.L) (x : FreeMonoid α) :
    μ {ω | ∃ z ∈ drawRoute S ω p x,
        z ∉ nodeReads (passBases S.seed (List.ofFn p) S.L)
          (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree ∧ ω ∈ flipAt S.O S.B S.F z}
      ≤ ((x.toList.length + 1) * (S.N + 1) : ℕ) * ENNReal.ofReal φ := by
  classical
  refine cell_bound S.O hV
    (fun ω => nodeReads (passBases S.seed (List.ofFn p) S.L)
      (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree)
    (fun ω => drawRoute S ω p x) (fun ω ω' h => ?_) (fun ω => ?_) Finset.subset_union_left
    (flipAt S.O S.B S.F) (measurableSet_flipAt S.O S.B S.F) (flipAt_le hflip hclean)
  · have := roundEnd_determined S.K S.O S.B S.F S.seed p hp ω ω' h
    constructor <;> simp only [drawRoute, this]
  · refine (Finset.card_le_card Finset.sdiff_subset).trans ?_
    unfold drawRoute
    refine Finset.card_biUnion_le.trans ?_
    calc _ ≤ ∑ _j ∈ Finset.range (x.toList.length + 1), (S.N + 1) :=
          Finset.sum_le_sum fun j _ => (List.toFinset_card_le _).trans
            ((DTree.route_length_le_depth _ _ _).trans
              (roundEnd_depth_le S.K S.O S.B S.F S.seed (ω, p)))
      _ = _ := by simp

theorem card_passFlip_le (S : RoundSetting α μ Q) (p : Fin S.N → FreeMonoid α) :
    (S.N + 1) * ((passBases S.seed (List.ofFn p) S.L).card
      * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1))) ≤ passReadBound S := by
  have hB := card_passBases S.seed (List.ofFn p) S.L
  simp only [List.length_ofFn] at hB
  unfold passReadBound
  have h1 : (S.N + 1) * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1))
      ≤ (1 + Fintype.card α) * ((S.N + 2) * (1 + Fintype.card α)) := by nlinarith
  calc (S.N + 1) * ((passBases S.seed (List.ofFn p) S.L).card
        * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1)))
      = (passBases S.seed (List.ofFn p) S.L).card
        * ((S.N + 1) * ((Fintype.card α + 1) * ((Fintype.card α + 1) * 1))) := by ring
    _ ≤ (S.seed.length + S.N * (S.L + 1))
        * ((1 + Fintype.card α) * ((S.N + 2) * (1 + Fintype.card α))) :=
        Nat.mul_le_mul hB h1
    _ = _ := by ring

/-- A draw the round counts against its hypothesis, all of whose prefixes are visited, has a
fresh flip on its prefixes' likelier paths when the pass's own reads did not flip. -/
theorem mem_fresh_of_disagree (S : RoundSetting α μ Q) [IsProbabilityMeasure S.D] {η φ uHi : ℝ}
    (hφ : 0 ≤ φ) (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ)
    (hclean : ∀ z, stateIndecision S.A S.O S.B S.F (S.A.state z) < uHi)
    (hη' : η < 1 - 2 * ((S.N + 1 : ℕ) : ℝ) * φ) (p : Fin S.N → FreeMonoid α) (ω : Ω)
    (hedge : ∀ y c, y.toList.length < S.L → 0 < S.D.real {q | y.toList <+: q.toList} →
      edgeDisagreeProb S.O S.B S.F (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp y c ≤ η)
    (hA : ¬ ∃ z ∈ nodeReads (passBases S.seed (List.ofFn p) S.L)
      (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree, ω ∈ flipAt S.O S.B S.F z)
    {x : FreeMonoid α} (hx0 : S.D {x} ≠ 0) (hxL : x.toList.length = S.L)
    (hdis : DFAandDTDisagree (readsAt S.O S.B S.F ω)
      (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp x) :
    ∃ z ∈ drawRoute S ω p x, z ∉ nodeReads (passBases S.seed (List.ofFn p) S.L)
      (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree ∧ ω ∈ flipAt S.O S.B S.F z := by
  set H := (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp
  have hx : 0 < S.D.real {x} := ENNReal.toReal_pos hx0 (measure_ne_top _ _)
  have hvis : ∀ j, 0 < S.D.real {q | (prefixOf x j).toList <+: q.toList} := fun j =>
    hx.trans_le (measureReal_mono (fun q hq => by
      rw [Set.mem_singleton_iff.1 hq]
      simpa [prefixOf] using List.take_prefix j x.toList) (measure_ne_top _ _))
  have hstep : ∀ j (hj : j < x.toList.length),
      majPath S.O S.B S.F H (prefixOf x (j + 1))
        = H.step (majPath S.O S.B S.F H (prefixOf x j)) x.toList[j] := by
    intro j hj
    rw [prefixOf_succ hj]
    exact majPath_step hflip H hclean (roundEnd_depth_le S.K S.O S.B S.F S.seed (ω, p)) hφ hη'
      (hedge _ _ (by simp [prefixOf]; omega) (hvis j))
  obtain ⟨j, hj, z, hz, hfl⟩ := flip_of_disagree H ω hstep hdis
  refine ⟨z, Finset.mem_biUnion.2 ⟨j, Finset.mem_range.2 (by omega), List.mem_toFinset.2 hz⟩,
    fun hzr => hA ⟨z, hzr, hfl⟩, hfl⟩

theorem tsum_singleton_eq_one {β : Type*} [MeasurableSpace β] [Countable β]
    [MeasurableSingletonClass β] (D : Measure β) [IsProbabilityMeasure D] :
    ∑' x, D {x} = 1 := by
  have := Measure.tsum_indicator_apply_singleton D Set.univ MeasurableSet.univ
  simpa using this

/-- For one draw of the probes: with every state read cleanly, the round's every visited edge read
on its edge but for `η` and the DFA/DT check failing by more than `ε` has chance at most the
pass's flips plus Markov's bound on the draws with a fresh flip. -/
theorem gate_flip_section (S : RoundSetting α μ Q) [IsProbabilityMeasure S.D] {ε η φ uHi : ℝ}
    (hε : 0 < ε) (hφ : 0 ≤ φ) (hlen : ∀ x, S.D {x} ≠ 0 → x.toList.length = S.L)
    (hV : SuffixFree (S.F ∪ S.K.train S.F)) (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ)
    (hclean : ∀ z, stateIndecision S.A S.O S.B S.F (S.A.state z) < uHi)
    (hη : η < 1 - 2 * (S.N + 1) * φ) (p : Fin S.N → FreeMonoid α)
    (hp : ∀ i, (p i).toList.length ≤ S.L) :
    μ {ω | (∀ y c, y.toList.length < S.L → 0 < S.D.real {q | y.toList <+: q.toList} →
          edgeDisagreeProb S.O S.B S.F (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp y c ≤ η)
        ∧ ¬ S.D.real {x | DFAandDTDisagree (readsAt S.O S.B S.F ω)
            (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp x} ≤ ε}
      ≤ ENNReal.ofReal (passReadBound S * φ)
        + ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ / ε) := by
  classical
  set Bs := passBases S.seed (List.ofFn p) S.L
  set A₁ := {ω | ∃ z ∈ nodeReads Bs (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree,
    ω ∈ flipAt S.O S.B S.F z} with hA₁
  set Ax : FreeMonoid α → Set Ω := fun x => {ω | ∃ z ∈ drawRoute S ω p x,
    z ∉ nodeReads Bs (roundEnd S.K S.O S.B S.F S.seed (ω, p)).tree ∧ ω ∈ flipAt S.O S.B S.F z}
    with hAx
  have hind : ∀ x, Measurable ((toMeasurable μ (Ax x)).indicator (1 : Ω → ℝ≥0∞)) := fun x =>
    measurable_one.indicator (measurableSet_toMeasurable μ _)
  set g : Ω → ℝ≥0∞ := fun ω => ∑' x, S.D {x} * (toMeasurable μ (Ax x)).indicator 1 ω with hg
  have hgm : Measurable g := Measurable.ennreal_tsum fun x => (hind x).const_mul _
  have hη' : η < 1 - 2 * ((S.N + 1 : ℕ) : ℝ) * φ := by push_cast; exact hη
  have hcover : {ω | (∀ y c, y.toList.length < S.L → 0 < S.D.real {q | y.toList <+: q.toList} →
          edgeDisagreeProb S.O S.B S.F (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp y c ≤ η)
        ∧ ¬ S.D.real {x | DFAandDTDisagree (readsAt S.O S.B S.F ω)
            (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp x} ≤ ε}
      ⊆ A₁ ∪ {ω | ENNReal.ofReal ε ≤ g ω} := by
    rintro ω ⟨hedge, hfail⟩
    by_cases hA : ω ∈ A₁
    · exact Or.inl hA
    right
    set H := (roundEnd S.K S.O S.B S.F S.seed (ω, p)).hyp
    have hpt : ∀ x, {x | DFAandDTDisagree (readsAt S.O S.B S.F ω) H x}.indicator
        (fun x => S.D {x}) x ≤ S.D {x} * (toMeasurable μ (Ax x)).indicator 1 ω := by
      intro x
      by_cases hdis : DFAandDTDisagree (readsAt S.O S.B S.F ω) H x
      · rw [Set.indicator_of_mem (show x ∈ {x | DFAandDTDisagree _ H x} from hdis)]
        by_cases hx0 : S.D {x} = 0
        · rw [hx0]; exact zero_le
        have hmem : ω ∈ Ax x := mem_fresh_of_disagree S hφ hflip hclean hη' p ω hedge hA hx0
          (hlen x hx0) hdis
        rw [Set.indicator_of_mem (subset_toMeasurable μ _ hmem), Pi.one_apply, mul_one]
      · rw [Set.indicator_of_notMem (show x ∉ {x | DFAandDTDisagree _ H x} from hdis)]
        exact zero_le
    have hle : S.D {x | DFAandDTDisagree (readsAt S.O S.B S.F ω) H x} ≤ g ω := by
      rw [← Measure.tsum_indicator_apply_singleton S.D _ (MeasurableSpace.measurableSet_top)]
      exact ENNReal.tsum_le_tsum hpt
    have hlt : ENNReal.ofReal ε < S.D {x | DFAandDTDisagree (readsAt S.O S.B S.F ω) H x} :=
      (ENNReal.ofReal_lt_iff_lt_toReal hε.le (measure_ne_top _ _)).2 (not_le.1 hfail)
    exact hlt.le.trans hle
  have hA₁le : μ A₁ ≤ ENNReal.ofReal (passReadBound S * φ) := by
    refine (passFlip_le S hV hflip hclean p hp).trans ?_
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]
    exact ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_right
      (by exact_mod_cast card_passFlip_le S p) hφ)
  have hint : ∫⁻ ω, g ω ∂μ ≤ ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ) := by
    rw [lintegral_tsum fun x => ((hind x).const_mul _).aemeasurable]
    simp_rw [lintegral_const_mul _ (hind _), lintegral_indicator_one (measurableSet_toMeasurable μ _),
      measure_toMeasurable]
    calc ∑' x, S.D {x} * μ (Ax x)
        ≤ ∑' x, S.D {x} * ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ) :=
          ENNReal.tsum_le_tsum fun x => by
            by_cases hx0 : S.D {x} = 0
            · simp [hx0]
            · gcongr
              have := freshFlip_le S hV hflip hclean p hp x
              rw [hlen x hx0] at this
              refine this.trans (le_of_eq ?_)
              rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]
              push_cast
              ring_nf
      _ = (∑' x, S.D {x}) * ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ) :=
          ENNReal.tsum_mul_right
      _ = _ := by rw [tsum_singleton_eq_one, one_mul]
  have hmark : μ {ω | ENNReal.ofReal ε ≤ g ω}
      ≤ ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ / ε) := by
    have hm := mul_meas_ge_le_lintegral₀ (μ := μ) hgm.aemeasurable (ENNReal.ofReal ε)
    rw [ENNReal.ofReal_div_of_pos hε]
    rw [ENNReal.le_div_iff_mul_le (Or.inl (ENNReal.ofReal_pos.2 hε).ne')
      (Or.inl ENNReal.ofReal_ne_top), mul_comm]
    exact hm.trans hint
  calc _ ≤ μ (A₁ ∪ {ω | ENNReal.ofReal ε ≤ g ω}) := measure_mono hcover
    _ ≤ μ A₁ + μ {ω | ENNReal.ofReal ε ≤ g ω} := measure_union_le _ _
    _ ≤ _ := add_le_add hA₁le hmark

/-- A product with a countable space, bounded by its sections. -/
theorem prod_le_tsum (μ' : Measure Ω) [SFinite μ'] {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (ν : Measure X) [SFinite ν] (E : Set (Ω × X)) :
    (μ'.prod ν) E ≤ ∑' x, ν {x} * μ' {ω | (ω, x) ∈ E} := by
  calc (μ'.prod ν) E ≤ (μ'.prod ν) (⋃ x, {ω | (ω, x) ∈ E} ×ˢ {x}) :=
        measure_mono fun q hq => Set.mem_iUnion.2 ⟨q.2, hq, rfl⟩
    _ ≤ ∑' x, (μ'.prod ν) ({ω | (ω, x) ∈ E} ×ˢ {x}) := measure_iUnion_le _
    _ = ∑' x, ν {x} * μ' {ω | (ω, x) ∈ E} := by
        congr 1 with x
        rw [Measure.prod_prod, mul_comm]

/-- A draw the gate counts against the hypothesis leaves its walk first through a middle reading
off its likelier side, every state being read cleanly and every visited edge on its edge but
for `η`.  Through a suffix-free vote family the readings the pass made flip with chance at most
`passReadBound·φ`, and the rest are fresh given the pass, so Markov bounds the mass of draws
with one. -/
theorem gate_flip_bound (S : RoundSetting α μ Q) [IsProbabilityMeasure S.D] {ε η φ uHi : ℝ}
    (hε : 0 < ε) (hφ : 0 ≤ φ) (hlen : ∀ᵐ x ∂S.D, x.toList.length = S.L)
    (hV : SuffixFree (S.F ∪ S.K.train S.F))
    (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ) (hη : η < 1 - 2 * (S.N + 1) * φ) :
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        (∀ y c, y.toList.length < S.L → 0 < S.D.real {p | y.toList <+: p.toList} →
            edgeDisagreeProb S.O S.B S.F s.hyp y c ≤ η)
          ∧ (∀ q, stateIndecision S.A S.O S.B S.F q < uHi)
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε}
      ≤ (S.L + 1) * (S.N + 1) * φ / ε + passReadBound S * φ := by
  have hRHS : 0 ≤ (S.L + 1) * (S.N + 1) * φ / ε + passReadBound S * φ := by positivity
  by_cases hall : ∀ q, stateIndecision S.A S.O S.B S.F q < uHi
  swap
  · refine le_trans (le_of_eq ?_) hRHS
    rw [show {θ : Ω × (Fin S.N → FreeMonoid α) | _} = ∅ from
      Set.eq_empty_of_forall_notMem fun θ hθ => hall hθ.2.1]
    simp
  have hclean : ∀ z, stateIndecision S.A S.O S.B S.F (S.A.state z) < uHi := fun z => hall _
  have hlen' : ∀ x, S.D {x} ≠ 0 → x.toList.length = S.L := by
    intro x hx
    by_contra hne
    exact hx (measure_mono_null (Set.singleton_subset_iff.2 hne) (ae_iff.1 hlen))
  set ν := Measure.pi fun _ : Fin S.N => S.D
  set c := ENNReal.ofReal (passReadBound S * φ) + ENNReal.ofReal ((S.L + 1) * (S.N + 1) * φ / ε)
  have hsec : ∀ p : Fin S.N → FreeMonoid α, ν {p} ≠ 0 →
      μ {ω | (ω, p) ∈ {θ : Ω × (Fin S.N → FreeMonoid α) |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        (∀ y c, y.toList.length < S.L → 0 < S.D.real {p | y.toList <+: p.toList} →
            edgeDisagreeProb S.O S.B S.F s.hyp y c ≤ η)
          ∧ (∀ q, stateIndecision S.A S.O S.B S.F q < uHi)
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε}} ≤ c := by
    intro p hp0
    have hp : ∀ i, (p i).toList.length ≤ S.L := by
      intro i
      rw [← Set.univ_pi_singleton, Measure.pi_pi] at hp0
      exact (hlen' _ (Finset.prod_ne_zero_iff.1 hp0 i (Finset.mem_univ i))).le
    refine le_trans (measure_mono fun ω hω => ?_)
      (gate_flip_section S hε hφ hlen' hV hflip hclean hη p hp)
    exact ⟨hω.1, hω.2.2⟩
  have hE := prod_le_tsum μ ν {θ : Ω × (Fin S.N → FreeMonoid α) |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        (∀ y c, y.toList.length < S.L → 0 < S.D.real {p | y.toList <+: p.toList} →
            edgeDisagreeProb S.O S.B S.F s.hyp y c ≤ η)
          ∧ (∀ q, stateIndecision S.A S.O S.B S.F q < uHi)
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε}
  have hE' : (μ.prod ν) {θ : Ω × (Fin S.N → FreeMonoid α) |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        (∀ y c, y.toList.length < S.L → 0 < S.D.real {p | y.toList <+: p.toList} →
            edgeDisagreeProb S.O S.B S.F s.hyp y c ≤ η)
          ∧ (∀ q, stateIndecision S.A S.O S.B S.F q < uHi)
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε} ≤ c := by
    refine hE.trans ?_
    calc _ ≤ ∑' p, ν {p} * c := ENNReal.tsum_le_tsum fun p => by
          by_cases hp0 : ν {p} = 0
          · simp [hp0]
          · gcongr; exact hsec p hp0
      _ = c := by rw [ENNReal.tsum_mul_right, tsum_singleton_eq_one, one_mul]
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hRHS (hE'.trans ?_)
  simp only [c]
  rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
  exact ENNReal.ofReal_le_ofReal (by linarith)

end Round

/-- But for the gate's flips, a round ends in one of `RoundProgress`'s outcomes.  Short of (5) and
(6) every state is read cleanly and every visited edge on its edge but for `2(N+1)φ`, so
`gate_flip_bound` applies. -/
theorem round_progress_holds : RoundProgress := by
  intro α _ _ Ω _ μ _ Q _ S ε nH nS σ f a r uHi φ hv hε hφ hφN hlen hV hflip
  have hD : IsProbabilityMeasure S.D := hv.2.1
  refine le_trans (measureReal_mono ?_)
    (gate_flip_bound S hε hφ hlen hV hflip (η := 2 * (S.N + 1) * φ) (by linarith))
  rintro θ' ⟨hfail, -, -, -, hbad, hedge, -⟩
  have hall : ∀ q, stateIndecision S.A S.O S.B S.F q < uHi := fun q =>
    not_le.1 fun h => hbad ⟨q, h⟩
  have hdepth := roundEnd_depth_le S.K S.O S.B S.F S.seed θ'
  refine ⟨fun y c hy hy' => ?_, hall, hfail⟩
  have := edgeDisagreeProb_le_of_clean hflip hall hdepth hφ (by push_cast; linarith)
    (not_lt.1 fun h => hedge ⟨y, c, hy, hy', h⟩)
  push_cast at this
  exact this

end OrthoDFA
