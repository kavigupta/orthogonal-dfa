import OrthoDFA.Proofs.RoundStrong

/-!
# A reading is decided by its reads

Two read functions agreeing on the family's reads of a reading's gate and refusal draws, and on
its certificate draws, give the reading the same gate, refusal sample and certificate.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Congr

variable {K : StageKnobs α} {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem hitsIn_iff {N : ℕ} (b : Fin N → FreeMonoid α) {P Q : FreeMonoid α → Prop}
    (h : ∀ i, P (b i) ↔ Q (b i)) (n : ℕ) : hitsIn b P n = hitsIn b Q n := by
  classical
  unfold hitsIn
  congr 1
  exact Finset.filter_congr fun i _ => by rw [h i]

theorem fires_iff {N : ℕ} (b : Fin N → FreeMonoid α) (a : ℝ) {T₁ T₂ : ClassTest α}
    (hθ : T₁.θ = T₂.θ) (hh : ∀ i, T₁.hits (b i) ↔ T₂.hits (b i))
    (ht : ∀ i, T₁.trials (b i) ↔ T₂.trials (b i)) (n : ℕ) :
    T₁.fires b a n ↔ T₂.fires b a n := by
  unfold ClassTest.fires
  rw [hθ, hitsIn_iff b ht, hitsIn_iff b hh]

/-- Agreement on a draw's reads from `k` on, against `t`. -/
def DrawAgree (K : StageKnobs α) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)
    (t : DTree α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∀ i, k ≤ i → ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf x i * m)

theorem deepUndecided_iff {t : DTree α} {y : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (y * m)) :
    DeepUndecided (rd B F f₁) t y ↔ DeepUndecided (rd B F f₂) t y := by
  unfold DeepUndecided
  rw [sift_congr (B := B) h, route_congr' (B := B) h]

theorem walkCheck_congr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : DrawAgree K F f₁ f₂ t k x) :
    walkCheck (rd B F f₁) t edges k x = walkCheck (rd B F f₂) t edges k x := by
  have hs : ∀ i, k ≤ i → t.sift (rd B F f₁).cut (prefixOf x i)
      = t.sift (rd B F f₂).cut (prefixOf x i) := fun i hi => sift_congr (h i hi)
  have hx : t.sift (rd B F f₁).cut x = t.sift (rd B F f₂).cut x := by
    have := hs (max k x.toList.length) (le_max_left _ _)
    rwa [show prefixOf x (max k x.toList.length) = x by
      unfold prefixOf; rw [List.take_of_length_le (le_max_right _ _)]; rfl] at this
  have hkw : kWalk (rd B F f₁) t edges k x = kWalk (rd B F f₂) t edges k x := by
    unfold kWalk; rw [hs k le_rfl]
  unfold walkCheck
  rw [hkw]
  split
  · rfl
  · rename_i s c j hj
    have hkj := kWalk_edge_ge _ hj
    rw [hs (j + 1) (by omega), hs j hkj]
    have hwt : walkTo (rd B F f₁) t edges k x j = walkTo (rd B F f₂) t edges k x j := by
      unfold walkTo; rw [hs k le_rfl]
    rw [hwt]
  · rw [hx]

theorem edgeAt_congr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : DrawAgree K F f₁ f₂ t k x) :
    edgeAt (rd B F f₁) t edges k x = edgeAt (rd B F f₂) t edges k x := by
  unfold edgeAt; rw [probeOutcome_congr h]

/-- On a draw whose reads agree, each harvest class and the pair test count it alike. -/
theorem harvestTests_fires_iff {t : DTree α} {edges : Edges α} {k L : ℕ} {f c θM a : ℝ}
    {N : ℕ} (br : Fin N → FreeMonoid α) (h : ∀ i, DrawAgree K F f₁ f₂ t k (br i)) (n : ℕ) :
    ((∃ T ∈ harvestTests (rd B F f₁) t edges k L f c θM, T.fires br a n)
      ↔ ∃ T ∈ harvestTests (rd B F f₂) t edges k L f c θM, T.fires br a n)
    ∧ ((pairTrip (rd B F f₁) t edges k).fires br a n
      ↔ (pairTrip (rd B F f₂) t edges k).fires br a n) := by
  have hpo := fun i => probeOutcome_congr (B := B) (edges := edges) (h i)
  have hsr : ∀ i, Searched (rd B F f₁) t edges k (br i) ↔ Searched (rd B F f₂) t edges k (br i) :=
    fun i => by unfold Searched; rw [walkCheck_congr (h i)]
  have hk : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf (br i) k * m) := fun i => h i k le_rfl
  have hwhole : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (br i * m) := fun i => by
    have := h i (max k (br i).toList.length) (le_max_left _ _)
    rwa [show prefixOf (br i) (max k (br i).toList.length) = br i by
      unfold prefixOf; rw [List.take_of_length_le (le_max_right _ _)]; rfl] at this
  have hkw : ∀ i, kWalk (rd B F f₁) t edges k (br i) = kWalk (rd B F f₂) t edges k (br i) :=
    fun i => by unfold kWalk; rw [sift_congr (B := B) (hk i)]
  have tr : ∀ (T₁ T₂ : ClassTest α), T₁.θ = T₂.θ → (∀ i, T₁.hits (br i) ↔ T₂.hits (br i)) →
      (∀ i, T₁.trials (br i) ↔ T₂.trials (br i)) → (T₁.fires br a n ↔ T₂.fires br a n) :=
    fun T₁ T₂ hθ hh ht => fires_iff br a hθ hh ht n
  refine ⟨?_, tr _ _ rfl (fun i => by simp only [pairTrip, IsPair, hpo i]) hsr⟩
  simp only [harvestTests, heldTests, pairTrip, List.mem_append, List.mem_cons, List.mem_nil_iff,
    or_false, or_and_right, exists_or, exists_eq_left]
  have e1 := tr ⟨StartDeep (rd B F f₁) t k, fun _ => True, (t.depth - 1 : ℕ) * f⟩
    ⟨StartDeep (rd B F f₂) t k, fun _ => True, (t.depth - 1 : ℕ) * f⟩ rfl
    (fun i => deepUndecided_iff (hk i)) (fun _ => Iff.rfl)
  have e2 := tr ⟨EndDeep (rd B F f₁) t, fun _ => True, (t.depth - 1 : ℕ) * f⟩
    ⟨EndDeep (rd B F f₂) t, fun _ => True, (t.depth - 1 : ℕ) * f⟩ rfl
    (fun i => deepUndecided_iff (hwhole i)) (fun _ => Iff.rfl)
  have e3 := tr ⟨IsTriple (rd B F f₁) t edges k, Searched (rd B F f₁) t edges k,
      c * f * t.depth * searchSteps L k⟩
    ⟨IsTriple (rd B F f₂) t edges k, Searched (rd B F f₂) t edges k,
      c * f * t.depth * searchSteps L k⟩ rfl
    (fun i => by simp only [IsTriple, hpo i]) hsr
  have e4 := tr ⟨IsPair (rd B F f₁) t edges k, Searched (rd B F f₁) t edges k,
      c * f * t.depth * searchSteps L k⟩
    ⟨IsPair (rd B F f₂) t edges k, Searched (rd B F f₂) t edges k,
      c * f * t.depth * searchSteps L k⟩ rfl
    (fun i => by simp only [IsPair, hpo i]) hsr
  have e5 := tr ⟨IsBlocked (rd B F f₁) t edges k, fun _ => True, 2 * c * f * t.depth⟩
    ⟨IsBlocked (rd B F f₂) t edges k, fun _ => True, 2 * c * f * t.depth⟩ rfl
    (fun i => by simp only [IsBlocked, hpo i, hkw i]) (fun _ => Iff.rfl)
  have e6 := tr ⟨IsMember (rd B F f₁) t edges k, fun _ => True, θM⟩
    ⟨IsMember (rd B F f₂) t edges k, fun _ => True, θM⟩ rfl
    (fun i => by simp only [IsMember, hpo i]) (fun _ => Iff.rfl)
  have e7 := tr ⟨IsPair (rd B F f₁) t edges k, Searched (rd B F f₁) t edges k, 1 / 2⟩
    ⟨IsPair (rd B F f₂) t edges k, Searched (rd B F f₂) t edges k, 1 / 2⟩ rfl
    (fun i => by simp only [IsPair, hpo i]) hsr
  rw [e1, e2, e3, e4, e5, e6, e7]

theorem refusalStop_congr {N : ℕ} (br : Fin N → FreeMonoid α) (a : ℝ)
    {ts₁ ts₂ : List (ClassTest α)} {l₁ l₂ : FreeMonoid α → Prop}
    (ht : ∀ n, (∃ T ∈ ts₁, T.fires br a n) ↔ ∃ T ∈ ts₂, T.fires br a n)
    (hl : ∀ i, l₁ (br i) ↔ l₂ (br i)) :
    refusalStop br a ts₁ l₁ = refusalStop br a ts₂ l₂ := by
  classical
  unfold refusalStop
  congr 2
  funext n
  simp only [ht n, hl]

theorem startDis_iff {edges : Edges α} {q : List Bool} {x : FreeMonoid α}
    (h : Agree K F f₁ f₂ x) :
    StartDis (rd B F f₁) edges q x ↔ StartDis (rd B F f₂) edges q x := by
  unfold StartDis
  have : (rd B F f₁).mid (x * 1) = (rd B F f₂).mid (x * 1) := mid_congr (by simpa using h)
  simp only [mul_one] at this
  rw [this]

theorem gate_congr {t : DTree α} {edges : Edges α} {N : ℕ} (bg : Fin N → FreeMonoid α)
    (h : ∀ i, Agree K F f₁ f₂ (bg i)) (acc a : ℝ) :
    gateStop (rd B F f₁) t edges bg acc a = gateStop (rd B F f₂) t edges bg acc a
      ∧ ∀ n, gateSide (rd B F f₁) t edges bg acc a n = gateSide (rd B F f₂) t edges bg acc a n
      ∧ gateBest (rd B F f₁) t edges bg n = gateBest (rd B F f₂) t edges bg n := by
  have hd : ∀ q n, hitsIn bg (fun x => ¬ StartDis (rd B F f₁) edges q x) n
      = hitsIn bg (fun x => ¬ StartDis (rd B F f₂) edges q x) n :=
    fun q => hitsIn_iff bg fun i => by rw [startDis_iff (B := B) (q := q) (edges := edges) (h i)]
  have hb : ∀ n, gateBest (rd B F f₁) t edges bg n = gateBest (rd B F f₂) t edges bg n := by
    intro n; unfold gateBest; simp only [hd]
  have hs : ∀ n, gateSide (rd B F f₁) t edges bg acc a n
      = gateSide (rd B F f₂) t edges bg acc a n := by
    intro n; unfold gateSide; rw [hb n, hd]
  refine ⟨?_, fun n => ⟨hs n, hb n⟩⟩
  unfold gateStop
  simp only [hs]

end Congr

end OrthoDFA

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Reading

variable (C : StrongCfg α) {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_max_min {x : FreeMonoid α} {k i : ℕ} (hi : k ≤ i) :
    prefixOf x i = prefixOf x (max k (min i x.toList.length)) := by
  unfold prefixOf
  congr 1
  rcases le_total i x.toList.length with h | h
  · rw [min_eq_left h, max_eq_right hi]
  · rw [min_eq_right h, List.take_of_length_le h]
    rcases le_total k x.toList.length with h' | h'
    · rw [max_eq_right h', List.take_length]
    · rw [max_eq_left h', List.take_of_length_le h']

theorem gate_agree {t : DTree α} {y : C.Draws}
    (hr : ∀ z ∈ readingReads C F t y, f₁ z = f₂ z) (i : Fin C.ng) :
    Agree C.K F f₁ f₂ (y.2.1 i) := fun v hv =>
  hr _ (Finset.mem_union_left _ (Finset.mem_union_left _
    (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, Finset.mem_image.2 ⟨v, hv, rfl⟩⟩)))

theorem refusal_agree {t : DTree α} {y : C.Draws}
    (hr : ∀ z ∈ readingReads C F t y, f₁ z = f₂ z) (i : Fin C.nr) :
    DrawAgree C.K F f₁ f₂ t C.k (y.2.2.1 i) := by
  classical
  intro i' hi' m hm v hv
  refine hr _ (Finset.mem_union_left _ (Finset.mem_union_right _
    (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, Finset.mem_image.2
      ⟨((prefixOf (y.2.2.1 i) i', m), v), ?_, rfl⟩⟩)))
  simp only [Finset.mem_product, List.mem_toFinset, List.mem_map, List.mem_range]
  refine ⟨⟨⟨min i' (y.2.2.1 i).toList.length, by omega, (prefixOf_max_min hi').symm⟩,
    (DTree.mem_midfixes_iff _).2 hm⟩, hv⟩

theorem cert_agree {t : DTree α} {y : C.Draws}
    (hr : ∀ z ∈ readingReads C F t y, f₁ z = f₂ z) (i : Fin C.nc) :
    f₁ (y.2.2.2 i) = f₂ (y.2.2.2 i) :=
  hr _ (Finset.mem_union_right _ (Finset.mem_image.2 ⟨i, Finset.mem_univ _, rfl⟩))

/-- A reading whose pass two read functions agree on, and whose gate, refusal sample and
certificate read alike, ends alike. -/
theorem strongReading_congr {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)}
    {y : C.Draws}
    (hpass : strongPass C (rd B F f₁) A (first ++ List.ofFn y.1)
      = strongPass C (rd B F f₂) A (first ++ List.ofFn y.1))
    (hr : ∀ z ∈ readingReads C F (strongPass C (rd B F f₁) A (first ++ List.ofFn y.1)).s.tree y,
      f₁ z = f₂ z) :
    strongReading C (rd B F f₁) j A first y = strongReading C (rd B F f₂) j A first y := by
  classical
  unfold strongReading
  simp only []
  rw [← hpass]
  generalize strongPass C (rd B F f₁) A (first ++ List.ofFn y.1) = A' at hr ⊢
  have hdr := refusal_agree C hr
  obtain ⟨hs1, hs2⟩ := gate_congr (B := B) (t := A'.s.tree) (edges := A'.s.edges) y.2.1
    (gate_agree C hr) C.acc (C.a / 2 ^ j)
  have hstart : gateStart (rd B F f₁) A'.s.tree A'.s.edges y.2.1 C.acc (C.a / 2 ^ j)
      = gateStart (rd B F f₂) A'.s.tree A'.s.edges y.2.1 C.acc (C.a / 2 ^ j) := by
    unfold gateStart; rw [hs1, (hs2 _).2]
  have hhf := harvestTests_fires_iff (B := B) (edges := A'.s.edges) (L := C.L) (f := C.f)
    (c := C.c) (θM := C.θM) (a := C.a) y.2.2.1 hdr
  have hlive : ∀ i, LiveEdge (rd B F f₁) A'.s.tree A'.s.edges C.k (fun _ => False) (y.2.2.1 i)
      ↔ LiveEdge (rd B F f₂) A'.s.tree A'.s.edges C.k (fun _ => False) (y.2.2.1 i) := fun i => by
    unfold LiveEdge; rw [edgeAt_congr (hdr i)]
  have hstop := refusalStop_congr y.2.2.1 C.a (fun n => (hhf n).1) hlive
  have hcert : (fun i => f₁ (y.2.2.2 i)) = (fun i => f₂ (y.2.2.2 i)) :=
    funext fun i => cert_agree C hr i
  have hfire := fun n => (hhf n).1
  have hpair := fun n => (hhf n).2
  simp only [rd] at hs1 hs2 hstart hstop hcert hfire hpair hlive ⊢
  simp only [hs1, (hs2 _).1, hstart, hcert, hstop, hfire, hpair, hlive]

end Reading

end OrthoDFA
