import OrthoDFA.Proofs.CheckWaste
import OrthoDFA.Proofs.RandomGood

/-!
# How many check strings read undecided

Over the reads, the share of check strings some prefix of which, followed by a midfix of at most
`n_Q` letters, reads undecided is the same bound for every tree whose midfixes are that short. A
prefix shorter than `h` is one of at most `|upTo h| |upTo n_Q|` strings, which all read decided
but for chance `|upTo h| |upTo n_Q| u`, `u` bounding each string's chance of reading undecided.
Off them, each check string's reads share its first `h` letters, so where no length-`h` prefix
holds more than `p₀` of the check strings, `block_tail` bounds the share above `2θ₀ + η` by
`exp(−η/p₀)`, `θ₀ = (L_c + 1)|upTo n_Q| u` (`head_le`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
/-- A tree of at most `n` splits has midfixes of at most `n` letters. -/
theorem midsLe_classSet : ∀ (n : ℕ) (T : DTree α), T ∈ classSet n →
    ∀ m ∈ mids T, m.toList.length ≤ n
  | 0, T, h => by
    classical
    simp only [classSet, Finset.mem_singleton] at h
    subst h
    intro m hm
    simp [mids] at hm
    subst hm
    simp
  | n + 1, T, h => by
    classical
    simp only [classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T', hT', q, -, rfl⟩
    · exact fun m hm => (midsLe_classSet n T h m hm).trans (by omega)
    · intro m hm
      rcases Finset.mem_insert.1 (mids_splitAt _ T' q.1 hm) with rfl | hm
      · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
          List.length_singleton]
        rcases midfixAt_mem T' (commonPrefix q.2.2.1 q.2.2.2) with h1 | h1
        · rw [h1]; simp
        · have := midsLe_classSet n T' hT' _ h1; omega
      · have := midsLe_classSet n T' hT' m hm; omega

/-- Some prefix of `x` at least `h` long, followed by at most `n` letters, reads undecided. -/
def lightUnd (rd : FreeMonoid α → ARU) (h n : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ j, h ≤ j ∧ j ≤ x.toList.length ∧ ∃ m ∈ upTo n, rd (prefixOf x j * m) = .undecided

/-- The strings a prefix shorter than `h` followed by at most `n` letters can be. -/
noncomputable def heavySet (h n : ℕ) : Finset (FreeMonoid α) :=
  ((upTo h) ×ˢ (upTo n)).image fun p => p.1 * p.2

omit [DecidableEq α] in
/-- A check string some prefix of which, followed by a midfix, reads undecided reads one of the
heavy strings undecided or is light-undecided. -/
theorem utype_sub {rd : FreeMonoid α → ARU} {T : DTree α} {n h : ℕ}
    (hT : ∀ m ∈ mids T, m.toList.length ≤ n)
    (hH : ∀ z ∈ heavySet (α := α) h n, rd z ≠ .undecided) {x : FreeMonoid α}
    (hx : Utype rd T x = true) : lightUnd rd h n x := by
  classical
  simp only [Utype, decide_eq_true_eq] at hx
  obtain ⟨z, hz, hu⟩ := hx
  unfold pot at hz
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨-, hjx⟩ := Finset.mem_Icc.1 hj
  have hmn : m ∈ upTo n := mem_upTo.2 (hT m hm)
  by_cases hjh : h ≤ j
  · exact ⟨j, hjh, hjx, m, hmn, hu⟩
  · exfalso
    refine hH _ (Finset.mem_image.2 ⟨(prefixOf x j, m), Finset.mem_product.2 ⟨mem_upTo.2 ?_,
      hmn⟩, rfl⟩) hu
    simp only [prefixOf, FreeMonoid.toList_ofList, List.length_take]
    omega

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

open scoped Classical in
/-- Over the reads, some tree with short midfixes has more than `2θ₀ + η` of the check strings
read undecided with chance at most `|heavySet| u + exp(−η/p₀)`. -/
theorem head_le (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : ProbabilityTheory.iIndepFun read μ) {u : ℝ} {Lh : ℕ}
    (hu : ∀ z : FreeMonoid α, z.toList.length ≤ Lh → μ.real {ω | read z ω = .undecided} ≤ u)
    (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc] {Lc n h : ℕ}
    (hlen : ∀ᵐ x ∂Dc, x.toList.length ≤ Lc) (hLh : Lc + n ≤ Lh) {p₀ η : ℝ} (hp₀ : 0 < p₀)
    (hpmax : ∀ v, Dc.real {x | prefixOf x h = v} ≤ p₀) (hhL : h + n ≤ Lh)
    (hθ8 : 8 * ((Lc + 1) * (upTo (α := α) n).card * u) ≤ 1) (hu0 : 0 ≤ u) :
    μ {ω | ∃ T : DTree α, (∀ m ∈ mids T, m.toList.length ≤ n) ∧
        2 * ((Lc + 1) * (upTo (α := α) n).card * u) + η
          < Dc.real {x | Utype (read · ω) T x = true}}
      ≤ ENNReal.ofReal ((heavySet (α := α) h n).card * u + Real.exp (-(η / p₀))) := by
  set θ₀ := (Lc + 1) * ((upTo (α := α) n).card : ℝ) * u with hθ₀
  set H := heavySet (α := α) h n
  set Ev₁ : Set Ω := {ω | ∃ z ∈ H, read z ω = .undecided}
  set Sx : FreeMonoid α → Finset (FreeMonoid α) := fun x =>
    (Finset.Icc h x.toList.length).biUnion fun j => (upTo n).image (prefixOf x j * ·)
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if lightUnd rd h n x then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun _ _ => 1
  have hlight : ∀ rd x, lightUnd rd h n x ↔ ∃ z ∈ Sx x, rd z = .undecided := by
    intro rd x
    constructor
    · rintro ⟨j, hj1, hj2, m, hm, hu⟩
      exact ⟨_, Finset.mem_biUnion.2 ⟨j, Finset.mem_Icc.2 ⟨hj1, hj2⟩,
        Finset.mem_image.2 ⟨m, hm, rfl⟩⟩, hu⟩
    · rintro ⟨z, hz, hu⟩
      obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
      exact ⟨j, (Finset.mem_Icc.1 hj).1, (Finset.mem_Icc.1 hj).2, m, hm, hu⟩
  have hSxk : ∀ x z, z ∈ Sx x → prefixOf z h = prefixOf x h := by
    intro x z hz
    obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
    obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
    obtain ⟨hj1, hj2⟩ := Finset.mem_Icc.1 hj
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
    rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hj1]
  have huf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → uf rd x = uf rd' x := by
    intro x rd rd' hr
    simp only [uf]
    congr 1
    apply propext
    rw [hlight, hlight]
    constructor
    · rintro ⟨z, hz, hu⟩; exact ⟨z, hz, (hr z hz) ▸ hu⟩
    · rintro ⟨z, hz, hu⟩; exact ⟨z, hz, (hr z hz).symm ▸ hu⟩
  have hSxcard : ∀ x ∈ upTo Lc, ((Sx x).card : ℝ) ≤ (Lc + 1) * (upTo (α := α) n).card := by
    intro x hx
    have hxl := mem_upTo.1 hx
    have : (Sx x).card ≤ (Lc + 1) * (upTo (α := α) n).card := by
      refine Finset.card_biUnion_le.trans ?_
      calc _ ≤ ∑ _j ∈ Finset.Icc h x.toList.length, (upTo (α := α) n).card :=
            Finset.sum_le_sum fun j _ => Finset.card_image_le
        _ ≤ _ := by
            rw [Finset.sum_const, smul_eq_mul, Nat.card_Icc]
            gcongr
            omega
    exact_mod_cast this
  have hlight_congr : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) →
      (lightUnd rd h n x ↔ lightUnd rd' h n x) := by
    intro x rd rd' hr
    rw [hlight, hlight]
    constructor
    · rintro ⟨z, hz, hu⟩; exact ⟨z, hz, (hr z hz) ▸ hu⟩
    · rintro ⟨z, hz, hu⟩; exact ⟨z, hz, (hr z hz).symm ▸ hu⟩
  have hθ₀0 : 0 ≤ θ₀ := by positivity
  have hmean : ∀ x ∈ upTo Lc, ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ := by
    intro x hx
    simp only [tf, integral_const, probReal_univ, smul_eq_mul, mul_one]
    simp only [uf]
    rw [integral_ite_strings read hmeas (Sx x) (fun rd => lightUnd rd h n x)
      fun rd rd' hr => hlight_congr x rd rd' hr]
    calc μ.real {ω | lightUnd (read · ω) h n x}
        ≤ μ.real (⋃ z ∈ Sx x, {ω | read z ω = .undecided}) := by
          refine measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)
          obtain ⟨z, hz, hu⟩ := (hlight (fun z => read z ω) x).1 hω
          exact Set.mem_biUnion hz hu
      _ ≤ ∑ _z ∈ Sx x, u := by
          refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun z hz => hu z ?_)
          obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
          obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
          have h1 := mem_upTo.1 hm
          have h2 := mem_upTo.1 hx
          have h3 : (prefixOf x j).toList.length ≤ x.toList.length := by simp [prefixOf]
          simp only [FreeMonoid.toList_mul, List.length_append]
          omega
      _ = (Sx x).card * u := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ θ₀ := mul_le_mul_of_nonneg_right (hSxcard x hx) hu0
  have hbt := block_tail hmeas hind Dc hp₀ hpmax hθ₀0 hθ8 Sx hSxk uf tf
    huf (fun _ _ _ _ => rfl)
    (fun rd x _ => by simp only [uf]; split_ifs <;> norm_num)
    (fun rd x _ => by simp only [tf]; norm_num) (η := η) (L := Lc) hmean
  have hEv₁ : μ Ev₁ ≤ ENNReal.ofReal (H.card * u) := by
    have hset : Ev₁ = ⋃ z ∈ H, {ω | read z ω = .undecided} := by
      ext ω; simp [Ev₁]
    rw [hset]
    calc _ ≤ ∑ z ∈ H, μ {ω | read z ω = .undecided} := measure_biUnion_finset_le _ _
      _ ≤ ∑ _z ∈ H, ENNReal.ofReal u := Finset.sum_le_sum fun z hz => by
          rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
          refine ENNReal.ofReal_le_ofReal (hu z ?_)
          obtain ⟨⟨v, m⟩, hvm, rfl⟩ := Finset.mem_image.1 hz
          obtain ⟨hv, hm⟩ := Finset.mem_product.1 hvm
          dsimp only at hv hm ⊢
          have := mem_upTo.1 hv
          have := mem_upTo.1 hm
          simp only [FreeMonoid.toList_mul, List.length_append]
          omega
      _ = _ := by
          rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
            ← ENNReal.ofReal_mul (by positivity)]
  have hsub : {ω | ∃ T : DTree α, (∀ m ∈ mids T, m.toList.length ≤ n) ∧
        2 * θ₀ + η < Dc.real {x | Utype (read · ω) T x = true}}
      ⊆ Ev₁ ∪ {ω | 2 * θ₀ * ∑ x ∈ upTo Lc, Dc.real {x} * tf (read · ω) x + η
        < ∑ x ∈ upTo Lc, Dc.real {x} * uf (read · ω) x} := by
    rintro ω ⟨T, hT, hlt⟩
    by_cases h1 : ω ∈ Ev₁
    · exact Or.inl h1
    right
    have hH : ∀ z ∈ H, read z ω ≠ .undecided := fun z hz hu' => h1 ⟨z, hz, hu'⟩
    have hle : Dc.real {x | Utype (read · ω) T x = true}
        ≤ Dc.real {x | lightUnd (read · ω) h n x} :=
      measureReal_mono (fun x hx => utype_sub hT hH hx) (measure_ne_top _ _)
    have e1 : Dc.real {x | lightUnd (read · ω) h n x}
        = ∑ x ∈ upTo Lc, Dc.real {x} * uf (read · ω) x := by
      rw [measureReal_eq_sum_upTo Dc hlen]
      rfl
    have e2 : ∑ x ∈ upTo Lc, Dc.real {x} * tf (read · ω) x = 1 := by
      have := measureReal_eq_sum_upTo Dc hlen Set.univ
      simp only [probReal_univ, Set.mem_univ, if_true] at this
      simp only [tf]
      exact this.symm
    simp only [Set.mem_ofPred_eq]
    rw [e2, mul_one, ← e1]
    linarith
  calc _ ≤ μ Ev₁ + μ {ω | 2 * θ₀ * ∑ x ∈ upTo Lc, Dc.real {x} * tf (read · ω) x + η
        < ∑ x ∈ upTo Lc, Dc.real {x} * uf (read · ω) x} :=
        (measure_mono hsub).trans (measure_union_le _ _)
    _ ≤ ENNReal.ofReal (H.card * u) + ENNReal.ofReal (Real.exp (-(η / p₀))) :=
        add_le_add hEv₁ hbt
    _ = _ := by rw [← ENNReal.ofReal_add (by positivity) (Real.exp_pos _).le]

end Eps

end Random

end OrthoDFA
