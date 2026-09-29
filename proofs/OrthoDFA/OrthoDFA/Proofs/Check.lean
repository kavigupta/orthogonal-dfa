import OrthoDFA.Proofs.CheckTails

/-!
# The merge check's guarantee

Both bounds split on the first `2n` members.  Where they repeat, fall in `U` or outside `Pre`,
or meet `U` after the suffix, the noise read is not fresh and the bound pays for it outright;
elsewhere pinning `U` is invisible to the reads and the tails of `CheckTails` apply.
-/

namespace OrthoDFA.CheckProof

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

omit [IsProbabilityMeasure μ] in
lemma measurableSet_of_sections {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (E : Set (Ω × X)) (hE : ∀ x, MeasurableSet {ω | (ω, x) ∈ E}) :
    MeasurableSet E := by
  have : E = ⋃ x, {ω | (ω, x) ∈ E} ×ˢ {x} := by
    ext ⟨ω, x⟩; simp
  rw [this]
  exact MeasurableSet.iUnion fun x => (hE x).prod (measurableSet_singleton x)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_checkFails {R : Type*} (O : Oracle μ S) (H : DFA S R) (n : ℕ) (t : ℝ)
    {N m : ℕ} (h : R) :
    MeasurableSet {z : Ω × ((Fin N → S) × (Fin m → S)) | checkFails O H n t h z.2 z.1} := by
  refine measurableSet_of_sections _ fun x => ?_
  simp only [checkFails, Set.mem_ofPred_eq]
  rw [Set.ofPred_and, Set.ofPred_exists]
  refine (MeasurableSet.const _).inter (MeasurableSet.iUnion fun j => ?_)
  rw [Set.ofPred_and]
  exact (MeasurableSet.const _).inter
    (measurableSet_le measurable_const (measurable_leaning O _ _ _))

omit [IsProbabilityMeasure μ] in
lemma leaning_take (O : Oracle μ S) (n : ℕ) (L : List S) (v : S) :
    leaning O n L v = leaning O n (L.take (2 * n)) v := by
  funext ω
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk : k < n := Finset.mem_range.1 hk
  simp only [List.getD_eq_getElem?_getD, List.getElem?_take,
    if_pos (show 2 * k < 2 * n by omega), if_pos (show 2 * k + 1 < 2 * n by omega)]

lemma ne_mul_of_flat {Pre Suf : Set S} (hflat : Flat Pre Suf) {v : S}
    (hv : v ≠ 1) (hvS : v ∈ Suf) {l : List S} (hl : ∀ x ∈ l, x ∈ Pre) :
    ∀ x ∈ l, ∀ z ∈ l, x ≠ z * v := by
  intro x hx z hz e
  have hxz : x = z := hflat x (hl x hx) z (hl z hz) 1 (Set.mem_insert _ _) v
    (Set.mem_insert_of_mem _ hvS) (by rw [mul_one]; exact e)
  rw [← hxz] at e
  exact hv (mul_left_cancel ((mul_one x).trans e)).symm

omit [IsProbabilityMeasure μ] in
lemma readsOn_leanN_lt (O : Oracle μ S) (n : ℕ) (l : List S) (v : S) (hl : l.length = 2 * n)
    (c : ℝ) : ReadsOn (reads l v) (fun y => leanN O y n l v < c) := fun y y' hy => by
  have h := leanN_readsOn O v n l hl y y' hy
  simp only at h ⊢
  rw [h]

omit [IsProbabilityMeasure μ] in
lemma readsOn_le_leanN (O : Oracle μ S) (n : ℕ) (l : List S) (v : S) (hl : l.length = 2 * n)
    (c : ℝ) : ReadsOn (reads l v) (fun y => c ≤ leanN O y n l v) := fun y y' hy => by
  have h := leanN_readsOn O v n l hl y y' hy
  simp only at h ⊢
  rw [h]

lemma disjoint_reads {l : List S} {v : S} {U : Finset S} (h₁ : ∀ x ∈ l, x ∉ U)
    (h₂ : ∀ x ∈ l, x * v ∉ U) : Disjoint (reads l v) U := by
  rw [Finset.disjoint_left]
  intro w hw
  rcases mem_reads.1 hw with hw | ⟨x, hx, rfl⟩
  · exact h₁ w hw
  · exact h₂ x hx

section reaching
variable {R : Type*} (H : DFA S R) (h : R) (D : Measure S) [IsProbabilityMeasure D]

omit [IsProbabilityMeasure D] in
lemma reaching_eq (hD : D {w | H.state w = h} ≠ 0) :
    reaching H D h = D[|{w | H.state w = h}] := by
  simp [reaching, hD]

lemma reaching_atom_le (q κ : ℝ) (hq : 0 < q) (hheavy : q ≤ D.real {w | H.state w = h})
    (hκ : ∀ a, D.real {a} ≤ κ) (a : S) : reaching H D h {a} ≤ ENNReal.ofReal (κ / q) := by
  have hle : ENNReal.ofReal q ≤ D {w | H.state w = h} :=
    ENNReal.ofReal_le_of_le_toReal hheavy
  have hD : D {w | H.state w = h} ≠ 0 :=
    (lt_of_lt_of_le (ENNReal.ofReal_pos.2 hq) hle).ne'
  rw [reaching_eq H h D hD, cond_apply MeasurableSet.of_discrete]
  calc (D {w | H.state w = h})⁻¹ * D ({w | H.state w = h} ∩ {a})
      ≤ (ENNReal.ofReal q)⁻¹ * ENNReal.ofReal κ := by
        gcongr
        calc D ({w | H.state w = h} ∩ {a}) ≤ D {a} := measure_mono Set.inter_subset_right
          _ = ENNReal.ofReal (D.real {a}) := (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
          _ ≤ ENNReal.ofReal κ := ENNReal.ofReal_le_ofReal (hκ a)
    _ = ENNReal.ofReal (κ / q) := by
        rw [← ENNReal.ofReal_inv_of_pos hq, ← ENNReal.ofReal_mul (inv_nonneg.2 hq.le),
          div_eq_mul_inv, mul_comm]

end reaching

lemma measure_preimage_le (ρ : Measure S) (κ' : ℝ≥0∞) (hat : ∀ a, ρ {a} ≤ κ') (f : S → S)
    (hf : Function.Injective f) (U : Finset S) : ρ {x | f x ∈ U} ≤ U.card * κ' := by
  calc ρ {x | f x ∈ U} ≤ ρ (⋃ w ∈ U, {x | f x = w}) :=
        measure_mono fun x hx => Set.mem_iUnion₂.2 ⟨f x, hx, rfl⟩
    _ ≤ ∑ w ∈ U, ρ {x | f x = w} := measure_biUnion_finset_le _ _
    _ ≤ ∑ _w ∈ U, κ' := Finset.sum_le_sum fun w _ => by
        rcases (show {x | f x = w}.Subsingleton from
            fun x hx y hy => hf (hx.trans hy.symm)).eq_empty_or_singleton with he | ⟨a, he⟩
        · rw [he]; simp
        · rw [he]; exact hat a
    _ = U.card * κ' := by simp

lemma choose_two_le (n : ℕ) : ((2 * n).choose 2 : ℝ) ≤ 2 * n ^ 2 := by
  have : (2 * n).choose 2 ≤ 2 * n ^ 2 := by
    rw [Nat.choose_two_right]
    refine Nat.div_le_of_le_mul ?_
    calc 2 * n * (2 * n - 1) ≤ 2 * n * (2 * n) := Nat.mul_le_mul_left _ (Nat.sub_le _ _)
      _ = 2 * (2 * n ^ 2) := by ring
  exact_mod_cast this

end OrthoDFA.CheckProof

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Real CheckProof
open scoped ENNReal

theorem check_guarantee_holds : CheckGuarantee := by
  intro Ω _ μ _ S _ Q R _ _ A O Dsamp Dsf Pre Suf H h U b N m n T η₀ ε κ t pAP w hD hDsf hL
    hflat hPre hSuf hκ hε hheavy hU ht
  classical
  set Hh : Set S := {v | H.state v = h}
  set ρ := reaching H Dsamp h
  have hRpos : (0 : ℝ) < Fintype.card R := by
    have : Nonempty R := ⟨h⟩
    exact_mod_cast Fintype.card_pos
  have hq : 0 < ε / Fintype.card R := div_pos hε hRpos
  have hHh : Dsamp Hh ≠ 0 := fun h0 => by
    have : Dsamp.real Hh = 0 := by simp [measureReal_def, h0]
    linarith
  have hρ : ρ = Dsamp[|Hh] := reaching_eq H h Dsamp hHh
  have : IsProbabilityMeasure ρ := by rw [hρ]; exact cond_isProbabilityMeasure hHh
  have hκ0 : 0 ≤ κ := measureReal_nonneg.trans (hκ 1)
  set κr : ℝ := κ * Fintype.card R / ε
  have hκr0 : 0 ≤ κr := by positivity
  set κ' := ENNReal.ofReal κr
  have hat : ∀ a, ρ {a} ≤ κ' := fun a => by
    have := reaching_atom_le H h Dsamp (ε / Fintype.card R) κ hq hheavy hκ a
    convert this using 2
    simp only [κr]
    field_simp
  have hpre : ∀ f : S → S, Function.Injective f → ρ {x | f x ∈ U} ≤ ENNReal.ofReal (T * κr) :=
    fun f hf => by
      refine (measure_preimage_le ρ κ' hat f hf U).trans ?_
      rw [ENNReal.ofReal_mul (Nat.cast_nonneg T), ENNReal.ofReal_natCast]
      gcongr
  have hρPre : ρ Preᶜ = 0 := by
    rw [hρ, cond_apply MeasurableSet.of_discrete]
    have : Dsamp (Hh ∩ Preᶜ) = 0 := measure_mono_null Set.inter_subset_right hPre
    simp [this]
  have hnodup : listInt ρ (2 * n) (fun l => if l.Nodup then 0 else 1)
      ≤ ENNReal.ofReal ((2 * n).choose 2 * κr) := by
    refine (listInt_not_nodup_le ρ κ' hat (2 * n)).trans (le_of_eq ?_)
    rw [ofReal_nat_mul]
  have hexU : ∀ f : S → S, Function.Injective f →
      listInt ρ (2 * n) (fun l => if ∃ x ∈ l, x ∈ {y | f y ∈ U} then 1 else 0)
        ≤ ENNReal.ofReal (2 * n * T * κr) := fun f hf => by
    refine (listInt_exists_le ρ _ (2 * n)).trans ?_
    calc ((2 * n : ℕ) : ℝ≥0∞) * ρ {y | f y ∈ U} ≤ ((2 * n : ℕ) : ℝ≥0∞) * ENNReal.ofReal (T * κr) :=
          by gcongr; exact hpre f hf
      _ = ENNReal.ofReal (2 * n * T * κr) := by rw [ofReal_nat_mul]; push_cast; ring_nf
  have hexPre : listInt ρ (2 * n) (fun l => if ∃ x ∈ l, x ∈ Preᶜ then 1 else 0) = 0 :=
    le_antisymm ((listInt_exists_le ρ _ (2 * n)).trans (by rw [hρPre, mul_zero])) zero_le
  set νN := Measure.pi fun _ : Fin N => Dsamp
  set νm := Measure.pi fun _ : Fin m => Dsf
  set μc := μ[|pinned O U b]
  have hdecomp : ∀ E : Set (Ω × ((Fin N → S) × (Fin m → S))), MeasurableSet E →
      (μc.prod (νN.prod νm)) E = ∫⁻ v, ∫⁻ u, μc {ω | (ω, (u, v)) ∈ E} ∂νN ∂νm := by
    intro E hE
    rw [Measure.prod_apply_symm hE, lintegral_prod_symm _ Measurable.of_discrete.aemeasurable]
    rfl
  have hE := measurableSet_checkFails (μ := μ) O H n t (N := N) (m := m) h
  have hsmall : ∀ (v : Fin m → S) (u : Fin N → S) (p : Prop) [Decidable p] (X : ℝ≥0∞), p →
      μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
        checkFails O H n t h z.2 z.1}ᶜ} ≤ (if p then 1 else 0) + X := fun v u p _ X hp => by
    rw [if_pos hp]
    exact (cond_pinned_le_one O U b _).trans le_self_add
  constructor
  · -- soundness
    set acc : Set S := {w | A.state w ∈ A.accept}
    set Mset : Set S := if ρ.real acc ≤ ρ.real accᶜ then acc else accᶜ
    have hMset : ρ Mset = ENNReal.ofReal (minorityShare A H Dsamp h) := by
      rw [← ENNReal.ofReal_toReal (measure_ne_top ρ Mset)]
      congr 1
      change ρ.real Mset = min (ρ.real acc) (ρ.real accᶜ)
      simp only [Mset]
      split_ifs with hc
      · exact (min_eq_left hc).symm
      · exact (min_eq_right (le_of_not_ge hc)).symm
    have hMlab : ∀ l : List S, ¬ (∃ x ∈ l, x ∈ Mset) → ∀ x ∈ l, ∀ z ∈ l, (x ∈ O.L ↔ z ∈ O.L) := by
      intro l hl x hx z hz
      have hx' : x ∉ Mset := fun hm => hl ⟨x, hx, hm⟩
      have hz' : z ∉ Mset := fun hm => hl ⟨z, hz, hm⟩
      rw [hL]
      simp only [Mset] at hx' hz'
      split_ifs at hx' hz'
      · exact iff_of_false hx' hz'
      · exact iff_of_true (not_not.1 hx') (not_not.1 hz')
    set e := exp (-2 * t ^ 2 / n)
    set Gs : (Fin m → S) → List S → ℝ≥0∞ := fun v l =>
      (if ∃ x ∈ l, x ∈ Mset then 1 else 0) + (if l.Nodup then 0 else 1)
        + (if ∃ x ∈ l, x ∈ {y | id y ∈ U} then 1 else 0)
        + (if ∃ x ∈ l, x ∈ Preᶜ then 1 else 0)
        + ∑ j, (if ∃ x ∈ l, x ∈ {y | y * v j ∈ U} then 1 else 0) + m * ENNReal.ofReal e
    have hpt : ∀ v, (∀ j, v j ∈ Suf) → ∀ u, μc {ω | (ω, (u, v)) ∈
        {z : Ω × ((Fin N → S) × (Fin m → S)) | checkFails O H n t h z.2 z.1}}
          ≤ if 2 * n ≤ (membersOf H h u).length
            then Gs v ((membersOf H h u).take (2 * n)) else 0 := by
      intro v hvS u
      split_ifs with hlen
      swap
      · have : {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
            checkFails O H n t h z.2 z.1}} = ∅ := by
          ext ω; simp [checkFails, hlen]
        rw [this, measure_empty]
      set l := (membersOf H h u).take (2 * n)
      have hl : l.length = 2 * n := by simp [l, hlen]
      by_cases hbad : (∃ x ∈ l, x ∈ Mset) ∨ ¬ l.Nodup ∨ (∃ x ∈ l, x ∈ {y | id y ∈ U})
          ∨ (∃ x ∈ l, x ∈ Preᶜ) ∨ (∃ j, ∃ x ∈ l, x ∈ {y | y * v j ∈ U})
      · refine (cond_pinned_le_one O U b _).trans ?_
        simp only [Gs]
        rcases hbad with hb | hb | hb | hb | ⟨j, hb⟩
        · rw [if_pos hb]
          exact le_self_add.trans (le_self_add.trans (le_self_add.trans
            (le_self_add.trans le_self_add)))
        · rw [if_neg hb]
          exact le_add_self.trans (le_self_add.trans (le_self_add.trans
            (le_self_add.trans le_self_add)))
        · rw [if_pos hb]
          exact le_add_self.trans (le_self_add.trans (le_self_add.trans le_self_add))
        · rw [if_pos hb]
          exact le_add_self.trans (le_self_add.trans le_self_add)
        · refine le_trans ?_ (le_add_self.trans le_self_add)
          refine le_trans ?_ (Finset.single_le_sum (fun _ _ => zero_le) (Finset.mem_univ j))
          rw [if_pos hb]
      push Not at hbad
      obtain ⟨hM, hnd, hUl, hPl, hUv⟩ := hbad
      have hPl' : ∀ x ∈ l, x ∈ Pre := fun x hx => by simpa using hPl x hx
      calc μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
            checkFails O H n t h z.2 z.1}}
          ≤ μc (⋃ j, {ω | v j ≠ 1 ∧ 2 * t ≤ leaning O n l (v j) ω}) := by
            refine measure_mono fun ω hω => ?_
            obtain ⟨_, j, hj1, hj2⟩ := hω
            exact Set.mem_iUnion.2 ⟨j, hj1, by rwa [leaning_take] at hj2⟩
        _ ≤ ∑ j, μc {ω | v j ≠ 1 ∧ 2 * t ≤ leaning O n l (v j) ω} :=
            measure_iUnion_fintype_le _ _
        _ ≤ ∑ _j : Fin m, ENNReal.ofReal e := Finset.sum_le_sum fun j _ => by
            by_cases hvj : v j = 1
            · simp [hvj]
            have hset : {ω | v j ≠ 1 ∧ 2 * t ≤ leaning O n l (v j) ω}
                = {ω | (fun y => 2 * t ≤ leanN O y n l (v j)) (fun w => O.noise w ω)} := by
              ext ω; simp [hvj, leaning_eq]
            rw [hset]
            refine (cond_pinned_le O U b (reads l (v j))
              (disjoint_reads (fun x hx => by simpa using hUl x hx)
                (fun x hx => by simpa using hUv j x hx)) _
              (measurableSet_le measurable_const (measurable_leanN O n l (v j)))
              (readsOn_le_leanN O n l (v j) hl _)).trans ?_
            exact sound_tail O (v j) n l t ht hl hnd (ne_mul_of_flat hflat hvj (hvS j) hPl')
              (hMlab l (fun ⟨x, hx, hm⟩ => hM x hx hm))
        _ = m * ENNReal.ofReal e := by simp
        _ ≤ Gs v l := le_add_self
    have hGs : ∀ v, listInt ρ (2 * n) (Gs v)
        ≤ ENNReal.ofReal (2 * n * minorityShare A H Dsamp h + (2 * n).choose 2 * κr
          + 2 * n * T * κr + m * (2 * n * T * κr) + m * e) := by
      intro v
      simp only [Gs]
      rw [listInt_add, listInt_add, listInt_add, listInt_add, listInt_add, listInt_sum,
        listInt_const, hexPre, add_zero]
      have h1 : listInt ρ (2 * n) (fun l => if ∃ x ∈ l, x ∈ Mset then 1 else 0)
          ≤ ENNReal.ofReal (2 * n * minorityShare A H Dsamp h) := by
        refine (listInt_exists_le ρ _ (2 * n)).trans (le_of_eq ?_)
        rw [hMset, ofReal_nat_mul]; push_cast; ring_nf
      have h5 : ∑ j, listInt ρ (2 * n)
            (fun l => if ∃ x ∈ l, x ∈ {y | y * v j ∈ U} then 1 else 0)
          ≤ ENNReal.ofReal (m * (2 * n * T * κr)) := by
        calc _ ≤ ∑ _j : Fin m, ENNReal.ofReal (2 * n * T * κr) :=
              Finset.sum_le_sum fun j _ => hexU _ (mul_left_injective (v j))
          _ = _ := by
              rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
                ofReal_nat_mul]
      have hms : 0 ≤ minorityShare A H Dsamp h :=
        le_min measureReal_nonneg measureReal_nonneg
      have h2n : (0 : ℝ) ≤ 2 * n := by positivity
      have hA : 0 ≤ 2 * n * minorityShare A H Dsamp h := mul_nonneg h2n hms
      have hB : (0 : ℝ) ≤ (2 * n).choose 2 * κr := by positivity
      have hC : (0 : ℝ) ≤ 2 * n * T * κr := by positivity
      have hD' : (0 : ℝ) ≤ m * (2 * n * T * κr) := by positivity
      have hE' : (0 : ℝ) ≤ m * e := mul_nonneg (Nat.cast_nonneg _) (exp_pos _).le
      calc _ ≤ ENNReal.ofReal (2 * n * minorityShare A H Dsamp h)
            + ENNReal.ofReal ((2 * n).choose 2 * κr) + ENNReal.ofReal (2 * n * T * κr)
            + ENNReal.ofReal (m * (2 * n * T * κr)) + m * ENNReal.ofReal e := by
            gcongr
            exact hexU id Function.injective_id
        _ = _ := by
            rw [ofReal_nat_mul, ← ENNReal.ofReal_add (by linarith) (by linarith),
              ← ENNReal.ofReal_add (by linarith) (by linarith),
              ← ENNReal.ofReal_add (by linarith) (by linarith),
              ← ENNReal.ofReal_add (by linarith) (by linarith)]
    have hms : 0 ≤ minorityShare A H Dsamp h := le_min measureReal_nonneg measureReal_nonneg
    refine ENNReal.toReal_le_of_le_ofReal
      (add_nonneg (add_nonneg (by positivity) (mul_nonneg (by positivity) hms)) (by positivity)) ?_
    rw [hdecomp _ hE]
    calc ∫⁻ v, ∫⁻ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
          checkFails O H n t h z.2 z.1}} ∂νN ∂νm
        ≤ ∫⁻ _v, ENNReal.ofReal (2 * n * minorityShare A H Dsamp h + (2 * n).choose 2 * κr
          + 2 * n * T * κr + m * (2 * n * T * κr) + m * e) ∂νm := by
          have hae : ∀ᵐ v ∂νm, ∀ j, v j ∈ Suf := by
            rw [ae_all_iff]
            exact fun j => ae_iff.2 (Measure.pi_eval_preimage_null
              (fun _ : Fin m => Dsf) (i := j) (s := Sufᶜ) hSuf)
          refine lintegral_mono_ae (hae.mono fun v hv => ?_)
          exact (lintegral_mono (hpt v hv)).trans
            ((lintegral_members_le H h Dsamp hHh N (2 * n) (Gs v)).trans (hGs v))
      _ = ENNReal.ofReal (2 * n * minorityShare A H Dsamp h + (2 * n).choose 2 * κr
          + 2 * n * T * κr + m * (2 * n * T * κr) + m * e) := by simp
      _ ≤ _ := by
          refine ENNReal.ofReal_le_ofReal ?_
          have := choose_two_le n
          simp only [κr] at this ⊢
          nlinarith [mul_le_mul_of_nonneg_right this hκr0]
  · -- power
    intro hη hη₀ hpAP hw hN hmargin
    set AP : Set S := {v | v ≠ 1 ∧ (∀ p, p * v ∈ O.L ↔ p ∈ O.L) ∧ v ∈ Suf}
    replace hpAP : pAP ≤ Dsf.real AP := by
      have hsub : {v | v ≠ 1 ∧ ∀ p, p * v ∈ O.L ↔ p ∈ O.L} ⊆ AP ∪ Sufᶜ := fun v hv => by
        by_cases hvS : v ∈ Suf
        exacts [Or.inl ⟨hv.1, hv.2, hvS⟩, Or.inr hvS]
      have h0 : Dsf.real Sufᶜ = 0 := by rw [measureReal_def, hSuf, ENNReal.toReal_zero]
      linarith [hpAP.trans ((measureReal_mono hsub (measure_ne_top _ _)).trans
        (measureReal_union_le AP Sufᶜ))]
    set eN := exp (-2 * (N * ε / Fintype.card R - 2 * n) ^ 2 / N)
    set eH := exp (-(n * w * (1 - 2 * η₀) ^ 2 - 2 * t) ^ 2 / (2 * n))
    set Bp := ENNReal.ofReal (eN + eH + κr * (2 * n ^ 2 + 2 * n * (m + 1) * T))
    have hshort : νN {u | (membersOf H h u).length < 2 * n} ≤ ENNReal.ofReal eN := by
      have := members_short_le H h Dsamp N n (ε / Fintype.card R) hheavy
        (by rw [← mul_div_assoc]; exact hN)
      simpa [eN, mul_div_assoc] using this
    have hwρ : w ≤ min (ρ.real O.L) (ρ.real O.Lᶜ) := by
      rw [hL, Set.compl_ofPred]
      exact hw
    have hper : ∀ v : Fin m → S, (∃ j, v j ∈ AP) →
        ∫⁻ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
          checkFails O H n t h z.2 z.1}ᶜ} ∂νN ≤ Bp := by
      intro v hAP
      obtain ⟨j₀, hj₀1, hj₀AP, hj₀S⟩ := hAP
      have hm1 : 1 ≤ m := Fin.pos j₀
      set vs := v j₀
      set G : List S → Prop := fun l => l.Nodup ∧ ∀ x ∈ l, x ∈ Pre ∧ x ∉ U ∧ x * vs ∉ U
      set Gp : List S → ℝ≥0∞ := fun l =>
        (if l.Nodup then 0 else 1) + (if ∃ x ∈ l, x ∈ {y | id y ∈ U} then 1 else 0)
          + (if ∃ x ∈ l, x ∈ Preᶜ then 1 else 0)
          + (if ∃ x ∈ l, x ∈ {y | y * vs ∈ U} then 1 else 0)
          + (if G l then μ {ω | leaning O n l vs ω < 2 * t} else 0)
      have hpt : ∀ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
          checkFails O H n t h z.2 z.1}ᶜ}
            ≤ {u : Fin N → S | (membersOf H h u).length < 2 * n}.indicator 1 u
              + if 2 * n ≤ (membersOf H h u).length
                then Gp ((membersOf H h u).take (2 * n)) else 0 := by
        intro u
        by_cases hlen : 2 * n ≤ (membersOf H h u).length
        swap
        · rw [Set.indicator_of_mem (show u ∈ {u : Fin N → S | (membersOf H h u).length < 2 * n}
            by simp only [Set.mem_ofPred_eq]; omega), if_neg hlen, add_zero]
          exact cond_pinned_le_one O U b _
        rw [Set.indicator_of_notMem (show u ∉ {u : Fin N → S | (membersOf H h u).length < 2 * n}
          by simp only [Set.mem_ofPred_eq]; omega), zero_add, if_pos hlen]
        set l := (membersOf H h u).take (2 * n)
        have hl : l.length = 2 * n := by simp [l, hlen]
        by_cases hbad : ¬ l.Nodup ∨ (∃ x ∈ l, x ∈ {y | id y ∈ U}) ∨ (∃ x ∈ l, x ∈ Preᶜ)
            ∨ (∃ x ∈ l, x ∈ {y | y * vs ∈ U})
        · refine (cond_pinned_le_one O U b _).trans ?_
          simp only [Gp]
          rcases hbad with hb | hb | hb | hb
          · rw [if_neg hb]
            exact le_self_add.trans (le_self_add.trans (le_self_add.trans le_self_add))
          · rw [if_pos hb]
            exact le_add_self.trans (le_self_add.trans (le_self_add.trans le_self_add))
          · rw [if_pos hb]
            exact le_add_self.trans (le_self_add.trans le_self_add)
          · rw [if_pos hb]
            exact le_add_self.trans le_self_add
        push Not at hbad
        obtain ⟨hnd, hUl, hPl, hUv⟩ := hbad
        have hGl : G l := ⟨hnd, fun x hx =>
          ⟨by simpa using hPl x hx, by simpa using hUl x hx, by simpa using hUv x hx⟩⟩
        calc μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
              checkFails O H n t h z.2 z.1}ᶜ}
            ≤ μc {ω | (fun y => leanN O y n l vs < 2 * t) (fun w => O.noise w ω)} := by
              refine measure_mono fun ω hω => ?_
              simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, checkFails, not_and, not_exists]
                at hω
              simp only [Set.mem_ofPred_eq, ← leaning_eq]
              by_contra hc
              push Not at hc
              exact hω hlen j₀ hj₀1 (by rwa [leaning_take])
          _ ≤ μ {ω | (fun y => leanN O y n l vs < 2 * t) (fun w => O.noise w ω)} :=
              cond_pinned_le O U b (reads l vs)
                (disjoint_reads (fun x hx => by simpa using hUl x hx)
                  (fun x hx => by simpa using hUv x hx)) _
                (measurableSet_lt (measurable_leanN O n l vs) measurable_const)
                (readsOn_leanN_lt O n l vs hl _)
          _ = if G l then μ {ω | leaning O n l vs ω < 2 * t} else 0 := by
              rw [if_pos hGl]; rfl
          _ ≤ Gp l := le_add_self
      have hGp : listInt ρ (2 * n) Gp
          ≤ ENNReal.ofReal ((2 * n).choose 2 * κr) + ENNReal.ofReal (2 * n * T * κr)
            + ENNReal.ofReal (2 * n * T * κr) + ENNReal.ofReal eH := by
        simp only [Gp]
        rw [listInt_add, listInt_add, listInt_add, listInt_add, hexPre, add_zero]
        gcongr
        · exact hexU id Function.injective_id
        · exact hexU _ (mul_left_injective vs)
        · exact power_avg O ρ vs hj₀AP G
            (fun l hG => ⟨hG.1, ne_mul_of_flat hflat hj₀1 hj₀S fun x hx => (hG.2 x hx).1⟩)
            n t η₀ w ht hη hη₀ hwρ hmargin
      calc ∫⁻ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
            checkFails O H n t h z.2 z.1}ᶜ} ∂νN
          ≤ ∫⁻ u, ({u : Fin N → S | (membersOf H h u).length < 2 * n}.indicator 1 u
              + if 2 * n ≤ (membersOf H h u).length
                then Gp ((membersOf H h u).take (2 * n)) else 0) ∂νN := lintegral_mono hpt
        _ = νN {u | (membersOf H h u).length < 2 * n}
            + ∫⁻ u, (if 2 * n ≤ (membersOf H h u).length
                then Gp ((membersOf H h u).take (2 * n)) else 0) ∂νN := by
            rw [lintegral_add_left Measurable.of_discrete,
              lintegral_indicator_one MeasurableSet.of_discrete]
        _ ≤ ENNReal.ofReal eN + (ENNReal.ofReal ((2 * n).choose 2 * κr)
              + ENNReal.ofReal (2 * n * T * κr) + ENNReal.ofReal (2 * n * T * κr)
              + ENNReal.ofReal eH) :=
            add_le_add hshort ((lintegral_members_le H h Dsamp hHh N (2 * n) Gp).trans hGp)
        _ ≤ Bp := by
            rw [← ENNReal.ofReal_add (by positivity) (by positivity),
              ← ENNReal.ofReal_add (by positivity) (by positivity),
              ← ENNReal.ofReal_add (by positivity) (by positivity),
              ← ENNReal.ofReal_add (by positivity) (by positivity)]
            refine ENNReal.ofReal_le_ofReal ?_
            have := choose_two_le n
            have hm1' : (1 : ℝ) ≤ m := by exact_mod_cast hm1
            have hT : (0 : ℝ) ≤ n * T * κr := by positivity
            nlinarith [mul_le_mul_of_nonneg_right this hκr0]
    set NoAP : Set (Fin m → S) := Set.univ.pi fun _ => APᶜ
    have hNoAP : νm NoAP ≤ ENNReal.ofReal ((1 - pAP) ^ m) := by
      simp only [NoAP, νm]
      rw [Measure.pi_pi, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
      have hAPr : Dsf.real AP ≤ 1 := measureReal_le_one
      have hc : Dsf APᶜ = ENNReal.ofReal (1 - Dsf.real AP) := by
        rw [← ENNReal.ofReal_toReal (measure_ne_top Dsf APᶜ)]
        congr 1
        change Dsf.real APᶜ = _
        rw [measureReal_compl MeasurableSet.of_discrete, probReal_univ]
      rw [hc, ← ENNReal.ofReal_pow (by linarith)]
      exact ENNReal.ofReal_le_ofReal (pow_le_pow_left₀ (by linarith) (by linarith) m)
    refine ENNReal.toReal_le_of_le_ofReal ?_ ?_
    · have : 0 ≤ 1 - pAP := by linarith [measureReal_le_one (μ := Dsf) (s := AP)]
      positivity
    rw [show {z : Ω × ((Fin N → S) × (Fin m → S)) | ¬ checkFails O H n t h z.2 z.1}
      = {z | checkFails O H n t h z.2 z.1}ᶜ from rfl, hdecomp _ hE.compl]
    calc ∫⁻ v, ∫⁻ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
          checkFails O H n t h z.2 z.1}ᶜ} ∂νN ∂νm
        ≤ ∫⁻ v, (NoAP.indicator 1 v + Bp) ∂νm := by
          refine lintegral_mono fun v => ?_
          by_cases hv : ∃ j, v j ∈ AP
          · exact (hper v hv).trans le_add_self
          · have hv' : v ∈ NoAP := by
              simp only [NoAP, Set.mem_pi, Set.mem_univ, Set.mem_compl_iff, true_implies]
              exact fun j hj => hv ⟨j, hj⟩
            rw [Set.indicator_of_mem hv']
            refine le_trans ?_ le_self_add
            calc ∫⁻ u, μc {ω | (ω, (u, v)) ∈ {z : Ω × ((Fin N → S) × (Fin m → S)) |
                  checkFails O H n t h z.2 z.1}ᶜ} ∂νN
                ≤ ∫⁻ _u, 1 ∂νN := lintegral_mono fun u => cond_pinned_le_one O U b _
              _ = 1 := by simp
      _ = νm NoAP + Bp := by
          rw [lintegral_add_left Measurable.of_discrete,
            lintegral_indicator_one MeasurableSet.of_discrete]
          simp
      _ ≤ ENNReal.ofReal ((1 - pAP) ^ m) + Bp := add_le_add hNoAP le_rfl
      _ = _ := by
          have : 0 ≤ 1 - pAP := by linarith [measureReal_le_one (μ := Dsf) (s := AP)]
          rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
          congr 1
          simp only [eN, eH, κr]
          ring

end OrthoDFA
