import OrthoDFA.Proofs.LoopE2Round

/-!
# Collisions across rounds

Round `r` repeats an earlier round's suffix with chance at most the pairs times `Dsf`'s largest
atom, and an earlier round's prefix with chance at most the pairs times its populations' largest
atom; a draw of the wrong length has chance zero (`loop_coll_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Laws

variable (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] [IsProbabilityMeasure Dsf]

instance : IsProbabilityMeasure (restLaw D Dsf K) := by unfold restLaw; infer_instance

instance (T : ℕ) : IsProbabilityMeasure (roundLaw D Dsf K T) := by unfold roundLaw; infer_instance

instance (T : ℕ) : IsProbabilityMeasure (loopMeasure μ D Dsf K T) := by
  unfold loopMeasure; infer_instance

lemma restLaw_prefDraw (H : List (Measure (FreeMonoid α))) (hpops : ∀ ν ∈ H, IsProbabilityMeasure ν)
    (j : Option (Fin K)) (a : ℕ) (b : Bool) (A : Set (FreeMonoid α)) :
    restLaw D Dsf K {c | prefDraw H c j a b ∈ A} = loopDistributions D H j A := by
  have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) D H j) :=
    isProbabilityMeasure_loopDistributions D H hpops
  have hcr := measurePreserving_clusterRun (μ := Measure.dirac ()) (K := K) D Dsf H hpops
  set L' := loopDistributions (K := K) D H
  have hev : ∀ j : Option (Fin K), MeasurePreserving (fun f : Option (Fin K) → ℕ → FreeMonoid α =>
      f j a) (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => L' j) (L' j) := fun j =>
    (show MeasurePreserving (fun g : ℕ → FreeMonoid α => g a)
        (Measure.infinitePi fun _ : ℕ => L' j) (L' j) from
      ⟨measurable_pi_apply a, Measure.infinitePi_map_eval _ a⟩).comp (measurePreserving_eval _ j)
  set coord : Run Unit (FreeMonoid α) (Option (Fin K)) → FreeMonoid α := fun x =>
    if b then x.2.2 j a else x.2.1.2 j a
  have hcoord : MeasurePreserving coord (runMeasure (Measure.dirac ()) L' Dsf) (L' j) := by
    cases b
    · exact (hev j).comp (measurePreserving_snd.comp (measurePreserving_fst.comp
        measurePreserving_snd))
    · exact (hev j).comp (measurePreserving_snd.comp measurePreserving_snd)
  have hA : MeasurableSet A := measurableSet_of_countable A
  have hset : (Set.univ : Set Unit) ×ˢ {c : ClusterRest α K | prefDraw H c j a b ∈ A}
      = clusterRun H ⁻¹' (coord ⁻¹' A) := by
    ext ⟨u, c⟩
    simp only [Set.mem_prod, Set.mem_univ, true_and, Set.mem_ofPred_eq, Set.mem_preimage]
    cases b <;> rfl
  calc restLaw D Dsf K {c | prefDraw H c j a b ∈ A}
      = (clusterLaw (Measure.dirac ()) D Dsf K)
          ((Set.univ : Set Unit) ×ˢ {c : ClusterRest α K | prefDraw H c j a b ∈ A}) := by
        unfold clusterLaw
        rw [Measure.prod_prod, measure_univ, one_mul]
    _ = L' j A := by
        rw [hset, hcr.measure_preimage (hcoord.measurable hA).nullMeasurableSet,
          hcoord.measure_preimage hA.nullMeasurableSet]

lemma restLaw_suffix (a : ℕ) (A : Set (FreeMonoid α)) :
    restLaw D Dsf K {c | c.1 a ∈ A} = Dsf A := by
  have hev : MeasurePreserving (fun c : ClusterRest α K => c.1 a) (restLaw D Dsf K) Dsf :=
    (show MeasurePreserving (fun f : ℕ → FreeMonoid α => f a)
        (Measure.infinitePi fun _ : ℕ => Dsf) Dsf from
      ⟨measurable_pi_apply a, Measure.infinitePi_map_eval _ a⟩).comp measurePreserving_fst
  exact hev.measure_preimage (measurableSet_of_countable A).nullMeasurableSet

end Laws

section Fresh

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  [IsProbabilityMeasure D] [IsProbabilityMeasure Dsf]

/-- Round `r`'s clustering draws land in a set the earlier rounds fix with chance at most the
bound the set has under `restLaw`. -/
theorem persist_rest_le (r : Fin K) {Y : Type*} [Countable Y]
    (stage : Ω → (Fin K → RoundDraws α K settings.level4.probeBudget) → Y)
    (hdep : ∀ ω (x x' : Fin K → RoundDraws α K settings.level4.probeBudget),
      (∀ i, i < r → x i = x' i) → stage ω x = stage ω x')
    (hmeas : ∀ y, MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
      stage p.1 p.2 = y})
    (W : Y → (Fin K → RoundDraws α K settings.level4.probeBudget) → Finset (FreeMonoid α))
    (G : Y → Prop)
    (hW : ∀ y x ω ω', G y → stage ω x = y → (∀ w ∈ W y x, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      stage ω' x = y)
    (Bad : Y → Set (ClusterRest α K)) (hG : ∀ y, ¬ G y → Bad y = ∅)
    (hmBad : ∀ y, MeasurableSet (Bad y)) (δ : ENNReal)
    (hBad : ∀ y, restLaw D Dsf K (Bad y) ≤ δ) :
    loopMeasure μ D Dsf K settings.level4.probeBudget
      {p | (p.2 r).1 ∈ Bad (stage p.1 p.2)} ≤ δ := by
  have hσ : MeasurePreserving (fun d : RoundDraws α K settings.level4.probeBudget => (d.2, d.1))
      (roundLaw D Dsf K settings.level4.probeBudget)
      ((Measure.pi fun _ : Fin settings.level4.probeBudget => (volume : Measure unitInterval)).prod
        (restLaw D Dsf K)) :=
    Measure.measurePreserving_swap
  have h := persist_stage_le O (roundLaw D Dsf K settings.level4.probeBudget)
    (Measure.pi fun _ : Fin settings.level4.probeBudget => (volume : Measure unitInterval))
    (restLaw D Dsf K)
    (fun d => (d.2, d.1)) hσ (fun w => (w.2, w.1)) (measurable_snd.prodMk measurable_fst)
    (fun w => rfl) r stage (fun ω x x' h _ => hdep ω x x' h) hmeas W G hW
    (fun y => (Set.univ : Set Ω) ×ˢ Bad y) (fun y hy => by rw [hG y hy, Set.prod_empty])
    (fun y => MeasurableSet.univ.prod (hmBad y)) (fun _ _ => ∅)
    (fun y x₂ ω ω' _ h => ⟨Set.mem_univ _, h.2⟩) (fun _ => Set.univ) (fun _ => MeasurableSet.univ)
    (fun y x _ _ => Finset.disjoint_empty_right _) δ
    (fun y => by rw [Measure.prod_prod, measure_univ, one_mul]; exact hBad y)
  refine le_trans (measure_mono fun p hp => ?_) h
  exact ⟨⟨Set.mem_univ _, hp⟩, Set.mem_univ _⟩

end Fresh

section Coll

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

omit [DecidableEq α] in
lemma measurePreserving_roundRest (i : Fin K) :
    MeasurePreserving (fun p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) =>
      (p.2 i).1) (loopMeasure μ P.D Dsf K settings.level4.probeBudget) (restLaw P.D Dsf K) :=
  measurePreserving_fst.comp ((measurePreserving_eval _ i).comp measurePreserving_snd)

/-- Round `r` repeats a suffix or a prefix an earlier round drew, or a draw has the wrong length,
with chance at most the pairs times the largest atoms. -/
theorem loop_coll_le (Ls NS NP N : ℕ) (qsf qa : ℝ) (hC : 2 ≤ settings.level4.maxLeaves)
    (hN : P.L + settings.level4.maxLeaves ≤ N) (hLs : P.L < Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0) (hqsf : ∀ t, Dsf.real {t} ≤ qsf)
    (hqD : ∀ t, P.D.real {t} ≤ qa) (r : ℕ) (hr : r < K) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (histCodes settings O P.D N p.1 p.2 r).map
            (decodePop P.D settings.level4.walkStart),
          ν {q | Ls ≤ q.toList.length} = 0 ∧ ∀ t, ν.real {t} ≤ qa)
        ∧ ¬ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr}
      ≤ r * NS ^ 2 * qsf + 16 * r * (K + 1) ^ 2 * NP ^ 2 * qa := by
  classical
  set T := settings.level4.probeBudget
  set dec := decodePop P.D settings.level4.walkStart with hdec
  set μL := loopMeasure μ P.D Dsf K T
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  have hqsf0 : 0 ≤ qsf := measureReal_nonneg.trans (hqsf 1)
  have hqa0 : 0 ≤ qa := measureReal_nonneg.trans (hqD 1)
  have hrest := measurePreserving_roundRest settings P Dsf (μ := μ)
  have hnil : ∀ ν ∈ ([] : List (Measure (FreeMonoid α))), IsProbabilityMeasure ν := by simp
  have hsuf : ∀ a, Measurable fun c : ClusterRest α K => c.1 a := fun a =>
    (measurable_pi_apply a).comp measurable_fst
  have hrr : ∀ i : Fin K, Measurable fun p : Ω × (Fin K → RoundDraws α K T) => (p.2 i).1 :=
    fun i => measurable_fst.comp ((measurable_pi_apply i).comp measurable_snd)
  -- draws of the wrong length
  set Z0 : Set (Ω × (Fin K → RoundDraws α K T)) :=
    (⋃ (i : Fin K) (a : ℕ), (fun p : Ω × (Fin K → RoundDraws α K T) => (p.2 i).1) ⁻¹'
        {c : ClusterRest α K | c.1 a ∈ {v : FreeMonoid α | v.toList.length ≠ Ls}})
      ∪ ⋃ (i : Fin K) (a : ℕ) (b : Bool), (fun p : Ω × (Fin K → RoundDraws α K T) => (p.2 i).1) ⁻¹'
        {c : ClusterRest α K | prefDraw ([] : List (Measure (FreeMonoid α))) c none a b
          ∈ {v : FreeMonoid α | P.L < v.toList.length}}
  have hZ0 : μL Z0 = 0 := by
    refine measure_union_null (measure_iUnion_null fun i => measure_iUnion_null fun a => ?_)
      (measure_iUnion_null fun i => measure_iUnion_null fun a => measure_iUnion_null fun b => ?_)
    · rw [(hrest i).measure_preimage (show MeasurableSet {c : ClusterRest α K | c.1 a ∈
        {v : FreeMonoid α | v.toList.length ≠ Ls}} from
          (hsuf a) (measurableSet_of_countable _)).nullMeasurableSet, restLaw_suffix, hDsf]
    · rw [(hrest i).measure_preimage (show MeasurableSet {c : ClusterRest α K |
        prefDraw ([] : List (Measure (FreeMonoid α))) c none a b
          ∈ {v : FreeMonoid α | P.L < v.toList.length}} from
          (measurable_prefDraw (K := K) [] hnil none a b)
        (measurableSet_of_countable _)).nullMeasurableSet, restLaw_prefDraw P.D Dsf [] hnil]
      exact measure_mono_null (fun x hx => by simpa using hx) (ae_iff.1 P.length_le)
  -- suffix repeats
  set SC : Fin r × Fin NS × Fin NS → Set (Ω × (Fin K → RoundDraws α K T)) := fun t =>
    {p | (p.2 rK).1 ∈ {c : ClusterRest α K | c.1 t.2.2 ∈
      ({(p.2 ⟨t.1, t.1.2.trans hr⟩).1.1 t.2.1} : Set (FreeMonoid α))}}
  have hSC : ∀ t, μL (SC t) ≤ ENNReal.ofReal qsf := by
    intro t
    refine persist_rest_le settings O P.D Dsf rK
      (fun _ x => (x ⟨t.1, t.1.2.trans hr⟩).1.1 t.2.1)
      (fun ω x x' h => by
        show (x _).1.1 _ = (x' _).1.1 _
        rw [h ⟨t.1, t.1.2.trans hr⟩ (show t.1.1 < r from t.1.2)])
      (fun s => measurableSet_setOfPred.2 ((measurable_of_countable (fun v : FreeMonoid α => v = s)).comp
        ((hsuf t.2.1).comp (hrr _))))
      (fun _ _ => ∅) (fun _ => True) (fun y x ω ω' _ h _ => h)
      (fun s => {c : ClusterRest α K | c.1 t.2.2 ∈ ({s} : Set (FreeMonoid α))})
      (fun _ h => absurd trivial h)
      (fun _ => (hsuf _) (measurableSet_of_countable _)) (ENNReal.ofReal qsf) (fun s => ?_)
    rw [restLaw_suffix, ← ofReal_measureReal]
    exact ENNReal.ofReal_le_ofReal (hqsf s)
  -- prefix repeats
  set Idx := Fin r × Option (Fin K) × Fin (2 * NP) × Bool × Option (Fin K) × Fin (2 * NP) × Bool
  set stP : Idx → Ω → (Fin K → RoundDraws α K T) → Option (List (HarvestCode α) × FreeMonoid α) :=
    fun t ω x => (stageH settings O P.D N r ω x).map fun c =>
      (c, prefDraw (popsAt settings P.D c t.1) (x ⟨t.1, t.1.2.trans hr⟩).1 t.2.1 t.2.2.1 t.2.2.2.1)
  set BadP : Idx → Option (List (HarvestCode α) × FreeMonoid α) → Set (ClusterRest α K) :=
    fun t y => match y with
    | some (c, s) => if c.length = r ∧ (∀ ν ∈ c.map dec, ∀ u, ν.real {u} ≤ qa) then
        {c1 | t.2.2.2.2.1 ∈ loopPopulations K (c.map dec).length
          ∧ prefDraw (c.map dec) c1 t.2.2.2.2.1 t.2.2.2.2.2.1 t.2.2.2.2.2.2 ∈
            ({s} : Set (FreeMonoid α))}
        else ∅
    | none => ∅
  set PC : Idx → Set (Ω × (Fin K → RoundDraws α K T)) := fun t =>
    {p | (p.2 rK).1 ∈ BadP t (stP t p.1 p.2)}
  have hpopsP : ∀ (c : List (HarvestCode α)) (i : ℕ), ∀ ν ∈ popsAt settings P.D c i,
      IsProbabilityMeasure ν := fun c i => decodePop_prob settings P.D (c.take i)
  have hPC : ∀ t, μL (PC t) ≤ ENNReal.ofReal qa := by
    intro t
    obtain ⟨i, j, a, b, j', a', b'⟩ := t
    refine persist_rest_le settings O P.D Dsf rK (stP (i, j, a, b, j', a', b'))
      (fun ω x x' h => by
        simp only [stP]
        rw [stageH_congr_draws settings O P.D N r ω x x' fun i hi => h i hi,
          h ⟨i, i.2.trans hr⟩ (show i.1 < r from i.2)])
      (fun y => ?_)
      (fun y x => match y with
        | some (c, _) => histReads settings P.D N c x r ω₀
        | none => ∅)
      (fun y => y.isSome)
      (fun y x ω ω' hG hs h => ?_)
      (BadP (i, j, a, b, j', a', b')) (fun y hy => by
        rcases y with _ | ⟨c, s⟩
        · rfl
        · exact absurd rfl hy)
      (fun y => by
        rcases y with _ | ⟨c, s⟩
        · exact MeasurableSet.empty
        · show MeasurableSet (if _ then _ else _)
          split_ifs
          · exact (MeasurableSet.const _).inter ((measurable_prefDraw _ (decodePop_prob settings
              P.D c) _ _ _) (measurableSet_of_countable _))
          · exact MeasurableSet.empty)
      (ENNReal.ofReal qa) (fun y => ?_)
    · -- the stage is measurable
      rcases y with _ | ⟨c, s⟩
      · have : {p : Ω × (Fin K → RoundDraws α K T) | stP (i, j, a, b, j', a', b') p.1 p.2 = none}
            = {p | stageH settings O P.D N r p.1 p.2 = none} := by
          ext p; simp [stP]
        rw [this]
        exact measurableSet_stageH settings O P.D N P.length_le hC hN r none
      · have : {p : Ω × (Fin K → RoundDraws α K T) |
            stP (i, j, a, b, j', a', b') p.1 p.2 = some (c, s)}
            = {p | stageH settings O P.D N r p.1 p.2 = some c}
              ∩ (fun p : Ω × (Fin K → RoundDraws α K T) => (p.2 ⟨i, i.2.trans hr⟩).1) ⁻¹'
                {c1 | prefDraw (popsAt settings P.D c i) c1 j a b ∈ ({s} : Set (FreeMonoid α))} := by
          ext p
          simp only [stP, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_preimage,
            Set.mem_singleton_iff, Option.map_eq_some_iff, Prod.mk.injEq]
          constructor
          · rintro ⟨c', h1, rfl, h2⟩
            exact ⟨h1, h2⟩
          · rintro ⟨h1, h2⟩
            exact ⟨c, h1, rfl, h2⟩
        rw [this]
        exact (measurableSet_stageH settings O P.D N P.length_le hC hN r (some c)).inter
          (hrr _ ((measurable_prefDraw _ (hpopsP c i) j a b) (measurableSet_of_countable _)))
    · -- the stage is fixed by the earlier rounds' reads
      rcases y with _ | ⟨c, s⟩
      · exact absurd hG (by simp)
      · simp only [stP, Option.map_eq_some_iff, Prod.mk.injEq] at hs ⊢
        obtain ⟨c', h1, rfl, h2⟩ := hs
        exact ⟨c', stageH_stop settings O P.D N P.length_le hC hN r c' x ω ω' ω₀ h1 h, rfl, h2⟩
    · -- a draw from round `r`'s populations hits a given string with chance at most `qa`
      rcases y with _ | ⟨c, s⟩
      · show restLaw P.D Dsf K ∅ ≤ _
        simp
      · show restLaw P.D Dsf K (if _ then _ else _) ≤ _
        split_ifs with hc
        · by_cases hj' : j' ∈ loopPopulations K (c.map dec).length
          · have hpm := decodePop_prob settings P.D c
            have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) P.D (c.map dec) j) :=
              isProbabilityMeasure_loopDistributions P.D _ hpm
            refine (measure_mono (t := {c1 : ClusterRest α K | prefDraw (c.map dec) c1 j' a' b'
              ∈ ({s} : Set (FreeMonoid α))}) fun c1 hc1 => hc1.2).trans ?_
            rw [restLaw_prefDraw P.D Dsf _ hpm, ← ofReal_measureReal]
            refine ENNReal.ofReal_le_ofReal ?_
            cases j' with
            | none => exact hqD s
            | some i' =>
              have hi' := lt_of_some_mem_loopPops hj'
              show ((c.map dec).getD i' (Measure.dirac 1)).real {s} ≤ qa
              rw [List.getD_eq_getElem (hn := hi')]
              exact hc.2 _ (List.getElem_mem _) s
          · rw [show {c1 : ClusterRest α K | j' ∈ loopPopulations K (c.map dec).length
                ∧ prefDraw (c.map dec) c1 j' a' b' ∈ ({s} : Set (FreeMonoid α))} = ∅ from
              Set.eq_empty_of_forall_notMem fun c1 hc1 => hj' hc1.1, measure_empty]
            exact zero_le
        · simp
  -- every bad draw collides or has the wrong length
  have hsub : {p : Ω × (Fin K → RoundDraws α K T) | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (histCodes settings O P.D N p.1 p.2 r).map dec,
          ν {q | Ls ≤ q.toList.length} = 0 ∧ ∀ t, ν.real {t} ≤ qa)
        ∧ ¬ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr}
      ⊆ (Z0 ∪ ⋃ t, SC t) ∪ ⋃ t, PC t := by
    rintro p ⟨hok, hlen, hgood, hnot⟩
    by_contra hout
    apply hnot
    set c := histCodes settings O P.D N p.1 p.2 r
    have hpop : popsAt settings P.D c r = c.map dec := by
      simp only [popsAt, List.take_of_length_le (le_of_eq hlen)]
      rfl
    have hst : stageH settings O P.D N r p.1 p.2 = some c := by simp only [stageH, if_pos hok]; rfl
    refine ⟨fun i hi a' ha => ?_, fun i hi j hj a' b ha => ?_, fun i hi a' ha a'' ha' heq => ?_,
      fun i hi j hj a' b ha j' hj' a'' b' ha' heq => ?_⟩
    · by_contra h
      exact hout (Or.inl (Or.inl (Or.inl (Set.mem_iUnion₂.2 ⟨i, a', h⟩))))
    · cases j with
      | none =>
        by_contra h
        refine hout (Or.inl (Or.inl (Or.inr (Set.mem_iUnion.2 ⟨i, Set.mem_iUnion₂.2 ⟨a', b, ?_⟩⟩))))
        show P.L < (prefDraw [] (p.2 i).1 none a' b).toList.length
        have : prefDraw [] (p.2 i).1 none a' b
            = prefDraw (popsAt settings P.D c i) (p.2 i).1 none a' b := by cases b <;> rfl
        rw [this]
        omega
      | some i' =>
        set H' := popsAt settings P.D c i
        have hi' := lt_of_some_mem_loopPops hj
        set ν := H'.getD i' (Measure.dirac 1)
        have hνm : ν ∈ c.map dec := by
          have h1 : ν ∈ H' := by
            simp only [ν]
            rw [List.getD_eq_getElem (hn := hi')]
            exact List.getElem_mem _
          obtain ⟨c', hc', hν⟩ := List.mem_map.1 h1
          exact List.mem_map.2 ⟨c', List.mem_of_mem_take hc', hν⟩
        have : IsProbabilityMeasure ν := hpopsP c i ν (by
          simp only [ν]; rw [List.getD_eq_getElem (hn := hi')]; exact List.getElem_mem _)
        have hdraw : prefDraw H' (p.2 i).1 (some i') a' b = drawFrom ν
            (if b then (p.2 i).1.2.2.2 i' a' else (p.2 i).1.2.2.1 i' a') := by cases b <;> rfl
        rw [hdraw]
        rcases drawFrom_pos_or_one ν (if b then (p.2 i).1.2.2.2 i' a' else (p.2 i).1.2.2.1 i' a')
          with h | h
        · by_contra hl
          have h0 : ν {drawFrom ν (if b then (p.2 i).1.2.2.2 i' a' else (p.2 i).1.2.2.1 i' a')} = 0 :=
            measure_mono_null (fun z hz => by
              rw [Set.mem_singleton_iff.1 hz]; exact (not_lt.1 hl)) (hgood ν hνm).1
          rw [measureReal_def, h0, ENNReal.toReal_zero] at h
          exact lt_irrefl _ h
        · rw [h]; simp only [FreeMonoid.toList_one, List.length_nil]; omega
    · exact hout (Or.inl (Or.inr (Set.mem_iUnion.2 ⟨(⟨i.1, hi⟩, ⟨a', ha⟩, ⟨a'', ha'⟩),
        by show (p.2 rK).1.1 a'' = (p.2 ⟨i.1, _⟩).1.1 a'; exact heq.symm⟩)))
    · refine hout (Or.inr (Set.mem_iUnion.2 ⟨(⟨i.1, hi⟩, j, ⟨a', ha⟩, b, j', ⟨a'', ha'⟩, b'), ?_⟩))
      show (p.2 rK).1 ∈ BadP _ (stP _ p.1 p.2)
      simp only [stP, hst, Option.map_some]
      show (p.2 rK).1 ∈ (if _ then _ else _)
      rw [if_pos ⟨hlen, fun ν hν => (hgood ν hν).2⟩]
      refine ⟨by rw [← hpop]; exact hj', ?_⟩
      show prefDraw (c.map dec) (p.2 rK).1 j' a'' b'
        = prefDraw (popsAt settings P.D c i) (p.2 ⟨i.1, _⟩).1 j a' b
      rw [← hpop]
      exact heq.symm
  -- the sum
  have hcardI : (Fintype.card Idx : ℝ) = 16 * r * (K + 1) ^ 2 * NP ^ 2 := by
    simp only [Idx, Fintype.card_prod, Fintype.card_fin, Fintype.card_option, Fintype.card_bool]
    push_cast
    ring
  have hcardS : (Fintype.card (Fin r × Fin NS × Fin NS) : ℝ) = r * NS ^ 2 := by
    simp only [Fintype.card_prod, Fintype.card_fin]
    push_cast
    ring
  have hbound : μL ((Z0 ∪ ⋃ t, SC t) ∪ ⋃ t, PC t)
      ≤ ENNReal.ofReal (r * NS ^ 2 * qsf + 16 * r * (K + 1) ^ 2 * NP ^ 2 * qa) := by
    calc μL ((Z0 ∪ ⋃ t, SC t) ∪ ⋃ t, PC t)
        ≤ μL Z0 + μL (⋃ t, SC t) + μL (⋃ t, PC t) :=
          (measure_union_le _ _).trans (add_le_add (measure_union_le _ _) le_rfl)
      _ ≤ 0 + ∑ t, ENNReal.ofReal qsf + ∑ t, ENNReal.ofReal qa := by
          rw [hZ0]
          gcongr
          · exact (measure_iUnion_fintype_le _ _).trans (Finset.sum_le_sum fun t _ => hSC t)
          · exact (measure_iUnion_fintype_le _ _).trans (Finset.sum_le_sum fun t _ => hPC t)
      _ = ENNReal.ofReal (r * NS ^ 2 * qsf + 16 * r * (K + 1) ^ 2 * NP ^ 2 * qa) := by
          rw [zero_add, Finset.sum_const, Finset.sum_const, Finset.card_univ, Finset.card_univ,
            nsmul_eq_mul, nsmul_eq_mul, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast,
            ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity),
            ← ENNReal.ofReal_add (by positivity) (by positivity), hcardS, hcardI]
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) ((measure_mono hsub).trans hbound)

end Coll

end OrthoDFA
