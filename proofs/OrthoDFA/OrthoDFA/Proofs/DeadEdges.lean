import OrthoDFA.Proofs.EdgeAttempts
import OrthoDFA.HalvingStrong

/-!
# Draws ending at a given-up edge

A computation scanning another's reads for the first one a test flags, tagging each string the
first time it is read where a second test holds (`Qry.hitScan`).
-/

namespace OrthoDFA

namespace Qry

variable {α β : Type*} [DecidableEq (FreeMonoid α)]

/-- `q`, each string tagged the first time it is read where `tg` holds, stopping with the first
string whose answer `hit` flags. -/
noncomputable def hitScan (tg : FreeMonoid α → Bool) (hit : FreeMonoid α → Option Bool → Bool) :
    Finset (FreeMonoid α) → Qry α β → Qry α (Option (FreeMonoid α))
  | _, pure _ => pure none
  | seen, ask w _ k => ask w (tg w && decide (w ∉ seen)) fun o =>
      if hit w o then pure (some w) else hitScan tg hit (insert w seen) (k o)

variable (tg : FreeMonoid α → Bool) (hit : FreeMonoid α → Option Bool → Bool)

theorem hitScan_asksIn {S : FreeMonoid α → Prop} :
    ∀ (seen : Finset (FreeMonoid α)) (q : Qry α β), q.AsksIn S → (hitScan tg hit seen q).AsksIn S
  | _, pure _, _ => trivial
  | seen, ask w g k, h => ⟨h.1, fun o => by
      simp only []
      split
      · trivial
      · exact hitScan_asksIn _ (k o) (h.2 o)⟩

theorem trace_asksIn {S : FreeMonoid α → Prop} (cut : FreeMonoid α → Option Bool) :
    ∀ q : Qry α β, q.AsksIn S → ∀ e ∈ q.trace cut, S e.1
  | pure _, _, e, he => by simp [trace] at he
  | ask w g k, h, e, he => by
    simp only [trace, List.mem_cons] at he
    rcases he with rfl | he
    · exact h.1
    · exact trace_asksIn cut (k (cut w)) (h.2 _) e he

theorem hitScan_none (cut : FreeMonoid α → Option Bool) :
    ∀ (seen : Finset (FreeMonoid α)) (q : Qry α β), (hitScan tg hit seen q).run cut = none →
      ∀ e ∈ q.trace cut, hit e.1 (cut e.1) = false
  | _, pure _, _, e, he => by simp [trace] at he
  | seen, ask w g k, h, e, he => by
    simp only [hitScan, run] at h
    split at h
    · simp [run] at h
    · rename_i hw
      simp only [trace, List.mem_cons] at he
      rcases he with rfl | he
      · simpa using hw
      · exact hitScan_none cut _ (k (cut w)) h e he

theorem hitScan_some (cut : FreeMonoid α → Option Bool) :
    ∀ (seen : Finset (FreeMonoid α)) (q : Qry α β), (∀ w ∈ seen, hit w (cut w) = false) →
      ∀ b, (hitScan tg hit seen q).run cut = some b →
        hit b (cut b) = true ∧ ∃ r, ∃ hr : r < ((hitScan tg hit seen q).trace cut).length,
          ((hitScan tg hit seen q).trace cut)[r] = (b, tg b)
            ∧ ∀ i, ∀ hi : i < r, (((hitScan tg hit seen q).trace cut)[i]'(hi.trans hr)).1 ≠ b
  | _, pure _, _, b, h => by simp [hitScan, run] at h
  | seen, ask w g k, hs, b, h => by
    simp only [hitScan, run] at h
    by_cases hw : hit w (cut w) = true
    · rw [if_pos hw] at h
      simp only [run, Option.some.injEq] at h
      subst h
      have hns : w ∉ seen := fun hm => by rw [hs w hm] at hw; exact absurd hw (by decide)
      refine ⟨hw, 0, by simp [hitScan, trace], ?_, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
      simp [hitScan, trace, hns]
    · rw [if_neg hw] at h
      have hw' : hit w (cut w) = false := by simpa using hw
      have hs' : ∀ v ∈ insert w seen, hit v (cut v) = false := fun v hv => by
        rcases Finset.mem_insert.1 hv with rfl | hv
        · exact hw'
        · exact hs v hv
      obtain ⟨hb, r, hr, hrb, hfirst⟩ := hitScan_some cut _ (k (cut w)) hs' b h
      have htr : (hitScan tg hit seen (ask w g k)).trace cut
          = (w, tg w && decide (w ∉ seen)) :: (hitScan tg hit (insert w seen) (k (cut w))).trace cut
          := by
        simp only [hitScan, trace, if_neg hw]
      rw [htr]
      refine ⟨hb, r + 1, by simp; omega, by simpa using hrb, fun i hi => ?_⟩
      rcases i with _ | i
      · intro he
        simp only [List.getElem_cons_zero] at he
        rw [← he, hw'] at hb
        exact absurd hb (by decide)
      · simpa using hfirst i (by omega)

theorem hitScan_countP (cut : FreeMonoid α → Option Bool) :
    ∀ (seen : Finset (FreeMonoid α)) (q : Qry α β),
      ((hitScan tg hit seen q).trace cut).countP (·.2)
        ≤ (((q.trace cut).map Prod.fst).toFinset.filter fun w => tg w ∧ w ∉ seen).card
  | _, pure _ => by simp [hitScan, trace]
  | seen, ask w g k => by
    have ih := hitScan_countP cut (insert w seen) (k (cut w))
    set S' := (((k (cut w)).trace cut).map Prod.fst).toFinset
    have hrest : ((if hit w (cut w) = true then (pure (some w) : Qry α (Option (FreeMonoid α)))
        else hitScan tg hit (insert w seen) (k (cut w))).trace cut).countP (·.2)
        ≤ (S'.filter fun v => tg v ∧ v ∉ insert w seen).card := by
      split
      · simp [trace]
      · exact ih
    have hsub1 : (S'.filter fun v => tg v ∧ v ∉ insert w seen)
        ⊆ ((((ask w g k : Qry α β).trace cut).map Prod.fst).toFinset.filter
          fun v => tg v ∧ v ∉ seen) := by
      intro v hv
      obtain ⟨hv1, hv2, hv3⟩ := Finset.mem_filter.1 hv
      simp only [trace, List.map_cons, List.toFinset_cons]
      exact Finset.mem_filter.2 ⟨Finset.mem_insert_of_mem hv1, hv2,
        fun h => hv3 (Finset.mem_insert_of_mem h)⟩
    have htr : (hitScan tg hit seen (ask w g k)).trace cut
        = (w, tg w && decide (w ∉ seen)) :: ((if hit w (cut w) = true then
          (pure (some w) : Qry α (Option (FreeMonoid α)))
          else hitScan tg hit (insert w seen) (k (cut w))).trace cut) := by
      simp only [hitScan, trace]
    have hcons : ∀ (b : Bool) (L : List (FreeMonoid α × Bool)),
        ((w, b) :: L).countP (·.2) = L.countP (·.2) + (if b then 1 else 0) := by
      intro b L; cases b <;> simp
    rw [htr, hcons]
    by_cases hh : (tg w && decide (w ∉ seen)) = true
    · rw [if_pos hh]
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
      have hnot : w ∉ S'.filter fun v => tg v ∧ v ∉ insert w seen := fun h =>
        (Finset.mem_filter.1 h).2.2 (Finset.mem_insert_self _ _)
      have hsub : insert w (S'.filter fun v => tg v ∧ v ∉ insert w seen)
          ⊆ ((((ask w g k : Qry α β).trace cut).map Prod.fst).toFinset.filter
            fun v => tg v ∧ v ∉ seen) := by
        intro v hv
        rcases Finset.mem_insert.1 hv with rfl | hv
        · simp only [trace, List.map_cons, List.toFinset_cons]
          exact Finset.mem_filter.2 ⟨Finset.mem_insert_self _ _, hh⟩
        · exact hsub1 hv
      have := Finset.card_le_card hsub
      rw [Finset.card_insert_of_notMem hnot] at this
      omega
    · rw [if_neg hh, Nat.add_zero]
      exact hrest.trans (Finset.card_le_card hsub1)

end Qry

section Ends

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The probe's walk and search, then, at an edge, the sifts of the prefixes either side of it. -/
noncomputable def qEnds (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Qry α Unit :=
  (qProbe t edges k x).bind fun o => match o with
    | .edge _ fd => (qSift (prefixOf x (max k (fd - 1))) true t).bind fun _ =>
        (qSift (prefixOf x (max k fd)) true t).map fun _ => ()
    | _ => .pure ()

theorem qEnds_asksIn (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qEnds t edges k x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := by
  have hs : ∀ i g, k ≤ i → (qSift (prefixOf x i) g t).AsksIn
      fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := fun i g hi =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, hi, m, hm, he⟩) _ (qSift_asksIn _ g t)
  refine Qry.asksIn_bind (fun o => ?_) _ (qProbe_asksIn_ge t edges k x)
  rcases o with _ | _ | _ | _ | ⟨_, fd⟩ | _ | _
  all_goals first
    | trivial
    | exact Qry.asksIn_bind (fun _ => Qry.asksIn_map _ _ (hs _ true (le_max_left _ _))) _
        (hs _ true (le_max_left _ _))

/-- A sift whose every read lands on the side `side` gives ends where `side` leads. -/
theorem sift_eq_sidePath {cut : FreeMonoid α → Option Bool} (side : FreeMonoid α → Bool) :
    ∀ (t : DTree α) (w : FreeMonoid α) (p : List Bool), t.sift cut w = .inl p →
      (∀ v ∈ (t.route cut w).1, cut v ≠ some (!side v)) → p = sidePath side t w
  | .leaf, w, p, h, _ => by simpa [DTree.sift, DTree.route, sidePath] using h.symm
  | .node m r a, w, p, h, hv => by
    have hm := hv (w * m) (by simp only [DTree.route]; split <;> simp)
    unfold DTree.sift DTree.route at h
    simp only [sidePath]
    rcases hc : cut (w * m) with _ | _ | _ <;> rw [hc] at h hm <;> simp only [] at h
    · simp at h
    · have hs : side (w * m) = false := by
        cases hsd : side (w * m) <;> simp_all
      rw [if_neg (by simp [hs])]
      rcases hr : (r.route cut w).2 with q | q <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        congr 1
        refine sift_eq_sidePath side r w q hr fun v hvr => hv v ?_
        simp only [DTree.route, hc, List.mem_cons]
        exact .inr hvr
      · simp at h
    · have hs : side (w * m) = true := by
        cases hsd : side (w * m) <;> simp_all
      rw [if_pos hs]
      rcases hr : (a.route cut w).2 with q | q <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        congr 1
        refine sift_eq_sidePath side a w q hr fun v hvr => hv v ?_
        simp only [DTree.route, hc, List.mem_cons]
        exact .inr hvr
      · simp at h

variable (R : CutReads α)

/-- At an edge the walk ends at, the edge is learned; and where it leads, for every string the
sides `side` place at its source, where they place the string's extension, some read of the
probe's walk, search or edge lands off its side. -/
theorem ends_minority (side : FreeMonoid α → Bool) {t : DTree α} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} {s1 : List Bool} {c : α} (he : edgeAt R t edges k x = some (s1, c)) :
    ∃ s2 y, edges s1 c = some (s2, y) ∧ (EdgeCorrect side t s1 c s2 →
      ∃ e ∈ (qEnds t edges k x).trace R.cut, R.cut e.1 = some (!side e.1)) := by
  unfold edgeAt at he
  split at he
  swap
  · simp at he
  rename_i ps fd h
  obtain ⟨c', hc', hsc⟩ := Option.map_eq_some_iff.1 he
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hsc
  obtain ⟨ps₀, hi, hw, hb⟩ := probeOutcome_search R h trivial
  obtain ⟨p₀, hk, hf, hkh, hhn, hpn⟩ := walkCheck_inr R hw
  set walkAt : ℕ → List Bool := fun j => ps₀.getD (j - k) [] with hwalk
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  obtain ⟨rfl, hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) ps₀ (hi - k) k hi ps fd hkh le_rfl hpk hpn hb
  have hfdn : fd - 1 < x.toList.length := by omega
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : ((x.toList.drop k).take (hi - k))[fd - 1 - k]'(by simp; omega)
      = x.toList[fd - 1] := by
    simp only [List.getElem_take, List.getElem_drop]
    congr 1
    omega
  rw [hidx, show fd - 1 - k + 1 = fd - k by omega] at hy
  have hc'' : x.toList[fd - 1] = c' := by
    rw [List.getElem?_eq_getElem hfdn] at hc'
    exact Option.some.inj hc'
  rw [hc''] at hy
  refine ⟨_, _, hy, fun hcor => ?_⟩
  by_contra hno
  push Not at hno
  rcases hs3 : t.sift R.cut (prefixOf x (fd - 1)) with p3 | _
  swap
  · simp [agreesAt, hs3] at hfd3
  rcases hs4 : t.sift R.cut (prefixOf x fd) with p4 | _
  swap
  · simp [agreesAt, hs4] at hfd4
  have h3 : p3 = walkAt (fd - 1) := by
    simpa [agreesAt, hs3] using hfd3
  have h4 : p4 ≠ walkAt fd := by
    simpa [agreesAt, hs4] using hfd4
  have htr : ∀ i, k ≤ i → ∀ v ∈ (t.route R.cut (prefixOf x i)).1,
      i = max k (fd - 1) ∨ i = max k fd →
      v ∈ ((qEnds t edges k x).trace R.cut).map Prod.fst := by
    intro i hi v hv hie
    simp only [qEnds, Qry.trace_bind, qProbe_run, h, List.map_append, List.mem_append]
    refine .inr ?_
    simp only [Qry.trace_bind, Qry.trace_map, qSift_trace, List.map_append, List.map_map,
      List.mem_append]
    rcases hie with rfl | rfl
    · exact .inl (by simpa using hv)
    · exact .inr (by simpa using hv)
  have hfa : ∀ i, (i = fd - 1 ∨ i = fd) → ∀ v ∈ (t.route R.cut (prefixOf x i)).1,
      R.cut v ≠ some (!side v) := by
    intro i hi v hv
    have hk' : k ≤ i := by rcases hi with rfl | rfl <;> omega
    obtain ⟨e, he', hev⟩ := List.mem_map.1 (htr i hk' v hv (by
      rcases hi with rfl | rfl
      · left; omega
      · right; omega))
    rw [← hev]
    exact hno e he'
  have e3 := sift_eq_sidePath side t _ _ hs3 (hfa _ (.inl rfl))
  have e4 := sift_eq_sidePath side t _ _ hs4 (hfa _ (.inr rfl))
  have := hcor (prefixOf x (fd - 1)) (by rw [← e3, h3])
  have hpre : prefixOf x (fd - 1) * FreeMonoid.of c' = prefixOf x fd := by
    rw [← hc'', ← prefixOf_succ hfdn, show fd - 1 + 1 = fd by omega]
  rw [hpre, ← e4] at this
  exact h4 this

end Ends

section Classes

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (k : ℕ)

/-- The walk, search and edge, scanned for the first read off its likelier side among the strings
`tg` picks, each tagged the first time it is read. -/
noncomputable def gScan (tg : FreeMonoid α → Bool) (t : DTree α) (edges : Edges α)
    (x : FreeMonoid α) : Qry α (Option (FreeMonoid α)) :=
  (qEnds t edges k x).hitScan tg (fun w o => tg w && decide (o = some (!majSide O B F w))) ∅

theorem gScan_spec (tg : FreeMonoid α → Bool) :
    HarvestSpec (fun w o => o = some (!majSide O B F w)) (gScan O B F k tg) Option.toList k where
  asks t e x := Qry.hitScan_asksIn _ _ _ _ (qEnds_asksIn t e k x)
  first R t e x h := by
    obtain ⟨b, hb⟩ : ∃ b, (gScan O B F k tg t e x).run R.cut = some b := by
      rcases hr : (gScan O B F k tg t e x).run R.cut with _ | b
      · simp [hr] at h
      · exact ⟨b, rfl⟩
    unfold gScan at hb ⊢
    obtain ⟨hh, r, hr, hrb, hfirst⟩ := Qry.hitScan_some _ _ R.cut ∅ (qEnds t e k x)
      (fun w hw => absurd hw (Finset.notMem_empty w)) b hb
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
    refine ⟨b, by simp [hb], hh.2, r, hr, ?_, hfirst⟩
    rw [hrb, hh.1]
  form R t e x b hb := by
    have hb' : (gScan O B F k tg t e x).run R.cut = some b := Option.mem_toList.1 hb
    obtain ⟨-, r, hr, hrb, -⟩ := Qry.hitScan_some _ _ R.cut ∅ (qEnds t e k x)
      (fun w hw => absurd hw (Finset.notMem_empty w)) b hb'
    obtain ⟨i, hi, m, -, he⟩ := Qry.trace_asksIn R.cut _
      (Qry.hitScan_asksIn _ _ _ _ (qEnds_asksIn t e k x)) _ (List.getElem_mem hr)
    rw [hrb] at he
    exact ⟨i, hi, m, he⟩

theorem gScan_tags_le (tg : FreeMonoid α → Bool) (R : CutReads α) (t : DTree α)
    (edges : Edges α) (x : FreeMonoid α) :
    gTags (gScan O B F k tg) R.cut t edges x
      ≤ ((Finset.Icc k (max k x.toList.length) ×ˢ t.midfixes).filter fun p =>
          tg (prefixOf x p.1 * p.2)).card := by
  refine (Qry.hitScan_countP _ _ R.cut ∅ (qEnds t edges k x)).trans ?_
  have hsub : ((((qEnds t edges k x).trace R.cut).map Prod.fst).toFinset.filter
      fun w => tg w ∧ w ∉ (∅ : Finset (FreeMonoid α)))
      ⊆ ((Finset.Icc k (max k x.toList.length) ×ˢ t.midfixes).filter fun p =>
          tg (prefixOf x p.1 * p.2)).image fun p => prefixOf x p.1 * p.2 := by
    intro w hw
    obtain ⟨hw1, hw2, -⟩ := Finset.mem_filter.1 hw
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hw1)
    obtain ⟨i, hi, m, hm, hem⟩ := Qry.trace_asksIn R.cut _ (qEnds_asksIn t edges k x) e he
    refine Finset.mem_image.2 ⟨(min i (max k x.toList.length), m), ?_, ?_⟩
    · refine Finset.mem_filter.2 ⟨Finset.mem_product.2 ⟨Finset.mem_Icc.2 ⟨?_, min_le_right _ _⟩,
        (DTree.mem_midfixes_iff t).2 hm⟩, ?_⟩
      · exact le_min hi (le_max_left _ _)
      · simp only []
        rw [prefixOf_clip k x hi, ← hem]
        exact hw2
    · simp only []
      rw [prefixOf_clip k x hi, ← hem]
  exact (Finset.card_le_card hsub).trans Finset.card_image_le

theorem gScan_tags_bnd (tg : FreeMonoid α → Bool) (R : CutReads α) (t : DTree α)
    (edges : Edges α) (x : FreeMonoid α) :
    gTags (gScan O B F k tg) R.cut t edges x
      ≤ (max k x.toList.length + 1) * t.midfixes.card := by
  refine (gScan_tags_le O B F k tg R t edges x).trans ((Finset.card_filter_le _ _).trans ?_)
  rw [Finset.card_product, Nat.card_Icc]
  exact Nat.mul_le_mul_right _ (by omega)

variable [IsProbabilityMeasure μ]

theorem minorProb_le_one (w : FreeMonoid α) : minorProb O B F w ≤ 1 := measureReal_le_one

theorem rhoMax_nonneg : 0 ≤ rhoMax O B F :=
  Real.iSup_nonneg fun _ => measureReal_nonneg

theorem crossWell_nonneg (u : ℝ) : 0 ≤ crossWell O B F u :=
  Real.iSup_nonneg fun _ => by split_ifs <;> simp [minorProb, measureReal_nonneg]

theorem minor_le_rhoMax (z : FreeMonoid α) :
    μ {ω | (readsAt O B F ω).cut z = some (!majSide O B F z)} ≤ ENNReal.ofReal (rhoMax O B F) := by
  rw [← ofReal_measureReal]
  exact ENNReal.ofReal_le_ofReal (le_ciSup (f := minorProb O B F)
    ⟨1, by rintro _ ⟨w, rfl⟩; exact minorProb_le_one O B F w⟩ z)

theorem minor_le_crossWell (u : ℝ) (z : FreeMonoid α) (hz : undecProb O B F z < u) :
    μ {ω | (readsAt O B F ω).cut z = some (!majSide O B F z)}
      ≤ ENNReal.ofReal (crossWell O B F u) := by
  rw [← ofReal_measureReal]
  refine ENNReal.ofReal_le_ofReal (le_of_eq_of_le ?_ (le_ciSup (f := fun w =>
    if undecProb O B F w < u then minorProb O B F w else 0)
    ⟨1, by rintro _ ⟨w, rfl⟩; dsimp only; split_ifs <;> simp [minorProb_le_one]⟩ z))
  simp only [if_pos hz, minorProb]

end Classes

section Union

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω}

/-- Sets each of measure at most `5·exp(−2ε_r²/prefixMax)`, one per candidate segment list of
each reading count, together have measure at most `δ`. -/
theorem candS_union_le (C : StrongCfg α) (D : Measure (FreeMonoid α)) (Rmax : ℕ)
    (d : Fin Rmax → C.Draws) {δ : ℝ} (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hpm : 0 < prefixMax D C.k)
    (Ef : ℕ → List (List (FreeMonoid α)) → Set Ω)
    (hEf : ∀ r p, μ.real (Ef r p) ≤ 5 * Real.exp (-2 * strongEps C D δ r ^ 2 / prefixMax D C.k)) :
    μ.real (⋃ r ∈ Finset.range (Rmax + 1), ⋃ p ∈ candS C Rmax d r {[]}, Ef r p) ≤ δ := by
  have hlog : 0 < Real.log (5 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ]; linarith)
  have hterm : ∀ r, ∑ p ∈ candS C Rmax d r {[]}, μ.real (Ef r p) ≤ δ / 2 ^ (r + 1) := by
    intro r
    have hcard : ((candS C Rmax d r {[]}).card : ℝ) ≤ 2 ^ ((C.nr + 1) * r) := by
      have := card_candS C Rmax d r {[]}
      simp only [Finset.card_singleton, max_self, one_mul] at this
      exact_mod_cast this
    have hexp : 5 * Real.exp (-2 * strongEps C D δ r ^ 2 / prefixMax D C.k)
        = δ / 2 ^ ((C.nr + 1) * r + r + 1) := by
      unfold strongEps
      rw [Real.sq_sqrt (by
        have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
          mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
        positivity)]
      rw [show -2 * (prefixMax D C.k * ((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2
          + Real.log (5 / δ)) / 2) / prefixMax D C.k
          = -((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2) - Real.log (5 / δ) by
        field_simp; ring]
      rw [Real.exp_sub, Real.exp_neg, ← Real.log_rpow two_pos, Real.exp_log (by positivity),
        Real.exp_log (by positivity), Real.rpow_natCast]
      field_simp
    calc ∑ p ∈ candS C Rmax d r {[]}, μ.real (Ef r p)
        ≤ ∑ _p ∈ candS C Rmax d r {[]}, δ / 2 ^ ((C.nr + 1) * r + r + 1) :=
          Finset.sum_le_sum fun p _ => (hEf r p).trans (le_of_eq hexp)
      _ = (candS C Rmax d r {[]}).card * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ 2 ^ ((C.nr + 1) * r) * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) :=
          mul_le_mul_of_nonneg_right hcard (by positivity)
      _ = δ / 2 ^ (r + 1) := by
          rw [pow_add, pow_add (2 : ℝ) ((C.nr + 1) * r)]
          field_simp
          ring
  calc μ.real _ ≤ ∑ r ∈ Finset.range (Rmax + 1), μ.real (⋃ p ∈ candS C Rmax d r {[]}, Ef r p) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1) :=
        Finset.sum_le_sum fun r _ => (measureReal_biUnion_finset_le _ _).trans (hterm r)
    _ ≤ δ := by
        have h := sum_geometric_two_le (Rmax + 1)
        have : ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1)
            = δ / 2 * ∑ r ∈ Finset.range (Rmax + 1), (1 / 2 : ℝ) ^ r := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [one_div_pow, pow_succ]; ring
        rw [this]
        nlinarith

end Union

section Sub

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : MeasureTheory.Measure Ω} (O : Oracle μ (FreeMonoid α)) (B : State)
  (F : Finset (FreeMonoid α))

/-- A draw at a given-up edge the likelier sides get right has a read off its likelier side,
which one of two scans, splitting the strings by `tgB` and `tgA`, finds. -/
theorem dead_sub (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ)
    (gu : List Bool × α → Prop) (tgB tgA : FreeMonoid α → Bool)
    (htot : ∀ w, tgB w = true ∨ tgA w = true) (x : FreeMonoid α)
    (hx : DeadEdge R t edges k gu x) :
    (((gScan O B F k tgB t edges x).run R.cut).toList ≠ []
        ∧ ∀ b ∈ ((gScan O B F k tgB t edges x).run R.cut).toList, tgB b = true)
      ∨ (((gScan O B F k tgA t edges x).run R.cut).toList ≠ []
        ∧ ∀ b ∈ ((gScan O B F k tgA t edges x).run R.cut).toList, tgA b = true)
      ∨ WrongEdgeAt (majSide O B F) R t edges k x := by
  obtain ⟨⟨s1, c⟩, he, -⟩ := hx
  by_cases hw : WrongEdgeAt (majSide O B F) R t edges k x
  · exact .inr (.inr hw)
  obtain ⟨s2, y, hed, himp⟩ := ends_minority R (majSide O B F) he
  have hcor : EdgeCorrect (majSide O B F) t s1 c s2 := by
    by_contra hn
    exact hw ⟨s1, c, s2, y, he, hed, hn⟩
  obtain ⟨e, het, hcut⟩ := himp hcor
  have scan : ∀ tg : FreeMonoid α → Bool, tg e.1 = true →
      ((gScan O B F k tg t edges x).run R.cut).toList ≠ []
        ∧ ∀ b ∈ ((gScan O B F k tg t edges x).run R.cut).toList, tg b = true := by
    intro tg htg
    rcases hrun : (gScan O B F k tg t edges x).run R.cut with _ | b
    · exfalso
      have := Qry.hitScan_none tg (fun w o => tg w && decide (o = some (!majSide O B F w))) R.cut
        ∅ (qEnds t edges k x) hrun e het
      simp [htg, hcut] at this
    · refine ⟨by simp, fun b' hb' => ?_⟩
      simp only [Option.toList_some, List.mem_singleton] at hb'
      subst hb'
      obtain ⟨hh, -⟩ := Qry.hitScan_some tg _ R.cut ∅ (qEnds t edges k x)
        (fun w hw => absurd hw (Finset.notMem_empty w)) b' hrun
      simp only [Bool.and_eq_true] at hh
      exact hh.1
  rcases htot e.1 with h | h
  · exact .inl (scan tgB h)
  · exact .inr (.inl (scan tgA h))

end Sub

open MeasureTheory in
theorem round_strong_dead_edge : RoundStrongDeadEdge := by
  intro α _ _ Ω _ μ _ C O B F D _ seed δ u Rmax d hδ hδ1 hkL hlen hV
  have hpm := prefixMax_pos D hkL hlen
  have hlog : 0 < Real.log (5 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ]; linarith)
  have heps : ∀ r, 0 < strongEps C D δ r := fun r => by
    unfold strongEps
    refine Real.sqrt_pos.2 (div_pos (mul_pos hpm ?_) two_pos)
    have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
      mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
    linarith
  set ρ := rhoMax O B F with hρdef
  set cw := crossWell O B F u with hcwdef
  have hρ0 : 0 ≤ ρ := rhoMax_nonneg O B F
  have hcw0 : 0 ≤ cw := crossWell_nonneg O B F u
  set tgB : FreeMonoid α → Bool := fun w => decide (u ≤ undecProb O B F w) with htgB
  set tgA : FreeMonoid α → Bool := fun w => decide (undecProb O B F w < u) with htgA
  set bndf : DTree α → ℕ → ℕ := fun t n => (max C.k n + 1) * t.midfixes.card with hbndf
  have hqB := fun (r : ℕ) (segs : List (List (FreeMonoid α))) => exists_good_of_le
    (harvest_holds_le_of (μ := μ) (gScan_spec O B F C.k tgB) bndf
      (fun R t e x => gScan_tags_bnd O B F C.k tgB R t e x) (fun b => tgB b = true) O B F C.K D
      seed segs.flatten _ (segRun_determined C O B F seed segs) hρ0
      (fun z hz => minor_le_rhoMax O B F z) (heps r) hkL hlen hV)
  have hqA := fun (r : ℕ) (segs : List (List (FreeMonoid α))) => exists_good_of_le
    (harvest_holds_le_of (μ := μ) (gScan_spec O B F C.k tgA) bndf
      (fun R t e x => gScan_tags_bnd O B F C.k tgA R t e x) (fun b => tgA b = true) O B F C.K D
      seed segs.flatten _ (segRun_determined C O B F seed segs) hcw0
      (fun z hz => minor_le_crossWell O B F u z (by simpa [htgA] using hz)) (heps r) hkL hlen hV)
  choose E1 hE1 hg1 using hqB
  choose E2 hE2 hg2 using hqA
  set E : Set Ω := ⋃ r ∈ Finset.range (Rmax + 1), ⋃ p ∈ candS C Rmax d r {[]}, (E1 r p ∪ E2 r p)
  refine ⟨E, candS_union_le C D Rmax d hδ hδ1 hpm _ fun r p => ?_, fun ω hω => ?_⟩
  · refine (measureReal_union_le _ _).trans ?_
    have := hE1 r p
    have := hE2 r p
    have := Real.exp_pos (-2 * strongEps C D δ r ^ 2 / prefixMax D C.k)
    linarith
  simp only []
  set R := readsAt O B F ω
  obtain ⟨segs, hs, hp, hst⟩ := strongRound_segs C R Rmax 0 (startAcc C R seed) [] d {[]}
    (Finset.mem_singleton_self _)
  have hr : (strongRound C R Rmax 0 (startAcc C R seed) [] d).2.2.length
      ∈ Finset.range (Rmax + 1) :=
    Finset.mem_range.2 (Nat.lt_succ_of_le (strongRound_len_certs C R Rmax 0 _ [] d).1)
  simp only [E, Set.mem_iUnion, not_exists, Set.mem_union, not_or] at hω
  obtain ⟨hn1, hn2⟩ := hω _ hr _ hs
  have hB := hg1 _ _ ω hn1
  have hA := hg2 _ _ ω hn2
  simp only [passOf] at hB hA
  rw [← hst, ← hp] at hB hA
  unfold strongRun
  set s := (strongRound C R Rmax 0 (startAcc C R seed) [] d).2.1.s
  set gu := givenUp C (strongRound C R Rmax 0 (startAcc C R seed) [] d).2.1
  set SB := {x | ((gScan O B F C.k tgB s.tree s.edges x).run R.cut).toList ≠ []
      ∧ ∀ b ∈ ((gScan O B F C.k tgB s.tree s.edges x).run R.cut).toList, tgB b = true}
  set SA := {x | ((gScan O B F C.k tgA s.tree s.edges x).run R.cut).toList ≠ []
      ∧ ∀ b ∈ ((gScan O B F C.k tgA s.tree s.edges x).run R.cut).toList, tgA b = true}
  set W := {x | DeadEdge R s.tree s.edges C.k gu x ∧ WrongEdgeAt (majSide O B F) R s.tree s.edges
    C.k x}
  have htot : ∀ w, tgB w = true ∨ tgA w = true := fun w => by
    rcases le_or_gt u (undecProb O B F w) with h | h
    · exact .inl (by simp [htgB, h])
    · exact .inr (by simp [htgA, h])
  have hsub : {x | DeadEdge R s.tree s.edges C.k gu x} ⊆ SB ∪ SA ∪ W := by
    intro x hx
    rcases dead_sub O B F R s.tree s.edges C.k gu tgB tgA htot x hx with h | h | h
    · exact .inl (.inl h)
    · exact .inl (.inr h)
    · exact .inr ⟨hx, h⟩
  have hD : D.real {x | DeadEdge R s.tree s.edges C.k gu x} ≤ D.real SB + D.real SA + D.real W :=
    (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans
      (add_le_add (measureReal_union_le _ _) le_rfl))
  have hbL : (bndf s.tree C.L : ℝ) = (C.L + 1) * s.tree.midfixes.card := by
    simp only [hbndf, max_eq_right hkL]
    push_cast
    ring
  have hIB : ∫ x, (gTags (gScan O B F C.k tgB) R.cut s.tree s.edges x : ℝ) ∂D
      ≤ ∫ x, (badReads O B F u s.tree C.k x : ℝ) ∂D := by
    refine integral_le_words D hlen fun x _ => ?_
    have := gScan_tags_le O B F C.k tgB R s.tree s.edges x
    have heq : ((Finset.Icc C.k (max C.k x.toList.length) ×ˢ s.tree.midfixes).filter fun p =>
        tgB (prefixOf x p.1 * p.2) = true).card = badReads O B F u s.tree C.k x := by
      unfold badReads
      congr 1
      exact Finset.filter_congr fun p _ => by simp [htgB]
    exact_mod_cast this.trans heq.le
  have hIA : ∫ x, (gTags (gScan O B F C.k tgA) R.cut s.tree s.edges x : ℝ) ∂D
      ≤ (C.L + 1) * s.tree.midfixes.card := by
    rw [← hbL]
    refine (integral_le_words D hlen (g := fun _ => (bndf s.tree C.L : ℝ)) fun x hx => ?_).trans
      (by simp)
    have := gScan_tags_bnd O B F C.k tgA R s.tree s.edges x
    rw [hx] at this
    exact_mod_cast this
  rw [hbL] at hB hA
  have h1 := mul_le_mul_of_nonneg_left hIB hρ0
  have h2 := mul_le_mul_of_nonneg_left hIA hcw0
  linarith

end OrthoDFA
