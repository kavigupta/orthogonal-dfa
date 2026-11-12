import RLStar.Proofs.LoopLaw
import RLStar.Proofs.RawWindow

/-!
# What a round reads lies within its draws

Where every scheduled state finds its samples among the first `n j` draws of each population `j`'s
streams (`ScanWithin`), two round's draws agreeing on those draws (`AgreeTo`) give the clustering
the same samples at every scheduled state (`sameReads_of_scan`), so the same round
(`roundOf_congr`).  A harvested population's table lies on the table side of the hash
(`prefixesOf_onSide`), so a harvested population's draws holding every scheduled state's samples
(`DrawsHold`) scan within themselves (`scanWithin_of_drawsHold`).  Where the filled draws' first
`scanCap` hit, scan within `scanCap` and return at a scheduled state, the round is the one on the
filled draws (`loopRound_eq_some`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

/-! ## Streams agreeing on a stretch -/

section Stretch

variable {S : Type*} [Stringlike S]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma image_nthDrawIn (A : Set S) (d : S) (w : ℕ → S) (M : ℕ) :
    (Finset.range (drawsIn A w M).length).image (nthDrawIn A d w) = (drawsIn A w M).toFinset := by
  classical
  ext s
  simp only [Finset.mem_image, Finset.mem_range, List.mem_toFinset]
  constructor
  · rintro ⟨k, hk', rfl⟩
    rw [Thin.sub_eq_getD A d w hk', List.getD_eq_getElem _ _ hk']
    exact List.getElem_mem hk'
  · intro hs
    obtain ⟨k, hk', rfl⟩ := List.mem_iff_getElem.1 hs
    exact ⟨k, hk', by rw [Thin.sub_eq_getD A d w hk', List.getD_eq_getElem _ _ hk']⟩

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Two streams agreeing on their first `M` draws, whose draws in `A` among them offer `m` fresh
values off `T`, sample their draws in `A` alike, whatever their spares. -/
lemma freshDraws_nthDrawIn_congr (A : Set S) (T : Finset S) {w w' : ℕ → S} {M m : ℕ}
    (hw : ∀ k < M, w k = w' k) (d d' : S) (hm : m ≤ ((drawsIn A w M).toFinset \ T).card) :
    freshDraws T (nthDrawIn A d w) m = freshDraws T (nthDrawIn A d' w') m := by
  classical
  set E := drawsIn A w M
  have hE : drawsIn A w' M = E := (Thin.entries_congr A hw).symm
  have hk : ∀ k < E.length, nthDrawIn A d w k = nthDrawIn A d' w' k := by
    intro k hk
    have hk' : k < (drawsIn A w' M).length := by rw [hE]; exact hk
    rw [Thin.sub_eq_getD A d w hk, Thin.sub_eq_getD A d' w' hk', hE,
      List.getD_eq_getElem _ _ hk, List.getD_eq_getElem _ _ hk]
  refine freshDraws_congr T (n := E.length) hk ?_
  rw [acceptedAt, image_nthDrawIn]
  exact hm

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Where the first `M` draws of `w` in `A` offer `m` fresh values off `T`, the sample lies in
`A`. -/
lemma freshDraws_nthDrawIn_subset (A : Set S) (T : Finset S) (d : S) (w : ℕ → S) {M m : ℕ}
    (hm : m ≤ ((drawsIn A w M).toFinset \ T).card) :
    ∀ s ∈ freshDraws T (nthDrawIn A d w) m, s ∈ A := by
  classical
  have h1 : acceptedAt T (nthDrawIn A d w) (drawsIn A w M).length
      = (drawsIn A w M).toFinset \ T := by
    rw [acceptedAt, image_nthDrawIn]
  intro s hs
  have h2 := freshDraws_subset_acceptedAt T (nthDrawIn A d w) (n := (drawsIn A w M).length)
    (by rw [h1]; exact hm) hs
  rw [h1, Finset.mem_sdiff, List.mem_toFinset] at h2
  unfold drawsIn at h2
  simpa using (List.mem_filter.1 h2.1).2

end Stretch

/-! ## The clustering's samples -/

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma seqsOf_agree {n : Option (Fin K) → ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none n x y) (j : Option (Fin K)) : seqsOf n x j = seqsOf n y j := by
  obtain ⟨-, -, hg, hp, hc⟩ := hxy
  have hmap : (List.range (n j)).map (x.1.2.1.2 j) = (List.range (n j)).map (y.1.2.1.2 j) :=
    List.map_congr_left fun m hm => hp j m (List.mem_range.1 hm)
  rcases j with _ | i
  · simp only [seqsOf]
    rw [hmap, List.map_congr_left fun m hm => hc m (List.mem_range.1 hm)]
  · simp only [seqsOf]
    rw [hmap, hg]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma drawsIn_onSide (b : Bool) (g : FreeMonoid α → Bool) (w : ℕ → FreeMonoid α) (M : ℕ) :
    drawsIn (onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))) b g) w M
      = ((List.range M).map w).filter fun s => g s == b := by
  classical
  unfold drawsIn
  refine List.filter_congr fun s _ => ?_
  simp [onSide, allHarvest] <;> cases g s <;> cases b <;> rfl

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma prefixDraw_some (x : Draws Ω (FreeMonoid α) (Option (Fin K))) (i : Fin K) :
    (fun k => prefixDraw (some i) k (streams (fun _ => ∅) none x))
      = nthDrawIn (onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α))))
          true x.2)
        (nthDrawIn (onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α))))
          true x.2) (x.1.2.2 (some i) 0) (x.1.2.1.2 (some i)) 0) (x.1.2.1.2 (some i)) := by
  funext n; simp [prefixDraw, streams, tableSeqOf, prependStream]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma prefixDraw_none (x : Draws Ω (FreeMonoid α) (Option (Fin K))) :
    (fun k => prefixDraw none k (streams (fun _ => ∅) none x)) = x.1.2.1.2 none := by
  funext n; simp [prefixDraw, streams]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma card_seqsOf_none {n : Option (Fin K) → ℕ} (x : Draws Ω (FreeMonoid α) (Option (Fin K)))
    (T : Finset (FreeMonoid α)) :
    ((seqsOf n x none).1.toFinset \ T).card = (acceptedAt T (x.1.2.1.2 none) (n none)).card := by
  classical
  have himg : ((List.range (n none)).map (x.1.2.1.2 none)).toFinset
      = (Finset.range (n none)).image (x.1.2.1.2 none) := by ext; simp
  simp only [seqsOf, himg, acceptedAt]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma card_seqsOf_some {n : Option (Fin K) → ℕ} (x : Draws Ω (FreeMonoid α) (Option (Fin K)))
    (i : Fin K) (T : Finset (FreeMonoid α)) :
    ((seqsOf n x (some i)).1.toFinset \ T).card
      = ((drawsIn (onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α))))
          true x.2) (x.1.2.1.2 (some i)) (n (some i))).toFinset \ T).card := by
  rw [drawsIn_onSide]
  simp only [seqsOf, beq_true]

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Agreeing draws give a population the same `m` prefixes, where its first `n j` table draws
offer `m`. -/
lemma prefixesOf_agree {n : Option (Fin K) → ℕ} {m : ℕ}
    {x y : Draws Ω (FreeMonoid α) (Option (Fin K))} (hxy : AgreeTo none n x y) (j : Option (Fin K))
    (hm : m ≤ (seqsOf n x j).1.toFinset.card) :
    prefixesOf j m (streams (fun _ => ∅) none x) = prefixesOf j m (streams (fun _ => ∅) none
        y) := by
  classical
  obtain ⟨-, -, hg, hp, -⟩ := hxy
  rcases j with _ | i
  · unfold prefixesOf
    rw [prefixDraw_none, prefixDraw_none]
    refine freshDraws_congr ∅ (n := n none) (fun k hk => hp none k hk) ?_
    rw [← card_seqsOf_none, Finset.sdiff_empty]
    exact hm
  · unfold prefixesOf
    rw [prefixDraw_some, prefixDraw_some, ← hg]
    refine freshDraws_nthDrawIn_congr _ ∅ (M := n (some i)) (fun k hk => hp (some i) k hk) _ _ ?_
    rw [← card_seqsOf_some, Finset.sdiff_empty]
    exact hm

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Agreeing draws give a population the same `m` certification prefixes off `T`, where its first
`n j` certification draws offer `m` off `T`. -/
lemma certOf_agree {n : Option (Fin K) → ℕ} {m : ℕ}
    {x y : Draws Ω (FreeMonoid α) (Option (Fin K))} (hxy : AgreeTo none n x y)
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (hm : m ≤ ((seqsOf n x j).2.toFinset \ T).card) :
    certOf T j m (streams (fun _ => ∅) none x) = certOf T j m (streams (fun _ => ∅) none y) := by
  classical
  obtain ⟨-, -, hg, hp, hc⟩ := hxy
  rcases j with _ | i
  · have hx : (fun k => certPrefix none k (streams (fun _ => ∅) none x)) = x.1.2.2 none := by
      funext k; simp [certPrefix, streams]
    have hy : (fun k => certPrefix none k (streams (fun _ => ∅) none y)) = y.1.2.2 none := by
      funext k; simp [certPrefix, streams]
    unfold certOf
    rw [hx, hy]
    refine freshDraws_congr T (n := n none) (fun k hk => hc k hk) ?_
    have himg : ((List.range (n none)).map (x.1.2.2 none)).toFinset
        = (Finset.range (n none)).image (x.1.2.2 none) := by ext; simp
    simp only [seqsOf, himg] at hm
    simpa [acceptedAt] using hm
  · set R := onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))) false x.2
    have hx : (fun k => certPrefix (some i) k (streams (fun _ => ∅) none x))
        = nthDrawIn R (nthDrawIn R (x.1.2.2 (some i) 1) (x.1.2.1.2 (some i)) 0)
          (x.1.2.1.2 (some i)) := by
      funext k; simp [certPrefix, streams, reserveSeqOf, R]
    have hy : (fun k => certPrefix (some i) k (streams (fun _ => ∅) none y))
        = nthDrawIn R (nthDrawIn R (y.1.2.2 (some i) 1) (y.1.2.1.2 (some i)) 0)
          (y.1.2.1.2 (some i)) := by
      funext k; simp [certPrefix, streams, reserveSeqOf, R, hg]
    unfold certOf
    rw [hx, hy]
    refine freshDraws_nthDrawIn_congr R T (M := n (some i)) (fun k hk => hp (some i) k hk) _ _ ?_
    rw [drawsIn_onSide]
    convert hm using 3
    simp only [seqsOf]
    exact congrArg List.toFinset (List.filter_congr fun s _ => by cases x.2 s <;> rfl)

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- A harvested population's `m` prefixes lie on the table side of the hash, where its first
`n (some i)` draws offer `m` there. -/
lemma prefixesOf_onSide {n : Option (Fin K) → ℕ} {m : ℕ}
    (x : Draws Ω (FreeMonoid α) (Option (Fin K))) (i : Fin K)
    (hm : m ≤ (seqsOf n x (some i)).1.toFinset.card) :
    ∀ s ∈ prefixesOf (some i) m (streams (fun _ => ∅) none x), x.2 s = true := by
  classical
  intro s hs
  unfold prefixesOf at hs
  rw [prefixDraw_some] at hs
  have hm' := hm
  rw [← Finset.sdiff_empty (s := (seqsOf n x (some i)).1.toFinset), card_seqsOf_some] at hm'
  exact (freshDraws_nthDrawIn_subset _ ∅ _ _ hm' s hs).1

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- The probes' population's `m` prefixes, where its first `n none` draws offer `m`, are the first
`m` distinct of any stream agreeing with its draws there. -/
lemma prefixesOf_none_eq {n : Option (Fin K) → ℕ} {m : ℕ}
    (x : Draws Ω (FreeMonoid α) (Option (Fin K))) {w : ℕ → FreeMonoid α}
    (hw : ∀ k < n none, x.1.2.1.2 none k = w k) (hm : m ≤ (seqsOf n x none).1.toFinset.card) :
    prefixesOf none m (streams (fun _ => ∅) none x) = freshDraws ∅ w m := by
  unfold prefixesOf
  rw [prefixDraw_none]
  refine freshDraws_congr ∅ (n := n none) hw ?_
  rw [← card_seqsOf_none, Finset.sdiff_empty]
  exact hm

/-! ## The same samples, the same round -/

/-- Two runs with the same noise and suffixes, and at every state of `sched` the same prefixes and
certification prefixes of each population. -/
def SameReads (sched : Finset State) (pops : Finset (Option (Fin K)))
    (r r' : Run Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  r.1 = r'.1 ∧ r.2.1.1 = r'.2.1.1 ∧ ∀ B ∈ sched, ∀ j ∈ pops,
    prefixesOf j B.npref r = prefixesOf j B.npref r'
      ∧ ∀ e ≤ B.npref, certOf (prefixesAt pops B.npref r) j (B.npref + e) r
        = certOf (prefixesAt pops B.npref r') j (B.npref + e) r'

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Where the first `n j` draws of each population hold every scheduled state's samples, draws
agreeing on them give the clustering the same samples. -/
theorem sameReads_of_scan {sched : Finset State} {pops : Finset (Option (Fin K))}
    {n : Option (Fin K) → ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none n x y) (hs : ScanWithin sched pops n x) :
    SameReads sched pops (streams (fun _ => ∅) none x) (streams (fun _ => ∅) none y) := by
  refine ⟨hxy.1, hxy.2.1, fun B hB j hj => ⟨prefixesOf_agree hxy j (hs B hB j hj).1, ?_⟩⟩
  intro e he
  have hP : prefixesAt pops B.npref (streams (fun _ => ∅) none x)
      = prefixesAt pops B.npref (streams (fun _ => ∅) none y) :=
    Finset.biUnion_congr rfl fun j' hj' => prefixesOf_agree hxy j' (hs B hB j' hj').1
  rw [← hP]
  exact certOf_agree hxy _ j ((hs B hB j hj).2 e he)

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Agreeing draws scan alike. -/
theorem scanWithin_agree {sched : Finset State} {pops : Finset (Option (Fin K))}
    {n : Option (Fin K) → ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none n x y) (hs : ScanWithin sched pops n x) : ScanWithin sched pops n y := by
  intro B hB j hj
  have hP : prefixesAt pops B.npref (streams (fun _ => ∅) none x)
      = prefixesAt pops B.npref (streams (fun _ => ∅) none y) :=
    Finset.biUnion_congr rfl fun j' hj' => prefixesOf_agree hxy j' (hs B hB j' hj').1
  rw [← seqsOf_agree hxy j, ← hP]
  exact hs B hB j hj

omit [DecidableEq α] in
lemma AgreeTo.symm {J : Type*} {uni : J} {n : J → ℕ} {x y : Draws Ω (FreeMonoid α) J}
    (h : AgreeTo uni n x y) : AgreeTo uni n y x :=
  ⟨h.1.symm, h.2.1.symm, h.2.2.1.symm, fun j m hm => (h.2.2.2.1 j m hm).symm,
    fun m hm => (h.2.2.2.2 m hm).symm⟩

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Where the first `M` draws of each population hold every scheduled state's samples, so do the
first `c j`: all `M` of the probes' population's, and a harvested population's where they number
`M` or hold every scheduled state's samples (`DrawsHold`) off `U`, the probes' population's
table. -/
lemma scanWithin_of_drawsHold {sched : Finset State} {pops : Finset (Option (Fin K))} {M : ℕ}
    {x : Draws Ω (FreeMonoid α) (Option (Fin K))} {c : Option (Fin K) → ℕ}
    {U : ℕ → Finset (FreeMonoid α)} (hs : ScanWithin sched pops (fun _ => M) x)
    (hnone : c none = M)
    (hU : ∀ B ∈ sched, U B.npref = prefixesOf none B.npref (streams (fun _ => ∅) none x))
    (hpop : ∀ i, some i ∈ pops → c (some i) = M
      ∨ DrawsHold sched x.2 U ((List.range (c (some i))).map (x.1.2.1.2 (some i)))) :
    ScanWithin sched pops c x := by
  classical
  intro B hB j hj
  rcases j with _ | i
  · have e : seqsOf c x none = seqsOf (fun _ => M) x none := by simp only [seqsOf, hnone]
    rw [e]
    exact hs B hB none hj
  rcases hpop i hj with hM | hD
  · have e : seqsOf c x (some i) = seqsOf (fun _ => M) x (some i) := by simp only [seqsOf, hM]
    rw [e]
    exact hs B hB (some i) hj
  obtain ⟨h1, h2⟩ := hD B hB
  refine ⟨h1, fun e he => le_trans (by omega) (h2.trans (Finset.card_le_card ?_))⟩
  intro s hs'
  rw [Finset.mem_sdiff] at hs' ⊢
  refine ⟨hs'.1, fun hT => hs'.2 ?_⟩
  have hside : x.2 s = false := by
    have := (List.mem_filter.1 (List.mem_toFinset.1 hs'.1)).2
    simpa using this
  obtain ⟨j', hj', hsj⟩ := Finset.mem_biUnion.1 hT
  rcases j' with _ | i'
  · rw [hU B hB]
    exact hsj
  · have := prefixesOf_onSide (n := fun _ => M) x i' (hs B hB (some i') hj').1 s hsj
    rw [hside] at this
    exact absurd this (by simp)

section Congr

variable {sched : Finset State} {pops : Finset (Option (Fin K))}
  {r r' : Run Ω (FreeMonoid α) (Option (Fin K))}

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.prefixesAt (h : SameReads sched pops r r') {B : State} (hB : B ∈ sched) :
    prefixesAt pops B.npref r = prefixesAt pops B.npref r' :=
  Finset.biUnion_congr rfl fun j hj => (h.2.2 B hB j hj).1

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.poolAt (h : SameReads sched pops r r') (n : ℕ) : poolAt n r = poolAt n r' := by
  unfold OrthoDFA.poolAt suffixDraw
  rw [h.2.1]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.clusterAt (h : SameReads sched pops r r') (mq : FreeMonoid α → Ω → ℝ)
    {B : State} (hB : B ∈ sched) : clusterAt mq pops r B = clusterAt mq pops r' B := by
  unfold OrthoDFA.clusterAt screenedAt oracleNoise
  rw [h.prefixesAt hB, h.poolAt, h.1]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.ret (h : SameReads sched pops r r') (mq : FreeMonoid α → Ω → ℝ)
    (uni : Option (Fin K)) (huni : uni ∈ pops) (η₀ ι εc a : ℝ) {B : State} (hB : B ∈ sched) :
    r ∈ OrthoDFA.ret mq pops uni η₀ ι εc a B ↔ r' ∈ OrthoDFA.ret mq pops uni η₀ ι εc a B := by
  have hcl := h.clusterAt mq hB
  have hP := h.prefixesAt hB
  have hfam : familyAt mq pops r B = familyAt mq pops r' B := by unfold familyAt; rw [hcl]
  have hrefs : refsAt mq pops r B = refsAt mq pops r' B := by unfold refsAt; rw [h.poolAt, hfam]
  have hω : oracleNoise r = oracleNoise r' := h.1
  have hc : ∀ j ∈ pops, ∀ e ≤ B.npref, certOf (OrthoDFA.prefixesAt pops B.npref r) j (B.npref + e) r
      = certOf (OrthoDFA.prefixesAt pops B.npref r') j (B.npref + e) r' :=
    fun j hj => (h.2.2 B hB j hj).2
  have hc0 : ∀ j ∈ pops, certOf (OrthoDFA.prefixesAt pops B.npref r) j B.npref r
      = certOf (OrthoDFA.prefixesAt pops B.npref r') j B.npref r' := fun j hj => by
    simpa using hc j hj 0 (Nat.zero_le _)
  have hkept : ∀ j ∈ pops, keptAt pops j B.npref r ↔ keptAt pops j B.npref r' := fun j hj => by
    unfold keptAt; rw [(h.2.2 B hB j hj).1, hc0 j hj]
  simp only [OrthoDFA.ret, Set.mem_setOf_eq]
  rw [hcl, hfam, hrefs, hω]
  refine and_congr Iff.rfl (and_congr (forall₂_congr fun j hj => ?_) (forall₂_congr fun j hj => ?_))
  · rw [hkept j hj, hc0 j hj]
  · rw [hkept j hj]
    exact imp_congr_right fun _ => exists_congr fun e => and_congr_right fun he => by
      rw [hc j hj e he, hc0 uni huni]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.roundPrefs (h : SameReads sched pops r r') :
    OrthoDFA.roundPrefs sched pops r = OrthoDFA.roundPrefs sched pops r' := by
  unfold OrthoDFA.roundPrefs
  refine Finset.biUnion_congr rfl fun B hB => ?_
  congr 1
  · exact h.prefixesAt hB
  · refine Finset.biUnion_congr rfl fun j hj => Finset.biUnion_congr rfl fun e he => ?_
    exact (h.2.2 B hB j hj).2 e (Nat.lt_succ_iff.1 (Finset.mem_range.1 he))

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma SameReads.roundSuffs (h : SameReads sched pops r r') :
    OrthoDFA.roundSuffs sched r = OrthoDFA.roundSuffs sched r' := by
  unfold OrthoDFA.roundSuffs
  exact Finset.biUnion_congr rfl fun B _ => h.poolAt B.nsuff

end Congr

variable {μ : Measure Ω}

/-- Runs with the same samples at every scheduled state, one of which returns, make the same
round. -/
theorem roundOf_congr (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : SpecHist α) {r r' : Run Ω (FreeMonoid α) (Option (Fin K))}
    (hr : SameReads (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) r r')
    (hret : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      r ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B)
    (u : Fin settings.level3.probeBudget → unitInterval) :
    roundOf settings O D h r u = roundOf settings O D h r' u := by
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  have hpred : (fun B => B ∈ sched ∧ r ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B)
      = fun B => B ∈ sched ∧ r' ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B := by
    funext B
    exact propext (and_congr_right fun hB => hr.ret O.mq none (Finset.mem_insert_self _ _) _ _ _ _
        hB)
  have hB := (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ r ∈ ret O.mq ps none
    settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    (let ⟨B, hB, hx⟩ := hret; ⟨B, hB, hx⟩)).1
  unfold roundOf
  simp only
  rw [← hpred, hr.clusterAt O.mq hB, hr.1, clusterKeys, clusterKeys, hr.roundPrefs, hr.roundSuffs]

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma exists_getElem_of_mem_loopPopulations {h : SpecHist α} {i : Fin K}
    (hi : some i ∈ loopPopulations K h.pops.length) : ∃ s, h.pops[i.val]? = some s := by
  simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or, Finset.mem_image,
    Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq] at hi
  obtain ⟨a, ha, rfl⟩ := hi
  exact ⟨_, List.getElem?_eq_getElem ha⟩

/-- Where the filled draws' first `scanCap` after the history `h` hit, scan within `scanCap` and
return at a scheduled state, the round is the one on the filled draws. -/
theorem loopRound_eq_some (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (hhit : (clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap
      settings.scanCap h (d.1, d.2.1)).isSome)
    (hscan : ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (fun _ => settings.scanCap)
      (idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)))
    (hret : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    loopRound settings O D Dsf h d = some (roundOf settings O D h
      (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1))
      d.2.2) := by
  classical
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set M := settings.scanCap
  set I := idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
    with hI
  obtain ⟨tabs, hdl, hag, hnone, hpop⟩ := drawLists_of_hit (sched := sched) hhit
  set dr := drawsOf d.1 D Dsf M h.prefs h.suffs d.2.1 tabs
  have hdl' : drawLists D settings.level3.walkStart settings.level3.harvestReadCap M sched h d.2.1
      = some tabs := hdl
  have hag' : AgreeTo none (fun j => (tabs j).length) dr I := hag
  have hlen : (tabs none).length = M := by rw [hnone, uniDraws, List.length_ofFn]
  have hU : ∀ B ∈ sched, uniTable D M h.prefs (d.2.1.2.1 none) B.npref
      = prefixesOf none B.npref (streams (fun _ => ∅) none I) := by
    intro B hB
    refine (prefixesOf_none_eq (n := fun _ => M)
      (w := padList (uniDraws D M h.prefs (d.2.1.2.1 none))) I (fun k hk => ?_)
      (hscan B hB none (Finset.mem_insert_self _ _)).1).symm
    simp only [hI, idealDraws, fillDraw, padList, uniDraws]
    rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_ofFn]
  have hsc : ScanWithin sched ps (fun j => (tabs j).length) I := by
    refine scanWithin_of_drawsHold (U := uniTable D M h.prefs (d.2.1.2.1 none)) hscan hlen hU
      fun i hi => ?_
    obtain ⟨s, hs⟩ := exists_getElem_of_mem_loopPopulations hi
    obtain ⟨-, hD, -⟩ := hpop i s hs
    rcases hD with hD | hM
    · right
      have hl : (List.range (tabs (some i)).length).map (I.1.2.1.2 (some i)) = tabs (some i) := by
        refine List.ext_getElem (by simp) fun n h1 h2 => ?_
        rw [List.getElem_map, List.getElem_range, ← hag'.2.2.2.1 (some i) n h2]
        simp only [dr, drawsOf, padList]
        rw [List.getD_eq_getElem _ _ h2]
      rw [hl]
      exact hD
    · exact Or.inl hM
  have hsr := sameReads_of_scan hag'.symm hsc
  have hsc' := scanWithin_agree hag'.symm hsc
  have hsr' := sameReads_of_scan hag' hsc'
  have hret' : ∃ B ∈ sched, streams (fun _ => ∅) none dr ∈ ret O.mq ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B := by
    obtain ⟨B, hB, hx⟩ := hret
    exact ⟨B, hB, (hsr.ret O.mq none (Finset.mem_insert_self _ _) _ _ _ _ hB).1 hx⟩
  have hro := roundOf_congr settings O D h hsr' hret' d.2.2
  unfold loopRound
  change (drawLists D settings.level3.walkStart settings.level3.harvestReadCap M sched h
    d.2.1).bind _ = _
  rw [hdl', Option.bind_some]
  rw [if_pos ⟨hsc', hret'⟩]
  exact congrArg some hro

end OrthoDFA

end
