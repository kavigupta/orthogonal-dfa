import OrthoDFA.Proofs.LoopLaw
import OrthoDFA.Proofs.RawWindow

/-!
# What a round reads lies within its first `scanCap` draws

Where every scheduled state finds its samples among the first `M` draws of each stream
(`ScanWithin`), two round's draws agreeing on those draws (`AgreeTo`) give the clustering the same
samples at every scheduled state (`sameReads_of_scan`), so the same round (`roundOf_congr`).  Where
the filled draws hit, scan within `M` and return at a scheduled state, the round is the one on the
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
  have himg : (Finset.range E.length).image (nthDrawIn A d w) = E.toFinset := by
    ext s
    simp only [Finset.mem_image, Finset.mem_range, List.mem_toFinset]
    constructor
    · rintro ⟨k, hk', rfl⟩
      rw [Thin.sub_eq_getD A d w hk', List.getD_eq_getElem _ _ hk']
      exact List.getElem_mem hk'
    · intro hs
      obtain ⟨k, hk', rfl⟩ := List.mem_iff_getElem.1 hs
      exact ⟨k, hk', by rw [Thin.sub_eq_getD A d w hk', List.getD_eq_getElem _ _ hk']⟩
  refine freshDraws_congr T (n := E.length) hk ?_
  rw [acceptedAt, himg]
  exact hm

end Stretch

/-! ## The clustering's samples -/

omit [DecidableEq α] [MeasurableSpace Ω] in
lemma seqsOf_agree {M : ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none M x y) (j : Option (Fin K)) : seqsOf M x j = seqsOf M y j := by
  obtain ⟨-, -, hg, hp, hc⟩ := hxy
  have hmap : (List.range M).map (x.1.2.1.2 j) = (List.range M).map (y.1.2.1.2 j) :=
    List.map_congr_left fun n hn => hp j n (List.mem_range.1 hn)
  rcases j with _ | i
  · simp only [seqsOf]
    rw [hmap, List.map_congr_left fun n hn => hc n (List.mem_range.1 hn)]
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
/-- Agreeing draws give a population the same `m` prefixes, where its first `M` table draws offer
`m`. -/
lemma prefixesOf_agree {M m : ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none M x y) (j : Option (Fin K)) (hm : m ≤ (seqsOf M x j).1.toFinset.card) :
    prefixesOf j m (streams (fun _ => ∅) none x) = prefixesOf j m (streams (fun _ => ∅) none y) := by
  classical
  obtain ⟨-, -, hg, hp, -⟩ := hxy
  rcases j with _ | i
  · have hx : (fun n => prefixDraw none n (streams (fun _ => ∅) none x)) = x.1.2.1.2 none := by
      funext n; simp [prefixDraw, streams]
    have hy : (fun n => prefixDraw none n (streams (fun _ => ∅) none y)) = y.1.2.1.2 none := by
      funext n; simp [prefixDraw, streams]
    unfold prefixesOf
    rw [hx, hy]
    refine freshDraws_congr ∅ (n := M) (fun k hk => hp none k hk) ?_
    have himg : ((List.range M).map (x.1.2.1.2 none)).toFinset
        = (Finset.range M).image (x.1.2.1.2 none) := by ext; simp
    simp only [seqsOf, himg] at hm
    simpa [acceptedAt] using hm
  · set A := onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))) true x.2
    have hx : (fun n => prefixDraw (some i) n (streams (fun _ => ∅) none x))
        = nthDrawIn A (nthDrawIn A (x.1.2.2 (some i) 0) (x.1.2.1.2 (some i)) 0) (x.1.2.1.2 (some i)) := by
      funext n; simp [prefixDraw, streams, tableSeqOf, prependStream, A]
    have hy : (fun n => prefixDraw (some i) n (streams (fun _ => ∅) none y))
        = nthDrawIn A (nthDrawIn A (y.1.2.2 (some i) 0) (y.1.2.1.2 (some i)) 0) (y.1.2.1.2 (some i)) := by
      funext n; simp [prefixDraw, streams, tableSeqOf, prependStream, A, hg]
    unfold prefixesOf
    rw [hx, hy]
    refine freshDraws_nthDrawIn_congr A ∅ (M := M) (fun k hk => hp (some i) k hk) _ _ ?_
    rw [Finset.sdiff_empty, drawsIn_onSide]
    convert hm using 3
    simp only [seqsOf, beq_true]

omit [DecidableEq α] [MeasurableSpace Ω] in
/-- Agreeing draws give a population the same `m` certification prefixes off `T`, where its first
`M` certification draws offer `m` off `T`. -/
lemma certOf_agree {M m : ℕ} {x y : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hxy : AgreeTo none M x y) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (hm : m ≤ ((seqsOf M x j).2.toFinset \ T).card) :
    certOf T j m (streams (fun _ => ∅) none x) = certOf T j m (streams (fun _ => ∅) none y) := by
  classical
  obtain ⟨-, -, hg, hp, hc⟩ := hxy
  rcases j with _ | i
  · have hx : (fun n => certPrefix none n (streams (fun _ => ∅) none x)) = x.1.2.2 none := by
      funext n; simp [certPrefix, streams]
    have hy : (fun n => certPrefix none n (streams (fun _ => ∅) none y)) = y.1.2.2 none := by
      funext n; simp [certPrefix, streams]
    unfold certOf
    rw [hx, hy]
    refine freshDraws_congr T (n := M) (fun k hk => hc k hk) ?_
    have himg : ((List.range M).map (x.1.2.2 none)).toFinset
        = (Finset.range M).image (x.1.2.2 none) := by ext; simp
    simp only [seqsOf, himg] at hm
    simpa [acceptedAt] using hm
  · set R := onSide (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))) false x.2
    have hx : (fun n => certPrefix (some i) n (streams (fun _ => ∅) none x))
        = nthDrawIn R (nthDrawIn R (x.1.2.2 (some i) 1) (x.1.2.1.2 (some i)) 0) (x.1.2.1.2 (some i)) := by
      funext n; simp [certPrefix, streams, reserveSeqOf, R]
    have hy : (fun n => certPrefix (some i) n (streams (fun _ => ∅) none y))
        = nthDrawIn R (nthDrawIn R (y.1.2.2 (some i) 1) (y.1.2.1.2 (some i)) 0) (y.1.2.1.2 (some i)) := by
      funext n; simp [certPrefix, streams, reserveSeqOf, R, hg]
    unfold certOf
    rw [hx, hy]
    refine freshDraws_nthDrawIn_congr R T (M := M) (fun k hk => hp (some i) k hk) _ _ ?_
    rw [drawsIn_onSide]
    convert hm using 3
    simp only [seqsOf]
    exact congrArg List.toFinset (List.filter_congr fun s _ => by cases x.2 s <;> rfl)

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
/-- Where the first `M` draws hold every scheduled state's samples, draws agreeing on them give
the clustering the same samples. -/
theorem sameReads_of_scan {sched : Finset State} {pops : Finset (Option (Fin K))} {M : ℕ}
    {x y : Draws Ω (FreeMonoid α) (Option (Fin K))} (hxy : AgreeTo none M x y)
    (hs : ScanWithin sched pops M x) :
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
theorem scanWithin_agree {sched : Finset State} {pops : Finset (Option (Fin K))} {M : ℕ}
    {x y : Draws Ω (FreeMonoid α) (Option (Fin K))} (hxy : AgreeTo none M x y)
    (hs : ScanWithin sched pops M x) : ScanWithin sched pops M y := by
  intro B hB j hj
  have hP : prefixesAt pops B.npref (streams (fun _ => ∅) none x)
      = prefixesAt pops B.npref (streams (fun _ => ∅) none y) :=
    Finset.biUnion_congr rfl fun j' hj' => prefixesOf_agree hxy j' (hs B hB j' hj').1
  rw [← seqsOf_agree hxy j, ← hP]
  exact hs B hB j hj

omit [DecidableEq α] in
lemma AgreeTo.symm {J : Type*} {uni : J} {M : ℕ} {x y : Draws Ω (FreeMonoid α) J}
    (h : AgreeTo uni M x y) : AgreeTo uni M y x :=
  ⟨h.1.symm, h.2.1.symm, h.2.2.1.symm, fun j n hn => (h.2.2.2.1 j n hn).symm,
    fun n hn => (h.2.2.2.2 n hn).symm⟩

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
    (uni : Option (Fin K)) (η₀ ι εc a : ℝ) {B : State} (hB : B ∈ sched) :
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
      rw [hc j hj e he]

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
    (D : Measure (FreeMonoid α)) (h : LoopHist α) {r r' : Run Ω (FreeMonoid α) (Option (Fin K))}
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
    exact propext (and_congr_right fun hB => hr.ret O.mq none _ _ _ _ hB)
  have hB := (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ r ∈ ret O.mq ps none
    settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    (let ⟨B, hB, hx⟩ := hret; ⟨B, hB, hx⟩)).1
  unfold roundOf
  simp only
  rw [← hpred, hr.clusterAt O.mq hB, hr.1, clusterKeys, clusterKeys, hr.roundPrefs, hr.roundSuffs]

theorem roundOf_result (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : LoopHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : Fin settings.level3.probeBudget → unitInterval) :
    (roundOf settings O D h x u).result = Random.indecisiveLStarLevel3 settings.level3
      (roundOf settings O D h x u).read fun i => drawFrom (avoidLaw D settings.level3.walkStart
        (clusterKeys (settings.schedule (loopPopulations K h.pops.length))
          (loopPopulations K h.pops.length) settings.level3.walkStart x)) (u i) := rfl

open scoped Classical in
/-- The round fails where level 3's uncapped run ends with more than `leafCap` leaves. -/
theorem loopRound_eq (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    loopRound settings O D Dsf h d =
      (clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
          h (d.1, d.2.1)).bind fun dr =>
        if ScanWithin (settings.schedule (loopPopulations K h.pops.length))
            (loopPopulations K h.pops.length) settings.scanCap dr
            ∧ (∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
              streams (fun _ => ∅) none dr ∈ ret O.mq (loopPopulations K h.pops.length) none
                settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
            ∧ (roundOf settings O D h (streams (fun _ => ∅) none dr)
              d.2.2).result.hypothesis.tree.leaves.length ≤ settings.level3.leafCap
        then some (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2)
        else none := by
  classical
  unfold loopRound
  dsimp only
  refine congrArg _ (funext fun dr => ?_)
  by_cases hc : ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) settings.scanCap dr
      ∧ ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
        streams (fun _ => ∅) none dr ∈ ret O.mq (loopPopulations K h.pops.length) none
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  · rw [if_pos hc]
    by_cases hl : (roundOf settings O D h (streams (fun _ => ∅) none dr)
        d.2.2).result.hypothesis.tree.leaves.length ≤ settings.level3.leafCap
    · rw [if_pos ⟨hc.1, hc.2, hl⟩, Random.level3Capped_eq, ← roundOf_result, if_pos hl,
        Option.map_some]
    · rw [if_neg fun h' => hl h'.2.2, Random.level3Capped_eq, ← roundOf_result, if_neg hl,
        Option.map_none]
  · rw [if_neg hc, if_neg fun h' => hc ⟨h'.1, h'.2.1⟩]

/-- Where the filled draws after the history `h` hit, scan within `scanCap` and return at a
scheduled state, and level 3 on them keeps within `leafCap` leaves, the round is the one on the
filled draws. -/
theorem loopRound_eq_some (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (hhit : (clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap
      settings.scanCap h (d.1, d.2.1)).isSome)
    (hscan : ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) settings.scanCap
      (idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)))
    (hret : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B)
    (hcap : (roundOf settings O D h
      (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1))
      d.2.2).result.hypothesis.tree.leaves.length ≤ settings.level3.leafCap) :
    loopRound settings O D Dsf h d = some (roundOf settings O D h
      (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1))
      d.2.2) := by
  classical
  obtain ⟨dr, hdr⟩ := Option.isSome_iff_exists.1 hhit
  have hag := clusterDraws_agree hdr
  have hsr := sameReads_of_scan hag.symm hscan
  have hsr' := sameReads_of_scan hag (scanWithin_agree hag.symm hscan)
  have hret' : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      streams (fun _ => ∅) none dr ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀
        settings.indecisionLimit settings.εcov settings.gateLevel B := by
    obtain ⟨B, hB, hx⟩ := hret
    exact ⟨B, hB, (hsr.ret O.mq none _ _ _ _ hB).1 hx⟩
  rw [loopRound_eq]
  simp only [hdr, Option.bind_some]
  have hro := roundOf_congr settings O D h hsr' hret' d.2.2
  rw [if_pos ⟨scanWithin_agree hag.symm hscan, hret', by rw [hro]; exact hcap⟩]
  exact congrArg some hro

end OrthoDFA

end
