import RLStar.Proofs.TrieAgree
import RLStar.Proofs.LoopQuery
import RLStar.LowLevelClaims.Trie

/-!
# The loop with local walk starts is the loop at `walkStart` where its probes' first letters differ

Two programs are twins at an oracle when they read the same strings and one's value is a fixed
function of the other's (`Twin`). Where `cap ≤ walkStart`, nothing is expanded past the complete
trie, so a harvested population's probes, each at least `walkStart` long (`BatchesGood`), walk from
`walkStart` and the top-up is the loop's. A round whose level-3 probes are at least `walkStart`
long (`L3Good`) runs as the loop's (`level3Trie_agree`). So where every round the
loop runs is good (`TrieRoundGood`), the loop with local walk starts returns the loop's rounds,
each with the complete trie as its expanded prefixes, and reads the same strings
(`twin_trieLoopRoundsQ`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Query Random

section Twin

variable {Q R β β' γ γ' : Type*} [DecidableEq Q] (o : Q → R)

/-- `p` reads what `p'` reads, and returns `f` of what `p'` returns. -/
def Twin (p : Query Q R β) (p' : Query Q R β') (f : β' → β) : Prop :=
  p.eval o = f (p'.eval o) ∧ p.reads o = p'.reads o

variable {o}

lemma Twin.refl (p : Query Q R β) : Twin o p p id := ⟨rfl, rfl⟩

lemma Twin.pure (f : β' → β) (b : β') : Twin o (Query.pure (f b)) (Query.pure b) f := ⟨rfl, rfl⟩

lemma Twin.bind {p : Query Q R β} {p' : Query Q R β'} {f : β' → β} {g : β → Query Q R γ}
    {g' : β' → Query Q R γ'} {f' : γ' → γ} (h : Twin o p p' f)
    (hg : Twin o (g (f (p'.eval o))) (g' (p'.eval o)) f') : Twin o (p.bind g) (p'.bind g') f' := by
  refine ⟨?_, ?_⟩
  · rw [eval_bind, eval_bind, h.1]; exact hg.1
  · rw [reads_bind, reads_bind, h.1, h.2, hg.2]

lemma Twin.map {p : Query Q R β} {p' : Query Q R β'} {f : β' → β} {g : β → γ} {g' : β' → γ'}
    {f' : γ' → γ} (h : Twin o p p' f) (hg : g (f (p'.eval o)) = f' (g' (p'.eval o))) :
    Twin o (p.map g) (p'.map g') f' := by
  refine ⟨?_, ?_⟩
  · rw [eval_map, eval_map, h.1, hg]
  · rw [reads_map, reads_map, h.2]

lemma Twin.subst {Q' R' : Type*} [DecidableEq Q'] {o' : Q' → R'} {hq : Q → Query Q' R' R}
    {p : Query Q R β} {p' : Query Q R β'} {f : β' → β}
    (h : Twin (fun q => (hq q).eval o') p p' f) : Twin o' (p.subst hq) (p'.subst hq) f := by
  refine ⟨?_, ?_⟩
  · rw [eval_subst, eval_subst, h.1]
  · rw [reads_subst, reads_subst, h.2]

lemma Twin.pi {ι : Type*} [Fintype ι] [DecidableEq ι] [Inhabited β] [Inhabited β']
    {F : ι → Query Q R β} {F' : ι → Query Q R β'} {f : β' → β} (h : ∀ j, Twin o (F j) (F' j) f) :
    Twin o (Query.pi F) (Query.pi F') fun g j => f (g j) := by
  refine ⟨?_, ?_⟩
  · rw [eval_pi, eval_pi]
    funext j
    exact (h j).1
  · rw [reads_pi, reads_pi]
    exact Finset.biUnion_congr rfl fun j _ => (h j).2

end Twin

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The lifts -/

section Lift

variable (k : ℕ)

omit [DecidableEq α] in
lemma trieHistOf_map (rs : List (LoopRound α)) :
    trieHistOf (rs.map (liftRound k)) = liftHist k (histOf rs) := by
  simp only [trieHistOf, histOf, liftHist, List.filterMap_map, List.foldr_map]
  congr 1
  rw [List.map_filterMap]
  refine List.filterMap_congr fun o _ => ?_
  simp only [Function.comp, TrieLoopRound.sampler, LoopRound.sampler, liftRound]
  rcases o.result.ending.harvested <;> rfl

end Lift

/-! ## The harvest's sampler -/

section Harvest

variable (read : FreeMonoid α → ARU) (k : ℕ)

open scoped Classical in
/-- Probes at least `k` long harvest past the complete trie what they harvest from `k`, and expand
nothing, where `cap ≤ k`. -/
lemma twin_trieHarvestFirst (cap : ℕ) (hcap : cap ≤ k) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) :
    ∀ ys : List (FreeMonoid α), (∀ y ∈ ys, k ≤ y.toList.length) →
      Twin read (trieHarvestFirstQ cap H site T (completeTrie k) ys) (harvestFirstQ k H site T ys)
        fun r => (r, completeTrie k)
  | [], _ => ⟨rfl, rfl⟩
  | y :: ys, hys => by
    have hyk := hys y List.mem_cons_self
    have ih := twin_trieHarvestFirst cap hcap H site T ys fun y' hy' =>
      hys y' (List.mem_cons_of_mem _ hy')
    have hw : walkStartIn (completeTrie k) y = k := walkStartIn_completeTrie hyk
    have hu : ¬ (prefixOf y k).toList.length < cap := by rw [length_prefixOf']; omega
    set l := (siteHarvestQ k H site y).eval read with hl
    unfold trieHarvestFirstQ harvestFirstQ
    rw [hw]
    simp only [Query.findSome]
    refine ⟨?_, ?_⟩
    · rw [eval_bind, eval_bind, eval_map, ← hl]
      rcases hh : l.head? with _ | z
      · simp only [Option.filter_none]
        exact ih.1
      · by_cases hzT : z ∈ T
        · have hf : Option.filter (fun z => !decide (z ∈ T)) (some z) = none := by
            simp [Option.filter_some, hzT]
          simp [hzT, hu, hf, ih.1]
        · have hf : Option.filter (fun z => !decide (z ∈ T)) (some z) = some z := by
            simp [Option.filter_some, hzT]
          simp [hzT, hf]
    · rw [reads_bind, reads_bind, reads_map, eval_map, ← hl]
      rcases hh : l.head? with _ | z
      · simp only [Option.filter_none]
        rw [ih.2]
        rfl
      · by_cases hzT : z ∈ T
        · have hf : Option.filter (fun z => !decide (z ∈ T)) (some z) = none := by
            simp [Option.filter_some, hzT]
          simp [hzT, hu, hf, ih.2]
        · have hf : Option.filter (fun z => !decide (z ∈ T)) (some z) = some z := by
            simp [Option.filter_some, hzT]
          simp [hzT, hf]

open scoped Classical in
lemma twin_trieTopUp (cap : ℕ) (hcap : cap ≤ k) (s : HarvestSampler α)
    (T : Finset (FreeMonoid α)) (stop : List (FreeMonoid α) → Prop) (xs : ℕ → List (FreeMonoid α)) :
    ∀ (M : ℕ) (l : List (FreeMonoid α)),
      (∀ n, l.length ≤ n → n < l.length + M → ∀ x ∈ xs n, k ≤ x.toList.length) →
      Twin read (trieTopUpQ cap (liftSampler k s) T stop xs M (completeTrie k) l)
        (Query.extendUntil (fun n => harvestFirstQ k s.hyp s.site ↑T (xs n)) stop M l) id
  | 0, l, _ => ⟨rfl, rfl⟩
  | M + 1, l, hb => by
    unfold trieTopUpQ Query.extendUntil
    split_ifs with hs
    · exact ⟨rfl, rfl⟩
    have hh := twin_trieHarvestFirst read k cap hcap s.hyp s.site ↑T (xs l.length)
      fun y hy => hb l.length le_rfl (by omega) y hy
    refine hh.bind ?_
    simp only [liftSampler]
    rcases (harvestFirstQ k s.hyp s.site ↑T (xs l.length)).eval read with _ | z
    · exact ⟨rfl, rfl⟩
    · refine twin_trieTopUp cap hcap s T stop xs M (l ++ [z]) fun n h1 h2 => ?_
      rw [List.length_append, List.length_singleton] at h1 h2
      exact hb n (by omega) (by omega)

end Harvest

/-! ## A round -/

section Round

variable {K : ℕ}

lemma twin_level3Trie (C : Level3Settings) (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) (hlen : ∀ i, C.walkStart ≤ (draws i).toList.length) :
    Twin read (level3TrieQ C draws) (level3Q C draws)
      fun r => ⟨r.hypothesis, completeTrie C.walkStart, r.ending⟩ := by
  refine ⟨level3Trie_agree C read draws hlen, ?_⟩
  have h := runTrieProbes_agree C read (List.ofFn draws) (trieInitialState C) rfl
    (fun x hx => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact hlen i)
  unfold level3TrieQ level3Q
  rw [reads_map, reads_map]
  exact h.2.2.2

open scoped Classical in
lemma twin_triePopDrawsQ (rd : FreeMonoid α → ℝ) (cap : ℕ) (D : Measure (FreeMonoid α))
    (k N M : ℕ) (hcap : cap ≤ k) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α))
    (stop : List (FreeMonoid α) → Prop) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval))
    (hg : ∀ i, j = some i → BatchesGood k D N M u) :
    Twin rd (triePopDrawsQ cap D N M (pops.map (liftSampler k)) T stop j u)
      (popDrawsQ D k N M pops T stop j u) id := by
  rcases j with _ | i
  · exact ⟨rfl, rfl⟩
  simp only [triePopDrawsQ, popDrawsQ, List.getElem?_map]
  rcases pops[i.val]? with _ | s
  · exact ⟨rfl, rfl⟩
  simp only [Option.map_some]
  refine Twin.subst (twin_trieTopUp _ k cap hcap s T stop _ M [] fun n _ hn x hx => ?_)
  obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
  exact hg i rfl n (by simpa using hn) t

lemma twin_trieDrawListsQ (rd : FreeMonoid α → ℝ) (cap : ℕ) (D : Measure (FreeMonoid α))
    (k N M : ℕ) (hcap : cap ≤ k) (sched : Finset State) (h : LoopHist α) (c : ClusterRest α K N)
    (hg : ∀ i : Fin K, BatchesGood k D N M (c.2.1 (some i))) :
    Twin rd (trieDrawListsQ cap D N M sched (liftHist k h) c) (drawListsQ D k N M sched h c) id := by
  unfold trieDrawListsQ drawListsQ
  refine (Twin.pi fun j => twin_triePopDrawsQ rd cap D k N M hcap h.pops h.prefs _ j (c.2.1 j)
    fun i hi => hi ▸ hg i).map ?_
  rfl

open scoped Classical in
lemma twin_trieRoundQ (rd : FreeMonoid α → ℝ) (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (hcap : cap ≤ settings.level3.walkStart) (hg : TrieRoundGood settings D Dsf h d rd) :
    Twin rd (trieRoundQ settings cap D Dsf (liftHist settings.level3.walkStart h) d)
      (roundQ settings D Dsf h d) (Option.map (liftRound settings.level3.walkStart)) := by
  set k := settings.level3.walkStart
  have hlen : (liftHist k h).pops.length = h.pops.length := by simp [liftHist]
  unfold trieRoundQ
  rw [hlen]
  have hdl := twin_trieDrawListsQ rd cap D k settings.level3.harvestReadCap settings.scanCap hcap
    (settings.schedule (loopPopulations K h.pops.length)) h d.1 hg.1
  have hroundQ : roundQ settings D Dsf h d = (drawListsQ D k settings.level3.harvestReadCap
      settings.scanCap (settings.schedule (loopPopulations K h.pops.length)) h d.1).bind
      (fun tabs => match tabs with
        | none => Query.pure none
        | some tabs =>
          let ps := loopPopulations K h.pops.length
          let sched := settings.schedule ps
          let dr := drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs
          let x := streams (fun _ => ∅) none dr
          if ScanWithin sched ps (fun j => (tabs j).length) dr then
            (Query.table (clusterReads sched ps x).toList).bind fun t =>
              let mq : FreeMonoid α → Unit → ℝ := fun w _ => t w
              if ∃ B ∈ sched, x ∈ ret mq ps none settings.η₀ settings.indecisionLimit
                  settings.εcov settings.gateLevel B then
                let B := Classical.epsilon fun B => B ∈ sched
                  ∧ x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
                    settings.gateLevel B
                let F := clusterAt mq ps x B
                ((Random.level3Q settings.level3 fun i =>
                    drawFrom (avoidLaw D settings.level3.walkStart
                      (clusterKeys sched ps settings.level3.walkStart x)) (d.2 i)).subst
                  (familyReadQ F B.lo (B.hi + 1))).map fun res =>
                  some ⟨B, F, res, roundPrefs sched ps x, roundSuffs sched x⟩
              else .pure none
          else .pure none) := rfl
  have hg2 := hg.2
  rw [hroundQ] at hg2 ⊢
  refine hdl.bind ?_
  simp only [id]
  rcases htabs : (drawListsQ D k settings.level3.harvestReadCap settings.scanCap
      (settings.schedule (loopPopulations K h.pops.length)) h d.1).eval rd with _ | tabs
  · exact ⟨rfl, rfl⟩
  simp only [liftHist]
  by_cases hsc : ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (fun j => (tabs j).length)
      (drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs)
  swap
  · simp only [hsc, ↓reduceIte]
    exact ⟨rfl, rfl⟩
  simp only [hsc, ↓reduceIte]
  refine (Twin.refl _).bind ?_
  simp only [id]
  by_cases hex : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      streams (fun _ => ∅) none (drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs)
        ∈ ret (fun w _ => (Query.table (clusterReads (settings.schedule
          (loopPopulations K h.pops.length)) (loopPopulations K h.pops.length)
          (streams (fun _ => ∅) none (drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1
            tabs))).toList).eval rd w) (loopPopulations K h.pops.length) none settings.η₀
          settings.indecisionLimit settings.εcov settings.gateLevel B
  swap
  · simp only [hex, ↓reduceIte]
    exact ⟨rfl, rfl⟩
  simp only [hex, ↓reduceIte]
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set x := streams (fun _ => ∅) none (drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs)
  set t := (Query.table (clusterReads sched ps x).toList).eval rd
  set B := Classical.epsilon fun B => B ∈ sched
    ∧ x ∈ ret (fun w _ => t w) ps none settings.η₀ settings.indecisionLimit settings.εcov
      settings.gateLevel B
  set F := clusterAt (fun w _ => t w) ps x B
  set draws := fun i => drawFrom (avoidLaw D k (clusterKeys sched ps k x)) (d.2 i)
  have hL3 : L3Good k D (clusterKeys sched ps k x) d.2 := by
    have e := hg2 ⟨B, F, (Random.level3Q settings.level3 draws).eval
        fun q => (familyReadQ F B.lo (B.hi + 1) q).eval rd, roundPrefs sched ps x,
        roundSuffs sched x⟩ (by
      simp only [eval_bind, htabs]
      split_ifs with h1
      · rw [eval_bind]
        split_ifs with h2
        · rw [eval_map, eval_subst]
        · exact absurd hex h2
      · exact absurd hsc h1)
    exact e
  refine (Twin.subst (twin_level3Trie settings.level3 _ draws hL3)).map ?_
  rfl

/-- Where every round the loop runs is good, the loop with local walk starts returns the loop's
rounds, each with the complete trie, and reads what it reads. -/
theorem twin_trieLoopRoundsQ (rd : FreeMonoid α → ℝ) (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (hcap : cap ≤ settings.level3.walkStart)
    (hg : ∀ r (hr : r < K) rs, (loopRoundsQ settings D Dsf draws r).eval rd = some rs →
      (∀ o ∈ rs, o.result.ending.harvested ≠ none) →
      TrieRoundGood settings D Dsf (histOf rs) (draws ⟨r, hr⟩) rd) :
    ∀ r, Twin rd (trieLoopRoundsQ settings cap D Dsf draws r) (loopRoundsQ settings D Dsf draws r)
      (Option.map (List.map (liftRound settings.level3.walkStart)))
  | 0 => ⟨rfl, rfl⟩
  | r + 1 => by
    unfold trieLoopRoundsQ loopRoundsQ
    refine (twin_trieLoopRoundsQ rd settings cap D Dsf draws hcap hg r).bind ?_
    rcases hrs : (loopRoundsQ settings D Dsf draws r).eval rd with _ | rs
    · exact ⟨rfl, rfl⟩
    simp only [Option.map_some]
    have hcond : (∀ o ∈ rs.map (liftRound settings.level3.walkStart),
        o.result.ending.harvested ≠ none) ↔ ∀ o ∈ rs, o.result.ending.harvested ≠ none := by
      simp [liftRound]
    by_cases hc : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none
    · rw [dif_pos ⟨hc.1, hcond.2 hc.2⟩, dif_pos hc, trieHistOf_map]
      refine (twin_trieRoundQ rd settings cap D Dsf _ _ hcap (hg r hc.1 rs hrs hc.2)).map ?_
      rcases (roundQ settings D Dsf (histOf rs) (draws ⟨r, hc.1⟩)).eval rd with _ | o <;> simp
    · rw [dif_neg (fun h => hc ⟨h.1, hcond.1 h.2⟩), dif_neg hc]
      exact ⟨rfl, rfl⟩

end Round

theorem level3TrieAgrees_holds : Level3TrieAgrees := by
  intro α _ _ C read P draws hlen
  exact level3Trie_agree C read draws hlen

theorem trieLoopAgrees_holds : TrieLoopAgrees := by
  intro α _ _ K rd settings cap D Dsf draws hcap hg
  exact twin_trieLoopRoundsQ rd settings cap D Dsf draws hcap hg K

end OrthoDFA

end
