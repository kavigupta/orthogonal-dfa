import OrthoDFA.Proofs.Select
import OrthoDFA.Proofs.SelectScore
import OrthoDFA.Proofs.LearnerLoop

/-!
# The loop's last round, scored on held-out strings

Level 5 reads only short strings beginning with some probe's first `k` letters
(`level5_congr_zone`). So what round `r` ends at is fixed by the answers at what the rounds before
it, its clustering and its level 5 read (`stageSel_stop`). Held-out strings of length `L` that
are distinct and whose first `k` letters are neither those of a prefix the round's clustering read
nor those of a probe are read at none of those strings, so the scores' deviation is that of fresh
noise (`loop_select_round_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Zone

/-- The first `k` letters of the probes. -/
def probeKeys (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) : Finset (FreeMonoid α) :=
  Finset.univ.image fun i => prefixOf (xs i) k

open scoped Classical in
/-- The strings level 5 reads over the probes `xs`: short strings beginning with a probe's first
`walkStart` letters, followed by the family's suffixes. -/
def level5Zone (C : Level5Settings) {T : ℕ} (xs : Fin T → FreeMonoid α)
    (F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  readStrings ((shortStrings (probesLen xs + 1 + C.maxLeaves)).filter fun z =>
    C.walkStart ≤ z.toList.length ∧ prefixOf z C.walkStart ∈ probeKeys C.walkStart xs) F

omit [IsProbabilityMeasure μ] in
theorem level5_congr_zone (O : Oracle μ (FreeMonoid α)) (C : Level5Settings)
    (hC : 2 ≤ C.maxLeaves) (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ w ∈ level5Zone C xs F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω') xs := by
  classical
  set K₀ : Set (FreeMonoid α) := (↑(probeKeys C.walkStart xs))ᶜ
  set mix : FreeMonoid α → ARU := fun z =>
    if KeyOff C.walkStart K₀ z then familyRead O.mq F kl kh z ω else familyRead O.mq F kl kh z ω'
  have h1 : indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel5 C mix xs :=
    level5_congr_off C xs (fun i _ => by simp [K₀, probeKeys])
      (fun z hz => (if_pos hz).symm)
  have h2 : indecisiveLStarLevel5 C mix xs
      = indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω') xs := by
    refine level5_congr_up C hC xs (le_probesLen xs) fun z hz => ?_
    by_cases hk : KeyOff C.walkStart K₀ z
    · have hz' : prefixOf z C.walkStart ∈ probeKeys C.walkStart xs := by simpa [K₀] using hk.2
      exact (if_pos hk).trans
        (familyRead_congr O h (Finset.mem_filter.2 ⟨mem_shortStrings.2 hz, hk.1, hz'⟩))
    · exact if_neg hk
  rw [h1, h2]

end Zone

section Stage

/-- A level 5 result as data: its tree, its edges' graph, and whether it ended consistent. -/
abbrev ResCode (α : Type*) :=
  DTree α × Finset ((List Bool × α) × (List Bool × FreeMonoid α)) × Bool

def resCode (res : Level5Result α) : ResCode α :=
  (res.hypothesis.tree, edgeGraph (Function.curry res.hypothesis.edges),
    match res.ending with
    | .consistent => true
    | _ => false)

/-- The hypothesis a code holds. -/
def ResCode.hyp (c : ResCode α) : HypothesisDFA α := ⟨c.1, Function.uncurry (decodeEdges c.2.1)⟩

/-- What reaches a round's level 5: the earlier rounds' data, the state, family and keys. -/
abbrev Stage4 (α : Type*) :=
  List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α)

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (N : ℕ)

/-- Level 5 at what reaches it, on the noise `ω` and the uniforms `u`. -/
def stageRes (t : Stage4 α) (ω : Ω) (u : Fin settings.level5.probeBudget → unitInterval) :
    Level5Result α :=
  indecisiveLStarLevel5 settings.level5
    (fun z => familyRead O.mq t.2.2.1 t.2.1.lo (t.2.1.hi + 1) z ω)
    (probesOf settings D t.2.2.2 u)

/-- What reaches round `r`'s level 5, and what level 5 returns. -/
def stageSel (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level5.probeBudget) : Option (Stage4 α × ResCode α) :=
  (stage4 settings O D Dsf N r hr ω g).map fun t =>
    (t, resCode (stageRes settings O D t ω (g ⟨r, hr⟩).2))

end Stage

section Fail

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (N Ls L n : ℕ) (t : ℝ) (r : ℕ) (hr : r < K)

/-- The stage values the selection is bounded at: round `r` ended consistent after `r` harvests. -/
def SelG (y : Option (Stage4 α × ResCode α)) : Prop :=
  ∃ t h, y = some (t, h) ∧ t.1.length = r ∧ h.2.2 = true

/-- Round `r` reads no prefix or suffix an earlier round did, reads its family among its own
suffixes, and keys its probes off its prefixes. -/
def SelOk : Option (Stage4 α × ResCode α) → Set (Fin K → RoundDraws α K settings.level5.probeBudget)
  | some (t, _) => {x | NoColl settings D Dsf Ls t.1 x r hr
      ∧ t.2.2.1 ⊆ suffsOf settings D Dsf (histAt settings D t.1 r) (x ⟨r, hr⟩).1
      ∧ t.2.2.2 = (prefsOf settings D Dsf (histAt settings D t.1 r) (x ⟨r, hr⟩).1).image
          fun p => prefixOf p settings.level5.walkStart}
  | none => Set.univ

/-- Held-out strings, distinct, of length `L`, whose first `k` letters are neither a key of round
`r`'s prefixes nor of its probes. -/
def SelDk : Option (Stage4 α × ResCode α) →
    Set ((Fin K → RoundDraws α K settings.level5.probeBudget) × (Fin n → FreeMonoid α))
  | some (t, _) => {q | Function.Injective q.2 ∧ ∀ i, (q.2 i).toList.length = L
      ∧ prefixOf (q.2 i) settings.level5.walkStart ∉ t.2.2.2
        ∪ probeKeys settings.level5.walkStart (probesOf settings D t.2.2.2 (q.1 ⟨r, hr⟩).2)}
  | none => Set.univ

open scoped Classical in
/-- Some leaf's score strays `n t` from its mean, at a suffix-free family and a tree of at most
`maxLeaves` leaves. -/
def SelBad : Option (Stage4 α × ResCode α) → Set (Ω × (Fin n → FreeMonoid α))
  | some (t', h) => if SuffixFree t'.2.2.1 ∧ h.hyp.tree.leaves.length ≤ settings.level5.maxLeaves
      then scoreBad O h.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅
  | none => ∅

/-- Round `r` ended consistent, and its scores strayed or a held-out string collided. -/
def selFail : Set ((Ω × (Fin K → RoundDraws α K settings.level5.probeBudget))
    × (Fin n → FreeMonoid α)) :=
  {q | SelG r (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
    ∧ q.1.2 ∈ SelOk settings D Dsf Ls r hr (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
    ∧ ((q.1.1, q.2) ∈ SelBad settings O L n t (stageSel settings O D Dsf N r hr q.1.1 q.1.2)
      ∨ (q.1.2, q.2) ∉ SelDk settings D L n r hr (stageSel settings O D Dsf N r hr q.1.1 q.1.2))}

end Fail

section Congr

omit [IsProbabilityMeasure μ] in
theorem scoreBad_congr (O : Oracle μ (FreeMonoid α)) (H : HypothesisDFA α)
    (F : Finset (FreeMonoid α)) (kl kh L n : ℕ) (t : ℝ) {ω ω' : Ω} (b : Fin n → FreeMonoid α)
    (h : ∀ w ∈ readStrings (Finset.univ.image b) F, (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (hb : (ω, b) ∈ scoreBad O H F kl kh L n t) : (ω', b) ∈ scoreBad O H F kl kh L n t := by
  have he : ∀ s i, agreeAt O H F kl kh s (b i) ω = agreeAt O H F kl kh s (b i) ω' := fun s i => by
    unfold agreeAt
    rw [familyRead_congr O h (Finset.mem_image_of_mem b (Finset.mem_univ i))]
  obtain ⟨hinj, s, hs, hdev⟩ := hb
  refine ⟨hinj, s, hs, ?_⟩
  simp only [he] at hdev
  exact hdev

end Congr

section Meas

omit [IsProbabilityMeasure μ] in
theorem measurableSet_selOk (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (Ls r : ℕ) (hr : r < K) (y : Option (Stage4 α × ResCode α)) :
    MeasurableSet (SelOk settings D Dsf Ls r hr y) := by
  classical
  rcases y with _ | ⟨⟨c, B, F, K₀⟩, h'⟩
  · exact MeasurableSet.univ
  set rK : Fin K := ⟨r, hr⟩
  have hx : Measurable fun x : Fin K → RoundDraws α K settings.level5.probeBudget => (x rK).1 :=
    measurable_fst.comp (measurable_pi_apply rK)
  have hlev : ∀ (d : (Fin K → RoundDraws α K settings.level5.probeBudget) → Finset (FreeMonoid α))
      (P : Finset (FreeMonoid α) → Prop), (∀ Q, MeasurableSet {x | d x = Q}) →
        MeasurableSet {x | P (d x)} := by
    intro d P hd
    have hset : {x | P (d x)} = ⋃ Q : {Q : Finset (FreeMonoid α) // P Q}, {x | d x = Q.1} := by
      ext x
      simp only [Set.mem_iUnion, Set.mem_ofPred_eq]
      exact ⟨fun h => ⟨⟨d x, h⟩, rfl⟩, fun ⟨Q, hQ⟩ => hQ ▸ Q.2⟩
    rw [hset]
    exact MeasurableSet.iUnion fun Q => hd Q.1
  exact (measurableSet_noColl settings D Dsf Ls c r hr).inter
    ((hlev (fun x => suffsOf settings D Dsf (histAt settings D c r) (x rK).1) (F ⊆ ·)
      fun Q => hx (measurableSet_suffsOf settings D Dsf _ Q)).inter
    (hlev (fun x => prefsOf settings D Dsf (histAt settings D c r) (x rK).1)
      (fun Q => K₀ = Q.image fun p => prefixOf p settings.level5.walkStart)
      fun Q => hx (measurableSet_prefsOf settings D Dsf _ Q)))

end Meas

section Bound

variable [Nonempty α]

set_option maxHeartbeats 1000000 in -- the selection's many obligations
theorem loop_select_round_le (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (Ls L n NP : ℕ) (t : ℝ) (ht : 0 ≤ t) (hC : 2 ≤ settings.level5.maxLeaves)
    (hkL : settings.level5.walkStart ≤ L)
    (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP) (r : ℕ) (hr : r < K) :
    ((loopMeasure μ α K settings.level5.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      (selFail settings O P.D Dsf (P.L + settings.level5.maxLeaves) Ls L n t r hr)
      ≤ settings.level5.maxLeaves * (4 * Real.exp (-(n * t ^ 2 / 2)))
        + (n * (6 * (K + 1) * NP + settings.level5.probeBudget)
            / (Fintype.card α : ℝ) ^ settings.level5.walkStart
          + n ^ 2 / (Fintype.card α : ℝ) ^ L) := by
  classical
  set T := settings.level5.probeBudget
  set k := settings.level5.walkStart
  set N := P.L + settings.level5.maxLeaves
  have hN : P.L + settings.level5.maxLeaves ≤ N := le_rfl
  set rK : Fin K := ⟨r, hr⟩
  set dec := decodeHist P.D k with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  have hA0 : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  set Y := Option (Stage4 α × ResCode α)
  set stage := stageSel settings O P.D Dsf N r hr
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some (t, _) => (histReads settings P.D Dsf N t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf P.D Dsf (dec t.1) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K))))
        ∪ level5Zone settings.level5 (probesOf settings P.D t.2.2.2 (x rK).2) t.2.2.1
    | none => ∅ with hWd
  set Z : Y → (Fin n → FreeMonoid α) → Finset (FreeMonoid α) := fun y b => match y with
    | some (t, _) => readStrings (Finset.univ.image b) t.2.2.1
    | none => ∅ with hZd
  set δ₁ : ℝ := settings.level5.maxLeaves * (4 * Real.exp (-(n * t ^ 2 / 2)))
  set δ₂ : ℝ := n * (6 * (K + 1) * NP + T) / (Fintype.card α : ℝ) ^ k
    + n ^ 2 / (Fintype.card α : ℝ) ^ L
  have hδ₁ : 0 ≤ δ₁ := by positivity
  have hδ₂ : 0 ≤ δ₂ := by positivity
  have hmain := select_le O (Measure.pi fun _ : Fin K => roundLaw α K T)
    (Measure.pi fun _ : Fin n => uniformStrings (α := α) L) stage (SelG r)
    (fun y hy => by
      obtain ⟨t', h', rfl, -, -⟩ := hy
      have hset : {p : Ω × (Fin K → RoundDraws α K T) | stage p.1 p.2 = some (t', h')}
          = {p | stage4 settings O P.D Dsf N r hr p.1 p.2 = some t'}
            ∩ (fun p : Ω × (Fin K → RoundDraws α K T) =>
                (p.1, probesOf settings P.D t'.2.2.2 (p.2 rK).2)) ⁻¹'
              {e : Ω × (Fin T → FreeMonoid α) | resCode (indecisiveLStarLevel5 settings.level5
                (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z e.1) e.2) = h'} := by
        ext p
        simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_preimage, stage, stageSel,
          Option.map_eq_some_iff]
        constructor
        · rintro ⟨t'', ht'', he⟩
          simp only [Prod.mk.injEq] at he
          obtain ⟨rfl, he⟩ := he
          exact ⟨ht'', he⟩
        · rintro ⟨ht, he⟩
          exact ⟨t', ht, by simp only [Prod.mk.injEq]; exact ⟨trivial, he⟩⟩
      rw [hset]
      refine (measurableSet_stage4 settings O P.D Dsf N P.length_le hC hN r hr _).inter ?_
      refine (measurable_fst.prodMk ((measurable_probesOf settings P.D _).comp
        (measurable_snd.comp ((measurable_pi_apply rK).comp measurable_snd)))) ?_
      refine measurableSet_prod_countable_right _ fun xs => ?_
      refine measurableSet_of_reads O (readStrings (shortStrings (probesLen xs + 1
        + settings.level5.maxLeaves)) t'.2.2.1) _ fun ω ω' h hω => ?_
      have e : indecisiveLStarLevel5 settings.level5
            (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω) xs
          = indecisiveLStarLevel5 settings.level5
            (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω') xs :=
        level5_result_congr O settings.level5 hC t'.2.2.1 _ _ xs h
      change resCode (indecisiveLStarLevel5 settings.level5
        (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω) xs) = h' at hω
      show resCode (indecisiveLStarLevel5 settings.level5
        (fun z => familyRead O.mq t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) z ω') xs) = h'
      rw [← e]
      exact hω)
    W
    (fun y x ω ω' hy hs h => by
      obtain ⟨t', h', rfl, -, -⟩ := hy
      obtain ⟨t'', ht4, he⟩ := Option.map_eq_some_iff.1 hs
      simp only [Prod.mk.injEq] at he
      obtain ⟨rfl, he⟩ := he
      obtain ⟨c, B, F, K₀⟩ := t''
      have h4 := stage4_stop settings O P.D Dsf N P.length_le hC hN r hr c B F K₀ x ω ω' ω₀ ht4
        (fun w hw => h w (Finset.mem_union_left _ hw))
      have hres : stageRes settings O P.D (c, B, F, K₀) ω (x rK).2
          = stageRes settings O P.D (c, B, F, K₀) ω' (x rK).2 :=
        level5_congr_zone O settings.level5 hC F B.lo (B.hi + 1) _ fun w hw =>
          h w (Finset.mem_union_right _ hw)
      show (stage4 settings O P.D Dsf N r hr ω' x).map _ = _
      rw [h4, Option.map_some, ← hres, he])
    (SelOk settings P.D Dsf Ls r hr)
    (measurableSet_selOk settings P.D Dsf Ls r hr)
    (SelBad settings O L n t)
    (fun y => by
      rcases y with _ | ⟨t', h'⟩
      · exact MeasurableSet.empty
      · show MeasurableSet (if _ then _ else _)
        split_ifs
        · exact measurableSet_scoreBad O _ _ _ _ L n t
        · exact MeasurableSet.empty)
    Z
    (fun y b ω ω' h hb => by
      rcases y with _ | ⟨t', h'⟩
      · exact hb
      · change (ω, b) ∈ (if SuffixFree t'.2.2.1
            ∧ h'.hyp.tree.leaves.length ≤ settings.level5.maxLeaves
          then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅) at hb
        show (ω', b) ∈ (if SuffixFree t'.2.2.1
            ∧ h'.hyp.tree.leaves.length ≤ settings.level5.maxLeaves
          then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅)
        split_ifs at hb ⊢
        · exact scoreBad_congr O _ _ _ _ L n t b h hb
        · exact hb)
    (SelDk settings P.D L n r hr)
    (fun y => by
      rcases y with _ | ⟨⟨c, B, F, K₀⟩, h'⟩
      · exact MeasurableSet.univ
      · refine measurableSet_prod_countable_right _ fun b => ?_
        have hset : {x : Fin K → RoundDraws α K T |
            (x, b) ∈ SelDk settings P.D L n r hr (some ((c, B, F, K₀), h'))}
            = (fun x : Fin K → RoundDraws α K T => probesOf settings P.D K₀ (x rK).2) ⁻¹'
              {xs | Function.Injective b ∧ ∀ i, (b i).toList.length = L
                ∧ prefixOf (b i) k ∉ K₀ ∪ probeKeys k xs} := rfl
        rw [hset]
        exact ((measurable_probesOf settings P.D K₀).comp
          (measurable_snd.comp (measurable_pi_apply rK))) (Set.to_countable _).measurableSet)
    (fun y x b hy hx hb => by
      obtain ⟨t', h', rfl, hlen, -⟩ := hy
      obtain ⟨c, B, F, K₀⟩ := t'
      obtain ⟨hnc, hF, hK⟩ := hx
      obtain ⟨hinj, hbi⟩ := hb
      change c.length = r at hlen
      have hpop : histAt settings P.D c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hlen)]
        rfl
      have hlenF : ∀ v ∈ F, v.toList.length = Ls := fun v hv => hnc.1 rK le_rfl v (hF hv)
      have hd1 := disjoint_hist_level5 settings P.D Dsf N Ls L c x r hr ω₀ hnc F hF
      rw [hpop] at hd1 hK
      show Disjoint ((histReads settings P.D Dsf N c x r ω₀
          ∪ schedReads (settings.schedule (loopPopulations K (dec c).pops.length))
            (loopPopulations K (dec c).pops.length)
            ((ω₀, runOf P.D Dsf (dec c) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K))))
          ∪ level5Zone settings.level5 (probesOf settings P.D K₀ (x rK).2) F)
        (readStrings (Finset.univ.image b) F)
      refine Finset.disjoint_union_left.2 ⟨hd1.mono_right ?_, ?_⟩
      · intro w hw
        obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hz, hv⟩ := Finset.mem_product.1 hzv
        obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hz
        refine Finset.mem_sdiff.2 ⟨Finset.mem_image.2 ⟨(b i, v), Finset.mem_product.2
          ⟨mem_shortStrings.2 (hbi i).1.le, hv⟩, rfl⟩, fun hw' => ?_⟩
        obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
        obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
        dsimp only at he
        obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he ((hlenF v' hv').trans (hlenF v hv).symm)
        apply (hbi i).2
        refine Finset.mem_union_left _ ?_
        rw [hK]
        exact Finset.mem_image.2 ⟨p, hp, by rw [hzp]⟩
      · refine Finset.disjoint_left.2 fun w hw hw' => ?_
        obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
        obtain ⟨hz, hv⟩ := Finset.mem_product.1 hzv
        obtain ⟨⟨z', v'⟩, hzv', he⟩ := Finset.mem_image.1 hw'
        obtain ⟨hz', hv'⟩ := Finset.mem_product.1 hzv'
        obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hz'
        dsimp only at he
        obtain ⟨-, hzz⟩ := suffix_eq_of_mul_eq he ((hlenF v' hv').trans (hlenF v hv).symm)
        apply (hbi i).2
        refine Finset.mem_union_right _ ?_
        have := (Finset.mem_filter.1 hz).2.2
        dsimp only
        rw [hzz]
        exact this)
    (ENNReal.ofReal δ₁) (ENNReal.ofReal δ₂)
    (fun y hy => by
      obtain ⟨t', h', rfl, -, -⟩ := hy
      show (μ.prod _) (if SuffixFree t'.2.2.1
          ∧ h'.hyp.tree.leaves.length ≤ settings.level5.maxLeaves
        then scoreBad O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) L n t else ∅) ≤ _
      split_ifs with hc
      · refine (scoreBad_le O h'.hyp t'.2.2.1 t'.2.1.lo (t'.2.1.hi + 1) hc.1 L n t ht).trans
          (ENNReal.ofReal_le_ofReal ?_)
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast hc.2) (by positivity)
      · simp)
    (fun y hy x hx => by
      obtain ⟨t', h', rfl, -, -⟩ := hy
      obtain ⟨c, B, F, K₀⟩ := t'
      obtain ⟨-, -, hK⟩ := hx
      set PK := probeKeys k (probesOf settings P.D K₀ (x rK).2)
      refine (measure_mono ?_).trans ((batch_collide_le L k n hkL (K₀ ∪ PK)).trans
        (ENNReal.ofReal_le_ofReal ?_))
      · intro b hb
        simp only [Set.mem_ofPred_eq] at hb ⊢
        by_contra hno
        push Not at hno
        exact hb ⟨hno.1, fun i => ⟨(hno.2 i).1, (hno.2 i).2⟩⟩
      · have hPK : PK.card ≤ T := by
          refine Finset.card_image_le.trans (le_of_eq ?_)
          simp [T]
        change K₀ = _ at hK
        have hK₀ : K₀.card ≤ 6 * (K + 1) * NP := by
          rw [hK]
          refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
          have hpops : (loopPopulations K (histAt settings P.D c r).pops.length).card ≤ K + 1 := by
            have := Finset.card_le_univ (loopPopulations K (histAt settings P.D c r).pops.length)
            simpa using this
          have := hNP (loopPopulations K (histAt settings P.D c r).pops.length)
          calc 3 * (loopPopulations K (histAt settings P.D c r).pops.length).card
                * ∑ B ∈ settings.schedule (loopPopulations K (histAt settings P.D c r).pops.length),
                  B.npref
              ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
            _ = 6 * (K + 1) * NP := by ring
        have hcard : ((K₀ ∪ PK).card : ℝ) ≤ 6 * (K + 1) * NP + T := by
          have h1 := Finset.card_union_le K₀ PK
          have h2 : ((K₀ ∪ PK).card : ℕ) ≤ 6 * (K + 1) * NP + T := by omega
          exact_mod_cast h2
        have hAk : (0 : ℝ) < (Fintype.card α : ℝ) ^ k := by positivity
        have h3 : (n : ℝ) * ((K₀ ∪ PK).card : ℝ) ≤ n * (6 * (K + 1) * NP + T) :=
          mul_le_mul_of_nonneg_left hcard (Nat.cast_nonneg _)
        exact add_le_add (div_le_div_of_nonneg_right h3 hAk.le) le_rfl)
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) (hmain.trans (le_of_eq ?_))
  rw [ENNReal.ofReal_add hδ₁ hδ₂]

end Bound

section Last

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α))

/-- Where the rounds ran fresh, the last round is what `stageSel` records, and its draws are
`SelOk`'s. -/
theorem stageSel_last (hC : 2 ≤ settings.level5.maxLeaves) (Ls : ℕ)
    (p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget))
    (hfr : RoundsFresh settings O P.D Dsf (P.L + settings.level5.maxLeaves) Ls p)
    (o : LoopRound Ω α K)
    (ho : o ∈ (indecisiveLStarLoop settings O P.D Dsf (persistDraws p)).getLast?)
    (hcons : o.result.ending = .consistent) :
    ∃ r, ∃ hr : r < K, r + 1 = (indecisiveLStarLoop settings O P.D Dsf (persistDraws p)).length
      ∧ ∃ t : Stage4 α,
      stageSel settings O P.D Dsf (P.L + settings.level5.maxLeaves) r hr p.1 p.2
        = some (t, resCode o.result)
      ∧ SelG r (some (t, resCode o.result))
      ∧ p.2 ∈ SelOk settings P.D Dsf Ls r hr (some (t, resCode o.result))
      ∧ t.2.1 = o.state ∧ t.2.2.1 = o.family
      ∧ o.read = (fun z => familyRead O.mq o.family o.state.lo (o.state.hi + 1) z p.1)
      ∧ (resCode o.result).hyp = o.result.hypothesis := by
  classical
  set N := P.L + settings.level5.maxLeaves
  have hN : P.L + settings.level5.maxLeaves ≤ N := le_rfl
  set rounds := loopRounds settings O P.D Dsf (persistDraws p) K with hrounds
  change o ∈ rounds.getLast? at ho
  change ∃ r, ∃ hr : r < K, r + 1 = rounds.length ∧ _
  have hne : rounds ≠ [] := by
    intro h
    rw [h] at ho
    simp at ho
  set r := rounds.length - 1 with hrd
  have hrl : r < rounds.length := by
    have := List.length_pos_of_ne_nil hne
    omega
  have hor : rounds[r] = o := by
    rw [List.getLast?_eq_getElem?] at ho
    exact (List.getElem?_eq_some_iff.1 (Option.mem_def.1 ho)).2
  obtain ⟨hK, hok, hret, hnc⟩ := hfr r hrl
  obtain ⟨hK', hlen, -, e⟩ := persist_bridge settings O P.D Dsf P.length_le hC hN p r hrl
  set c := histCodes settings O P.D Dsf N p.1 p.2 r
  set H := decodeHist P.D settings.level5.walkStart c
  set cd : ClusterDraws Ω α K := (p.1, (p.2 ⟨r, hK⟩).1)
  set B := roundState settings O P.D Dsf H cd
  set F := clusterAt O.mq (loopPopulations K H.pops.length) (clusterRun P.D Dsf H cd) B
  set K₀ := roundKeys settings P.D Dsf H cd
  have hst4 : stage4 settings O P.D Dsf N r hK p.1 p.2 = some (c, B, F, K₀) :=
    (stage4_eq_some settings O P.D Dsf N r hK p.1 p.2 c B F K₀).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have ho' : o = mkRound settings O P.D (clusterRun P.D Dsf H cd) B F K₀
      (roundSets settings P.D Dsf H cd).1 (roundSets settings P.D Dsf H cd).2
      (p.1, (p.2 ⟨r, hK⟩).2) := by
    rw [← hor, e, loopRound_eq_mkRound]
  have hres : o.result = stageRes settings O P.D (c, B, F, K₀) p.1 (p.2 ⟨r, hK⟩).2 := by
    rw [ho']
    rfl
  have hBs : B ∈ settings.schedule (loopPopulations K H.pops.length) :=
    (Classical.epsilon_spec hret).1
  have hpop : histAt settings P.D c r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen)]
    rfl
  obtain ⟨T', xs, hxs⟩ := loopRounds_result settings O P.D Dsf (persistDraws p) K o
    (hor ▸ List.getElem_mem hrl)
  refine ⟨r, hK, by omega, (c, B, F, K₀), ?_, ⟨(c, B, F, K₀), resCode o.result, rfl, hlen, ?_⟩,
    ?_, ?_, ?_, ?_, ?_⟩
  · simp only [stageSel, hst4, Option.map_some, hres]
  · simp [resCode, hcons]
  · refine ⟨hnc, ?_, ?_⟩
    · rw [hpop]
      exact clusterAt_subset_suffsOf settings O P.D Dsf H p.1 (p.2 ⟨r, hK⟩).1 hBs
    · rw [hpop]
      exact roundKeys_eq_prefsOf settings P.D Dsf H p.1 (p.2 ⟨r, hK⟩).1
  · rw [ho']
    rfl
  · rw [ho']
    rfl
  · rw [ho']
    rfl
  · have hfin := result_fin settings.level5 o.read xs
    rw [← hxs] at hfin
    unfold resCode ResCode.hyp
    dsimp only
    rw [decodeEdges_edgeGraph hfin, Function.uncurry_curry]

end Last

end OrthoDFA

end
