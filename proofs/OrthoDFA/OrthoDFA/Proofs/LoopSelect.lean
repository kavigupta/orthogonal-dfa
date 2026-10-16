import OrthoDFA.Proofs.Select
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

theorem level5_congr_zone (O : Oracle μ (FreeMonoid α)) (C : Level5Settings)
    (hC : 2 ≤ C.maxLeaves) (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ w ∈ level5Zone C xs F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω') xs := by
  sorry

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

section Bound

variable [Nonempty α]

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
  sorry

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
  sorry

end Last

end OrthoDFA

end
