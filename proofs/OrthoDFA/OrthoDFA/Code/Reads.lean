import OrthoDFA.Code.Level3Harvest
import OrthoDFA.Code.Clustering
import OrthoDFA.Code.FamilyRead
import OrthoDFA.Code.Probe
import OrthoDFA.Code.TripartiteSearch
import OrthoDFA.Code.HypothesisDFA
import OrthoDFA.Code.DTree

/-!
# What the learner reads from the oracle

The strings a level-3 run asks the oracle about, and a count of the calls it makes. `familyRead` at
`w` queries only `{w · v : v ∈ F}`, so one read is `|F|` calls and `level3Count` is `runReadCount`
times `|F|`. `runReadsFrom` and `runReadCount` walk the same `applyProbe` recursion the run itself
walks, so the count tallies exactly the reads the run performs.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*}

open scoped Classical in
/-- The midfixes of the tree's nodes. -/
noncomputable def mids : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (mids r ∪ mids a)

open MeasureTheory in
/-- The largest mass the probes' distribution puts on one length-`k` prefix. -/
noncomputable def prefixMax [Fintype α] (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ :=
  ⨆ u, D.real {x | prefixOf x k = u}

namespace Kappa

open scoped Classical in
/-- The strings sifting `z` reads. -/
noncomputable def siftReads (read : FreeMonoid α → ARU) :
    DTree α → FreeMonoid α → Finset (FreeMonoid α)
  | .leaf, _ => ∅
  | .node m r a, z => insert (z * m) (match read (z * m) with
    | .accept => siftReads read a z
    | .reject => siftReads read r z
    | .undecided => ∅)

/-- The positions whose sifts a search reads: those it queries, and those its result names. -/
def searchPos (res : SearchResult × List ℕ) (hi : ℕ) : List ℕ :=
  res.2 ++ match res.1 with
    | .edge p => [p - 1, p]
    | .undecidedMid p => [p]
    | .undecidedHi => [hi]
    | _ => []

open scoped Classical in
/-- The strings the sifts of the prefixes of `x` at the positions `ps` read. -/
noncomputable def posReads (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α)
    (ps : List ℕ) : Finset (FreeMonoid α) :=
  ps.toFinset.biUnion fun p => siftReads read T (prefixOf x p)

open scoped Classical in
/-- The strings a probe reads. -/
noncomputable def probeReads (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  if x.toList.length < k then ∅ else
  siftReads read T (prefixOf x k) ∪
    match T.sift read (prefixOf x k) with
    | .inr _ => ∅
    | .inl a =>
      let hi := k + (walk E a (x.toList.drop k)).length - 1
      let sr := posReads read T x (searchPos (tripartiteSearch false
        (siftAgrees read T x fun p => (walk E a (x.toList.drop k)).getD (p - k) []) k hi) hi)
      match x.toList[hi]? with
      | some _ => siftReads read T (prefixOf x (hi + 1)) ∪
        match T.sift read (prefixOf x (hi + 1)) with
        | .inr _ => ∅
        | .inl _ => sr
      | none => sr

/-- The sifts one probe of a length-`L` string makes: `min (L + 1) (3 + ⌈log₂ L⌉)`. -/
def probeSifts (L : ℕ) : ℕ := min (L + 1) (3 + Nat.clog 2 L)

end Kappa

section
variable [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

namespace Eps

open scoped Classical in
/-- Each learned edge's witness's successor followed by a midfix of `T'`. -/
noncomputable def resifts (s : Level3State α) (T' : DTree α) : Finset (FreeMonoid α) :=
  (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).biUnion fun e =>
    match s.edges e with
    | some (_, w) => (mids T').image ((w * FreeMonoid.of e.2) * ·)
    | none => ∅

/-- What a step's split re-sifts. -/
noncomputable def splitReads (s : Level3State α) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) =>
      resifts s (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
    | none => ∅
  | _ => ∅

end Eps

end

open Eps Kappa

section
variable [Fintype α] [DecidableEq α]

open scoped Classical in
/-- Every string a level-3 run from `s` over `xs` reads. -/
noncomputable def runReadsFrom (r : FreeMonoid α → ARU) (C : Level3Settings) :
    Level3State α → List (FreeMonoid α) → Finset (FreeMonoid α)
  | _, [] => ∅
  | s, x :: xs =>
    (probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x) ∪
      (match applyProbe r C s x with
       | .inl s' => runReadsFrom r C s' xs
       | .inr _ => ∅)

open scoped Classical in
/-- The oracle reads a level-3 run from `s` over `xs` makes, with multiplicity across steps. -/
noncomputable def runReadCount (r : FreeMonoid α → ARU) (C : Level3Settings) :
    Level3State α → List (FreeMonoid α) → ℕ
  | _, [] => 0
  | s, x :: xs =>
    (probeReads r s.tree s.edges C.walkStart x).card + (splitReads r C s x).card +
      (match applyProbe r C s x with
       | .inl s' => runReadCount r C s' xs
       | .inr _ => 0)

end

end Random

open Random MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Oracle calls the level-3 run over `xs` makes: each `runReadCount` read is one `familyRead`,
hence `|F|` calls. -/
noncomputable def level3Count (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) {P : ℕ} (xs : Fin P → FreeMonoid α) (ω : Ω) : ℕ :=
  runReadCount (fun z => familyRead O.mq F kl kh z ω) C initialState (List.ofFn xs) * F.card

end OrthoDFA
