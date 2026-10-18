import OrthoDFA.Level5

/-! The probes whose start or end sift stops undecided. -/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The probes whose start or end sift stops undecided. -/
def undSiftSet (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : EdgeTable α) (k : ℕ) :
    Set (FreeMonoid α) :=
  {x | match probeBy cut T edges k x with
    | .startUndecided _ | .endUndecided _ => True
    | _ => False}

/-- Those of them, read through `read` and walked under `H`, where `sift` holds. -/
def siftedProbes (sift : Bool) (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | sift = true ∧ x ∈ undSiftSet (fun z => (read z).cut) H.tree (Function.curry H.edges) k}

end OrthoDFA
