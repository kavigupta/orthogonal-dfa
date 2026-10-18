import OrthoDFA.Level5

/-! The probes whose start or end sift stops undecided. -/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The probes whose start or end sift stops undecided. -/
def undSiftSet (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : Edges α) (k : ℕ) :
    Set (FreeMonoid α) :=
  {x | match probeOrAgree (ofCut cut) T edges k x with
    | .undecidedStart _ | .undecided _ true => True
    | _ => False}

/-- Those of them, read through `read` and walked under `H`, where `sift` holds. -/
def siftedProbes (sift : Bool) (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | sift = true ∧ x ∈ undSiftSet (fun z => (read z).cut) H.tree H.edges k}

end OrthoDFA
