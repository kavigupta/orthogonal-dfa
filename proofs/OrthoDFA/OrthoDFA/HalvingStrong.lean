import OrthoDFA.RoundStrong

/-!
# What a halving leaves to given-up edges

`RoundStrongNoStop`: no attempt in the round stops, so it holds no stopped strings.
-/

namespace OrthoDFA

/-- `RoundStrongNoStop`: the round holds no string that stopped an attempt's guards. -/
def RoundStrongNoStop : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    (strongRun C R seed Rmax d).2.1.stopped = []

end OrthoDFA
