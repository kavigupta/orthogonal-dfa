import OrthoDFA.Level3

/-!
# Level 3.5 of indecisive L*: rare wrong reads

Level 3's algorithm, unchanged, run where a read can fall on its state's wrong side.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [DecidableEq α] [Fintype α]

noncomputable abbrev indecisiveLStarLevel3_5 (settings : Level3Settings)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α) : Level3Result α :=
  indecisiveLStarLevel3 settings read draws

end Random

end OrthoDFA
