import OrthoDFA.Stage

/-!
# Strings as a measurable, countable space
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

end OrthoDFA
