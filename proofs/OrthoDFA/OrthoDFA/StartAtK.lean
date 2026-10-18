import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead
import OrthoDFA.RateTest
import OrthoDFA.TripartiteSearch

/-!
# A tree's midfixes
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

open scoped Classical in
noncomputable def midfixes : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (r.midfixes ∪ a.midfixes)

end DTree

end OrthoDFA
