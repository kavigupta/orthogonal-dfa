import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.GateFlip
import OrthoDFA.Proofs.RoundAtK
import OrthoDFA.Proofs.StartState
import OrthoDFA.Proofs.Ends
import OrthoDFA.Proofs.Trichotomy
import OrthoDFA.Proofs.RoundLevel

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, `RoundOutcome`, stated in `OrthoDFA.Stage`, `RoundProgress`, stated
in `OrthoDFA.Round`, `RoundAtK`, stated in `OrthoDFA.StartAtK`,
`StartExists`, stated in `OrthoDFA.StartState`, `StartRootCovered`, `EndsCovered` and
`BadShare`, stated in `OrthoDFA.Ends`, `RoundTrichotomy`, stated in `OrthoDFA.Trichotomy`, and
`RoundTrichotomyLevel` and `RoundQualityLevel`, stated in `OrthoDFA.RoundLevel`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem round_outcome : RoundOutcome := round_outcome_holds

#print axioms round_outcome

theorem round_progress : RoundProgress := round_progress_holds

#print axioms round_progress

theorem round_at_k : RoundAtK := round_at_k_holds

#print axioms round_at_k

theorem start_exists : StartExists := start_exists_holds

#print axioms start_exists

theorem start_root_covered' : StartRootCovered := start_root_covered

#print axioms start_root_covered'

theorem ends_covered' : EndsCovered := ends_covered

#print axioms ends_covered'

theorem bad_share' : BadShare := bad_share

#print axioms bad_share'

theorem round_trichotomy' : RoundTrichotomy := round_trichotomy

#print axioms round_trichotomy'

theorem round_trichotomy_level' : RoundTrichotomyLevel := round_trichotomy_level

#print axioms round_trichotomy_level'

theorem round_quality_level' : RoundQualityLevel := round_quality_level

#print axioms round_quality_level'

end OrthoDFA
