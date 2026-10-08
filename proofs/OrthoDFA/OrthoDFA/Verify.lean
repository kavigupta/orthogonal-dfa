import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.GateFlip

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, `RoundOutcome`, stated in `OrthoDFA.Stage`, and `RoundTetrachotomy`,
stated in `OrthoDFA.Round`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem round_outcome : RoundOutcome := round_outcome_holds

#print axioms round_outcome

theorem round_tetrachotomy : RoundTetrachotomy := round_tetrachotomy_holds

#print axioms round_tetrachotomy

end OrthoDFA
