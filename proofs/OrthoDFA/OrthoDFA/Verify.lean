import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Stage
import OrthoDFA.Proofs.Replay

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, and `PassDichotomy`, `ReplaySpread` and `ReplaySpreadUniform`, stated
in `OrthoDFA.Stage`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem pass_dichotomy : PassDichotomy := pass_dichotomy_holds

#print axioms pass_dichotomy

theorem replay_spread : ReplaySpread := replay_spread_holds

#print axioms replay_spread

theorem replay_spread_uniform : ReplaySpreadUniform := replay_spread_uniform_holds

#print axioms replay_spread_uniform

end OrthoDFA
