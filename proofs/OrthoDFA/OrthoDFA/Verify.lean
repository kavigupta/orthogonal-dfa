import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

end OrthoDFA
