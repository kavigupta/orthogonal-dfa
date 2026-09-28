import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.Quality

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `QualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem quality_guarantee : QualityGuarantee := quality_guarantee_holds

#print axioms quality_guarantee

end OrthoDFA
