import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Lloyd

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold.  Both are stated for every `Clusterer`, and
`clusterer_nonempty` exhibits one.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

#print axioms clusterer_nonempty

end OrthoDFA
