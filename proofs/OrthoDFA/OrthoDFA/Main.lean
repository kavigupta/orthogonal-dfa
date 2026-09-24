import OrthoDFA.Budget

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Model`, holds.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

end OrthoDFA
