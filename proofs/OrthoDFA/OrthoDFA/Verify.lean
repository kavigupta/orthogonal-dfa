import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.FamilyRead

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem family_read_independent : FamilyReadIndependent := family_read_independent_holds

#print axioms family_read_independent

theorem family_read_by_state : FamilyReadByState := family_read_by_state_holds

#print axioms family_read_by_state

theorem family_read_trichotomy : FamilyReadTrichotomy := family_read_trichotomy_holds

#print axioms family_read_trichotomy

theorem band_holds_shipped : BandHolds 62 20 42 (1 / 5) (1 / 5) := shipped_band_holds

#print axioms band_holds_shipped

theorem family_read_guarantee : FamilyReadGuarantee := family_read_guarantee_holds

#print axioms family_read_guarantee

end OrthoDFA
