import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.RandomRound

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`, the idealized round's, stated in `OrthoDFA.IdealRound`, and the round's
with random reads, stated in `OrthoDFA.RandomRound`.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem family_read_trichotomy : FamilyReadTrichotomy := family_read_trichotomy_holds

#print axioms family_read_trichotomy

theorem ideal_round_correct : Ideal.IdealRoundCorrect := Ideal.idealRoundCorrect_holds

#print axioms ideal_round_correct

theorem random_round_correct : Random.RandomRoundCorrect := Random.randomRoundCorrect_holds

#print axioms random_round_correct

end OrthoDFA
