import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.RandomRound
import OrthoDFA.Proofs.EpsRound
import OrthoDFA.Proofs.RateTest
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`, `TripartiteSearchCorrect`, stated in `OrthoDFA.TripartiteSearch`,
`RateTestSound`, stated in `OrthoDFA.RateTest`, level 2's ideal-read claim, stated in
`OrthoDFA.Level2Claim`, level 3's random-read claim, stated in `OrthoDFA.Level3Claim`, and level
3.5's claim with rare wrong reads, stated in `OrthoDFA.Level3_5Claim`.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem family_read_bound : FamilyReadBound := family_read_bound_holds

#print axioms family_read_bound

theorem tripartite_search_correct : TripartiteSearchCorrect := OrthoDFA.tripartiteSearch_correct

#print axioms tripartite_search_correct

theorem rate_test_sound : RateTestSound := rateTestSound

#print axioms rate_test_sound

theorem ideal_round_correct : Ideal.IdealRoundCorrect := Ideal.idealRoundCorrect_holds

#print axioms ideal_round_correct

theorem random_round_correct : Random.RandomRoundCorrect := Random.randomRoundCorrect_holds

#print axioms random_round_correct

theorem eps_round_correct : Random.EpsRoundCorrect := Random.epsRoundCorrect_holds

#print axioms eps_round_correct

end OrthoDFA
