import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.RandomRound
import OrthoDFA.Proofs.EpsRound
import OrthoDFA.Proofs.RateTest
import OrthoDFA.Proofs.SeqTest
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`, `TripartiteSearchCorrect`, stated in `OrthoDFA.TripartiteSearch`,
`SiftWithPathCorrect` and `SiftAndSplitCorrect`, stated in `OrthoDFA.DTree`, `WalkCorrect`, stated
in `OrthoDFA.HypothesisDFA`, `RateTestSound`, stated in `OrthoDFA.RateTest`, `SeqRateSound`,
stated in `OrthoDFA.SeqTest`, level 2's ideal-read claim, stated in `OrthoDFA.Level2Claim`, level
3's random-read claim, stated in `OrthoDFA.Level3Claim`, and level 4's claim with rare wrong
reads, stated in `OrthoDFA.Level4Claim`.
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

theorem sift_with_path_correct : SiftWithPathCorrect := siftWithPathCorrect

#print axioms sift_with_path_correct

theorem sift_and_split_correct : SiftAndSplitCorrect := siftAndSplitCorrect

#print axioms sift_and_split_correct

theorem walk_correct : WalkCorrect := walkCorrect

#print axioms walk_correct

theorem rate_test_sound : RateTestSound := rateTestSound

#print axioms rate_test_sound

theorem seq_rate_sound : SeqRateSound := seqRateSound

#print axioms seq_rate_sound

theorem ideal_round_correct : Ideal.IdealRoundCorrect := Ideal.idealRoundCorrect_holds

#print axioms ideal_round_correct

theorem random_round_correct : Random.RandomRoundCorrect := Random.randomRoundCorrect_holds

#print axioms random_round_correct

theorem eps_round_correct : Random.Eps.EpsRoundCorrect := Random.Eps.epsRoundCorrect_holds

#print axioms eps_round_correct

end OrthoDFA
