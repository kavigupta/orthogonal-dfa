import OrthoDFA.Proofs.AdaptiveReads
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.CoveredStart
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.Probe
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.EpsRound
import OrthoDFA.Proofs.KappaRound
import OrthoDFA.Proofs.RaceReads
import OrthoDFA.Proofs.RateTest
import OrthoDFA.Proofs.SeqTest
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`, `TripartiteSearchCorrect`, stated in `OrthoDFA.TripartiteSearch`,
`SiftWithPathCorrect` and `SiftAndSplitCorrect`, stated in `OrthoDFA.DTree`, `WalkCorrect`, stated
in `OrthoDFA.HypothesisDFA`, `SplitAndLearnCorrect`, stated in `OrthoDFA.HypothesisDFA`,
`ProbeCorrect`, stated in `OrthoDFA.Probe`, `RateTestSound`, stated in `OrthoDFA.RateTest`,
`SeqRateSound`, stated in `OrthoDFA.SeqTest`, `AdaptiveReadsBound`, stated in
`OrthoDFA.Utils.AdaptiveReads`, `RaceReadsBound`, stated in `OrthoDFA.Utils.RaceReads`,
`CoveredStartCorrect`, stated in `OrthoDFA.Utils.CoveredStart`, level 2's
claim, stated in `OrthoDFA.Level2Claim`, level 3's random-read claim, stated in
`OrthoDFA.Level3Claim`, level 4's claim with rare wrong reads, stated in `OrthoDFA.Level4Claim`,
level 5's claim with wrong reads scaling with indecision, stated in `OrthoDFA.Level5Claim`, and
`FamilyKappaOracle`, stated in `OrthoDFA.FamilyOracle`.
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

theorem split_and_learn_correct : SplitAndLearnCorrect := splitAndLearnCorrect

#print axioms split_and_learn_correct

theorem probe_correct : ProbeCorrect := probeCorrect

#print axioms probe_correct

theorem rate_test_sound : RateTestSound := rateTestSound

#print axioms rate_test_sound

theorem seq_rate_sound : SeqRateSound := seqRateSound

#print axioms seq_rate_sound

theorem level2_correct : Level2.Correct := Level2.correct_holds

#print axioms level2_correct

theorem adaptive_reads_bound : AdaptiveReadsBound := adaptiveReadsBound_holds

#print axioms adaptive_reads_bound

theorem random_round_correct : Random.RandomRoundCorrect := Random.randomRoundCorrect_holds

#print axioms random_round_correct

theorem eps_round_correct : Random.Eps.EpsRoundCorrect := Random.Eps.epsRoundCorrect_holds

#print axioms eps_round_correct

theorem race_reads_bound : RaceReadsBound := raceReadsBound_holds

#print axioms race_reads_bound

theorem kappa_round_correct : Random.Kappa.KappaRoundCorrect :=
  Random.Kappa.kappaRoundCorrect_holds

#print axioms kappa_round_correct

theorem family_kappa_oracle : FamilyKappaOracle := familyKappaOracle_holds

#print axioms family_kappa_oracle

theorem covered_start_correct : CoveredStartCorrect := coveredStartCorrect_holds

#print axioms covered_start_correct

end OrthoDFA
