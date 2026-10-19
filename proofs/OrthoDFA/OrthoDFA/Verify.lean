import OrthoDFA.Proofs.AdaptiveReads
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.Level5Noise
import OrthoDFA.Proofs.Level5Query
import OrthoDFA.Proofs.Level5Event
import OrthoDFA.Proofs.Level5Correct
import OrthoDFA.Proofs.Level5Risk
import OrthoDFA.Proofs.Level5RiskBound
import OrthoDFA.Proofs.Probe
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
`OrthoDFA.Utils.AdaptiveReads`, level 2's claim, stated in `OrthoDFA.Level2Claim`, level 5's, stated
in `OrthoDFA.Level5Claim`, and level 5 read by the suffix family, stated in `OrthoDFA.Level5Family`.
Level 5's detailed bound and its closed form, stated in `OrthoDFA.Proofs.Level5RiskDefs`, are
internal: the claims above are derived from them.
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

#print axioms runEnds_of_budget

#print axioms truthInv_step

#print axioms pi_wrong_le

#print axioms site_le

#print axioms goodStart_le

#print axioms goodEdge_le

#print axioms goodPT_le

#print axioms noiseEvent_le

theorem level5_correct : Level5Correct := level5_correct_holds

#print axioms level5_correct

theorem level5_round_risk : Level5RoundRisk := level5_round_risk_holds

#print axioms level5_round_risk

theorem level5_risk_bound : Level5RiskBound := level5_risk_bound_holds

#print axioms level5_risk_bound

#print axioms QTree.firsts_lintegral_le

#print axioms run_probeQuery

#print axioms asks_probeQuery

#print axioms edge_first_mean

theorem level5_family_correct : Level5FamilyCorrect := level5_family_correct_holds

#print axioms level5_family_correct

theorem seq_rate_sound : SeqRateSound := seqRateSound

#print axioms seq_rate_sound

theorem level2_correct : Level2.Correct := Level2.correct_holds

#print axioms level2_correct

theorem adaptive_reads_bound : AdaptiveReadsBound := adaptiveReadsBound_holds

#print axioms adaptive_reads_bound

end OrthoDFA
