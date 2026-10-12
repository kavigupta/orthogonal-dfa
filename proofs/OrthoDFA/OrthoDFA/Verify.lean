import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.Level4Noise
import OrthoDFA.Proofs.Level4Query
import OrthoDFA.Proofs.Level4Keep
import OrthoDFA.Proofs.Level4Race
import OrthoDFA.Proofs.Level4Success
import OrthoDFA.Proofs.Level4Event
import OrthoDFA.Proofs.Level4SpurNoise
import OrthoDFA.Proofs.Level4Correct
import OrthoDFA.Proofs.Level4Risk
import OrthoDFA.Proofs.Level4RiskBound
import OrthoDFA.Proofs.TripartiteSearch
import OrthoDFA.Proofs.LoopSimple

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claim stated in `OrthoDFA.FamilyRead`,
`TripartiteSearchCorrect`, stated in `OrthoDFA.TripartiteSearch`, level 2's ideal-read claim, stated
in `OrthoDFA.Level2Claim`, level 4's, stated in `OrthoDFA.Level4Claim`, and level 4 read by the
suffix family, stated in `OrthoDFA.Level4Family`, and the learner's loop, stated in
`OrthoDFA.LearnerLoop`, at free settings and at the theorem's own. Level 4's detailed bound and
its closed form, stated in `OrthoDFA.Proofs.Level4RiskDefs`, are internal: the claims above are
derived from them.
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

#print axioms round_budget

#print axioms roundGiven_of

#print axioms harvest_good_holds

#print axioms sub_round_holds

#print axioms fake_race_holds

#print axioms goodStart_le

#print axioms goodEdge_le

#print axioms goodPT_le

#print axioms noiseEvent_le

#print axioms spurious_le

#print axioms success_sound_holds

theorem level4_correct : Level4Correct := level4_correct_holds

#print axioms level4_correct

theorem level4_round_risk : Level4RoundRisk := level4_round_risk_holds

#print axioms level4_round_risk

theorem level4_risk_bound : Level4RiskBound := level4_risk_bound_holds

#print axioms level4_risk_bound

#print axioms QTree.firsts_lintegral_le

#print axioms run_probeQuery

#print axioms asks_probeQuery

#print axioms untrue_minority

#print axioms und_twins

#print axioms spurious_mean

#print axioms edge_first_mean

theorem level4_family_correct : Level4FamilyCorrect := level4_family_correct_holds

#print axioms level4_family_correct

theorem learner_loop_correct : LearnerLoopCorrect := learner_loop_correct_holds

#print axioms learner_loop_correct

theorem learner_loop_simple : LearnerLoopSimple := learner_loop_simple_holds

#print axioms learner_loop_simple

theorem ideal_round_correct : Ideal.IdealRoundCorrect := Ideal.idealRoundCorrect_holds

#print axioms ideal_round_correct

end OrthoDFA
