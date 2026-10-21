import OrthoDFA.Proofs.AdaptiveReads
import OrthoDFA.Proofs.Allowance
import OrthoDFA.Proofs.BitSum
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.CoveredStart
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.Probe
import OrthoDFA.Proofs.LoopDFA
import OrthoDFA.Proofs.Level5
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.EpsRound
import OrthoDFA.Proofs.KappaRound
import OrthoDFA.Proofs.RaceReads
import OrthoDFA.Proofs.RaceTail
import OrthoDFA.Proofs.RateTest
import OrthoDFA.Proofs.SeqTest
import OrthoDFA.Proofs.SpurAccount
import OrthoDFA.Proofs.SpurCount
import OrthoDFA.Proofs.SuccessiveDraws
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The theorem

The headline: `LearnerLoopDFA`, stated in `OrthoDFA.LoopDFA`, holds: the learner's loop outputs a
hypothesis that agrees with the target DFA but for chance `δ`.

The claims it rests on are `ClusteringGuarantee`, stated in `OrthoDFA.Clustering`,
`ClusteringQualityGuarantee`, stated in `OrthoDFA.ClusteringQuality`, the family-read claims, stated
in `OrthoDFA.FamilyRead`, `ProbeCorrect`, stated in `OrthoDFA.Probe`, `Level2RoundCorrect`, stated
in `OrthoDFA.Level2Claim`, `Level3RoundCorrect`, `Level4RoundCorrect` and `Level4KappaRoundCorrect`,
stated in `OrthoDFA.Level3Claim`, `OrthoDFA.Level4Claim` and `OrthoDFA.Level4KappaClaim`,
`Level3AllowanceCorrect` and `Level5RoundCorrect`, stated in `OrthoDFA.AllowanceClaim` and
`OrthoDFA.Level5Claim`, and
`FamilyKappaOracle`, stated in `OrthoDFA.FamilyOracle`. `LearnerLoopDFASimple` is the same claim
with the simpler premises.

The supporting claims are stated under `OrthoDFA.Utils`: `AdaptiveReadsBound`, `RaceReadsBound`,
`RaceTailBound`, `BitSumTails`, `SuccessiveDrawsTails`, `CoveredStartCorrect`, `RateTestSound`, `SeqRateSound`, `TripartiteSearchCorrect`,
`SiftWithPathCorrect`, `SiftAndSplitCorrect`, `WalkCorrect` and `SplitAndLearnCorrect`.
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

theorem learner_loop_dfa_simple : LearnerLoopDFASimple := learner_loop_dfa_simple_holds

#print axioms learner_loop_dfa_simple

theorem learner_loop_dfa : LearnerLoopDFA := learner_loop_dfa_holds

#print axioms learner_loop_dfa

theorem split_and_learn_correct : SplitAndLearnCorrect := splitAndLearnCorrect

#print axioms split_and_learn_correct

theorem probe_correct : ProbeCorrect := probeCorrect

#print axioms probe_correct

theorem rate_test_sound : RateTestSound := rateTestSound

#print axioms rate_test_sound

theorem bit_sum_tails : BitSumTails := bitSumTails

#print axioms bit_sum_tails

theorem successive_draws_tails : SuccessiveDrawsTails := successiveDrawsTails_holds

#print axioms successive_draws_tails

theorem seq_rate_sound : SeqRateSound := seqRateSound

#print axioms seq_rate_sound

theorem level2_round_correct : Level2RoundCorrect := Level2.level2RoundCorrect_holds

#print axioms level2_round_correct

theorem adaptive_reads_bound : AdaptiveReadsBound := adaptiveReadsBound_holds

#print axioms adaptive_reads_bound

theorem level3_round_correct : Level3RoundCorrect := Random.level3RoundCorrect_holds

#print axioms level3_round_correct

theorem level4_round_correct : Level4RoundCorrect := Random.Eps.level4RoundCorrect_holds

#print axioms level4_round_correct

theorem race_reads_bound : RaceReadsBound := raceReadsBound_holds

#print axioms race_reads_bound

theorem race_tail_bound : RaceTailBound := raceTailBound_holds

#print axioms race_tail_bound

theorem level4_kappa_round_correct : Level4KappaRoundCorrect :=
  Random.Kappa.level4KappaRoundCorrect_holds

#print axioms level4_kappa_round_correct

theorem level3_allowance_correct : Level3AllowanceCorrect := level3AllowanceCorrect_holds

#print axioms level3_allowance_correct

theorem level5_round_correct : Level5RoundCorrect := level5RoundCorrect_holds

#print axioms level5_round_correct

theorem level3_allowance_correct : Level3AllowanceCorrect := level3AllowanceCorrect_holds

#print axioms level3_allowance_correct

theorem family_kappa_oracle : FamilyKappaOracle := familyKappaOracle_holds

#print axioms family_kappa_oracle

theorem covered_start_correct : CoveredStartCorrect := coveredStartCorrect_holds

#print axioms covered_start_correct

end OrthoDFA
