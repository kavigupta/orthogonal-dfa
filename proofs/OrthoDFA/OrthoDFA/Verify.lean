import OrthoDFA.Proofs.AdaptiveReads
import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.Probe
import OrthoDFA.Proofs.LoopDFA
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.EpsRound
import OrthoDFA.Proofs.KappaRound
import OrthoDFA.Proofs.KappaShort
import OrthoDFA.Proofs.RaceReads
import OrthoDFA.Proofs.RateTest
import OrthoDFA.Proofs.SeqTest
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The theorem

The headline: `LearnerLoopDFA`, stated in `OrthoDFA.LoopDFA`, holds: the learner's loop outputs a
hypothesis that agrees with the target DFA but for chance `δ`.

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claims stated in
`OrthoDFA.FamilyRead`, `TripartiteSearchCorrect`, stated in `OrthoDFA.TripartiteSearch`,
`SiftWithPathCorrect` and `SiftAndSplitCorrect`, stated in `OrthoDFA.DTree`, `WalkCorrect`, stated
in `OrthoDFA.HypothesisDFA`, `SplitAndLearnCorrect`, stated in `OrthoDFA.HypothesisDFA`,
`ProbeCorrect`, stated in `OrthoDFA.Probe`, `RateTestSound`, stated in `OrthoDFA.RateTest`,
`SeqRateSound`, stated in `OrthoDFA.SeqTest`, `AdaptiveReadsBound`, stated in
`OrthoDFA.Utils.AdaptiveReads`, `RaceReadsBound`, stated in `OrthoDFA.Utils.RaceReads`, level 2's
claim, stated in `OrthoDFA.Level2Claim`, level 3's random-read claim, stated in
`OrthoDFA.Level3Claim`, level 4's claim with rare wrong reads, stated in `OrthoDFA.Level4Claim`,
level 5's claim with wrong reads scaling with indecision, stated in `OrthoDFA.Level5Claim`, whose
oracle the clustering's family reads are (`FamilyKappaOracle`, stated in `OrthoDFA.FamilyOracle`),
with its version reading the prefix mass past the walk's start (`KappaRoundCorrectAt`), and
`LearnerLoopDFASimple`, stated in `OrthoDFA.LoopDFA`.
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

theorem kappa_round_correct_at : Random.Kappa.KappaRoundCorrectAt :=
  Random.Kappa.kappaRoundCorrectAt_holds

#print axioms kappa_round_correct_at

theorem family_kappa_oracle : FamilyKappaOracle := familyKappaOracle_holds

#print axioms family_kappa_oracle

end OrthoDFA
