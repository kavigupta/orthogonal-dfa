import RLStar.Proofs.AdaptiveReads
import RLStar.Proofs.BandKappa
import RLStar.Proofs.Allowance
import RLStar.Proofs.BitSum
import RLStar.Proofs.BlockMiss
import RLStar.Proofs.Budget
import RLStar.Proofs.ClaimTrainFamily
import RLStar.Proofs.ClaimCounterexSynth
import RLStar.Proofs.ClaimRLStar
import RLStar.Proofs.ClusteringQuality
import RLStar.Proofs.CoveredStart
import RLStar.Proofs.DTree
import RLStar.Proofs.FamilyOracle
import RLStar.Proofs.FamilyRead
import RLStar.Proofs.HypothesisDFA
import RLStar.Proofs.Level2Round
import RLStar.Proofs.LoopRuns
import RLStar.Proofs.Probe
import RLStar.Proofs.LoopDFA
import RLStar.Proofs.LoopDFARelative
import RLStar.Proofs.Level4
import RLStar.Proofs.Level5
import RLStar.Proofs.RandomCorrect
import RLStar.Proofs.EpsRound
import RLStar.Proofs.RaceTail
import RLStar.Proofs.RateTest
import RLStar.Proofs.SeqTest
import RLStar.Proofs.SpurAccount
import RLStar.Proofs.SpurCount
import RLStar.Proofs.SuccessiveDraws
import RLStar.Proofs.TrieMass
import RLStar.Proofs.TripartiteSearch
import RLStar.Proofs.Trie
import RLStar.Proofs.TrieLoop
import RLStar.Proofs.TrieTable
import RLStar.Proofs.UnaskedFresh

/-!
# The theorem

The headline: `LearnerLoopDFA`, stated in `RLStar.LowLevelClaims.LoopDFA`, holds: the learner's
loop outputs a hypothesis that agrees with the target DFA, with every round's tree within `|Q| + 2`
leaves and each round's harvested draws within `⌈HB⌉₊` probes (`HarvestProbesWithin`), but for
chance `δ`.

The claims it rests on are `ClusteringGuarantee`, stated in `RLStar.LowLevelClaims.Clustering`,
`ClusteringQualityGuarantee`, stated in `RLStar.LowLevelClaims.ClusteringQuality`, the family-read
claims, stated in `RLStar.LowLevelClaims.FamilyRead`, `ProbeCorrect`, stated in
`RLStar.LowLevelClaims.Probe`, `Level2RoundCorrect`, stated in
`RLStar.LowLevelClaims.Level2Claim`, `Level3RoundCorrect`, `Level4RoundCorrect` and
`Level5RoundCorrect`, stated in `RLStar.LowLevelClaims.Level3Claim`,
`RLStar.LowLevelClaims.Level4Claim` and `RLStar.LowLevelClaims.Level5Claim`, and
`FamilyKappaOracle`, stated in `RLStar.LowLevelClaims.FamilyOracle`. `LearnerLoopDFASimple` is the
same claim with the simpler premises, and `LearnerLoopDFADist` the simpler one over strings drawn
from an input distribution. Level 3's reads are randomly indecisive; level 4's are rarely
wrong, and level 5's wrong at most `κ` times as often as undecided, a few wrong reads tolerated at
both. The loop runs level 5.

The supporting claims are stated under `RLStar.LowLevelClaims`: `AdaptiveReadsBound`,
`UnaskedFresh`, `RaceTailBound`, `Level3AllowanceCorrect`, `BitSumTails`, `BandKappaBound`,
`SuccessiveDrawsTails`, `BlockMiss`, `CoveredStartCorrect`, `TrieProbeAgreeCorrect`,
`TrieSinglePieceCorrect`, `TrieTableCorrect`, `TrieTableReads`, `Level3TrieRootOne`,
`Level3TrieAgrees`, `TrieLoopAgrees`, `TrieMassBound`, `WalkStartMoves`, `RateTestSound`,
`SeqRateSound`, `TripartiteSearchCorrect`, `SiftWithPathCorrect`, `SiftAndSplitCorrect`,
`WalkCorrect`, `SplitAndLearnCorrect` and `RunsKeepEdgesAtLeaves`.

The costs, as big-O shapes in the inputs alone, are stated under `RLStar.Claims`:
`TrainFamilyClaim`, `CounterexSynthClaim`, `RLStarClaim`,
`RLStarClaimUniform` and `RLStarClaimUniformSimple`. The `RLStar` claims are of the learner with
local walk starts (`indecisiveLStarTrieLoop`).
-/

namespace OrthoDFA

theorem train_family_claim : TrainFamilyClaim := trainFamilyClaim_holds

#print axioms train_family_claim

theorem counterex_synth_claim : CounterexSynthClaim := counterexSynthClaim_holds

#print axioms counterex_synth_claim

theorem rl_star_claim : RLStarClaim := rlStarClaim_holds

#print axioms rl_star_claim

theorem rl_star_claim_uniform : RLStarClaimUniform := rlStarClaimUniform_holds

#print axioms rl_star_claim_uniform

theorem rl_star_claim_uniform_simple : RLStarClaimUniformSimple := rlStarClaimUniformSimple_holds

#print axioms rl_star_claim_uniform_simple

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

theorem learner_loop_dfa_dist : LearnerLoopDFADist := learner_loop_dfa_dist_holds

#print axioms learner_loop_dfa_dist

theorem learner_loop_dfa_relative_simple : LearnerLoopDFARelativeSimple :=
  learner_loop_dfa_relative_simple_holds

#print axioms learner_loop_dfa_relative_simple

theorem learner_loop_dfa_relative : LearnerLoopDFARelative := learner_loop_dfa_relative_holds

#print axioms learner_loop_dfa_relative

theorem split_and_learn_correct : SplitAndLearnCorrect := splitAndLearnCorrect

#print axioms split_and_learn_correct

theorem runs_keep_edges_at_leaves : RunsKeepEdgesAtLeaves := runsKeepEdgesAtLeaves_holds

#print axioms runs_keep_edges_at_leaves

theorem probe_correct : ProbeCorrect := probeCorrect

#print axioms probe_correct

theorem rate_test_sound : RateTestSound := rateTestSound

#print axioms rate_test_sound

theorem bit_sum_tails : BitSumTails := bitSumTails

#print axioms bit_sum_tails

theorem band_kappa_bound : BandKappaBound := bandKappaBound_holds

#print axioms band_kappa_bound

theorem successive_draws_tails : SuccessiveDrawsTails := successiveDrawsTails_holds

#print axioms successive_draws_tails

theorem block_miss : BlockMiss := blockMiss_holds

#print axioms block_miss

theorem seq_rate_sound : SeqRateSound := seqRateSound

#print axioms seq_rate_sound

theorem level2_round_correct : Level2RoundCorrect := Level2.level2RoundCorrect_holds

#print axioms level2_round_correct

theorem adaptive_reads_bound : AdaptiveReadsBound := adaptiveReadsBound_holds

#print axioms adaptive_reads_bound

theorem unasked_fresh : UnaskedFresh := unaskedFresh_holds

#print axioms unasked_fresh

theorem level3_round_correct : Level3RoundCorrect := Random.level3RoundCorrect_holds

#print axioms level3_round_correct

theorem level4_round_correct : Level4RoundCorrect := level4RoundCorrect_holds

#print axioms level4_round_correct

theorem race_tail_bound : RaceTailBound := raceTailBound_holds

#print axioms race_tail_bound

theorem level3_allowance_correct : Level3AllowanceCorrect := level3AllowanceCorrect_holds

#print axioms level3_allowance_correct

theorem level5_round_correct : Level5RoundCorrect := level5RoundCorrect_holds

#print axioms level5_round_correct

theorem family_kappa_oracle : FamilyKappaOracle := familyKappaOracle_holds

#print axioms family_kappa_oracle

theorem covered_start_correct : CoveredStartCorrect := coveredStartCorrect_holds

#print axioms covered_start_correct

theorem trie_probe_agree_correct : TrieProbeAgreeCorrect := trieProbeAgreeCorrect

#print axioms trie_probe_agree_correct

theorem trie_single_piece_correct : TrieSinglePieceCorrect := trieSinglePieceCorrect

#print axioms trie_single_piece_correct

theorem trie_table_correct : TrieTableCorrect := trieTableCorrect_holds

#print axioms trie_table_correct

theorem trie_table_reads : TrieTableReads := trieTableReads_holds

#print axioms trie_table_reads

theorem level3_trie_root_one : Level3TrieRootOne := level3TrieRootOne

#print axioms level3_trie_root_one

theorem level3_trie_agrees : Level3TrieAgrees := level3TrieAgrees_holds

#print axioms level3_trie_agrees

theorem trie_loop_agrees : TrieLoopAgrees := trieLoopAgrees_holds

#print axioms trie_loop_agrees

theorem trie_mass_bound : TrieMassBound := trieMassBound_holds

#print axioms trie_mass_bound

theorem walk_start_moves : WalkStartMoves := walkStartMoves_holds

#print axioms walk_start_moves

end OrthoDFA
