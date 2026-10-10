import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.GateFlip
import OrthoDFA.Proofs.RoundAtK
import OrthoDFA.Proofs.StartState
import OrthoDFA.Proofs.Ends
import OrthoDFA.Proofs.Trichotomy
import OrthoDFA.Proofs.RoundLevel
import OrthoDFA.Proofs.RoundStrong
import OrthoDFA.Proofs.EdgeAttempts
import OrthoDFA.Proofs.Spurious
import OrthoDFA.Proofs.Exhausted
import OrthoDFA.Proofs.TallyFix
import OrthoDFA.Proofs.Freedman
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.TallyRatio

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, `RoundOutcome`, stated in `OrthoDFA.Stage`, `RoundProgress`, stated
in `OrthoDFA.Round`, `RoundAtK`, stated in `OrthoDFA.StartAtK`,
`StartExists`, stated in `OrthoDFA.StartState`, `StartRootCovered`, `EndsCovered` and
`BadShare`, stated in `OrthoDFA.Ends`, `RoundTrichotomy`, stated in `OrthoDFA.Trichotomy`, and
`RoundTrichotomyLevel` and `RoundQualityLevel`, stated in `OrthoDFA.RoundLevel`, the
`RoundStrong` claims, stated in `OrthoDFA.RoundStrong`, `OrthoDFA.Spurious` and
`OrthoDFA.Exhausted`, the family-read claims stated in `OrthoDFA.FamilyRead`, and `TallyRound`,
stated in `OrthoDFA.TallyRound`, hold.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem round_outcome : RoundOutcome := round_outcome_holds

#print axioms round_outcome

theorem round_progress : RoundProgress := round_progress_holds

#print axioms round_progress

theorem round_at_k : RoundAtK := round_at_k_holds

#print axioms round_at_k

theorem start_exists : StartExists := start_exists_holds

#print axioms start_exists

theorem start_root_covered' : StartRootCovered := start_root_covered

#print axioms start_root_covered'

theorem ends_covered' : EndsCovered := ends_covered

#print axioms ends_covered'

theorem bad_share' : BadShare := bad_share

#print axioms bad_share'

theorem round_trichotomy' : RoundTrichotomy := round_trichotomy

#print axioms round_trichotomy'

theorem round_trichotomy_level' : RoundTrichotomyLevel := round_trichotomy_level

#print axioms round_trichotomy_level'

theorem round_quality_level' : RoundQualityLevel := round_quality_level

#print axioms round_quality_level'

theorem round_strong_readings' : RoundStrongReadings := round_strong_readings

#print axioms round_strong_readings'

theorem round_strong_leaves' : RoundStrongLeaves := round_strong_leaves

#print axioms round_strong_leaves'

theorem round_strong_same_state' : RoundStrongSameState := round_strong_same_state

#print axioms round_strong_same_state'

theorem round_strong_leaf_paths' : RoundStrongLeafPaths := round_strong_leaf_paths

#print axioms round_strong_leaf_paths'

theorem round_strong_trichotomy' : RoundStrongTrichotomy := round_strong_trichotomy

#print axioms round_strong_trichotomy'

theorem round_strong_quality' : RoundStrongQuality := round_strong_quality

#print axioms round_strong_quality'

theorem round_strong_no_stop' : RoundStrongNoStop := round_strong_no_stop

#print axioms round_strong_no_stop'

theorem edge_cause' : EdgeCause := edge_cause

#print axioms edge_cause'

theorem spurious_draw' : SpuriousDraw := spurious_draw

#print axioms spurious_draw'

theorem round_strong_spurious' : RoundStrongSpurious := round_strong_spurious

#print axioms round_strong_spurious'

theorem round_strong_power' : RoundStrongPower := by
  intro α _ _ Ω _ μ _ C O B F seed τ Rmax d h0 hτ
  exact power_tail C O B F seed τ Rmax d h0 hτ

#print axioms round_strong_power'

theorem round_strong_exhausted' : RoundStrongExhausted := round_strong_exhausted

#print axioms round_strong_exhausted'

#print axioms fix_in_time

#print axioms tally_fix_in_time

#print axioms bennett_tail

theorem family_read_trichotomy : FamilyReadTrichotomy := family_read_trichotomy_holds

#print axioms family_read_trichotomy

theorem tally_round : TallyRound := tally_round_holds

#print axioms tally_round

#print axioms spurious_tail_ratio

end OrthoDFA
