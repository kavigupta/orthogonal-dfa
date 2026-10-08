import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.GateFlip
import OrthoDFA.Proofs.StartAtK

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, `RoundOutcome`, stated in `OrthoDFA.Stage`, `RoundProgress`, stated
in `OrthoDFA.Round`, and `RoundAtK`, `CheckYield` and `SourceSpread`, stated in `OrthoDFA.StartAtK`,
hold.
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

theorem check_yield : CheckYield := check_yield_holds

#print axioms check_yield

theorem source_spread : SourceSpread := source_spread_holds

#print axioms source_spread

end OrthoDFA
