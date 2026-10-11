import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.FamilyRead
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.TallyNoise
import OrthoDFA.Proofs.TallyQuery
import OrthoDFA.Proofs.TallyKeep
import OrthoDFA.Proofs.TallyRace
import OrthoDFA.Proofs.TallySuccess
import OrthoDFA.Proofs.TallyEvent
import OrthoDFA.Proofs.TallySpurNoise
import OrthoDFA.Proofs.TallyFamily
import OrthoDFA.Proofs.TallyRisk

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and `ClusteringQualityGuarantee`, stated in
`OrthoDFA.ClusteringQuality`, hold, and so do the family-read claim stated in `OrthoDFA.FamilyRead`,
the idealized round's, stated in `OrthoDFA.IdealRound`, the tally round's, stated in
`OrthoDFA.TallyRound`, and the tally round read by the suffix family, stated in
`OrthoDFA.TallyFamily`.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem family_read_bound : FamilyReadBound := family_read_bound_holds

#print axioms family_read_bound

#print axioms round_budget

#print axioms tally_round_of

#print axioms harvest_good_holds

#print axioms sub_round_holds

#print axioms fake_race_holds

#print axioms goodStart_le

#print axioms goodEdge_le

#print axioms goodPT_le

#print axioms tallyE_le

#print axioms spurious_le

#print axioms success_sound_holds

theorem tally_round : TallyRound := tally_round_holds

#print axioms tally_round

#print axioms QTree.firsts_lintegral_le

#print axioms run_probeQ

#print axioms asks_probeQ

#print axioms untrue_minority

#print axioms und_twins

#print axioms spurious_mean

#print axioms edge_first_mean

theorem tally_round_family : TallyRoundFamily := tally_round_family_holds

#print axioms tally_round_family

theorem ideal_round_correct : Ideal.IdealRoundCorrect := Ideal.idealRoundCorrect_holds

#print axioms ideal_round_correct

end OrthoDFA
