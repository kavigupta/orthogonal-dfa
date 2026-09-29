import OrthoDFA.Proofs.Budget
import OrthoDFA.Proofs.Check
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Learner
import OrthoDFA.Proofs.ReturnAccuracy
import OrthoDFA.Proofs.Termination

/-!
# The theorem

The claims hold: `ClusteringGuarantee`, stated in `OrthoDFA.Clustering`, and the others, each
stated in the file named after it.
-/

namespace OrthoDFA

theorem clustering_guarantee : ClusteringGuarantee := clustering_guarantee_of_correct

#print axioms clustering_guarantee

theorem clustering_quality_guarantee : ClusteringQualityGuarantee :=
  clustering_quality_guarantee_holds

#print axioms clustering_quality_guarantee

theorem return_accuracy : ReturnAccuracy := return_accuracy_holds

#print axioms return_accuracy

theorem termination : Termination := termination_holds

#print axioms termination

theorem check_guarantee : CheckGuarantee := check_guarantee_holds

#print axioms check_guarantee

theorem learner_correct : LearnerCorrect := learner_correct_of_check check_guarantee_holds

#print axioms learner_correct

end OrthoDFA
