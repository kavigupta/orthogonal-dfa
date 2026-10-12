import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyRisk

namespace OrthoDFA

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ fam θ P C κs hκ
  exact tally_round_holds fam.M fam.oracle θ P C κs hκ

end OrthoDFA
