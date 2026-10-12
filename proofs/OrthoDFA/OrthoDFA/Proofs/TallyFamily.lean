import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyRisk

namespace OrthoDFA

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ fam P C θ κs hκ
  exact tally_round_holds (fam.readModel θ) (fam.readOracle θ) P C κs hκ

end OrthoDFA
