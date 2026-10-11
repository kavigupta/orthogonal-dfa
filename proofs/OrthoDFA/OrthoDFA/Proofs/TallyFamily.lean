import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyRisk

namespace OrthoDFA

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ fam D _ L hlen C θ κs t q hκ ht hq hpre
  exact tally_round_holds (fam.readModel θ) (fam.readOracle θ) D L hlen C κs t q hκ ht hq hpre

end OrthoDFA
