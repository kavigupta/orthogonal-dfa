import OrthoDFA.TallyFamily
import OrthoDFA.Proofs.TallyRisk

namespace OrthoDFA

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ σ _ Ω _ μ _ fam D _ L hlen C θ κs hκ
  exact tally_round_holds (fam.readModel θ) (fun z ω => fam.read ω z) fam.read_measurable
    fam.read_indep fam.read_law D L hlen C κs hκ

end OrthoDFA
