import OrthoDFA.Proofs.TrichotomyBatch

/-!
# `RoundTrichotomy`

The classes' quality holds off one noise set (`quality_holds`), and for any reads the batch
claims hold but for the gate's and refusal sample's failure chances (`trichotomy_batch`).
-/

namespace OrthoDFA

theorem round_trichotomy : RoundTrichotomy := by
  intro α _ _ Ω _ μ _ Q A O B F K D _ L ng nr seed probes f c θM acc a δc η minCov ν ε gu
    hacc0 hacc1 hf hc ha hδc0 hδc hν hε hlen hV
  obtain ⟨E, hE, hq⟩ := quality_holds A O B F K D L seed probes hf hc hε hlen hV
  exact ⟨E, hE, fun ω hω => ⟨hq ω hω, trichotomy_batch (readsAt O B F ω) A _ D _ L ng nr gu
    hacc0 hacc1 hf ha hδc0 hδc hν⟩⟩

end OrthoDFA
