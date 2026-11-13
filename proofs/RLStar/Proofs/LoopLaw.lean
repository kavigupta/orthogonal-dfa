import RLStar.Proofs.LoopAux
import RLStar.Proofs.RawRun
import RLStar.Proofs.DrawFrom
import Mathlib.Probability.ProductMeasure

/-!
# A round's draws as streams

`idealDraws` draws every population's stream, and lets the named populations certify on draws of
their own; it is drawn by `runMeasure` at the populations off the earlier prefixes
(`measurePreserving_idealDraws`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## A round's draws -/

/-- A stream of draws, each measure preserving, is measure preserving. -/
lemma measurePreserving_stream {β γ : Type*} [MeasurableSpace β] [MeasurableSpace γ]
    {ρ : Measure β} {τ : Measure γ} [IsProbabilityMeasure ρ] [IsProbabilityMeasure τ] {f : β → γ}
    (hf : MeasurePreserving f ρ τ) :
    MeasurePreserving (fun u : ℕ → β => fun n => f (u n))
      (Measure.infinitePi fun _ : ℕ => ρ) (Measure.infinitePi fun _ : ℕ => τ) :=
  ⟨measurable_pi_lambda _ fun n => hf.measurable.comp (measurable_pi_apply n), by
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => ρ) (hf := fun _ => hf.measurable), hf.map_eq]⟩

/-- Each population's stream, each measure preserving, is measure preserving. -/
lemma measurePreserving_streams {K : ℕ} {β : Type*} [MeasurableSpace β] {ρ : Measure β}
    [IsProbabilityMeasure ρ] (L : Option (Fin K) → Measure (FreeMonoid α))
    [∀ j, IsProbabilityMeasure (L j)] (f : Option (Fin K) → β → FreeMonoid α)
    (hf : ∀ j, MeasurePreserving (f j) ρ (L j)) :
    MeasurePreserving (fun U : Option (Fin K) → ℕ → β => fun j n => f j (U j n))
      (Measure.pi fun _ : Option (Fin K) => Measure.infinitePi fun _ : ℕ => ρ)
      (Measure.pi fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => L j) := by
  refine ⟨measurable_pi_lambda _ fun j => (measurePreserving_stream (hf j)).measurable.comp
    (measurable_pi_apply j), ?_⟩
  rw [Measure.pi_map_pi (fun j => (measurePreserving_stream (hf j)).measurable.aemeasurable)]
  congr 1
  funext j
  exact (measurePreserving_stream (hf j)).map_eq

variable {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

/-- The round's draws after the history `h` as streams, the named populations certifying on the
uniforms of their own. -/
def idealDraws (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (c : ClusterDraws Ω α K) :
    Draws Ω (FreeMonoid α) (Option (Fin K)) :=
  ((c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n),
      fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.1 j n)),
    fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.2.1 j n))),
    c.2.2.2.2)

/-- The run the clustering reads off `idealDraws`. -/
def idealRun (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (c : ClusterDraws Ω α K) :
    Run Ω (FreeMonoid α) (Option (Fin K)) :=
  streams (fun _ => ∅) none (idealDraws D Dsf h c)

/-- After the history `h`, the draws are drawn by `runMeasure` at the populations and suffixes
off what the earlier rounds read. -/
theorem measurePreserving_idealDraws {μ : Measure Ω} [IsProbabilityMeasure μ]
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (h : SpecHist α) :
    MeasurePreserving (idealDraws (Ω := Ω) (K := K) D Dsf h) (clusterLaw μ α K)
      (runMeasure μ (fun j => offLaw (loopDistributions D h.pops j) h.prefs)
        (offLaw Dsf h.suffs)) := by
  sorry -- closes by: main's proof (LoopLaw.lean:246) with hP from measurePreserving_streams over volume

lemma measurable_idealDraws (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (h : SpecHist α) : Measurable (idealDraws (Ω := Ω) (K := K) D Dsf h) := by
  rcases isEmpty_or_nonempty Ω with hΩ | hΩ
  · exact Subsingleton.measurable
  · exact (measurePreserving_idealDraws (μ := Measure.dirac hΩ.some) D Dsf h).measurable

lemma measurable_idealRun (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (h : SpecHist α) : Measurable (idealRun (Ω := Ω) (K := K) D Dsf h) :=
  (measurable_streams _ none).comp (measurable_idealDraws D Dsf h)


end OrthoDFA

end
