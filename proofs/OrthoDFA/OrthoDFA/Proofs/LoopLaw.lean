import OrthoDFA.Proofs.LoopRounds
import OrthoDFA.Proofs.DrawFrom
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.RawRun
import Mathlib.Probability.ProductMeasure
import Mathlib.MeasureTheory.Constructions.UnitInterval
import Mathlib.MeasureTheory.Function.Floor
import Mathlib.Data.List.GetD

/-!
# A round's clustering run, at given populations, is drawn as the clustering's theorem draws it

A uniform `u` lands where `ν`'s running mass over the strings' numbering first passes it, at a
string with chance its mass, so a draw is from `ν` (`map_drawFrom`). The round's draws, run at
given populations, are then drawn by `runMeasure` at those populations, the hash's sides
included (`measurePreserving_clusterDraws`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

lemma isProbabilityMeasure_loopDistributions {K : ℕ} (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (pops : List (Measure (FreeMonoid α)))
    (hpops : ∀ ν ∈ pops, IsProbabilityMeasure ν) (j : Option (Fin K)) :
    IsProbabilityMeasure (loopDistributions D pops j) := by
  cases j with
  | none => simp only [loopDistributions]; infer_instance
  | some i =>
    simp only [loopDistributions]
    by_cases hi : i.val < pops.length
    · rw [List.getD_eq_getElem (hn := hi)]
      exact hpops _ (List.getElem_mem _)
    · rw [List.getD_eq_default _ _ (not_lt.1 hi)]
      infer_instance

section Run

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- A stream of uniforms, each drawing from `ν`, is a stream of draws from `ν`. -/
lemma measurePreserving_drawStream (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] :
    MeasurePreserving (fun u : ℕ → unitInterval => fun n => drawFrom ν (u n))
      (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.infinitePi fun _ : ℕ => ν) :=
  ⟨measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp (measurable_pi_apply n), by
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => (volume : Measure unitInterval))
      (hf := fun _ => measurable_drawFrom _), map_drawFrom]⟩

/-- Each population's stream of uniforms drawing from its law. -/
lemma measurePreserving_popDraws (L : Option (Fin K) → Measure (FreeMonoid α))
    [∀ j, IsProbabilityMeasure (L j)] :
    MeasurePreserving (fun U : Option (Fin K) → ℕ → unitInterval => fun j n => drawFrom (L j) (U j n))
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.pi fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => L j) := by
  refine ⟨measurable_pi_lambda _ fun j => (measurePreserving_drawStream (L j)).measurable.comp
    (measurable_pi_apply j), ?_⟩
  rw [Measure.pi_map_pi (fun j => (measurePreserving_drawStream (L j)).measurable.aemeasurable)]
  congr 1
  funext j
  exact (measurePreserving_drawStream (L j)).map_eq

/-- After the history `h`, a round's clustering draws are drawn by `runMeasure` at the
populations and suffixes conditioned off what the earlier rounds read. -/
lemma measurePreserving_clusterDraws (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) :
    MeasurePreserving (clusterDraws (Ω := Ω) (K := K) D Dsf h) (clusterLaw μ α K)
      (runMeasure μ (fun j => offLaw (loopDistributions D h.pops j) h.prefs)
        (offLaw Dsf h.suffs)) := by
  set L : Option (Fin K) → Measure (FreeMonoid α) :=
    fun j => offLaw (loopDistributions D h.pops j) h.prefs
  have hS := measurePreserving_drawStream (offLaw Dsf h.suffs)
  have hP := measurePreserving_popDraws L
  set US := Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  set UP := Measure.pi fun _ : Option (Fin K) =>
    Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  set C := Measure.infinitePi fun _ : FreeMonoid α => fairCoin
  have e1 := (MeasurePreserving.id US).prod (measurePreserving_prodAssoc UP UP C).symm
  have e2 := (measurePreserving_prodAssoc US (UP.prod UP) C).symm
  have e3 := (measurePreserving_prodAssoc US UP UP).symm.prod (MeasurePreserving.id C)
  have hin := (((hS.prod hP).prod hP).prod (MeasurePreserving.id C)).comp (e3.comp (e2.comp e1))
  have hall := ((MeasurePreserving.id μ).prod hin)
  exact (measurePreserving_prodAssoc μ (((Measure.infinitePi fun _ : ℕ => offLaw Dsf h.suffs).prod
      (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => L j)).prod
      (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => L j)) C).symm.comp hall

omit [IsProbabilityMeasure μ] in
lemma measurable_clusterDraws (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) :
    Measurable (clusterDraws (Ω := Ω) (K := K) D Dsf h) := by
  unfold clusterDraws
  refine (measurable_fst.prodMk (Measurable.prodMk (Measurable.prodMk ?_ ?_) ?_)).prodMk ?_
  · exact measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp
      ((measurable_pi_apply n).comp (measurable_fst.comp measurable_snd))
  · exact measurable_pi_lambda _ fun j => measurable_pi_lambda _ fun n =>
      (measurable_drawFrom _).comp ((measurable_pi_apply n).comp ((measurable_pi_apply j).comp
        (measurable_fst.comp (measurable_snd.comp measurable_snd))))
  · exact measurable_pi_lambda _ fun j => measurable_pi_lambda _ fun n =>
      (measurable_drawFrom _).comp ((measurable_pi_apply n).comp ((measurable_pi_apply j).comp
        (measurable_fst.comp (measurable_snd.comp (measurable_snd.comp measurable_snd)))))
  · exact measurable_snd.comp (measurable_snd.comp (measurable_snd.comp measurable_snd))

omit [IsProbabilityMeasure μ] in
lemma measurable_clusterRun (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) :
    Measurable (clusterRun (Ω := Ω) (K := K) D Dsf h) :=
  (measurable_streams _ none).comp (measurable_clusterDraws D Dsf h)

end Run

end OrthoDFA
