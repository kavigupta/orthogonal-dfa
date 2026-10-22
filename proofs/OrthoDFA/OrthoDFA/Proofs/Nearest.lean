import OrthoDFA.Proofs.Adaptive

/-!
# The count rule is a `Clusterer`

`clusterAt`, the clustering the claim is stated for, is `nearestClusterer` read off the oracle's
answers, so what the proof shows of `clusterBy nearestClusterer` it shows of the claim's family.
-/

namespace OrthoDFA

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S]

open scoped Classical in
lemma screenCount_eq_readLoss (mq : S → Ω → ℝ) (P : Finset S) (ω : Ω) :
    (fun v => (screenCount mq P v ω : ℝ)) = readLoss (fun w => mq w ω = 1) P := by
  funext v
  unfold screenCount readLoss
  congr 2

lemma clusterAt_eq {J : Type*} (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) :
    clusterAt mq populations x B = clusterBy nearestClusterer mq populations x B := by
  rw [clusterAt, clusterBy, nearestRows, screenCount_eq_readLoss]
  rfl

lemma familyAt_eq {J : Type*} (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) :
    familyAt mq populations x B = familyBy nearestClusterer mq populations x B := by
  rw [familyAt, familyBy, clusterAt_eq]

lemma ret_eq {J : Type*} (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) :
    ret (Ω := Ω) mq populations uni η₀ indecisionLimit εcov α B
      = retBy nearestClusterer mq populations uni η₀ indecisionLimit εcov α B := by
  have h : clusterAt (Ω := Ω) mq populations
      = fun x B => clusterBy nearestClusterer mq populations x B := by
    funext x B
    exact clusterAt_eq mq populations x B
  have h' : familyAt (Ω := Ω) mq populations
      = fun x B => familyBy nearestClusterer mq populations x B := by
    funext x B
    exact familyAt_eq mq populations x B
  have h'' : refsAt (Ω := Ω) mq populations
      = fun x B => refsBy nearestClusterer mq populations x B := by
    funext x B
    unfold refsAt refsBy
    rw [familyAt_eq]
  unfold ret
  rw [h, h', h'']
  rfl

end OrthoDFA
