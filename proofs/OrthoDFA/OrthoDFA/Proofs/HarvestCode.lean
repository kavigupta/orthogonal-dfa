import OrthoDFA.Code.Level3HarvestCode
import OrthoDFA.Code.LearnerLoop
import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.KappaReads
import OrthoDFA.Proofs.RandomSearch

/-! # Decoupling the harvest sampler from the harvest law

The three facts that make `harvestCode` a drop-in for `harvestLaw` in the loop:

* `harvestCodeLaw_popLaw_eq` — (A1) conditioned on a hit, its output law is exactly
  `popLaw (harvestLaw …)`.  No `δ`, no good-event hypothesis: rejection sampling reproduces the
  conditional law exactly.
* `harvestCodeLaw_miss_le` — (A2) the all-miss mass is `(1 - hitProb) ^ N`; with the round's yield
  bound `y ≤ hitProb` and `N = harvestN y δₛ`, it is `≤ δₛ`.
* `harvestCode_reads_card_le` — (B) the sampler reads only what its (at most `|xs|`) walked probes
  read, `≤ |xs|` probe-walks each of `≤ 2 + clog₂ L` reads. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Level3Site) (N : ℕ) (xs : List (FreeMonoid α))

open scoped Classical in
/-- The strings the bounded-rejection sampler reads: the union of `probeReads` over the probes it
walks (all of `xs` in the worst case). -/
noncomputable def harvestFirstReads (read : FreeMonoid α → ARU) (T : DTree α)
    (E : Edges α) (k : ℕ) (xs : List (FreeMonoid α)) : Finset (FreeMonoid α) :=
  xs.foldr (fun x acc => Kappa.probeReads read T E k x ∪ acc) ∅

/-- `popLaw` ignores a positive finite scaling: rejection sampling's output is a `c`-multiple of the
harvest law (`c` the geometric first-hit factor), and normalisation divides `c` back out. -/
lemma popLaw_smul {ν : Measure (FreeMonoid α)} {c : ℝ≥0∞} (hc0 : c ≠ 0) (hctop : c ≠ ⊤) :
    popLaw (c • ν) = popLaw ν := by
  have huniv : (c • ν) Set.univ = c * ν Set.univ := by
    simp [Measure.smul_apply, smul_eq_mul]
  unfold popLaw
  by_cases h0 : ν Set.univ = 0
  · rw [if_pos (Or.inl h0), if_pos (Or.inl (by rw [huniv, h0, mul_zero]))]
  by_cases htop : ν Set.univ = ∞
  · rw [if_pos (Or.inr htop), if_pos (Or.inr (by rw [huniv, htop, ENNReal.mul_top hc0]))]
  · have hcz : c * ν Set.univ ≠ 0 := mul_ne_zero hc0 h0
    have hct : c * ν Set.univ ≠ ∞ := ENNReal.mul_ne_top hctop htop
    rw [if_neg (by rw [huniv]; exact not_or.2 ⟨hcz, hct⟩), if_neg (not_or.2 ⟨h0, htop⟩)]
    ext A
    simp only [Measure.smul_apply, smul_eq_mul, huniv]
    rw [ENNReal.mul_inv (Or.inl hc0) (Or.inl hctop)]
    rw [show c⁻¹ * (ν Set.univ)⁻¹ * (c * ν A)
        = (c⁻¹ * c) * ((ν Set.univ)⁻¹ * ν A) from by ring]
    rw [ENNReal.inv_mul_cancel hc0 hctop, one_mul]

/-- The sampler's output is a positive finite multiple of the harvest law.  Writing `p = D(miss)`
for the single-probe miss mass, the first-hit decomposition over the `N` i.i.d. probes gives, for
every `A`, `harvestCodeLaw A = (∑_{j<N} p^j) · harvestLaw A`, i.e. `harvestCodeLaw = c • harvestLaw`
with `c = ∑_{j<N} p^j`.  For `0 < N` the `j = 0` term makes `c ≥ 1 > 0`, and `c ≤ N < ∞`. -/
theorem harvestCodeLaw_smul [IsProbabilityMeasure D] (hN : 0 < N) :
    ∃ c : ℝ≥0∞, c ≠ 0 ∧ c ≠ ∞ ∧
      harvestCodeLaw read D k H site N = c • harvestLaw read D k H site := by
  sorry

/-- (A1) Agreement — exact.  Conditioned on a hit, the sampler's output law is the harvest law
(both normalised by `popLaw`), for every cap `N ≥ 1`. -/
theorem harvestCodeLaw_popLaw_eq [IsProbabilityMeasure D] (hN : 0 < N) :
    popLaw (harvestCodeLaw read D k H site N) = popLaw (harvestLaw read D k H site) := by
  obtain ⟨c, hc0, hctop, hpr⟩ := harvestCodeLaw_smul read D k H site N hN
  rw [hpr, popLaw_smul hc0 hctop]

/-- (A2) Failure mass.  The chance all `N` probes miss is `(1 - hitProb) ^ N`; with the (round-good)
yield bound `y ≤ hitProb = (harvestLaw …).real univ` and `N = harvestN y δₛ`, it is `≤ δₛ`. -/
theorem harvestCodeLaw_miss_le [IsProbabilityMeasure D] {y δₛ : ℝ}
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ)
    (hyield : y ≤ (harvestLaw read D k H site).real Set.univ) :
    harvestMiss read D k H site (harvestN y δₛ) ≤ ENNReal.ofReal δₛ := by
  classical
  set N := harvestN y δₛ with hNdef
  set miss : Set (FreeMonoid α) := {x | (siteHarvest read k H site x).head? = none} with hmissdef
  have hmeas : MeasurableSet miss := (Set.to_countable _).measurableSet
  -- (harvestLaw …).real univ equals the single-probe hit mass D(missᶜ).
  have hhit : (harvestLaw read D k H site).real Set.univ = D.real missᶜ := by
    rw [harvestLaw_real_univ]
    have hfun : (fun x => ((siteHarvest read k H site x).length : ℝ))
        = fun x => missᶜ.indicator (fun _ => (1 : ℝ)) x := by
      funext x
      have hlen := siteHarvest_length read k H site x
      rcases hx : siteHarvest read k H site x with _ | ⟨a, tl⟩
      · simp [hx, hmissdef, Set.indicator, List.head?]
      · have htl : tl = [] := by
          rcases tl with _ | ⟨b, l⟩
          · rfl
          · rw [hx] at hlen; simp at hlen
        subst htl
        simp [hx, hmissdef, Set.indicator, List.head?]
    rw [hfun, MeasureTheory.integral_indicator_const (1 : ℝ) hmeas.compl]
    simp
  -- D(miss) ≤ 1 - y as reals.
  have hmissreal : D.real miss ≤ 1 - y := by
    have huniv1 : D.real Set.univ = 1 := by simp [measureReal_def]
    have hcompl : D.real missᶜ = 1 - D.real miss := by
      rw [measureReal_compl hmeas, huniv1]
    rw [hhit.symm] at hcompl
    linarith [hyield]
  -- The real inequality (1 - y) ^ N ≤ δₛ.
  have h1y : (0 : ℝ) ≤ 1 - y := by linarith
  have hyN : Real.log (1 / δₛ) ≤ y * (N : ℝ) := by
    have h1 : Real.log (1 / δₛ) / y ≤ (N : ℝ) := by rw [hNdef]; exact Nat.le_ceil _
    calc Real.log (1 / δₛ) = Real.log (1 / δₛ) / y * y := by field_simp
      _ ≤ (N : ℝ) * y := by exact mul_le_mul_of_nonneg_right h1 hy0.le
      _ = y * (N : ℝ) := by ring
  have hpow : (1 - y) ^ N ≤ δₛ := by
    have hexpstep : (1 - y) ^ N ≤ Real.exp (-y) ^ N :=
      pow_le_pow_left₀ h1y (by linarith [Real.add_one_le_exp (-y)]) N
    have hlogδ : Real.log (1 / δₛ) = - Real.log δₛ := by rw [one_div, Real.log_inv]
    calc (1 - y) ^ N ≤ Real.exp (-y) ^ N := hexpstep
      _ = Real.exp (-(y * (N : ℝ))) := by rw [← Real.exp_nat_mul]; ring_nf
      _ ≤ Real.exp (Real.log δₛ) := by
          apply Real.exp_le_exp.2; rw [hlogδ] at hyN; linarith
      _ = δₛ := Real.exp_log hδ
  -- harvestMiss = D(miss) ^ N as the product measure of a pi-set.
  have hset : {xs : Fin N → FreeMonoid α | harvestFirst read k H site (List.ofFn xs) = none}
      = Set.univ.pi (fun _ => miss) := by
    ext xs
    simp only [Set.mem_setOf_eq, Set.mem_univ_pi, hmissdef]
    rw [harvestFirst, List.findSome?_eq_none_iff]
    constructor
    · intro h i; exact h (xs i) (List.mem_ofFn.2 ⟨i, rfl⟩)
    · intro h x hx; obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact h i
  have hmisseq : harvestMiss read D k H site N = (D miss) ^ N := by
    rw [harvestMiss, hset, Measure.pi_pi, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [hmisseq]
  -- Push the real bound through to ENNReal.
  have hDmiss_top : D miss ≠ ∞ := measure_ne_top D _
  have hDmiss_le : D miss ≤ ENNReal.ofReal (1 - y) := by
    rw [← ENNReal.ofReal_toReal hDmiss_top]
    exact ENNReal.ofReal_le_ofReal hmissreal
  calc (D miss) ^ N ≤ (ENNReal.ofReal (1 - y)) ^ N := pow_le_pow_left' hDmiss_le N
    _ = ENNReal.ofReal ((1 - y) ^ N) := (ENNReal.ofReal_pow h1y N).symm
    _ ≤ ENNReal.ofReal δₛ := ENNReal.ofReal_le_ofReal hpow

/-- (B) Cost.  The sampler reads only the strings its walked probes read; when every probe is at
most `L` long each probe-walk sifts `≤ 3 + clog₂ L` positions, each reading `≤ |mids T|` strings, so
the total is `≤ |xs| · ((3 + clog₂ L) · |mids T|)`.  (At the loop's `read = familyRead … F`, each
such read is `|F|` raw oracle queries; that factor is applied where the sampler is wired in.) -/
theorem harvestCode_reads_card_le {L : ℕ} (hxs : ∀ x ∈ xs, x.toList.length ≤ L) :
    (harvestFirstReads read H.tree H.edges k xs).card
      ≤ xs.length * ((3 + Nat.clog 2 L) * (mids H.tree).card) := by
  classical
  revert hxs
  induction xs with
  | nil => intro _; simp [harvestFirstReads]
  | cons x rest ih =>
    intro hxs
    have hx : x.toList.length ≤ L := hxs x (List.mem_cons.mpr (Or.inl rfl))
    have hper : (Kappa.probeReads read H.tree H.edges k x).card
        ≤ (3 + Nat.clog 2 L) * (mids H.tree).card := by
      obtain ⟨ps, hps, hsub⟩ := Kappa.probeReads_sub (rd := read) (T := H.tree) H.edges k hx
      calc (Kappa.probeReads read H.tree H.edges k x).card
          ≤ (Kappa.posReads read H.tree x ps).card := Finset.card_le_card hsub
        _ ≤ ps.toFinset.card * (mids H.tree).card := Kappa.posReads_card _ _ _ _
        _ ≤ (3 + Nat.clog 2 L) * (mids H.tree).card := Nat.mul_le_mul_right _ hps
    have hrest := ih (fun y hy => hxs y (List.mem_cons.mpr (Or.inr hy)))
    have hfold : harvestFirstReads read H.tree H.edges k (x :: rest)
        = Kappa.probeReads read H.tree H.edges k x ∪ harvestFirstReads read H.tree H.edges k rest :=
      rfl
    rw [hfold, List.length_cons]
    calc (Kappa.probeReads read H.tree H.edges k x
            ∪ harvestFirstReads read H.tree H.edges k rest).card
        ≤ (Kappa.probeReads read H.tree H.edges k x).card
            + (harvestFirstReads read H.tree H.edges k rest).card := Finset.card_union_le _ _
      _ ≤ (3 + Nat.clog 2 L) * (mids H.tree).card
            + rest.length * ((3 + Nat.clog 2 L) * (mids H.tree).card) := Nat.add_le_add hper hrest
      _ = (rest.length + 1) * ((3 + Nat.clog 2 L) * (mids H.tree).card) := by ring

end Random

end OrthoDFA
