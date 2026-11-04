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

open scoped Classical in
/-- `findSome?` over `ofFn` hits at the first index whose value is `some`. -/
private lemma findSome?_ofFn_eq_some {β γ : Type*} (g : β → Option γ) (v : γ) :
    ∀ {M : ℕ} (ys : Fin M → β),
      (List.ofFn ys).findSome? g = some v ↔
        ∃ j : Fin M, (∀ i : Fin M, i < j → g (ys i) = none) ∧ g (ys j) = some v := by
  intro M
  induction M with
  | zero =>
    intro ys
    simp only [List.ofFn_zero, List.findSome?_nil, reduceCtorEq, false_iff, not_exists]
    rintro j; exact j.elim0
  | succ M ih =>
    intro ys
    rw [List.ofFn_succ, List.findSome?_cons]
    rcases hy0 : g (ys 0) with _ | w
    · simp only [hy0]
      rw [ih (fun i => ys i.succ)]
      constructor
      · rintro ⟨j', hmiss', hhit'⟩
        refine ⟨j'.succ, ?_, hhit'⟩
        intro i hi
        rcases Fin.eq_zero_or_eq_succ i with rfl | ⟨i', rfl⟩
        · exact hy0
        · exact hmiss' i' (Fin.succ_lt_succ_iff.1 hi)
      · rintro ⟨j, hmiss, hhit⟩
        rcases Fin.eq_zero_or_eq_succ j with rfl | ⟨j', rfl⟩
        · rw [hy0] at hhit; exact absurd hhit (by simp)
        · exact ⟨j', fun i' hi' => hmiss i'.succ (Fin.succ_lt_succ_iff.2 hi'), hhit⟩
    · simp only [hy0, Option.some.injEq]
      constructor
      · rintro rfl
        exact ⟨0, fun i hi => by simp [Fin.lt_def] at hi, hy0⟩
      · rintro ⟨j, hmiss, hhit⟩
        rcases Fin.eq_zero_or_eq_succ j with rfl | ⟨j', rfl⟩
        · rw [hy0] at hhit; exact Option.some_inj.1 hhit
        · exact absurd (hmiss 0 (Fin.succ_pos j')) (by rw [hy0]; simp)

/-- The sampler's output is a positive finite multiple of the harvest law.  Writing `p = D(miss)`
for the single-probe miss mass, the first-hit decomposition over the `N` i.i.d. probes gives, for
every `A`, `harvestCodeLaw A = (∑_{j<N} p^j) · harvestLaw A`, i.e. `harvestCodeLaw = c • harvestLaw`
with `c = ∑_{j<N} p^j`.  For `0 < N` the `j = 0` term makes `c ≥ 1 > 0`, and `c ≤ N < ∞`. -/
theorem harvestCodeLaw_smul [IsProbabilityMeasure D] (hN : 0 < N) :
    ∃ c : ℝ≥0∞, c ≠ 0 ∧ c ≠ ∞ ∧
      harvestCodeLaw read D k H site N = c • harvestLaw read D k H site := by
  classical
  set g : FreeMonoid α → Option (FreeMonoid α) :=
    fun x => (siteHarvest read k H site x).head? with hgdef
  set miss : Set (FreeMonoid α) := {x | g x = none} with hmissdef
  set p : ℝ≥0∞ := D miss with hpdef
  have hmiss_meas : MeasurableSet miss := (Set.to_countable _).measurableSet
  have hp1 : p ≤ 1 := by
    rw [hpdef]
    calc D miss ≤ D Set.univ := measure_mono (Set.subset_univ _)
      _ = 1 := measure_univ
  refine ⟨∑ j ∈ Finset.range N, p ^ j, ?_, ?_, ?_⟩
  · have h1 : (1 : ℝ≥0∞) ≤ ∑ j ∈ Finset.range N, p ^ j := by
      calc (1 : ℝ≥0∞) = p ^ 0 := (pow_zero p).symm
        _ ≤ ∑ j ∈ Finset.range N, p ^ j := by
            refine Finset.single_le_sum ?_ (Finset.mem_range.2 hN)
            intro i _; exact zero_le'
    intro hc; rw [hc] at h1; simp at h1
  · have hle : (∑ j ∈ Finset.range N, p ^ j) ≤ (N : ℝ≥0∞) := by
      calc ∑ j ∈ Finset.range N, p ^ j
          ≤ ∑ _j ∈ Finset.range N, 1 := Finset.sum_le_sum (fun j _ => pow_le_one' hp1 j)
        _ = (N : ℝ≥0∞) := by simp
    exact ne_top_of_le_ne_top (ENNReal.natCast_ne_top N) hle
  · refine Measure.ext fun A hA => ?_
    rw [Measure.smul_apply, smul_eq_mul]
    set hitA : Set (FreeMonoid α) := {x | ∃ v, g x = some v ∧ v ∈ A} with hhitAdef
    have hhitA_meas : MeasurableSet hitA := (Set.to_countable _).measurableSet
    have hlaw : harvestLaw read D k H site A = D hitA := by
      rw [harvestLaw_apply read D k H site A]
      have hint : (fun x => (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ≥0∞))
          = fun x => hitA.indicator (fun _ => (1 : ℝ≥0∞)) x := by
        funext x
        have hlen := siteHarvest_length read k H site x
        rcases hx : siteHarvest read k H site x with _ | ⟨s, tl⟩
        · have hgn : g x = none := by simp [hgdef, hx]
          have hxnot : x ∉ hitA := by rintro ⟨v, hv, -⟩; rw [hgn] at hv; simp at hv
          rw [Set.indicator_of_notMem hxnot]; simp [hx]
        · have htl : tl = [] := by
            rcases tl with _ | ⟨b, l⟩
            · rfl
            · rw [hx] at hlen; simp at hlen
          subst htl
          have hgx : g x = some s := by simp [hgdef, hx]
          by_cases hs : s ∈ A
          · have hmem : x ∈ hitA := ⟨s, hgx, hs⟩
            rw [Set.indicator_of_mem hmem (fun _ => (1 : ℝ≥0∞))]
            simp [hx, hs]
          · have hxnot : x ∉ hitA := by
              rintro ⟨v, hv, hvA⟩
              rw [hgx] at hv
              exact hs (by rw [Option.some_inj.1 hv]; exact hvA)
            rw [Set.indicator_of_notMem hxnot]; simp [hx, hs]
      rw [hint, lintegral_indicator_const hhitA_meas, one_mul]
    have hfmeas : Measurable fun ys : Fin N → FreeMonoid α =>
        (harvestFirst read k H site (List.ofFn ys)).getD 1 := measurable_of_countable _
    unfold harvestCodeLaw
    rw [Measure.map_apply hfmeas hA, Measure.restrict_apply (hfmeas hA)]
    set sset : Fin N → Fin N → Set (FreeMonoid α) :=
      fun j i => if i < j then miss else if i = j then hitA else Set.univ with hssetdef
    set E : Fin N → Set (Fin N → FreeMonoid α) := fun j => Set.univ.pi (sset j) with hEdef
    have hS : ((fun ys : Fin N → FreeMonoid α =>
          (harvestFirst read k H site (List.ofFn ys)).getD 1) ⁻¹' A)
          ∩ {ys | (harvestFirst read k H site (List.ofFn ys)).isSome = true}
        = ⋃ j, E j := by
      ext ys
      simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_setOf_eq, Set.mem_iUnion]
      constructor
      · rintro ⟨hAmem, hsome⟩
        rcases ho : harvestFirst read k H site (List.ofFn ys) with _ | v
        · rw [ho] at hsome; simp at hsome
        · rw [ho, Option.getD_some] at hAmem
          have hfs : (List.ofFn ys).findSome? g = some v := ho
          obtain ⟨j, hmiss', hhit'⟩ := (findSome?_ofFn_eq_some g v ys).1 hfs
          refine ⟨j, ?_⟩
          simp only [hEdef, Set.mem_univ_pi, hssetdef]
          intro i
          rcases lt_trichotomy i j with h | h | h
          · rw [if_pos h]; exact hmiss' i h
          · rw [if_neg (by rw [h]; exact lt_irrefl j), if_pos h]
            exact ⟨v, by rw [h]; exact hhit', hAmem⟩
          · rw [if_neg (not_lt.2 h.le), if_neg h.ne']; exact Set.mem_univ _
      · rintro ⟨j, hj⟩
        simp only [hEdef, Set.mem_univ_pi, hssetdef] at hj
        have hhitj : ys j ∈ hitA := by
          have := hj j; rw [if_neg (lt_irrefl j), if_pos rfl] at this; exact this
        obtain ⟨v, hgv, hvA⟩ := hhitj
        have hmissbefore : ∀ i, i < j → g (ys i) = none := by
          intro i hi; have := hj i; rw [if_pos hi] at this; exact this
        have hfs : (List.ofFn ys).findSome? g = some v :=
          (findSome?_ofFn_eq_some g v ys).2 ⟨j, hmissbefore, hgv⟩
        have hhf : harvestFirst read k H site (List.ofFn ys) = some v := hfs
        refine ⟨?_, ?_⟩
        · show (harvestFirst read k H site (List.ofFn ys)).getD 1 ∈ A
          rw [hhf, Option.getD_some]; exact hvA
        · show (harvestFirst read k H site (List.ofFn ys)).isSome = true
          rw [hhf]; rfl
    rw [hS]
    have hEmeas : ∀ j, MeasurableSet (E j) := by
      intro j
      simp only [hEdef]
      apply MeasurableSet.univ_pi
      intro i
      simp only [hssetdef]
      split_ifs
      · exact hmiss_meas
      · exact hhitA_meas
      · exact MeasurableSet.univ
    have hdisj : Pairwise (Function.onFun Disjoint E) := by
      intro a b hab
      simp only [Function.onFun, hEdef, Set.disjoint_left, Set.mem_univ_pi]
      intro ys hya hyb
      simp only [hssetdef] at hya hyb
      rcases lt_trichotomy a b with h | h | h
      · have h1 := hya a; rw [if_neg (lt_irrefl a), if_pos rfl] at h1
        have h2 := hyb a; rw [if_pos h] at h2
        obtain ⟨v, hgv, -⟩ := h1
        have h2' : g (ys a) = none := h2
        rw [h2'] at hgv; simp at hgv
      · exact hab h
      · have h1 := hyb b; rw [if_neg (lt_irrefl b), if_pos rfl] at h1
        have h2 := hya b; rw [if_pos h] at h2
        obtain ⟨v, hgv, -⟩ := h1
        have h2' : g (ys b) = none := h2
        rw [h2'] at hgv; simp at hgv
    rw [measure_iUnion hdisj hEmeas, tsum_fintype]
    have hEjmeasure : ∀ j : Fin N,
        (Measure.pi fun _ : Fin N => D) (E j) = p ^ (j : ℕ) * D hitA := by
      intro j
      simp only [hEdef]
      have hval : ∀ i : Fin N,
          D (sset j i) = (if i < j then p else 1) * (if i = j then D hitA else 1) := by
        intro i
        simp only [hssetdef]
        rcases lt_trichotomy i j with h | h | h
        · rw [if_pos h, if_pos h, if_neg (ne_of_lt h), mul_one, ← hpdef]
        · rw [if_neg (show ¬ i < j by rw [h]; exact lt_irrefl j), if_pos h,
            if_neg (show ¬ i < j by rw [h]; exact lt_irrefl j), if_pos h, one_mul]
        · rw [if_neg (not_lt.2 h.le), if_neg h.ne', if_neg (not_lt.2 h.le), if_neg h.ne',
            measure_univ, mul_one]
      calc (Measure.pi fun _ : Fin N => D) (Set.univ.pi (sset j))
          = ∏ i, D (sset j i) := Measure.pi_pi _ _
        _ = ∏ i, ((if i < j then p else 1) * (if i = j then D hitA else (1 : ℝ≥0∞))) :=
            Finset.prod_congr rfl (fun i _ => hval i)
        _ = (∏ i, if i < j then p else 1) * (∏ i, if i = j then D hitA else (1 : ℝ≥0∞)) :=
            Finset.prod_mul_distrib
        _ = p ^ (j : ℕ) * D hitA := by
            rw [Finset.prod_ite_eq']
            simp only [Finset.mem_univ, if_true]
            congr 1
            rw [← Finset.prod_filter, Finset.prod_const]
            congr 1
            have hfilt : (Finset.univ.filter fun i : Fin N => i < j) = Finset.Iio j := by
              ext i; simp [Finset.mem_Iio]
            rw [hfilt, Fin.card_Iio]
    simp_rw [hEjmeasure]
    rw [← Finset.sum_mul, Fin.sum_univ_eq_sum_range (fun j => p ^ j) N, hlaw]

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

open MeasureTheory in
/-- The operational population equals the handed-on one in law: the bounded-rejection sampler at
any cap `N ≥ 1` reproduces `popLaw (harvestLaw …)` exactly (via `harvestCodeLaw_popLaw_eq`), so the
loop can be rewired to `populationOp` without changing any population it hands on. -/
theorem LoopRound.populationOp_eq_population {α : Type*} [Fintype α]
    {Ω : Type*} [MeasurableSpace Ω] {K N : ℕ}
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hN : 0 < N) (k : ℕ)
    (o : LoopRound Ω α K) :
    o.populationOp D k N = o.population D k := by
  unfold LoopRound.populationOp LoopRound.population
  rcases o.result.ending.harvested with _ | site
  · rfl
  · simp only [Option.map_some]
    rw [Random.harvestCodeLaw_popLaw_eq (read := o.read) (D := D) (k := k)
      (H := o.result.hypothesis) (site := site) (N := N) hN]

end OrthoDFA
