import OrthoDFA.Proofs.RawEvents

/-!
# The loop returns, whatever the named populations are

At the top rung every population's FNR test and gate pass, except with chance `δ/2`.  The
uniform pool's tests are priced at each assignment of the hash by the stream-level lemmas.  A
named population's are priced against its own table: the hash splits a window of its draws
evenly, so the family's members flip on its certification sample about as often as on its
table, which the screen holds them to.  Nothing is asked of a named population's law.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]
variable {η₀ indecisionLimit : ℝ}

/-- Population `j`'s window: its first `K` distinct draws outside the harvested strings. -/
noncomputable def winAt (H : J → Finset S) (K : ℕ) (j : J) (r : Run Ω S J) : Finset S :=
  window (↑(allHarvest H))ᶜ K (r.2.1.2 j)

open scoped Classical in
/-- The hash splits some pool member's flips on the window unevenly. -/
noncomputable def coinSplitBad (O : Oracle μ S) (H : J → Finset S) (K M : ℕ) (L : ℝ) (j : J) :
    Set (Draws Ω S J) :=
  {x | ∃ v ∈ poolAt M x.1,
    Real.sqrt (2 * (((winAt H K j x.1).filter (fun p => O.flip v p = 1)).card : ℝ) * L)
      < ((((winAt H K j x.1).filter (fun p => O.flip v p = 1)).filter
          (fun s => x.2 s = false)).card : ℝ)
        - ((((winAt H K j x.1).filter (fun p => O.flip v p = 1)).filter
          (fun s => x.2 s = true)).card : ℝ)}

open scoped Classical in
/-- The hash puts `G` more than half of the window on a side. -/
noncomputable def coinCardBad (H : J → Finset S) (K G : ℕ) (j : J) : Set (Draws Ω S J) :=
  {x | ((winAt H K j x.1).card : ℝ) / 2 + G
        ≤ (((winAt H K j x.1).filter (fun s => x.2 s = true)).card : ℝ)
    ∨ ((winAt H K j x.1).card : ℝ) / 2 + G
        ≤ (((winAt H K j x.1).filter (fun s => x.2 s = false)).card : ℝ)}

/-- The uniform pool's first `n` draws land `a` times in population `j`'s window. -/
def hitsBad (H : J → Finset S) (K n : ℕ) (a : ℝ) (uni j : J) : Set (Run Ω S J) :=
  {r | a ≤ ((winAt H K j r ∩ acceptedAt ∅ (r.2.1.2 uni) n).card : ℝ)}

section Streams

variable (H : J → Finset S) (uni : J)

lemma prefixDraw_streams_uni (x : Draws Ω S J) :
    (fun i => prefixDraw uni i (streams H uni x)) = x.1.2.1.2 uni := by
  funext i
  simp [prefixDraw, streams]

lemma prefixDraw_streams_named {j : J} (hju : j ≠ uni) (x : Draws Ω S J) :
    (fun i => prefixDraw j i (streams H uni x))
      = tableSeqOf (H j).toList (onSide (allHarvest H) true x.2) (x.1.2.1.2 j) (x.1.2.2 j) := by
  funext i
  simp [prefixDraw, streams, hju]

lemma certPrefix_streams_named {j : J} (hju : j ≠ uni) (x : Draws Ω S J) :
    (fun i => certPrefix j i (streams H uni x))
      = reserveSeqOf (onSide (allHarvest H) false x.2) (x.1.2.1.2 j) (x.1.2.2 j) := by
  funext i
  simp [certPrefix, streams, hju]

lemma poolAt_streams (M : ℕ) (x : Draws Ω S J) : poolAt M (streams H uni x) = poolAt M x.1 :=
  rfl

end Streams

lemma harvest_subset (H : J → Finset S) (j : J) : H j ⊆ allHarvest H :=
  Finset.subset_biUnion_of_mem H (Finset.mem_univ j)

open scoped Classical in
set_option maxHeartbeats 2000000 in
/-- A named population the round keeps passes its FNR test and gate at the top rung unless the
family stalls or has no clean reference, the screen or the hash misbehaves, the uniform pool
lands in the window too often, a light count misses, a side is dead or the pool is short. -/
theorem named_retAtK (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (H : J → Finset S) (j : J) (hj : j ∈ populations)
    (hju : j ≠ uni) (B : State) (x : Draws Ω S J) (hN2 : 2 ≤ B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lcut f Δ t L θ : ℝ) (r rs kmin kmax G : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hl2 : 2 * l ≤ indecisionLimit) (hf0 : 0 < f) (hΔ : 0 < Δ) (hL : 0 ≤ L) (hθ : 0 ≤ θ)
    (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 0 < B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + 5 / 8 * lcut) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hszG : (allHarvest H).card + G ≤ B.npref)
    (hheavyN : ((populations.card : ℝ) * Δ * B.npref
          + 2 * Real.sqrt (L * ((populations.card : ℝ) * Δ * B.npref)) + 2 * L) / f
        + (allHarvest H).card + 2 * G + θ * B.npref ≤ lcut * B.npref)
    (h1 : streams H uni x ∉ stalled rule O populations B kmin kmax)
    (h2 : streams H uni x ∉ {y : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff y).erase 1,
      ∀ p ∈ prefixesAt populations B.npref y, O.flip w₀ p = 0})
    (h3 : streams H uni x ∉ screenBad O populations B Δ)
    (h4 : x ∉ coinSplitBad O H (2 * (B.npref - (H j).card - G)) B.nsuff L j)
    (h5 : x ∉ coinCardBad H (2 * (B.npref - (H j).card - G)) G j)
    (h6 : tabGood uni B.npref r (streams H uni x))
    (h7 : x.1 ∉ hitsBad H (2 * (B.npref - (H j).card - G)) (B.npref + r) (θ * B.npref) uni j)
    (h8 : streams H uni x ∉ lightMiss rule O populations j B l lcut (5 / 8 * lcut) f kmin kmax)
    (h10 : streams H uni x ∉ sidesDead rule O populations j B B.npref t)
    (h11 : poolGood B.nsuff rs (streams H uni x)) :
    streams H uni x ∈ retAtK rule O populations uni η₀ indecisionLimit εcov α B j := by
  classical
  set y := streams H uni x with hy
  set N : ℕ := B.npref with hN
  set K : ℕ := 2 * (N - (H j).card - G) with hK
  set Hs : Finset S := allHarvest H with hHs
  set g : S → Bool := x.2 with hg
  set w : ℕ → S := x.1.2.1.2 j with hw
  set z : ℕ → S := x.1.2.2 j with hz
  set Z : Finset S := winAt H K j x.1 with hZ
  set F : Finset S := clusterBy rule O.mq populations y B with hF
  set P : Finset S := prefixesAt populations N y with hP
  set C : Finset S := certOf P j N y with hC
  have hsize : kmin ≤ F.card ∧ F.card ≤ kmax := by
    by_contra h; exact h1 h
  refine ⟨?_, fun hkept => ?_⟩
  · show B.k ≤ F.card + 1
    have := hsize.1
    omega
  obtain ⟨hT, hCn⟩ := hkept
  have hTeq : prefixesOf j N y
      = freshDraws ∅ (tableSeqOf (H j).toList (onSide Hs true g) w z) N := by
    show freshDraws ∅ (fun i => prefixDraw j i y) N = _
    rw [hy, prefixDraw_streams_named H uni hju x]
  have hCeq : C = freshDraws P (reserveSeqOf (onSide Hs false g) w z) N := by
    show freshDraws P (fun i => certPrefix j i y) N = _
    rw [hy, certPrefix_streams_named H uni hju x]
  have hTP : prefixesOf j N y ⊆ P := fun q hq => Finset.mem_biUnion.2 ⟨j, hj, hq⟩
  have hNP : N ≤ P.card := by
    have := Finset.card_le_card hTP
    rw [hT] at this
    exact this
  -- the screen
  have hclean : ∃ w₀ ∈ (poolAt B.nsuff y).erase 1, ∀ p ∈ P, O.flip w₀ p = 0 := by
    by_contra h; exact h2 h
  have hscr : ∀ v ∈ screenedAt O.mq populations B y,
      ¬ (Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p) := by
    by_contra h; exact h3 ⟨hNP, hclean, h⟩
  have hscreen : ∀ v ∈ F, ∑ p ∈ P, O.flip v p < Δ * P.card := fun v hv =>
    not_le.1 (clusterAt_flip_bound (rule := rule) O populations B y Δ hscr v hv)
  -- the window's sides
  have hZQ : ∀ s ∈ Z, s ∉ Hs := fun s hs => by
    have := mem_of_mem_rawIn hs
    simpa using this
  have hcard2 : ¬ (((Z.card : ℝ) / 2 + G ≤ ((Z.filter (fun s => g s = true)).card : ℝ))
      ∨ ((Z.card : ℝ) / 2 + G ≤ ((Z.filter (fun s => g s = false)).card : ℝ))) := h5
  push Not at hcard2
  have hZK : Z.card ≤ K := window_card_le _ K _
  have hZKR : (Z.card : ℝ) ≤ 2 * ((N : ℝ) - (H j).card - G) := by
    have hle : (H j).card + G ≤ N := le_trans (Nat.add_le_add_right
      (Finset.card_le_card (harvest_subset H j)) G) hszG
    have : (K : ℝ) = 2 * ((N : ℝ) - (H j).card - G) := by
      rw [hK]; push_cast [Nat.sub_sub, Nat.cast_sub hle]; ring
    rw [← this]; exact_mod_cast hZK
  have hZA : (Z.filter (fun s => g s = true)).card + (H j).toList.length ≤ N := by
    rw [Finset.length_toList]
    have : ((Z.filter (fun s => g s = true)).card : ℝ) + (H j).card < N + 1 := by
      linarith [hcard2.1]
    have h' : (Z.filter (fun s => g s = true)).card + (H j).card < N + 1 := by exact_mod_cast this
    omega
  have hZAP : Z.filter (fun s => g s = true) ⊆ P := by
    have hsub := window_table_subset (H j).toList (Finset.nodup_toList _) Hs
      (fun s hs => harvest_subset H j (Finset.mem_toList.1 hs)) g w z K N
      (by rw [← hTeq]; exact hT) hZA
    exact fun s hs => hTP (by rw [hTeq]; exact hsub hs)
  -- the members' flips
  have hbal : ∀ v ∈ F,
      (((Z.filter (fun p => O.flip v p = 1)).filter (fun s => g s = false)).card : ℝ)
        - (((Z.filter (fun p => O.flip v p = 1)).filter (fun s => g s = true)).card : ℝ)
      ≤ Real.sqrt (2 * ((Z.filter (fun p => O.flip v p = 1)).card : ℝ) * L) := by
    intro v hv
    have hvp : v ∈ poolAt B.nsuff x.1 := clusterAt_subset O populations B y hv
    by_contra hc
    exact h4 ⟨v, hvp, not_le.1 hc⟩
  have hCZ : ∀ p ∈ C, p ∈ Z → g p = false := by
    intro p hp _
    rw [hCeq] at hp
    obtain ⟨i, hi, rfl⟩ := mem_cert_of_two _ w z P hN2 hp
    exact hi.1
  -- what the sample holds beyond the window
  have hRP : ∀ p ∈ Z.filter (fun s => g s = false), p ∈ P → p ∈ prefixesOf uni N y := by
    intro p hpR hpP
    obtain ⟨j', hj', hpj'⟩ := Finset.mem_biUnion.1 hpP
    by_cases hj'u : j' = uni
    · rw [← hj'u]; exact hpj'
    · exfalso
      have hpj'' : p ∈ freshDraws ∅ (tableSeqOf (H j').toList (onSide Hs true g)
          (x.1.2.1.2 j') (x.1.2.2 j')) N := by
        have : prefixesOf j' N y = freshDraws ∅ (tableSeqOf (H j').toList (onSide Hs true g)
            (x.1.2.1.2 j') (x.1.2.2 j')) N := by
          show freshDraws ∅ (fun i => prefixDraw j' i y) N = _
          rw [hy, prefixDraw_streams_named H uni hj'u x]
        rw [← this]; exact hpj'
      obtain ⟨hpZ, hpg⟩ := Finset.mem_filter.1 hpR
      rcases mem_table_of_two _ _ _ _ hN2 hpj'' with hL' | hA'
      · exact hZQ p hpZ (harvest_subset H j' (Finset.mem_toList.1 hL'))
      · rw [hA'.1] at hpg; exact Bool.noConfusion hpg
  have hT_uni : prefixesOf uni N y ⊆ acceptedAt ∅ (x.1.2.1.2 uni) (N + r) := by
    have h := freshDraws_subset_acceptedAt ∅ (fun i => prefixDraw uni i y) h6
    rw [hy, prefixDraw_streams_uni H uni x] at h
    show freshDraws ∅ (fun i => prefixDraw uni i y) N ⊆ _
    rw [hy, prefixDraw_streams_uni H uni x]
    exact h
  have hhits : ((Z ∩ acceptedAt ∅ (x.1.2.1.2 uni) (N + r)).card : ℝ) < θ * N := by
    by_contra hc; exact h7 (not_lt.1 hc)
  have hEC : ((C \ Z).card : ℝ) ≤ (H j).card + 2 * G + θ * N := by
    by_cases hfull : Z.card = K
    · set ZR := Z.filter (fun s => g s = false) with hZR
      have hsub : C \ Z ⊆ C \ (ZR \ P) := Finset.sdiff_subset_sdiff le_rfl
        (fun s hs => (Finset.mem_filter.1 (Finset.mem_sdiff.1 hs).1).1)
      have hle : ((C \ (ZR \ P)).card : ℝ) ≤ max 0 ((N : ℝ) - ((ZR \ P).card : ℝ)) := by
        have := card_cert_sdiff_le Hs g w z K N P (by rw [← hCeq]; exact hCn)
        rw [← hCeq] at this
        exact this
      have hZsplit : ((Z.filter (fun s => g s = true)).card : ℝ) + ZR.card = Z.card := by
        have h := Finset.card_filter_add_card_filter_not (s := Z) (fun s => g s = true)
        have e : Z.filter (fun s => ¬ g s = true) = ZR :=
          Finset.filter_congr (fun s _ => by simp)
        rw [e] at h
        exact_mod_cast h
      have hRPc : ((ZR ∩ P).card : ℝ) ≤ ((Z ∩ acceptedAt ∅ (x.1.2.1.2 uni) (N + r)).card : ℝ) := by
        have hsub' : ZR ∩ P ⊆ Z ∩ acceptedAt ∅ (x.1.2.1.2 uni) (N + r) := by
          intro s hs
          obtain ⟨hsR, hsP⟩ := Finset.mem_inter.1 hs
          exact Finset.mem_inter.2 ⟨(Finset.mem_filter.1 hsR).1, hT_uni (hRP s hsR hsP)⟩
        exact_mod_cast Finset.card_le_card hsub'
      have hRsplit : ((ZR \ P).card : ℝ) + (ZR ∩ P).card = ZR.card := by
        exact_mod_cast Finset.card_sdiff_add_card_inter ZR P
      have hfullR : (Z.card : ℝ) = 2 * ((N : ℝ) - (H j).card - G) := by
        have hle' : (H j).card + G ≤ N := le_trans (Nat.add_le_add_right
          (Finset.card_le_card (harvest_subset H j)) G) hszG
        rw [hfull, hK]; push_cast [Nat.sub_sub, Nat.cast_sub hle']; ring
      have hmax : max 0 ((N : ℝ) - ((ZR \ P).card : ℝ)) ≤ (H j).card + 2 * G + θ * N := by
        refine max_le (by have := mul_nonneg hθ (Nat.cast_nonneg (α := ℝ) N); positivity) ?_
        linarith [hcard2.1]
      calc ((C \ Z).card : ℝ) ≤ ((C \ (ZR \ P)).card : ℝ) := by
            exact_mod_cast Finset.card_le_card hsub
        _ ≤ _ := le_trans hle hmax
    · have hsub : C ⊆ Z := by
        rw [hCeq]; exact cert_subset_window Hs g w z K N hN2 P hfull
      rw [Finset.sdiff_eq_empty_iff_subset.2 hsub, Finset.card_empty, Nat.cast_zero]
      have := mul_nonneg hθ (Nat.cast_nonneg (α := ℝ) N)
      positivity
  -- the heavy prefixes
  have hheavy := heavy_count_le O F C Z P g hf0 hL hΔ.le hscreen hZAP hbal hCZ hEC
  have hPJ : (P.card : ℝ) ≤ (populations.card : ℝ) * N := by
    exact_mod_cast card_prefixesAt_le populations N y
  have hHj : ((H j).card : ℝ) ≤ (allHarvest H).card := by
    exact_mod_cast Finset.card_le_card (harvest_subset H j)
  have hmono : Δ * P.card ≤ (populations.card : ℝ) * Δ * N := by nlinarith
  have hsq : Real.sqrt (L * (Δ * P.card)) ≤ Real.sqrt (L * ((populations.card : ℝ) * Δ * N)) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hmono hL)
  have hdiv : (Δ * P.card + 2 * Real.sqrt (L * (Δ * P.card)) + 2 * L) / f
      ≤ ((populations.card : ℝ) * Δ * N + 2 * Real.sqrt (L * ((populations.card : ℝ) * Δ * N))
        + 2 * L) / f :=
    div_le_div_of_nonneg_right (by linarith) hf0.le
  have hhv : ((C.filter (fun p => ¬ famGood O f kmin kmax F p)).card : ℝ)
      ≤ lcut * (C.card : ℝ) := by
    have e : C.filter (fun p => ¬ famGood O f kmin kmax F p)
        = C.filter (fun p => (F.card : ℝ) * f < flipCount O F p) := by
      refine Finset.filter_congr (fun p _ => ?_)
      unfold famGood
      simp only [hsize.1, hsize.2, and_true, not_le]
    rw [e, hCn]
    linarith
  have hret := retAt_of_light O populations uni huni j B y (by omega) hηle hη₀ εcov α hα0 hα1
    l lcut (5 / 8 * lcut) f t r rs kmin kmax hkmin hl2 hlohi ht hkpos hpool hgate hsize hCn hhv
    h8 h10 h11
  exact ⟨hret.2.1, hret.2.2⟩

lemma card_poolAt_le (M : ℕ) (r : Run Ω S J) : (poolAt M r).card ≤ M + 1 :=
  le_trans (Finset.card_insert_le _ _) (Nat.add_le_add_right (card_freshDraws_le _ _ _) 1)

open scoped Classical in
lemma measurableSet_winAt_eq (H : J → Finset S) (K : ℕ) (j : J) (z : Finset S) :
    MeasurableSet {r : Run Ω S J | winAt H K j r = z} :=
  measurableSet_window_of _ K j z

open scoped Classical in
lemma measurableSet_coinSplitBad (O : Oracle μ S) (H : J → Finset S) (K M : ℕ) (L : ℝ)
    (j : J) : MeasurableSet (coinSplitBad (Ω := Ω) O H K M L j) := by
  classical
  refine measurableSet_coinEvent (fun r : Run Ω S J => (winAt H K j r, poolAt M r))
    (fun a => ?_) (fun a => a.1)
    (fun a g => ∃ v ∈ a.2,
      Real.sqrt (2 * ((a.1.filter (fun p => O.flip v p = 1)).card : ℝ) * L)
        < (((a.1.filter (fun p => O.flip v p = 1)).filter (fun s => g s = false)).card : ℝ)
          - (((a.1.filter (fun p => O.flip v p = 1)).filter (fun s => g s = true)).card : ℝ))
    (fun a g g' hgg => ?_)
  · have : {r : Run Ω S J | (winAt H K j r, poolAt M r) = a}
        = {r | winAt H K j r = a.1} ∩ {r | poolAt M r = a.2} := by
      ext r; simp [Prod.ext_iff]
    rw [this]
    exact (measurableSet_winAt_eq H K j a.1).inter (measurableSet_poolAt M a.2)
  · have hf : ∀ (v : S) (b : Bool), (a.1.filter (fun p => O.flip v p = 1)).filter
        (fun s => g s = b) = (a.1.filter (fun p => O.flip v p = 1)).filter
          (fun s => g' s = b) :=
      fun v b => Finset.filter_congr (fun s hs => by
        rw [hgg s (Finset.mem_filter.1 hs).1])
    simp only [hf]

open scoped Classical in
lemma measurableSet_coinCardBad (H : J → Finset S) (K G : ℕ) (j : J) :
    MeasurableSet (coinCardBad (Ω := Ω) H K G j) := by
  classical
  refine measurableSet_coinEvent (winAt H K j) (measurableSet_winAt_eq H K j) (fun z => z)
    (fun z g => (z.card : ℝ) / 2 + G ≤ ((z.filter (fun s => g s = true)).card : ℝ)
      ∨ (z.card : ℝ) / 2 + G ≤ ((z.filter (fun s => g s = false)).card : ℝ))
    (fun z g g' hgg => ?_)
  have hf : ∀ b : Bool, z.filter (fun s => g s = b) = z.filter (fun s => g' s = b) :=
    fun b => Finset.filter_congr (fun s hs => by rw [hgg s hs])
  simp only [hf]

open scoped Classical in
lemma measurableSet_hitsBad (H : J → Finset S) (K n : ℕ) (a : ℝ) (uni j : J) :
    MeasurableSet (hitsBad (Ω := Ω) H K n a uni j) := by
  classical
  have hcov : hitsBad (Ω := Ω) H K n a uni j = ⋃ z : Finset S, {r | winAt H K j r = z}
      ∩ (fun r : Run Ω S J => r.2.1.2 uni) ⁻¹' {w | a ≤ ((z ∩ acceptedAt ∅ w n).card : ℝ)} := by
    ext r
    simp only [hitsBad, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨z, hz, h⟩ => hz ▸ h⟩
  rw [hcov]
  exact MeasurableSet.iUnion (fun z => (measurableSet_winAt_eq H K j z).inter
    (measurable_prefixStreamOf uni (measurableSet_acceptedAt ∅ n
      (fun F => a ≤ ((z ∩ F).card : ℝ)))))

open scoped Classical in
set_option maxHeartbeats 8000000 in
/-- A named population fails the round at the top rung with chance at most `namedFail`. -/
theorem measureReal_named_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (H : J → Finset S)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations)
    (hju : j ≠ uni) (B : State) (hN2 : 2 ≤ B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lcut f E Δ t L θ γscr γdirty tap pAP ρ ρsf : ℝ) (r rs G : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hElm : E ≤ 5 / 8 * lcut)
    (hl2 : 2 * l ≤ indecisionLimit) (hf0 : 0 < f) (hΔ : 0 < Δ) (hL : 0 ≤ L) (hθ : 0 < θ)
    (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 2 ≤ B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      (lcut + 5 / 8 * lcut) * B.npref / gateKappa η₀
          + 2 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hszG : (allHarvest H).card + G ≤ B.npref)
    (hheavyN : ((populations.card : ℝ) * Δ * B.npref
          + 2 * Real.sqrt (L * ((populations.card : ℝ) * Δ * B.npref)) + 2 * L) / f
        + (allHarvest H).card + 2 * G + θ * B.npref ≤ lcut * B.npref)
    (hγscr : 0 ≤ γscr) (hγdirty : 0 ≤ γdirty) (htap : 0 ≤ tap) (hpAP0 : 0 ≤ pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}) (hscd : 0 < B.scd)
    (hscLow : (B.scd : ℝ) * (2 * γscr) ≤ (B.sc : ℝ))
    (hscHigh : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γdirty))
    (hcount : (B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - tap))
    (hρu : ∀ t, (D uni).real {t} ≤ ρ) (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real
        {x | streams H uni x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B j}
      ≤ namedFail l lcut E γscr γdirty tap ρ ρsf t L θ r rs G B := by
  classical
  have hsig : O.η ≤ 1 / 2 := by linarith [hηle]
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρu 1)
  have hρc : collisionMass (D uni) ≤ ρ := collisionMass_le_of_atoms (D uni) hρu
  set N : ℕ := B.npref with hN
  set M : ℕ := B.nsuff with hM
  set K : ℕ := 2 * (N - (H j).card - G) with hK
  have hNpos : 0 < N := by omega
  have hNR : (0 : ℝ) < N := by exact_mod_cast hNpos
  set kk : ℕ := B.k - 1 with hkk
  set Y := streams (Ω := Ω) H uni with hY
  -- the events
  set Es : Set (Run Ω S J) := stalled rule O populations B kk kk with hEs
  set En : Set (Run Ω S J) := {y | ¬ ∃ w₀ ∈ (poolAt B.nsuff y).erase 1,
    ∀ p ∈ prefixesAt populations B.npref y, O.flip w₀ p = 0} with hEn
  set Eb : Set (Run Ω S J) := screenBad O populations B Δ with hEb
  set Et : Set (Run Ω S J) := {y | ¬ tabGood uni N r y} with hEt
  set El : Set (Run Ω S J) := lightMiss rule O populations j B l lcut (5 / 8 * lcut) f kk kk
    with hEl
  set Ed : Set (Run Ω S J) := sidesDead rule O populations j B N t with hEd
  set Ep : Set (Run Ω S J) := {y | ¬ poolGood M rs y} with hEp
  set Ec1 : Set (Draws Ω S J) := coinSplitBad O H K M L j with hEc1
  set Ec2 : Set (Draws Ω S J) := coinCardBad H K G j with hEc2
  set Eh : Set (Draws Ω S J) := {x | x.1 ∈ hitsBad H K (N + r) (θ * N) uni j} with hEh
  have hsub : {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B j}
      ⊆ Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh ∪ Y ⁻¹' El
        ∪ Y ⁻¹' Ed ∪ Y ⁻¹' Ep := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, not_or, Set.mem_preimage] at hc
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h10⟩, h11⟩ := hc
    have h6' : tabGood uni N r (Y x) := by by_contra h; exact h6 h
    have h11' : poolGood M rs (Y x) := by by_contra h; exact h11 h
    exact hx (named_retAtK O populations uni huni H j hj hju B x hN2 hηle hη₀ εcov α hα0 hα1
      l lcut f Δ t L θ r rs kk kk G (by omega) hl2 hf0 hΔ hL hθ.le hlohi ht (by omega) hpool hgate
      hszG hheavyN h1 h2 h3 h4 h5 h6' h7 h8 h10 h11')
  -- each event's price
  have hsT : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawT H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawT_mem H uni D g populations hsupp hH
  have hsC : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawC H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawC_mem H uni D g populations hsupp
  have bs : (runMeasure μ D Dsf).real (Y ⁻¹' Es)
      ≤ ((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + (Real.exp (-2 * (M : ℝ) * tap ^ 2)
        + (((N + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
          + ((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (N : ℝ) * γscr ^ 2))) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_stalled O populations B kk kk) (fun g =>
      measureReal_stalled_le hflat O populations _ _ Dsf (hsT g) hsuppSf B (by omega) uni huni
        (D uni) (sideLawT_uni H uni D g) γscr pAP tap ρsf ρ rs r hγscr hpAP0 htap hpAPBound hscd
        hsig hscLow hcount hρsf hρsf0 hρc hρ0)
  have bn : (runMeasure μ D Dsf).real (Y ⁻¹' En)
      ≤ Real.exp (-2 * (M : ℝ) * tap ^ 2) + ((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_noClean O populations B)
      (fun g => measureReal_noCleanRef_le _ _ Dsf O populations B pAP tap ρsf rs hpAP0 htap
        hpAPBound hρsf hρsf0 (by
          have : (2 : ℝ) ≤ B.k := by exact_mod_cast hkpos
          linarith))
  have bb : (runMeasure μ D Dsf).real (Y ⁻¹' Eb)
      ≤ ((M : ℝ) + 2) ^ 2 * Real.exp (-2 * (N : ℝ) * γdirty ^ 2) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_screenBad O populations B Δ) (fun g =>
      measureReal_screenBad_le hflat O populations _ _ Dsf (hsT g) hsuppSf B hsig hNpos Δ
        γdirty hΔ hγdirty hscd hscHigh)
  have bc1 : (runMeasure μ D Dsf).real Ec1 ≤ ((M : ℝ) + 1) * Real.exp (-L) := by
    refine runMeasure_coins_le D Dsf _ (measurableSet_coinSplitBad O H K M L j)
      (by positivity) (fun x => ?_)
    have h := coins_family_balance (poolAt M x)
      (fun v => (winAt H K j x).filter (fun p => O.flip v p = 1)) hL
    refine le_trans h (mul_le_mul_of_nonneg_right ?_ (Real.exp_nonneg _))
    exact_mod_cast card_poolAt_le M x
  have bc2 : (runMeasure μ D Dsf).real Ec2 ≤ 2 * Real.exp (-((G : ℝ) ^ 2 / N)) := by
    refine runMeasure_coins_le D Dsf _ (measurableSet_coinCardBad H K G j) (by positivity)
      (fun x => ?_)
    exact coins_window_card_le (winAt H K j x) N G hNpos
      (le_trans (window_card_le _ K _) (by omega))
  have bt : (runMeasure μ D Dsf).real (Y ⁻¹' Et) ≤ ((N + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_notTabGood uni N r) (fun g =>
      measureReal_tabBad_le _ _ Dsf uni (D uni) (sideLawT_uni H uni D g) N r ρ hρc hρ0)
  have bh : (runMeasure μ D Dsf).real Eh ≤ 2 * ((N + r : ℕ) : ℝ) * ρ / θ := by
    refine runMeasure_base_le D Dsf _ (measurableSet_hitsBad H K (N + r) (θ * N) uni j) ?_
    refine le_trans (measureReal_hits_le D Dsf uni j hju hρu _ K (N + r) (by positivity)) ?_
    have hKN : (K : ℝ) ≤ 2 * N := by
      have : K ≤ 2 * N := by omega
      exact_mod_cast this
    rw [div_le_div_iff₀ (by positivity) hθ]
    have h0 : (0 : ℝ) ≤ ((N + r : ℕ) : ℝ) * ρ := by positivity
    have h1 := mul_le_mul_of_nonneg_left hKN (mul_nonneg h0 hθ.le)
    nlinarith [h1]
  have bl : (runMeasure μ D Dsf).real (Y ⁻¹' El)
      ≤ Real.exp (-2 * (N : ℝ) * (l - E) ^ 2) + Real.exp (-2 * (N : ℝ) * (5 / 8 * lcut - E) ^ 2) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_lightMiss O populations j B l lcut
      (5 / 8 * lcut) f kk kk) (fun g => measureReal_lightMiss_le hflat O populations _ _ Dsf
        (hsT g) (hsC g) hsuppSf j hj B l lcut (5 / 8 * lcut) f E kk kk hE hElcut hlcl hElm hdec
        hcut)
  have bd : (runMeasure μ D Dsf).real (Y ⁻¹' Ed)
      ≤ 2 * (((M : ℝ) + 3) * Real.exp (-2 * t ^ 2 / N)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_sidesDead O populations j B N t)
      (fun g => measureReal_sidesDead_le hflat O populations _ _ Dsf (hsT g) (hsC g) hsuppSf j hj
        B N ht)
  have bp : (runMeasure μ D Dsf).real (Y ⁻¹' Ep) ≤ ((M + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_notPoolGood M rs) (fun g =>
      measureReal_poolBad_le _ _ Dsf M rs ρsf hρsf hρsf0)
  set ν := runMeasure μ D Dsf with hν
  have u : ∀ A B' : Set (Draws Ω S J), ν.real (A ∪ B') ≤ ν.real A + ν.real B' :=
    fun A B' => measureReal_union_le A B'
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have := u (Y ⁻¹' Es) (Y ⁻¹' En)
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En) (Y ⁻¹' Eb)
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb) Ec1
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1) Ec2
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2) (Y ⁻¹' Et)
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et) Eh
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh) (Y ⁻¹' El)
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh ∪ Y ⁻¹' El)
    (Y ⁻¹' Ed)
  have := u (Y ⁻¹' Es ∪ Y ⁻¹' En ∪ Y ⁻¹' Eb ∪ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh ∪ Y ⁻¹' El
    ∪ Y ⁻¹' Ed) (Y ⁻¹' Ep)
  unfold namedFail
  linarith

open scoped Classical in
set_option maxHeartbeats 4000000 in
/-- The loop returns: except with chance `δ/2`, the top rung passes at every population, wild
or tame, its gate read at the first look. -/
theorem raw_terminates {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (H : J → Finset S)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (εcov α δ ρ pAP crossLimit : ℝ)
    (hsig : O.η < 1 / 2) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρu : ∀ t, (D uni).real {t} ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit
      (allHarvest H).card)
    (hρsf : collisionMass Dsf
        ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State // B ∈ stoppable η₀ populations
        indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card ρ},
      streams H uni x ∉ retAll rule O.mq populations uni η₀ indecisionLimit εcov α B.val}
      ≤ δ / 2 := by
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρu 1)
  have hρc : collisionMass (D uni) ≤ ρ := collisionMass_le_of_atoms (D uni) hρu
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hsigle : O.η ≤ 1 / 2 := hsig.le
  obtain ⟨B, hB, hpass⟩ := exists_passable (crossLimit := crossLimit) (hsz := (allHarvest H).card)
    O populations Dsf indecisionLimit εcov α δ ρ (collisionMass Dsf) pAP hsig hpop hηle hη₀
    hεcov hε1 hδ hδ1 hαpos hα hindLim hind1 hpAPPositive hpAPBound hρ0 le_rfl hρsf0 hρcap hρsf
  obtain ⟨th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, f, c, t, L, θ, r, rs, G, hmpos, hkpos,
    hindLim', hind1', hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hf0, hpAP0, hlcut, hlcl,
    hc0, hr, hpAPBound', hρsf', hheavy, hL, hθ, hN2, hszG, hheavyN, hscd, hscLow, hscHigh,
    hcount, hpool, hhiUp, hloUp, hhiLo, hloLo, hlohi, ht, hgate, hEhalf, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hl2 : 2 * l ≤ indecisionLimit := by rw [hl]; linarith
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  have hE0 : 0 ≤ E := Real.exp_nonneg _
  have hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * f →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E := by
    intro F p h1 hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    have hcardI : ((insert 1 F).card : ℝ) = (B.k : ℝ) := by
      rw [Finset.card_insert_of_notMem h1, hcardF, hκ]
      have : 1 ≤ B.k := by omega
      push_cast [this]
      ring
    have hcle : (F.card : ℝ) ≤ ((insert 1 F).card : ℝ) := by
      exact_mod_cast Finset.card_le_card (Finset.subset_insert _ _)
    have hflI : flipCount O (insert 1 F) p ≤ ((insert 1 F).card : ℝ) * f :=
      le_trans (flipCount_insert_one O F p)
        (le_trans hfl (mul_le_mul_of_nonneg_right hcle hf0.le))
    refine le_trans (decided_whp O (insert 1 F) p B.lo (B.hi + 1) f γdec hγdec hsigle hflI
      ?_ ?_) ?_
    · rw [hcardI]; push_cast; exact hhiUp
    · rw [hcardI]; exact hloUp
    · rw [hE, hcardI]
      refine Real.exp_le_exp.2 ?_
      have hk : (κ : ℝ) ≤ (B.k : ℝ) := by rw [hκ]; exact_mod_cast Nat.sub_le _ _
      nlinarith [sq_nonneg γdec]
  have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E := by
    intro F p hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    refine le_trans (cutCorrect_whp O F p B.lo B.hi f γdec hsigle hfl hγdec ?_ ?_) ?_
    · rw [hcardF]; linarith [hhiLo]
    · rw [hcardF]; exact hloLo.le
    · rw [hE, hcardF]
  set Y := streams (Ω := Ω) H uni with hY
  set ν := runMeasure μ D Dsf with hν
  have hsT : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawT H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawT_mem H uni D g populations hsupp hH
  have hsC : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawC H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawC_mem H uni D g populations hsupp
  -- the uniform pool
  set Δp : ℝ := (populations.card : ℝ) * Δ + c + gdirty with hΔp
  have hΔp0 : 0 ≤ Δp := by rw [hΔp]; positivity
  set U : Set (Run Ω S J) := {x | ¬ certGood populations uni B.npref r x}
      ∪ stalled rule O populations B κ κ
      ∪ {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D uni) v ≤ Δp}
      ∪ heavyHits rule O populations (D uni) uni B f (Δp / f) th
      ∪ lightMiss rule O populations uni B l lcut (5 / 8 * lcut) f κ κ
      ∪ sidesDead rule O populations uni B B.npref t ∪ {x | ¬ poolGood B.nsuff rs x} with hU
  have hUm : MeasurableSet U :=
    ((((((measurableSet_notCertGood populations uni B.npref r).union
      (measurableSet_stalled O populations B κ κ)).union
      (measurableSet_dirty O populations B (D uni) Δp)).union
      (measurableSet_heavyHits O populations (D uni) uni B f (Δp / f) th)).union
      (measurableSet_lightMiss O populations uni B l lcut (5 / 8 * lcut) f κ κ)).union
      (measurableSet_sidesDead O populations uni B B.npref t)).union
      (measurableSet_notPoolGood B.nsuff rs)
  have hsubU : {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B uni}
      ⊆ Y ⁻¹' U := by
    intro x hx
    have hx' : Y x ∉ retAt rule O populations uni η₀ indecisionLimit εcov α B uni :=
      fun h => hx (retAt_subset_retAtK O populations uni η₀ indecisionLimit εcov α B uni h)
    exact notRetAt_subset (rule := rule) O populations uni huni uni (D uni) B hmpos hηle hη₀
      εcov α hαpos (by linarith) l lcut (5 / 8 * lcut) f Δp th c t r rs κ κ (by omega) hl2 hΔp0
      hf0 hr hheavy hlohi ht (by omega) hpool hgate hx'
  have hstallB : ∀ g, (streamMeasure μ (sideLawT H uni D g) (sideLawC H uni D g) Dsf).real
      (stalled rule O populations B κ κ)
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * collisionMass Dsf / (2 * ((rs : ℝ) + 1))
        + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
        + (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))) := fun g =>
    measureReal_stalled_le hflat O populations _ _ Dsf (hsT g) hsuppSf B (by omega) uni huni
      (D uni) (sideLawT_uni H uni D g) γscr pAP tap (collisionMass Dsf) ρ rs r hγscr hpAP0 htap
      hpAPBound' hscd hsigle hscLow hcount hρsf' hρsf0 hρc hρ0
  have huniB : ν.real {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B uni}
      ≤ roundFail populations l lcut th E γscr γdirty gdirty tap ρ (collisionMass Dsf) t r rs B := by
    refine le_trans (measureReal_mono hsubU (measure_ne_top _ _)) ?_
    refine runMeasure_streams_le H uni D Dsf U hUm (fun g => ?_)
    set Pp := sideLawT H uni D g
    set Pc := sideLawC H uni D g
    have hcertB : (streamMeasure μ Pp Pc Dsf).real {x | ¬ certGood populations uni B.npref r x}
        ≤ (((2 * B.npref + r : ℕ) : ℝ) * ((populations.card : ℝ) * B.npref * ρ)
          + ((2 * B.npref + r : ℕ) : ℝ) ^ 2 * ρ / 2) / ((r : ℝ) + 1) :=
      le_trans (measureReal_mono (fun x hx h2 => hx (certGood_of_certGood2 populations uni
          B.npref r x h2)) (measure_ne_top _ _))
        (measureReal_certBad2_le Pp Pc Dsf populations uni (D uni) (sideLawC_uni H uni D g)
          B.npref r ρ hρu)
    have hdirtyB : (streamMeasure μ Pp Pc Dsf).real
        {x | ¬ ∀ v ∈ clusterBy rule O.mq populations x B, flipMass O (D uni) v ≤ Δp}
        ≤ (((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
            + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
          + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
            + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * collisionMass Dsf / (2 * ((rs : ℝ) + 1))) := by
      have hdirtyI := measureReal_dirtyMember_le (rule := rule) hflat O populations Pp Pc Dsf
        (hsT g) hsuppSf uni huni (D uni) (sideLawT_uni H uni D g) B hsigle hmpos Δ γdirty gdirty
        c ρ r hΔ hγdirty hgdirty hc0 hρ0 hr hρc hscd hscHigh
      have hnoclean := measureReal_noCleanRef_le Pp Pc Dsf O populations B pAP tap
        (collisionMass Dsf) rs hpAP0 htap hpAPBound' le_rfl hρsf0 (by
          have hk2 : (2 : ℝ) ≤ (B.k : ℝ) := by exact_mod_cast hkpos
          linarith [hcount, hk2])
      have hcov : {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
            flipMass O (D uni) v ≤ Δp}
          ⊆ ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
                ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
              ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
                flipMass O (D uni) v ≤ (populations.card : ℝ) * Δ + c + gdirty})
            ∪ {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
                ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} := by
        intro x hx
        by_cases hc : ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
            ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0
        · exact Or.inl ⟨hc, hx⟩
        · exact Or.inr hc
      refine le_trans (measureReal_mono hcov (measure_ne_top _ _)) ?_
      refine le_trans (measureReal_union_le _ _) ?_
      gcongr
    refine le_trans (measureReal_failUnion_le hflat O populations Pp Pc Dsf (hsT g) (hsC g)
      hsuppSf uni huni (D uni) (sideLawC_uni H uni D g) B l lcut (5 / 8 * lcut) f E Δp th t
      (collisionMass Dsf) r rs κ κ hE0 (by linarith) (by rw [hl]; exact hlcl) (by linarith) hth
      ht hρsf' hρsf0 _ hcertB _ (hstallB g) _ hdirtyB hdec hcut) (le_of_eq ?_)
    unfold roundFail
    ring
  -- the named populations
  have hnamedB : ∀ j ∈ populations, j ≠ uni →
      ν.real {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B j}
        ≤ namedFail l lcut E γscr γdirty tap ρ (collisionMass Dsf) t L θ r rs G B := fun j hj hju =>
    measureReal_named_le hflat O populations uni huni H D Dsf hsupp hH hsuppSf j hj hju B hN2
      hηle hη₀ εcov α hαpos (by linarith) l lcut f E Δ t L θ γscr γdirty tap pAP ρ
      (collisionMass Dsf) r rs G hE0 (by linarith) (by rw [hl]; exact hlcl) (by linarith) hl2 hf0
      hΔ hL hθ hlohi ht hkpos hpool hgate hszG hheavyN hγscr hγdirty htap hpAP0 hpAPBound' hscd
      hscLow hscHigh hcount hρu hρsf' hρsf0 hdec hcut
  -- every population
  have hRF0 : 0 ≤ roundFail populations l lcut th E γscr γdirty gdirty tap ρ (collisionMass Dsf)
      t r rs B := by unfold roundFail; positivity
  have hNF0 : 0 ≤ namedFail l lcut E γscr γdirty tap ρ (collisionMass Dsf) t L θ r rs G B := by
    unfold namedFail; positivity
  have hsub : {x : Draws Ω S J | ∀ B' : {B : State // B ∈ stoppable η₀ populations
          indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card ρ},
        Y x ∉ retAll rule O.mq populations uni η₀ indecisionLimit εcov α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_ofPred_eq, not_and,
      not_not] at hc
    exact hx ⟨B, hB⟩ (mem_retAll_of_retAtK O populations hpop uni η₀ indecisionLimit εcov α B
      (Y x) hc)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ j ∈ populations,
      ν.real {x | Y x ∉ retAtK rule O populations uni η₀ indecisionLimit εcov α B j}
        ≤ roundFail populations l lcut th E γscr γdirty gdirty tap ρ (collisionMass Dsf) t r rs B
          + namedFail l lcut E γscr γdirty tap ρ (collisionMass Dsf) t L θ r rs G B := by
    intro j hj
    by_cases hju : j = uni
    · rw [hju]; linarith [huniB]
    · linarith [hnamedB j hj hju]
  refine le_trans (Finset.sum_le_sum hper) ?_
  rw [Finset.sum_const, nsmul_eq_mul]
  exact hbudget

end OrthoDFA
