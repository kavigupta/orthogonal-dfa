import OrthoDFA.Proofs.RawEvents

/-!
# The loop returns, whatever the named populations are

At the top rung every population's FNR test and gate pass, except with chance `δ/2`.  The
uniform pool's tests are priced at each assignment of the hash by the stream-level lemmas.  A
named population's are priced against its own table: the hash splits a window of its draws
evenly, so the family's members flip on its certification sample about as often as on its
table, where the count rule holds them to the table budget on average.  Nothing is asked of a
named population's law.  The count rule's table event and the pool's are events of the whole
round, charged once.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]
variable {η₀ indecisionLimit εs : ℝ}

/-- Population `j`'s window: its first `K` distinct draws outside the harvested strings. -/
noncomputable def winAt (H : J → Finset S) (K : ℕ) (j : J) (r : Run Ω S J) : Finset S :=
  window (↑(allHarvest H))ᶜ K (r.2.1.2 j)

open scoped Classical in
/-- The hash splits some pool member's flips on the window's prefixes of some class unevenly. -/
noncomputable def coinSplitBad (O : Oracle μ S) (H : J → Finset S) (K M : ℕ) (L : ℝ) (j : J) :
    Set (Draws Ω S J) :=
  {x | ∃ v ∈ poolAt M x.1, ∃ b : Bool,
    Real.sqrt (2 * (((winAt H K j x.1).filter (fun p => O.flip v p = 1
        ∧ O.label p = sideClass b)).card : ℝ) * L)
      < ((((winAt H K j x.1).filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
          (fun s => x.2 s = false)).card : ℝ)
        - ((((winAt H K j x.1).filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
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
family stalls, the pool is short of accept-preserving rows, the count rule's table event or the
hash misbehaves, the uniform pool lands in the window too often, a light count misses, a side is
dead or the pool is short. -/
theorem named_retAtK (O : Oracle μ S) (populations : Finset J) (uni : J)
    (huni : uni ∈ populations) (H : J → Finset S) (j : J) (hj : j ∈ populations)
    (hju : j ≠ uni) (B : State) (x : Draws Ω S J) (hN2 : 2 ≤ B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lF lG lmis Δ t L θ a pAP tap : ℝ) (r rs kmin kmax G : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hscreen : B.scd ≤ B.sc) (ha : a ≤ (B.nsuff : ℝ) * (pAP - tap) - rs - 1)
    (hlF : 2 * lF + l ≤ indecisionLimit) (hlmis : 0 ≤ lmis) (hΔ : 0 < Δ) (hL : 0 ≤ L)
    (hθ : 0 ≤ θ) (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 0 < B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      4 * (2 * lG + lmis / gateA η₀ ^ 2) * B.npref
          + 4 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hszG : (allHarvest H).card + G ≤ B.npref)
    (hNF : ∀ b, namedQ O η₀ Δ L θ (allHarvest H).card G B.npref b ≤ lF * B.npref)
    (hNG : ∀ b, sideCharge O η₀ (!b) * namedQ O η₀ Δ L θ (allHarvest H).card G B.npref b
      ≤ lG * B.npref)
    (h1 : streams H uni x ∉ stalled nearestClusterer O populations B kmin kmax)
    (h2 : streams H uni x ∉ apShort O B.nsuff pAP tap)
    (h3 : streams H uni x ∉ nearShift O populations B Δ a)
    (h4 : x ∉ coinSplitBad O H (2 * (B.npref - (H j).card - G)) B.nsuff L j)
    (h5 : x ∉ coinCardBad H (2 * (B.npref - (H j).card - G)) G j)
    (h6 : tabGood uni B.npref r (streams H uni x))
    (h7 : x.1 ∉ hitsBad H (2 * (B.npref - (H j).card - G)) (B.npref + r) (θ * B.npref) uni j)
    (h8 : streams H uni x ∉ lightMiss nearestClusterer O populations j B l lmis (fracAt O η₀)
      kmin kmax)
    (h10 : streams H uni x ∉ sidesDead nearestClusterer O populations j B B.npref t)
    (h11 : poolGood B.nsuff rs (streams H uni x)) :
    streams H uni x ∈ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B j := by
  classical
  set y := streams H uni x with hy
  set N : ℕ := B.npref with hN
  set K : ℕ := 2 * (N - (H j).card - G) with hK
  set Hs : Finset S := allHarvest H with hHs
  set g : S → Bool := x.2 with hg
  set w : ℕ → S := x.1.2.1.2 j with hw
  set z : ℕ → S := x.1.2.2 j with hz
  set Z : Finset S := winAt H K j x.1 with hZ
  set F : Finset S := clusterBy nearestClusterer O.mq populations y B with hF
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
  -- the count rule
  have hAPc := apRows_ge O B.nsuff rs pAP tap a y ha h11 h2
  have hNPR : (N : ℝ) ≤ P.card := by exact_mod_cast hNP
  have hPJ : (P.card : ℝ) ≤ (populations.card : ℝ) * N := by
    exact_mod_cast card_prefixesAt_le populations N y
  have hshift0 := famShift_of_not_nearShift O populations B hscreen Δ a y h3 hNPR hPJ hAPc
    (card_poolAt_le' _ _)
  have hκF : ((B.k - 1 : ℕ) : ℝ) ≤ F.card := by
    have : B.k - 1 ≤ F.card := by omega
    exact_mod_cast this
  have hshift : ∑ v ∈ F, ∑ p ∈ P, flipShift O v p ≤ (F.card : ℝ) * (Δ * N) :=
    le_trans hshift0 (mul_le_mul_of_nonneg_right hκF (by positivity))
  -- on each class, the members' flips on the tables, from their shift
  have hflips : ∀ b, ∑ v ∈ F, ((P.filter (fun p => O.flip v p = 1
      ∧ O.label p = sideClass b)).card : ℝ) ≤ (F.card : ℝ) * ((Δ / clsShift O b) * N) := by
    intro b
    have hw := clsShift_pos O hηle hη₀ b
    have hv : ∀ v, ((P.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).card : ℝ)
        * clsShift O b ≤ ∑ p ∈ P, flipShift O v p := by
      intro v
      rw [Finset.card_filter, Nat.cast_sum, Finset.sum_mul]
      refine Finset.sum_le_sum (fun p _ => ?_)
      split_ifs with h
      · rw [← clsW_mul_flip O b v p h.2]
        unfold clsW; rw [if_pos h.2, h.1]; simp
      · simp only [Nat.cast_zero, zero_mul]
        exact flipShift_nonneg O (by linarith) v p
    have hsum := Finset.sum_le_sum (fun v (_ : v ∈ F) => hv v)
    rw [← Finset.sum_mul] at hsum
    have e : (F.card : ℝ) * ((Δ / clsShift O b) * N) = (F.card : ℝ) * (Δ * N) / clsShift O b := by
      field_simp
    rw [e, le_div_iff₀ hw]
    linarith
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
  have hbal : ∀ b, ∀ v ∈ F,
      (((Z.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
          (fun s => g s = false)).card : ℝ)
        - (((Z.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
          (fun s => g s = true)).card : ℝ)
      ≤ Real.sqrt (2 * ((Z.filter (fun p => O.flip v p = 1
          ∧ O.label p = sideClass b)).card : ℝ) * L) := by
    intro b v hv
    have hvp : v ∈ poolAt B.nsuff x.1 := clusterAt_subset O populations B y hv
    by_contra hc
    exact h4 ⟨v, hvp, b, not_le.1 hc⟩
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
  -- each class's heavy prefixes
  have hHj : ((H j).card : ℝ) ≤ (allHarvest H).card := by
    exact_mod_cast Finset.card_le_card (harvest_subset H j)
  have hM : ∀ b, ((C.filter (fun p => clsBadP O η₀ b F p)).card : ℝ)
      ≤ namedQ O η₀ Δ L θ (allHarvest H).card G N b := by
    intro b
    have hw := clsShift_pos O hηle hη₀ b
    have hf0 : 0 < clsFrac O η₀ b := lt_of_lt_of_le (by
      have := sig_pos η₀ hη₀; positivity) (clsFrac_ge O hηle hη₀ b)
    have hheavy := heavy_count_le O F C Z P g (fun p => O.label p = sideClass b) hf0 hL
      (by positivity) (hflips b) hZAP (hbal b) hCZ hEC
    have hsub : C.filter (fun p => clsBadP O η₀ b F p)
        ⊆ C.filter (fun p => O.label p = sideClass b
          ∧ (F.card : ℝ) * clsFrac O η₀ b < flipCount O F p) := by
      intro p hp
      obtain ⟨hpC, hl, hfl⟩ := Finset.mem_filter.1 hp
      rw [fracAt_eq O η₀ hl] at hfl
      exact Finset.mem_filter.2 ⟨hpC, hl, not_le.1 hfl⟩
    refine le_trans (Nat.cast_le.2 (Finset.card_le_card hsub)) (le_trans hheavy ?_)
    unfold namedQ
    linarith
  have hret := retAt_of_light O populations uni huni j B y (by omega) hηle hη₀ εcov α hα0 hα1
    l lF lG lmis t (fun b => namedQ O η₀ Δ L θ (allHarvest H).card G N b) r rs kmin kmax hkmin
    hlF hlmis hlohi ht hkpos hpool hgate hsize hCn hM hNF
    (fun b => (mul_comm _ _).trans_le (hNG b)) h8 h10 h11
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
    (fun a g => ∃ v ∈ a.2, ∃ b : Bool,
      Real.sqrt (2 * ((a.1.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).card : ℝ)
          * L)
        < (((a.1.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
            (fun s => g s = false)).card : ℝ)
          - (((a.1.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
            (fun s => g s = true)).card : ℝ))
    (fun a g g' hgg => ?_)
  · have : {r : Run Ω S J | (winAt H K j r, poolAt M r) = a}
        = {r | winAt H K j r = a.1} ∩ {r | poolAt M r = a.2} := by
      ext r; simp [Prod.ext_iff]
    rw [this]
    exact (measurableSet_winAt_eq H K j a.1).inter (measurableSet_poolAt M a.2)
  · have hf : ∀ (v : S) (b c : Bool), (a.1.filter (fun p => O.flip v p = 1
        ∧ O.label p = sideClass b)).filter (fun s => g s = c)
        = (a.1.filter (fun p => O.flip v p = 1 ∧ O.label p = sideClass b)).filter
          (fun s => g' s = c) :=
      fun v b c => Finset.filter_congr (fun s hs => by
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

/-- The pool's events, which every population's round shares: the family's size, the pool short
of accept-preserving rows, and its sampler redrawing too much. -/
def poolEvents (O : Oracle μ S) (populations : Finset J) (B : State) (pAP tap : ℝ) (rs : ℕ) :
    Set (Run Ω S J) :=
  stalled nearestClusterer O populations B (B.k - 1) (B.k - 1) ∪ apShort O B.nsuff pAP tap
    ∪ {y | ¬ poolGood B.nsuff rs y}

theorem measureReal_poolEvents_le (O : Oracle μ S) (populations : Finset J) (uni : J)
    (H : J → Finset S) (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (B : State) (hkpos : 2 ≤ B.k) (hpool : B.k < B.nsuff)
    (hscreen : B.scd ≤ B.sc) (pAP tap ρsf : ℝ) (rs : ℕ) (htap : 0 ≤ tap) (hpAP0 : 0 ≤ pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf) :
    (runMeasure μ D Dsf).real (streams H uni ⁻¹' poolEvents O populations B pAP tap rs)
      ≤ poolFail tap pAP ρsf rs B := by
  classical
  set ν := runMeasure μ D Dsf with hν
  set Y := streams (Ω := Ω) H uni with hY
  have bs : ν.real (Y ⁻¹' stalled nearestClusterer O populations B (B.k - 1) (B.k - 1))
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_stalled O populations B _ _) (fun g =>
      measureReal_stalled_nearest_le O populations _ _ Dsf B (by omega) hscreen hpool.le ρsf rs
        hρsf hρsf0)
  have bn : ν.real (Y ⁻¹' apShort O B.nsuff pAP tap)
      ≤ Real.exp (-((B.nsuff : ℝ) * (tap ^ 2 / (2 * pAP)))) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_apShort O B.nsuff pAP tap)
      (fun g => measureReal_apShort_le _ _ Dsf O B.nsuff pAP tap hpAP0 htap hpAPBound)
  have bp : ν.real (Y ⁻¹' {y | ¬ poolGood B.nsuff rs y})
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_notPoolGood B.nsuff rs) (fun g =>
      measureReal_poolBad_le _ _ Dsf B.nsuff rs ρsf hρsf hρsf0)
  have u1 := measureReal_union_le (μ := ν)
    (Y ⁻¹' stalled nearestClusterer O populations B (B.k - 1) (B.k - 1))
    (Y ⁻¹' apShort O B.nsuff pAP tap)
  have u2 := measureReal_union_le (μ := ν)
    (Y ⁻¹' stalled nearestClusterer O populations B (B.k - 1) (B.k - 1)
      ∪ Y ⁻¹' apShort O B.nsuff pAP tap)
    (Y ⁻¹' {y | ¬ poolGood B.nsuff rs y})
  rw [poolEvents, Set.preimage_union, Set.preimage_union, poolFail]
  linarith

open scoped Classical in
set_option maxHeartbeats 8000000 in
/-- A named population fails the round at the top rung, off the count rule's table event and the
pool's, with chance at most `namedFail`. -/
theorem measureReal_named_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (H : J → Finset S)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations)
    (hju : j ≠ uni) (B : State) (hN2 : 2 ≤ B.npref)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (εcov α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1)
    (l lF lG lmis E Δ t L θ tap pAP a ρ : ℝ) (r rs G : ℕ)
    (hE : 0 ≤ E) (hEl : E ≤ l) (hElm : E ≤ lmis)
    (hlF : 2 * lF + l ≤ indecisionLimit) (hlmis : 0 ≤ lmis) (hΔ : 0 < Δ) (hL : 0 ≤ L)
    (hθ : 0 < θ) (hlohi : B.lo ≤ B.hi) (ht : 0 ≤ t) (hkpos : 2 ≤ B.k) (hpool : B.k < B.nsuff)
    (hgate : ∀ N : ℕ, B.npref ≤ N → N ≤ B.npref + r →
      4 * (2 * lG + lmis / gateA η₀ ^ 2) * B.npref
          + 4 * ((gateDevT η₀ + gateDevU η₀) * t
            + gateDevD η₀ * hoeffDev B.npref (sizeLevel (gateLevel α populations B) B.npref))
          + ((N : ℝ) - B.npref)
        ≤ N * min εcov (1 / 2) - hoeffDev N (sizeLevel (gateLevel α populations B) N))
    (hszG : (allHarvest H).card + G ≤ B.npref)
    (hNF : ∀ b, namedQ O η₀ Δ L θ (allHarvest H).card G B.npref b ≤ lF * B.npref)
    (hNG : ∀ b, sideCharge O η₀ (!b) * namedQ O η₀ Δ L θ (allHarvest H).card G B.npref b
      ≤ lG * B.npref)
    (hscreen : B.scd ≤ B.sc) (ha : a ≤ (B.nsuff : ℝ) * (pAP - tap) - rs - 1)
    (hρu : ∀ t, (D uni).real {t} ≤ ρ)
    (hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * fracAt O η₀ p →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * fracAt O η₀ p →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E) :
    (runMeasure μ D Dsf).real
        (({x | streams H uni x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov
          α B j} \ streams H uni ⁻¹' nearShift O populations B Δ a)
          \ streams H uni ⁻¹' poolEvents O populations B pAP tap rs)
      ≤ namedFail l lmis E ρ t L θ r G B := by
  classical
  have hsig : O.η ≤ 1 / 2 := by linarith [hηle]
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρu 1)
  set N : ℕ := B.npref with hN
  set M : ℕ := B.nsuff with hM
  set K : ℕ := 2 * (N - (H j).card - G) with hK
  have hNpos : 0 < N := by omega
  have hNR : (0 : ℝ) < N := by exact_mod_cast hNpos
  set kk : ℕ := B.k - 1 with hkk
  set Y := streams (Ω := Ω) H uni with hY
  -- the events
  set Et : Set (Run Ω S J) := {y | ¬ tabGood uni N r y} with hEt
  set El : Set (Run Ω S J) := lightMiss nearestClusterer O populations j B l lmis
    (fracAt O η₀) kk kk with hEldef
  set Ed : Set (Run Ω S J) := sidesDead nearestClusterer O populations j B N t with hEd
  set Ec1 : Set (Draws Ω S J) := coinSplitBad O H K M L j with hEc1
  set Ec2 : Set (Draws Ω S J) := coinCardBad H K G j with hEc2
  set Eh : Set (Draws Ω S J) := {x | x.1 ∈ hitsBad H K (N + r) (θ * N) uni j} with hEh
  have hsub : ({x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B j}
        \ Y ⁻¹' nearShift O populations B Δ a) \ Y ⁻¹' poolEvents O populations B pAP tap rs
      ⊆ Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh ∪ Y ⁻¹' El ∪ Y ⁻¹' Ed := by
    rintro x ⟨⟨hx, h3⟩, hP⟩
    by_contra hc
    simp only [Set.mem_union, not_or, Set.mem_preimage] at hc
    obtain ⟨⟨⟨⟨⟨h4, h5⟩, h6⟩, h7⟩, h8⟩, h10⟩ := hc
    simp only [Set.mem_preimage, poolEvents, Set.mem_union, Set.mem_setOf_eq, not_or] at hP
    obtain ⟨⟨h1, h2⟩, h11⟩ := hP
    have h6' : tabGood uni N r (Y x) := by by_contra h; exact h6 h
    have h11' : poolGood M rs (Y x) := by by_contra h; exact h11 h
    exact hx (named_retAtK O populations uni huni H j hj hju B x hN2 hηle hη₀ εcov α hα0 hα1
      l lF lG lmis Δ t L θ a pAP tap r rs kk kk G (by omega) hscreen ha hlF hlmis hΔ hL hθ.le hlohi
      ht (by omega) hpool hgate hszG hNF hNG h1 h2 h3 h4 h5 h6' h7 h8 h10 h11')
  -- each event's price
  have hsT : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawT H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawT_mem H uni D g populations hsupp hH
  have hsC : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawC H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawC_mem H uni D g populations hsupp
  have bc1 : (runMeasure μ D Dsf).real Ec1 ≤ 2 * (((M : ℝ) + 1) * Real.exp (-L)) := by
    refine runMeasure_coins_le D Dsf _ (measurableSet_coinSplitBad O H K M L j)
      (by positivity) (fun x => ?_)
    have h := coins_family_balance ((poolAt M x) ×ˢ (Finset.univ : Finset Bool))
      (fun vb => (winAt H K j x).filter (fun p => O.flip vb.1 p = 1
        ∧ O.label p = sideClass vb.2)) hL
    have hsub : {g : S → Bool | ((x, g) : Draws Ω S J) ∈ Ec1}
        ⊆ {g | ∃ vb ∈ (poolAt M x) ×ˢ (Finset.univ : Finset Bool),
          Real.sqrt (2 * (((winAt H K j x).filter (fun p => O.flip vb.1 p = 1
              ∧ O.label p = sideClass vb.2)).card : ℝ) * L)
            < ((((winAt H K j x).filter (fun p => O.flip vb.1 p = 1
                ∧ O.label p = sideClass vb.2)).filter (fun s => g s = false)).card : ℝ)
              - ((((winAt H K j x).filter (fun p => O.flip vb.1 p = 1
                ∧ O.label p = sideClass vb.2)).filter (fun s => g s = true)).card : ℝ)} := by
      rintro g ⟨v, hv, b, hb⟩
      exact ⟨(v, b), Finset.mem_product.2 ⟨hv, Finset.mem_univ b⟩, hb⟩
    refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) (le_trans h ?_)
    rw [Finset.card_product, Finset.card_univ, Fintype.card_bool]
    have : ((poolAt M x).card : ℝ) ≤ (M : ℝ) + 1 := by exact_mod_cast card_poolAt_le M x
    push_cast
    nlinarith [Real.exp_nonneg (-L)]
  have bc2 : (runMeasure μ D Dsf).real Ec2 ≤ 2 * Real.exp (-((G : ℝ) ^ 2 / N)) := by
    refine runMeasure_coins_le D Dsf _ (measurableSet_coinCardBad H K G j) (by positivity)
      (fun x => ?_)
    exact coins_window_card_le (winAt H K j x) N G hNpos
      (le_trans (window_card_le _ K _) (by omega))
  have bt : (runMeasure μ D Dsf).real (Y ⁻¹' Et)
      ≤ rejTail (N + r) (((N + r : ℕ) : ℝ) * ρ) r :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_notTabGood uni N r) (fun g =>
      measureReal_tabBad_le _ _ Dsf uni (D uni) (sideLawT_uni H uni D g) N r ρ hρu)
  have bh : (runMeasure μ D Dsf).real Eh
      ≤ rejTail (N + r) (((2 * N : ℕ) : ℝ) * ρ) (⌈θ * N⌉₊ - 1) := by
    refine runMeasure_base_le D Dsf _ (measurableSet_hitsBad H K (N + r) (θ * N) uni j) ?_
    refine le_trans (measureReal_hits_le D Dsf uni j hju hρu _ K (N + r) (by positivity)) ?_
    have hKN : ((K : ℕ) : ℝ) ≤ ((2 * N : ℕ) : ℝ) := by
      have : K ≤ 2 * N := by omega
      exact_mod_cast this
    exact rejTail_mono (by positivity) le_rfl (mul_le_mul_of_nonneg_right hKN hρ0) _
  have bl : (runMeasure μ D Dsf).real (Y ⁻¹' El)
      ≤ Real.exp (-((N : ℝ) * relRate l E)) + Real.exp (-((N : ℝ) * relRate lmis E)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_lightMiss O populations j B l
      lmis (fracAt O η₀) kk kk) (fun g => measureReal_lightMiss_le hflat O populations _ _
        Dsf (hsT g) (hsC g) hsuppSf j hj B l lmis E (fracAt O η₀) kk kk hE hEl hElm hdec hcut)
  have bd : (runMeasure μ D Dsf).real (Y ⁻¹' Ed)
      ≤ 2 * (((M : ℝ) + 3) * Real.exp (-2 * t ^ 2 / N)) :=
    runMeasure_streams_le H uni D Dsf _ (measurableSet_sidesDead O populations j B N t)
      (fun g => measureReal_sidesDead_le hflat O populations _ _ Dsf (hsT g) (hsC g) hsuppSf j hj
        B N ht)
  set ν := runMeasure μ D Dsf with hν
  have u : ∀ A B' : Set (Draws Ω S J), ν.real (A ∪ B') ≤ ν.real A + ν.real B' :=
    fun A B' => measureReal_union_le A B'
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have := u Ec1 Ec2
  have := u (Ec1 ∪ Ec2) (Y ⁻¹' Et)
  have := u (Ec1 ∪ Ec2 ∪ Y ⁻¹' Et) Eh
  have := u (Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh) (Y ⁻¹' El)
  have := u (Ec1 ∪ Ec2 ∪ Y ⁻¹' Et ∪ Eh ∪ Y ⁻¹' El) (Y ⁻¹' Ed)
  unfold namedFail
  linarith

open scoped Classical in
set_option maxHeartbeats 4000000 in
/-- The loop returns: except with chance `δ/2`, the top rung passes at every population, wild
or tame, its gate read at the first look.  The count rule's table event and the pool's are charged
once; every population's own failures are priced off them. -/
theorem raw_terminates {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (uni : J) (huni : uni ∈ populations) (H : J → Finset S)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (εcov α δ ρ pAP crossLimit : ℝ)
    (hsig : O.η < 1 / 2) (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hεs : 0 < εs) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρu : ∀ t, (D uni).real {t} ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
      (allHarvest H).card)
    (hρsf : collisionMass Dsf
        ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State // B ∈ stoppable η₀ populations
        indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card ρ},
      streams H uni x ∉ retAll nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
        B.val}
      ≤ δ / 2 := by
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρu 1)
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hsigle : O.η ≤ 1 / 2 := hsig.le
  obtain ⟨B, hB, hpass⟩ := exists_passable (crossLimit := crossLimit) (hsz := (allHarvest H).card)
    O populations Dsf indecisionLimit εcov α δ ρ (collisionMass Dsf) pAP hsig hpop hηle hη₀
    hεcov hε1 hεs hδ hδ1 hαpos hα hindLim hind1 hpAPPositive hpAPBound hρ0 le_rfl hρsf0 hρcap
    hρsf
  obtain ⟨tap, q, E, Δ, g, u, x, a, e1, e2, lF, lG, lmis, c, t, L, θ, r, rs, G, hmpos, hkpos,
    hindLim', hind1', htap, hΔ, hg, hpAP0, hlmis, hc0, hr, hpAPBound', hρsf', hUF, hUG, hL, hθ,
    hN2, hszG, hNF, hNG, hlF, hscreen, hu, hx, hsignal, ha, hκa, hmass, hE1, hE2,
    hcount, hpool, hq0, hq1, hqp, hhiUp, hloUp, hhiLo, hloLo, hhiκ, hlohi, ht, hgate, hE0,
    hElm, hEl, hEa, hEb, hEc, hEd, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  set κ : ℕ := B.k - 1 with hκ
  have hdec : ∀ (F : Finset S) (p : S), (1 : S) ∉ F → flipCount O F p ≤ (F.card : ℝ) * fracAt O η₀ p →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p ω} ≤ E := by
    intro F p h1 hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    have hcardI : (insert 1 F).card = B.k := by
      rw [Finset.card_insert_of_notMem h1, hcardF, hκ]
      omega
    have hcle : (F.card : ℝ) ≤ ((insert 1 F).card : ℝ) := by
      exact_mod_cast Finset.card_le_card (Finset.subset_insert _ _)
    have hflI : flipCount O (insert 1 F) p ≤ ((insert 1 F).card : ℝ) * fracAt O η₀ p :=
      le_trans (flipCount_insert_one O F p)
        (le_trans hfl (mul_le_mul_of_nonneg_right hcle (hqp p).1))
    refine decided_exact O (insert 1 F) p B.lo (B.hi + 1) (fracAt O η₀ p) q E hsigle hflI hq0 hq1
      (hqp p).2.1 (hqp p).2.2 ?_ ?_ ?_ ?_ ?_ <;> rw [hcardI]
    · push_cast; exact hhiUp
    · exact hloUp
    · omega
    · exact hEa
    · exact hEb
  have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * fracAt O η₀ p →
      κ ≤ F.card → F.card ≤ κ →
      μ.real {ω | ¬ cutCorrect O B.lo B.hi F p ω} ≤ E := by
    intro F p hfl hmin hmax
    have hcardF : F.card = κ := le_antisymm hmax hmin
    refine cutCorrect_exact O F p B.lo B.hi (fracAt O η₀ p) q E hsigle hfl hq0 hq1 (hqp p).2.1
      (hqp p).2.2 ?_ ?_ ?_ ?_ ?_ <;> rw [hcardF]
    · exact hhiLo
    · exact hloLo
    · exact hhiκ
    · exact hEc
    · exact hEd
  set Y := streams (Ω := Ω) H uni with hY
  set ν := runMeasure μ D Dsf with hν
  have hsT : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawT H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawT_mem H uni D g populations hsupp hH
  have hsC : ∀ g, ∀ j ∈ populations, ∀ᵐ y ∂(sideLawC H uni D g j), ∀ i, y i ∈ Pre :=
    fun g => ae_sideLawC_mem H uni D g populations hsupp
  -- the count rule's table event, once
  set N : Set (Run Ω S J) := nearShift O populations B Δ a with hNdef
  have hNm : MeasurableSet N := measurableSet_nearShift O populations B Δ a
  have hNb : ν.real (Y ⁻¹' N) ≤ e1 + e2 := by
    refine runMeasure_streams_le H uni D Dsf N hNm (fun g' => ?_)
    have h := measureReal_nearShift_le hflat O populations (sideLawT H uni D g')
      (sideLawC H uni D g') Dsf (hsT g') hsuppSf hpop B hsigle
      hkpos hmpos Δ (15 / 16) u x a hΔ (by norm_num) (by norm_num) hu hx hsignal hκa
    linarith
  -- the uniform pool
  set Dt : Set (Run Ω S J) := {x | ¬ ∑ v ∈ clusterBy nearestClusterer O.mq populations x B,
      wMass O (D uni) (clsW O true) v ≤ ((clusterBy nearestClusterer O.mq populations x B).card : ℝ)
        * (Δ + c * clsShift O true + g)} with hDt
  set Df : Set (Run Ω S J) := {x | ¬ ∑ v ∈ clusterBy nearestClusterer O.mq populations x B,
      wMass O (D uni) (clsW O false) v
        ≤ ((clusterBy nearestClusterer O.mq populations x B).card : ℝ)
          * (Δ + c * clsShift O false + g)} with hDf
  set Ht : Set (Run Ω S J) := heavyHits nearestClusterer O populations (D uni) uni B
    (clsBadP O η₀ true) (clsQ O η₀ Δ c g true) (clsQ O η₀ Δ c g true / 32) with hHt
  set Hf : Set (Run Ω S J) := heavyHits nearestClusterer O populations (D uni) uni B
    (clsBadP O η₀ false) (clsQ O η₀ Δ c g false) (clsQ O η₀ Δ c g false / 32) with hHf
  set U : Set (Run Ω S J) := {x | ¬ certGood populations uni B.npref r x}
      ∪ stalled nearestClusterer O populations B κ κ
      ∪ ((Dt \ N) ∪ (Df \ N)) ∪ (Ht ∪ Hf)
      ∪ lightMiss nearestClusterer O populations uni B l lmis (fracAt O η₀) κ κ
      ∪ sidesDead nearestClusterer O populations uni B B.npref t
      ∪ {x | ¬ poolGood B.nsuff rs x} with hU
  have hUm : MeasurableSet U :=
    ((((((measurableSet_notCertGood populations uni B.npref r).union
      (measurableSet_stalled O populations B κ κ)).union
      (((measurableSet_dirty O populations B (D uni) (clsW O true) _).diff hNm).union
        ((measurableSet_dirty O populations B (D uni) (clsW O false) _).diff hNm))).union
      ((measurableSet_heavyHits O populations (D uni) uni B _ _ _).union
        (measurableSet_heavyHits O populations (D uni) uni B _ _ _))).union
      (measurableSet_lightMiss O populations uni B l lmis (fracAt O η₀) κ κ)).union
      (measurableSet_sidesDead O populations uni B B.npref t)).union
      (measurableSet_notPoolGood B.nsuff rs)
  have hsubU : {x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B uni}
        \ Y ⁻¹' N ⊆ Y ⁻¹' U := by
    rintro x ⟨hx, hxN⟩
    have hx' : Y x ∉ retAt nearestClusterer O populations uni η₀ indecisionLimit εcov α B uni :=
      fun h => hx (retAt_subset_retAtK O populations uni η₀ indecisionLimit εcov α B uni h)
    have h := notRetAt_subset (rule := nearestClusterer) O populations uni huni uni (D uni) B
      hmpos hηle hη₀ εcov α hαpos (by linarith) l lF lG lmis Δ c g t r rs κ κ
      (by omega) hlF hlmis hΔ.le hc0 hg.le hr hUF hUG hlohi ht (by omega) hpool hgate hx'
    simp only [Set.mem_union, Set.mem_preimage] at h ⊢
    rcases h with ((((((h | h) | (h | h)) | (h | h)) | h) | h) | h)
    · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl h)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr h)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (Or.inl ⟨h, hxN⟩)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (Or.inr ⟨h, hxN⟩)))))
    · exact Or.inl (Or.inl (Or.inl (Or.inr (Or.inl h))))
    · exact Or.inl (Or.inl (Or.inl (Or.inr (Or.inr h))))
    · exact Or.inl (Or.inl (Or.inr h))
    · exact Or.inl (Or.inr h)
    · exact Or.inr h
  have huniB : ν.real ({x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit
        εcov α B uni} \ Y ⁻¹' N)
      ≤ roundFail populations l lmis (g / 1056) E L tap pAP ρ (collisionMass Dsf) t r rs B := by
    refine le_trans (measureReal_mono hsubU (measure_ne_top _ _)) ?_
    refine runMeasure_streams_le H uni D Dsf U hUm (fun g' => ?_)
    set Pp := sideLawT H uni D g'
    set Pc := sideLawC H uni D g'
    have hcertB : (streamMeasure μ Pp Pc Dsf).real {x | ¬ certGood populations uni B.npref r x}
        ≤ rejTail (2 * B.npref + r)
          (((populations.card * B.npref + (2 * B.npref + r) : ℕ) : ℝ) * ρ) r :=
      le_trans (measureReal_mono (fun x hx h2 => hx (certGood_of_certGood2 populations uni
          B.npref r x h2)) (measure_ne_top _ _))
        (measureReal_certBad2_le Pp Pc Dsf populations uni (D uni) (sideLawC_uni H uni D g')
          B.npref r ρ hρu)
    have hstallB := measureReal_stalled_nearest_le O populations Pp Pc Dsf B (by omega) hscreen
      hpool.le (collisionMass Dsf) rs hρsf' hρsf0
    have hdirty : ∀ b, (streamMeasure μ Pp Pc Dsf).real ({x : Run Ω S J |
        ¬ ∑ v ∈ clusterBy nearestClusterer O.mq populations x B,
        wMass O (D uni) (clsW O b) v ≤ ((clusterBy nearestClusterer O.mq populations x B).card : ℝ)
          * (Δ + c * clsShift O b + g)} \ N)
        ≤ rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
          + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * (collisionMass Dsf) / (2 * ((rs : ℝ) + 1))
          + Real.exp (-((B.nsuff : ℝ) * (tap ^ 2 / (2 * pAP))))
          + (B.nsuff : ℝ) * Real.exp (-L) := by
      intro b
      have hw0 := clsShift_pos O hηle hη₀ b
      have hw1 := clsShift_le_one O hηle hη₀ b
      have hcb : (r : ℝ) * clsShift O b ≤ c * clsShift O b * B.npref := by
        have := mul_le_mul_of_nonneg_right hr hw0.le
        linarith
      exact measureReal_famShift_le O populations Pp Pc Dsf uni huni (D uni)
        (sideLawT_uni H uni D g') B hscreen hkpos hmpos (clsW O b) (clsShift O b)
        (clsW_nonneg O hηle hη₀ b) (clsW_le O hηle hη₀ b) hw0 hw1
        (fun v p => clsW_mul_flip_le O hsigle b v p) Δ (c * clsShift O b) g L tap a pAP ρ
        (collisionMass Dsf) r rs hΔ (by positivity) hcb hg hL htap ha hκa hpAP0 hpAPBound'
        (by nlinarith [mul_le_mul_of_nonneg_left hw1 hc0]) hρu hρsf' hρsf0
    have hDB : (streamMeasure μ Pp Pc Dsf).real ((Dt \ N) ∪ (Df \ N))
        ≤ 2 * (rejTail (B.npref + r) (((B.npref + r : ℕ) : ℝ) * ρ) r
          + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * (collisionMass Dsf) / (2 * ((rs : ℝ) + 1))
          + Real.exp (-((B.nsuff : ℝ) * (tap ^ 2 / (2 * pAP))))
          + (B.nsuff : ℝ) * Real.exp (-L)) := by
      refine le_trans (measureReal_union_le _ _) ?_
      have := hdirty true
      have := hdirty false
      linarith
    have hQ : ∀ b, 2 * g ≤ clsQ O η₀ Δ c g b := by
      intro b
      have hσ := clsSig_pos O hηle hη₀ b
      have hσ1 := clsSig_le O hηle hη₀ b
      have hw0 := (clsShift_pos O hηle hη₀ b).le
      unfold clsQ
      rw [le_div_iff₀ hσ]
      nlinarith [mul_nonneg hc0 hw0]
    have hheavy : ∀ b, (streamMeasure μ Pp Pc Dsf).real (heavyHits nearestClusterer O populations
        (D uni) uni B (clsBadP O η₀ b) (clsQ O η₀ Δ c g b) (clsQ O η₀ Δ c g b / 32))
        ≤ Real.exp (-((B.npref : ℝ) * (g / 1056))) := by
      intro b
      have hq0 : 0 ≤ clsQ O η₀ Δ c g b := le_trans (by linarith) (hQ b)
      refine le_trans (measureReal_heavyHits_le Pp Pc Dsf O populations uni (D uni)
        (sideLawC_uni H uni D g') B _ _ _ (by positivity)) (Real.exp_le_exp.2 (neg_le_neg
          (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))))
      have e : relRate (clsQ O η₀ Δ c g b + clsQ O η₀ Δ c g b / 32) (clsQ O η₀ Δ c g b)
          = clsQ O η₀ Δ c g b / 2112 := by
        unfold relRate
        rcases eq_or_lt_of_le hq0 with h0 | hpos
        · rw [← h0]; simp
        · field_simp; ring
      rw [e]
      linarith [hQ b]
    have hHB : (streamMeasure μ Pp Pc Dsf).real (Ht ∪ Hf)
        ≤ 2 * Real.exp (-((B.npref : ℝ) * (g / 1056))) := by
      refine le_trans (measureReal_union_le _ _) ?_
      have := hheavy true
      have := hheavy false
      linarith
    refine le_trans (measureReal_failUnion_le hflat O populations Pp Pc Dsf (hsT g') (hsC g')
      hsuppSf uni huni B l lmis E t (collisionMass Dsf) (fracAt O η₀) r rs κ κ hE0
      (by rw [hl]; exact hEl) hElm ht hρsf' hρsf0 _ hcertB _ hstallB _ _ hDB _ _ hHB hdec hcut) ?_
    unfold roundFail
    linarith
  -- the pool, once
  set P : Set (Run Ω S J) := poolEvents O populations B pAP tap rs with hPdef
  have hPb : ν.real (Y ⁻¹' P) ≤ poolFail tap pAP (collisionMass Dsf) rs B :=
    measureReal_poolEvents_le O populations uni H D Dsf B hkpos hpool hscreen pAP tap
      (collisionMass Dsf) rs htap hpAP0 hpAPBound' hρsf' hρsf0
  -- the named populations
  have hnamedB : ∀ j ∈ populations.erase uni,
      ν.real (({x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B j}
          \ Y ⁻¹' N) \ Y ⁻¹' P)
        ≤ namedFail l lmis E ρ t L θ r G B := fun j hj =>
    measureReal_named_le hflat O populations uni huni H D Dsf hsupp hH hsuppSf j
      (Finset.mem_of_mem_erase hj) (Finset.ne_of_mem_erase hj) B hN2
      hηle hη₀ εcov α hαpos (by linarith) l lF lG lmis E Δ t L θ tap pAP a ρ
      r rs G hE0 (by rw [hl]; exact hEl) hElm hlF hlmis
      hΔ hL hθ hlohi ht hkpos hpool hgate hszG hNF hNG hscreen ha hρu hdec hcut
  -- every population
  have hNF0 : 0 ≤ namedFail l lmis E ρ t L θ r G B := namedFail_nonneg _ _ _ ρ t L θ r G B hρ0
  have hsub : {x : Draws Ω S J | ∀ B' : {B : State // B ∈ stoppable η₀ populations
          indecisionLimit εcov εs δ α pAP crossLimit (allHarvest H).card ρ},
        Y x ∉ retAll nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α B'.val}
      ⊆ Y ⁻¹' N ∪ Y ⁻¹' P
        ∪ ({x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B uni}
          \ Y ⁻¹' N)
        ∪ ⋃ j ∈ populations.erase uni,
          (({x | Y x ∉ retAtK nearestClusterer O populations uni η₀ indecisionLimit εcov α B j}
            \ Y ⁻¹' N) \ Y ⁻¹' P) := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, not_or] at hc
    obtain ⟨⟨⟨hxN, hxP⟩, hxu⟩, hxj⟩ := hc
    refine hx ⟨B, hB⟩ (mem_retAll_of_retAtK O populations hpop uni η₀ indecisionLimit εcov α B
      (Y x) (fun j hj => by_contra fun h => ?_))
    by_cases hju : j = uni
    · rw [hju] at h
      exact hxu ⟨h, hxN⟩
    · exact hxj (Set.mem_iUnion₂.2 ⟨j, Finset.mem_erase.2 ⟨hju, hj⟩, ⟨h, hxN⟩, hxP⟩)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  refine le_trans (add_le_add (measureReal_union_le _ _) (measureReal_biUnion_finset_le _ _)) ?_
  have hNP := measureReal_union_le (μ := ν) (Y ⁻¹' N) (Y ⁻¹' P)
  have hsum := Finset.sum_le_sum hnamedB
  rw [Finset.sum_const, nsmul_eq_mul] at hsum
  have hcardle : ((populations.erase uni).card : ℝ) ≤ populations.card := by
    exact_mod_cast Finset.card_erase_le
  have := mul_le_mul_of_nonneg_right hcardle hNF0
  linarith

end OrthoDFA
