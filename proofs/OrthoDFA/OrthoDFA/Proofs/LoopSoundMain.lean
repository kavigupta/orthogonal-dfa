import OrthoDFA.LoopSound
import OrthoDFA.Proofs.LoopSound
import OrthoDFA.Proofs.UniformStrings

/-!
# The learner's rate test certifies its DFA: the proof

The schedule is the clustering's at gate level and per-round share `δc`, and the pool's
class-preserving share `pAP/2`, which the suffixes off what the earlier rounds read keep. Its
counts are at most `PB` suffixes and `RB` prefixes per state (`counts_le`), so a round reads at
most `6 J ⌊RB⌋` prefixes and `⌊PB⌋ + 1` suffixes, and `loop_sound_core` applies with the uniform
strings' atoms `|Σ|^(−L)` and `|Σ|^(−Ls)`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

theorem learnerLoopSoundSimple_holds : LearnerLoopSoundSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L R η₀ ε εcov ι δ pAP A Ls a J δc s τ PB RB NP hL hη hη₀ hε0 hε1
    hc0 hc1 hι0 hι1 hδ0 hδ1 hpAP hR hpAPu h1 h2 h3 h4 y budget
  classical
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hA0 : (0 : ℝ) < A := by linarith
  have hR1 : (1 : ℝ) ≤ R := by exact_mod_cast hR
  have hR0 : (0 : ℝ) < R := by linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hpAP1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hδc0 : 0 < δc := by show 0 < δ / (4 * R); positivity
  have hδc1 : δc ≤ 1 / 4 := by
    show δ / (4 * R) ≤ 1 / 4
    rw [div_le_iff₀ (by positivity)]; nlinarith
  have hτe : rateScale η₀ ι εcov = τ := rfl
  have hτb : budgetScale η₀ ι εcov = τ := by
    show min (min εcov (1 / 2) * gateA η₀ ^ 2) ι = min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    rw [min_eq_left hc1]
    rfl
  have hτ0 : 0 < τ := by
    have ha : 0 < (1 - 2 * η₀) / (1 - η₀) := gateA_pos hη0 hη₀
    exact lt_min (mul_pos hc0 (pow_pos ha 2)) hι0
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  -- the clustering's schedule
  obtain ⟨schedule₀, hband₀, hsize₀, hsum₀, -, huni₀⟩ := clustering_schedule_exists O R
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι δc 1 εcov δc 1 (pAP / 2) hη hη₀ (flat_of_length Ls) (by positivity) hι0 hι1 hδc0
    (by linarith) one_pos hc0 hc1 hδc0 (by linarith) one_pos
  have h14 : (1 : ℝ) ^ 4 = Real.exp (-0) := by simp
  rw [h14] at hsize₀ hsum₀
  set schedule : Finset (Option (Fin R)) → Finset State := fun pops =>
    if none ∈ pops then schedule₀ pops else ∅ with hschedd
  set C : Level3Settings := ⟨0, 4 * δ / R, y, ε, budget⟩ with hCd
  set settings : LoopSettings R := ⟨schedule, η₀, ι, εcov, δc, C, a⟩ with hset
  refine ⟨settings, rfl, rfl, rfl, rfl, ?_⟩
  -- the counts
  have hJ : ∀ pops : Finset (Option (Fin R)), (pops.card : ℝ) ≤ J := fun pops => by
    have h := Finset.card_le_univ pops
    simp only [Fintype.card_option, Fintype.card_fin] at h
    show (pops.card : ℝ) ≤ R + 1
    exact_mod_cast h
  have hPBe : capPB η₀ ι εcov δc (pAP / 2) 0 J = PB := by
    unfold capPB
    rw [hτb]
    show _ = 256 * Real.log (2 / τ) / ((1 / 2 - η₀) ^ 2 * pAP)
      + 64 * Real.log ((J + 2) / (δc / 2)) / pAP ^ 2
    have : (1 / 2 - η₀) ≠ 0 := by linarith
    field_simp
    ring
  have hRBe : capRB η₀ ι εcov δc δc (pAP / 2) 0 J = RB := by
    unfold capRB
    rw [hPBe, hτb]
  have hcounts : ∀ pops : Finset (Option (Fin R)), none ∈ pops →
      (poolCount η₀ pops ι εcov (δc / 2) (pAP / 2) (Real.exp (-0)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι εcov (δc / 2) δc (pAP / 2) (Real.exp (-0)) 0 : ℝ) ≤ RB := by
    intro pops hpops
    have h := counts_le η₀ pops ι εcov δc δc (pAP / 2) 0 J hη0 hη₀ ⟨none, hpops⟩ (hJ pops) hι0 hc0
      (by linarith) hδc0 (by linarith) hδc0 (by linarith) (by positivity) (by linarith) le_rfl
    rw [hPBe, hRBe] at h
    exact h
  obtain ⟨hPB0', hRB0'⟩ := hcounts {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  set NPr : ℕ := 6 * (R + 1) * ⌊RB⌋₊ with hNPr
  set NSr : ℕ := ⌊PB⌋₊ + 1 with hNSr
  have hpref : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin R))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ NPr := by
    intro pops x
    refine (card_roundPrefs_le _ _ _).trans ?_
    have hpc : pops.card ≤ R + 1 := by
      have h := Finset.card_le_univ pops
      simpa using h
    have hsum : ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * ⌊RB⌋₊ := by
      show ∑ B ∈ schedule pops, B.npref ≤ _
      simp only [hschedd]
      split_ifs with hp
      · exact (hsum₀ pops).trans (Nat.mul_le_mul_left 2 (Nat.le_floor (hcounts pops hp).2))
      · simp
    calc 3 * pops.card * ∑ B ∈ settings.schedule pops, B.npref
        ≤ 3 * (R + 1) * (2 * ⌊RB⌋₊) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpc) hsum
      _ = NPr := by rw [hNPr]; ring
  have hsuff : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin R))),
      (roundSuffs (settings.schedule pops) x).card ≤ NSr := by
    intro pops x
    show (roundSuffs (schedule pops) x).card ≤ NSr
    simp only [hschedd]
    split_ifs with hp
    · refine (card_roundSuffs_le _ (fun B hB => (hsize₀ pops B hB).1) x).trans ?_
      exact Nat.succ_le_succ (Nat.le_floor (hcounts pops hp).1)
    · simp [roundSuffs]
  -- the uniform strings' atoms
  set capU : ℝ := 1 / A ^ L
  set capSf : ℝ := 1 / A ^ Ls
  have hcapU0 : 0 ≤ capU := by positivity
  have hcapS0 : 0 ≤ capSf := by positivity
  have hAa : A ^ a ≤ A ^ L := pow_le_pow_right₀ hA1 (by omega)
  have hALs : A ^ L ≤ A ^ Ls := pow_le_pow_right₀ hA1 (by omega)
  have hNPr : (NPr : ℝ) ≤ NP := by
    show ((6 * (R + 1) * ⌊RB⌋₊ : ℕ) : ℝ) ≤ 6 * J * RB
    push_cast
    have := Nat.floor_le hRB0
    show 6 * ((R : ℝ) + 1) * (⌊RB⌋₊ : ℝ) ≤ 6 * ((R : ℝ) + 1) * RB
    gcongr
  have hNSrr : (NSr : ℝ) ≤ PB + 1 := by
    show ((⌊PB⌋₊ + 1 : ℕ) : ℝ) ≤ PB + 1
    push_cast
    linarith [Nat.floor_le hPB0]
  set m : ℝ := R * NPr * capU
  set ms : ℝ := R * NSr * capSf
  set mA : ℝ := NPr * (1 / A ^ a)
  have hmmA : mA + m ≤ ε / 96 := by
    have hNP0 : (0 : ℝ) ≤ NPr := Nat.cast_nonneg _
    have e1 : m ≤ R * NPr * (1 / A ^ a) := by
      show R * NPr * (1 / A ^ L) ≤ _
      gcongr
    have e2 : mA + R * NPr * (1 / A ^ a) = NPr * (R + 1) / A ^ a := by
      show NPr * (1 / A ^ a) + _ = _
      field_simp
      ring
    have e3 : (NPr : ℝ) * (R + 1) / A ^ a ≤ NP * (R + 1) / A ^ a := by gcongr
    have e4 : NP * (R + 1) / A ^ a ≤ ε / 96 := by
      rw [div_le_iff₀ (by positivity)]
      rw [div_le_iff₀ (by positivity)] at h4
      linarith
    linarith
  have hm0 : 0 ≤ m := by positivity
  have hmA0 : 0 ≤ mA := by positivity
  have hms : ms ≤ pAP / 2 := by
    show R * NSr * (1 / A ^ Ls) ≤ pAP / 2
    have : (R : ℝ) * NSr ≤ R * (PB + 1) := by gcongr
    calc (R : ℝ) * NSr * (1 / A ^ Ls) ≤ R * (PB + 1) * (1 / A ^ Ls) := by gcongr
      _ = 2 * R * (PB + 1) / A ^ Ls / 2 := by field_simp
      _ ≤ pAP / 2 := by linarith
  -- the probes and the suffixes
  let P : PrefixDistribution α := ⟨uniformStrings L, inferInstance, L, by
    rw [ae_iff]
    refine measure_mono_null (fun x hx => ?_) (uniformStrings_length L)
    simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
    omega⟩
  have hDat : ∀ t, P.D.real {t} ≤ capU := fun t => uniformStrings_singleton_le L t
  have hsfat : ∀ t, (uniformStrings (α := α) Ls).real {t} ≤ capSf := fun t =>
    uniformStrings_singleton_le Ls t
  have hmA : (NPr : ℝ) * prefixMax P.D settings.avoidDepth ≤ mA :=
    mul_le_mul_of_nonneg_left (prefixMax_uniformStrings_le (by show a ≤ L; omega))
      (Nat.cast_nonneg _)
  have huni : ∀ pops : Finset (Option (Fin R)), none ∈ pops →
      ∀ (Lw : Option (Fin R) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (Lw j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, Lw j {p | Ls ≤ p.toList.length} = 0) →
        (∀ t, (Lw none).real {t} ≤ 2 * capU) →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * capSf) →
        (runMeasure μ Lw Lsf).real
          {x | ¬ UniSpec O pops (settings.schedule pops) settings.η₀ settings.indecisionLimit
            settings.εcov settings.gateLevel (Lw none) (streams (fun _ => ∅) none x)}
          ≤ δc + δc := by
    intro pops hpops Lw Lsf hLw hLsf hLs hLat hLsfLen hpAPL hLsfat
    have hs : settings.schedule pops = schedule₀ pops := if_pos hpops
    rw [hs]
    refine huni₀ pops hpops Lw Lsf hLw hLsf (fun j hj => ?_) ?_ (by linarith) (fun B hB t => ?_)
      (fun B hB => ?_)
    · refine measure_mono_null (fun p hp => ?_) (hLs j hj)
      simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hp ⊢
      exact hp
    · refine measure_mono_null (fun v hv => ?_) hLsfLen
      simpa using hv
    · have hnp : (B.npref : ℝ) ≤ RB :=
        (Nat.cast_le.2 (hsize₀ pops B hB).2).trans (hcounts pops hpops).2
      have hpc : ((pops.card : ℝ) + 1) ^ 2 ≤ (J + 1) ^ 2 := by
        have := hJ pops
        gcongr
      have hat := hLat t
      have hat0 : 0 ≤ (Lw none).real {t} := measureReal_nonneg
      rw [hτe]
      calc 2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (Lw none).real {t}
          ≤ 2 ^ 21 * (J + 1) ^ 2 * RB * (2 * capU) := by gcongr
        _ = 2 ^ 22 * (J + 1) ^ 2 * RB / A ^ L := by
            simp only [capU]
            field_simp
        _ ≤ δc * τ ^ 2 := h1
    · have hns : (B.nsuff : ℝ) ≤ PB := by
        rw [(hsize₀ pops B hB).1]
        exact (hcounts pops hpops).1
      have hpc : (pops.card : ℝ) + 1 ≤ J + 1 := by linarith [hJ pops]
      have hcol : collisionMass Lsf ≤ 2 * capSf := collisionMass_le_of_atoms Lsf hLsfat
      have hcol0 : 0 ≤ collisionMass Lsf := tsum_nonneg fun t => sq_nonneg _
      calc 2 ^ 13 * ((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Lsf
          ≤ 2 ^ 13 * (J + 1) * PB * (2 * capSf) := by gcongr
        _ = 2 ^ 15 * (J + 1) * PB / A ^ Ls / 2 := by
            simp only [capSf]
            field_simp
        _ ≤ δc * pAP / 2 := by linarith
        _ = δc * (pAP / 2) := by ring
  have hcore := loop_sound_core settings O P (uniformStrings Ls) M hL Ls (L + budget + 2) NPr NSr
    (δc + δc) capU capSf m ms mA pAP le_rfl (by show L < L + Fintype.card σ + 3; omega)
    (uniformStrings_length Ls) hpAPu hDat hsfat
    (fun pops B hB => by
      show B.lo ≤ B.hi
      have hB' : B ∈ schedule pops := hB
      simp only [hschedd] at hB'
      split_ifs at hB'
      · exact (hband₀ pops B hB').1
      · simp at hB')
    hpref hsuff rfl (by show 0 < 4 * δ / R; positivity) hε0 (by positivity) le_rfl
    (by linarith) le_rfl (by linarith) hmA (by linarith) huni
  refine (measureReal_mono (fun p hp => ?_) (measure_ne_top _ _)).trans (hcore.trans (le_of_eq ?_))
  · obtain ⟨o, ho, hco, hbad⟩ := hp
    refine ⟨o, ho, hco, fun ⟨s', hs', hacc⟩ => hbad ⟨s', hs', hacc.trans ?_⟩⟩
    show ε + mA + 513 / 512 * εcov + 2 * ι + m ≤ ε + ε / 96 + 513 / 512 * εcov + 2 * ι
    linarith
  · show (R : ℝ) * (δc + δc + 4 * δ / R / 8) = δ
    simp only [δc]
    field_simp
    ring

theorem learnerLoopSound_holds : LearnerLoopSound := by
  intro α _ _ _ σ _ Ω _ μ _ M O L R η₀ ε εcov ι δ pAP A Ls a J δc s τ PB RB NP E L₀ hL hη hη₀
    hε0 hε1 hc0 hc1 hι0 hι1 hδ0 hδ1 hpAP hR hA2 hpAPu hL₀
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hlogA : 0 < Real.log A := Real.log_pos (by linarith)
  have hE1 : 1 ≤ E := le_max_left _ _
  have hE0 : 0 < E := by linarith
  -- `|Σ|^a` clears `E`
  have hEa : E ≤ A ^ a := by
    have h2a : (L : ℝ) ≤ 2 * a := by
      have : L ≤ 2 * a := by show L ≤ 2 * ((L + 1) / 2); omega
      exact_mod_cast this
    have hlog : Real.log E ≤ a * Real.log A := by
      have := hL₀
      simp only [L₀] at this
      rw [div_le_iff₀ hlogA] at this
      nlinarith
    calc E = Real.exp (Real.log E) := (Real.exp_log hE0).symm
      _ ≤ Real.exp (a * Real.log A) := Real.exp_le_exp.2 hlog
      _ = A ^ a := by rw [Real.exp_nat_mul, Real.exp_log (by linarith)]
  have hA1 : (1 : ℝ) ≤ A := by linarith
  have hAa0 : 0 < A ^ a := by positivity
  have hAL : A ^ a ≤ A ^ L := pow_le_pow_right₀ hA1 (by show (L + 1) / 2 ≤ L; omega)
  have hALs : A ^ L ≤ A ^ Ls := pow_le_pow_right₀ hA1 (by show L ≤ L + Fintype.card σ + 3; omega)
  have hR1 : (1 : ℝ) ≤ R := by exact_mod_cast hR
  have hR0 : (0 : ℝ) < R := by linarith
  have hδc0 : 0 < δc := by show 0 < δ / (4 * R); positivity
  have hτ0 : 0 < τ := by
    have ha : 0 < (1 - 2 * η₀) / (1 - η₀) := gateA_pos (le_trans (eta_nonneg O) hη) hη₀
    exact lt_min (mul_pos hc0 (pow_pos ha 2)) hι0
  have hle : ∀ {t : ℝ}, t ≤ E → ∀ {n : ℕ}, A ^ a ≤ A ^ n → t ≤ A ^ n := fun ht _ hn =>
    ht.trans (hEa.trans hn)
  have hm1 := le_max_right (1 : ℝ) (max (max (2 ^ 22 * (J + 1) ^ 2 * RB / (δc * τ ^ 2))
      (2 ^ 15 * (J + 1) * PB / (δc * pAP)))
    (max (2 * R * (PB + 1) / pAP) (96 * NP * (R + 1) / ε)))
  refine learnerLoopSoundSimple_holds M O L R η₀ ε εcov ι δ pAP hL hη hη₀ hε0 hε1 hc0 hc1 hι0 hι1
    hδ0 hδ1 hpAP hR hpAPu ?_ ?_ ?_ ?_
  · have h := hle (((le_max_left _ _).trans (le_max_left _ _)).trans hm1) hAL
    rw [div_le_iff₀ (by positivity)] at h
    rw [div_le_iff₀ (by positivity)]
    show 2 ^ 22 * (J + 1) ^ 2 * RB ≤ δc * τ ^ 2 * A ^ L
    linarith [mul_comm (A ^ L) (δc * τ ^ 2)]
  · have h := hle (((le_max_right _ _).trans (le_max_left _ _)).trans hm1) (hAL.trans hALs)
    rw [div_le_iff₀ (by positivity)] at h
    rw [div_le_iff₀ (by positivity)]
    show 2 ^ 15 * (J + 1) * PB ≤ δc * pAP * A ^ Ls
    linarith [mul_comm (A ^ Ls) (δc * pAP)]
  · have h := hle (((le_max_left _ _).trans (le_max_right _ _)).trans hm1) (hAL.trans hALs)
    rw [div_le_iff₀ (by positivity)] at h
    rw [div_le_iff₀ (by positivity)]
    show 2 * R * (PB + 1) ≤ pAP * A ^ Ls
    linarith [mul_comm (A ^ Ls) pAP]
  · have h := hle (((le_max_right _ _).trans (le_max_right _ _)).trans hm1) le_rfl
    rw [div_le_iff₀ (by positivity)] at h
    rw [div_le_iff₀ (by positivity)]
    show 96 * NP * (R + 1) ≤ ε * A ^ a
    linarith [mul_comm (A ^ a) ε]

end OrthoDFA

end
