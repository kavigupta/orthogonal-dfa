import OrthoDFA.Proofs.LoopStop

/-!
# Rounds that do not collide read new strings

A round reads its prefixes alone, its prefixes followed by its pool's suffixes, and short strings
followed by its family's suffixes. With every suffix `Ls` long and every prefix shorter, two such
strings of different rounds can only coincide where the rounds drew the same suffix or the same
prefix (`NoColl`). So off a collision round `r` reads none of the strings the rounds before it
read (`disjoint_of_noColl`), and level 4 reads none of the strings round `r`'s clustering read
but at its prefixes, which the re-noised read leaves to a second noise.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

/-- `z · v = z' · v'` with `v` and `v'` the same length forces `v = v'`. -/
lemma suffix_eq_of_mul_eq {z v z' v' : FreeMonoid α} (h : z * v = z' * v')
    (hl : v.toList.length = v'.toList.length) : v = v' ∧ z = z' := by
  have h' : z.toList ++ v.toList = z'.toList ++ v'.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul, h]
  obtain ⟨h1, h2⟩ := List.append_inj' h' hl
  exact ⟨FreeMonoid.toList.injective h2, FreeMonoid.toList.injective h1⟩

lemma length_mul (z v : FreeMonoid α) : (z * v).toList.length = z.toList.length + v.toList.length := by
  simp

variable (settings : LoopSettings K) (D : Measure (FreeMonoid α))

/-- Round `i`'s populations, given the earlier rounds' harvest data `c`. -/
abbrev popsAt (c : List (HarvestCode α)) (i : ℕ) : List (Measure (FreeMonoid α)) :=
  (c.take i).map (decodePop D settings.level4.walkStart)

/-- The prefixes a round's clustering reads, after `harvests`. -/
def prefsOf (harvests : List (Measure (FreeMonoid α))) (c : ClusterRest α K) :
    Finset (FreeMonoid α) :=
  schedPrefs (settings.schedule (loopPopulations K harvests.length))
    (loopPopulations K harvests.length)
    (((), runOf harvests c) : Run Unit (FreeMonoid α) (Option (Fin K)))

/-- A round's first `NS` suffix draws. -/
def suffsOf (NS : ℕ) (c : ClusterRest α K) : Finset (FreeMonoid α) := (Finset.range NS).image c.1

/-- Through round `r`, every suffix drawn is `Ls` long and every prefix read shorter, and round `r`
draws no suffix and reads no prefix an earlier round did. -/
def NoColl (Ls NS : ℕ) (c : List (HarvestCode α))
    (x : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) (hr : r < K) : Prop :=
  (∀ i : Fin K, i.1 ≤ r → ∀ v ∈ suffsOf NS (x i).1, v.toList.length = Ls)
    ∧ (∀ i : Fin K, i.1 ≤ r → ∀ p ∈ prefsOf settings (popsAt settings D c i) (x i).1,
      p.toList.length < Ls)
    ∧ (∀ i : Fin K, i.1 < r → Disjoint (suffsOf NS (x i).1) (suffsOf NS (x ⟨r, hr⟩).1))
    ∧ (∀ i : Fin K, i.1 < r → Disjoint (prefsOf settings (popsAt settings D c i) (x i).1)
      (prefsOf settings (popsAt settings D c r) (x ⟨r, hr⟩).1))

lemma schedPrefs_noise (sched : Finset State) (pops : Finset (Option (Fin K))) (ω : Ω)
    (dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α)) :
    schedPrefs sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K)))
      = schedPrefs sched pops (((), dr) : Run Unit (FreeMonoid α) (Option (Fin K))) := rfl

@[simp] lemma runOf_suffix (harvests : List (Measure (FreeMonoid α))) (c : ClusterRest α K)
    (a : ℕ) : (runOf harvests c).1.1 a = c.1 a := rfl

/-- A string a round reads is one of its prefixes, or a string followed by one of its suffixes. -/
lemma mem_schedReads_shape {sched : Finset State} {pops : Finset (Option (Fin K))} {NS : ℕ}
    (hNS : schedNS sched ≤ NS) {ω : Ω}
    {dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α)}
    {w : FreeMonoid α} (hw : w ∈ schedReads sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K)))) :
    ∃ p ∈ schedPrefs sched pops (((), dr) : Run Unit (FreeMonoid α) (Option (Fin K))),
      w = p ∨ ∃ a < NS, w = p * dr.1.1 a := by
  classical
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  refine ⟨p, hp, ?_⟩
  rcases Finset.mem_insert.1 hv with rfl | hv
  · left; simp
  rcases Finset.mem_insert.1 hv with rfl | hv
  · left; simp
  obtain ⟨a, ha, rfl⟩ := Finset.mem_image.1 hv
  exact Or.inr ⟨a, (Finset.mem_range.1 ha).trans_le hNS, rfl⟩

/-- A string the rounds before `r` read is one of their prefixes, or a string followed by one of
their suffixes. -/
lemma mem_histReads_shape (N NS : ℕ) (c : List (HarvestCode α))
    (x : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) (ω : Ω)
    (hNS : ∀ i : Fin K, schedNS (settings.schedule
      (loopPopulations K (popsAt settings D c i).length)) ≤ NS)
    {w : FreeMonoid α} (hw : w ∈ histReads settings D N c x r ω) :
    ∃ i : Fin K, i.1 < r ∧ ((∃ p ∈ prefsOf settings (popsAt settings D c i) (x i).1, w = p)
      ∨ ∃ z, ∃ v ∈ suffsOf NS (x i).1, w = z * v) := by
  classical
  obtain ⟨i, hi, hw⟩ := Finset.mem_biUnion.1 hw
  split_ifs at hw with h
  swap
  · exact absurd hw (Finset.notMem_empty _)
  refine ⟨⟨i, h.1⟩, Finset.mem_range.1 hi, ?_⟩
  rcases Finset.mem_union.1 hw with hw | hw
  · obtain ⟨p, hp, hpw⟩ := mem_schedReads_shape (hNS ⟨i, h.1⟩) hw
    rcases hpw with h' | ⟨a, ha, h'⟩
    · exact Or.inl ⟨p, hp, h'⟩
    · exact Or.inr ⟨p, _, Finset.mem_image.2 ⟨a, Finset.mem_range.2 ha, rfl⟩, h'⟩
  · obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨-, hv⟩ := Finset.mem_product.1 hzv
    obtain ⟨hv1, hv⟩ := Finset.mem_erase.1 hv
    rcases Finset.mem_insert.1 hv with rfl | hv
    · exact absurd rfl hv1
    obtain ⟨a, ha, rfl⟩ := Finset.mem_image.1 hv
    exact Or.inr ⟨z, _, Finset.mem_image.2 ⟨a, Finset.mem_range.2
      ((Finset.mem_range.1 ha).trans_le (hNS ⟨i, h.1⟩)), rfl⟩, rfl⟩

/-- Off a collision, round `r` reads none of the strings the rounds before it read. -/
theorem disjoint_hist_round (N Ls NS : ℕ) (c : List (HarvestCode α))
    (x : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) (hr : r < K) (ω : Ω)
    (hNS : ∀ i : Fin K, schedNS (settings.schedule
      (loopPopulations K (popsAt settings D c i).length)) ≤ NS)
    (hnc : NoColl settings D Ls NS c x r hr) :
    Disjoint (histReads settings D N c x r ω)
      (schedReads (settings.schedule (loopPopulations K (popsAt settings D c r).length))
        (loopPopulations K (popsAt settings D c r).length)
        ((ω, runOf (popsAt settings D c r) (x ⟨r, hr⟩).1) : Run Ω (FreeMonoid α) (Option (Fin K)))) := by
  classical
  obtain ⟨hsl, hpl, hsd, hpd⟩ := hnc
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨i, hir, hsh⟩ := mem_histReads_shape settings D N NS c x r ω hNS hw
  obtain ⟨p', hp', hpw'⟩ := mem_schedReads_shape (hNS ⟨r, hr⟩) hw'
  simp only [runOf_suffix] at hpw'
  have hp'l := hpl ⟨r, hr⟩ le_rfl p' hp'
  rcases hsh with ⟨p, hp, hwp⟩ | ⟨z, v, hv, hwz⟩
  · have hpl' := hpl i hir.le p hp
    rcases hpw' with h | ⟨a, ha, h⟩
    · rw [hwp] at h
      exact Finset.disjoint_left.1 (hpd i hir) hp (h ▸ hp')
    · rw [hwp] at h
      have := congrArg (fun t : FreeMonoid α => t.toList.length) h
      simp only [length_mul] at this
      have hvl := hsl ⟨r, hr⟩ le_rfl _ (Finset.mem_image.2 ⟨a, Finset.mem_range.2 ha, rfl⟩)
      omega
  · have hvl := hsl i hir.le v hv
    rcases hpw' with h | ⟨a, ha, h⟩
    · rw [hwz] at h
      have := congrArg (fun t : FreeMonoid α => t.toList.length) h
      simp only [length_mul] at this
      omega
    · have hva : (x ⟨r, hr⟩).1.1 a ∈ suffsOf NS (x ⟨r, hr⟩).1 :=
        Finset.mem_image.2 ⟨a, Finset.mem_range.2 ha, rfl⟩
      have hvl' := hsl ⟨r, hr⟩ le_rfl _ hva
      rw [hwz] at h
      obtain ⟨hvv, -⟩ := suffix_eq_of_mul_eq h (hvl.trans hvl'.symm)
      exact Finset.disjoint_left.1 (hsd i hir) hv (hvv ▸ hva)

/-- Off a collision, level 4 reads, through a family of round `r`'s suffixes, none of the strings
the earlier rounds or round `r`'s clustering read but at round `r`'s prefixes. -/
theorem disjoint_hist_level4 (N Ls NS N' : ℕ) (c : List (HarvestCode α))
    (x : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) (hr : r < K) (ω : Ω)
    (hNS : ∀ i : Fin K, schedNS (settings.schedule
      (loopPopulations K (popsAt settings D c i).length)) ≤ NS)
    (hnc : NoColl settings D Ls NS c x r hr) (F : Finset (FreeMonoid α))
    (hF : F ⊆ suffsOf NS (x ⟨r, hr⟩).1) :
    Disjoint (histReads settings D N c x r ω
      ∪ schedReads (settings.schedule (loopPopulations K (popsAt settings D c r).length))
        (loopPopulations K (popsAt settings D c r).length)
        ((ω, runOf (popsAt settings D c r) (x ⟨r, hr⟩).1) : Run Ω (FreeMonoid α) (Option (Fin K))))
      (readStrings (shortStrings N') F
        \ readSet (prefsOf settings (popsAt settings D c r) (x ⟨r, hr⟩).1) F) := by
  classical
  obtain ⟨hsl, hpl, hsd, hpd⟩ := hnc
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨hw', hnot⟩ := Finset.mem_sdiff.1 hw'
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw'
  obtain ⟨-, hvF⟩ := Finset.mem_product.1 hzv
  have hvl := hsl ⟨r, hr⟩ le_rfl v (hF hvF)
  dsimp only at hnot ⊢
  rcases Finset.mem_union.1 hw with hw | hw
  · obtain ⟨i, hir, hsh⟩ := mem_histReads_shape settings D N NS c x r ω hNS hw
    rcases hsh with ⟨p, hp, h⟩ | ⟨z', v', hv', h⟩
    · have := congrArg (fun t : FreeMonoid α => t.toList.length) h
      simp only [length_mul] at this
      have := hpl i hir.le p hp
      omega
    · have hvl' := hsl i hir.le v' hv'
      obtain ⟨hvv, -⟩ := suffix_eq_of_mul_eq h (hvl.trans hvl'.symm)
      exact Finset.disjoint_left.1 (hsd i hir) hv' (hvv ▸ hF hvF)
  · obtain ⟨p', hp', hpw'⟩ := mem_schedReads_shape (hNS ⟨r, hr⟩) hw
    simp only [runOf_suffix] at hpw'
    have hp'l := hpl ⟨r, hr⟩ le_rfl p' hp'
    rcases hpw' with h | ⟨a, ha, h⟩
    · have := congrArg (fun t : FreeMonoid α => t.toList.length) h
      simp only [length_mul] at this
      omega
    · have hva : (x ⟨r, hr⟩).1.1 a ∈ suffsOf NS (x ⟨r, hr⟩).1 :=
        Finset.mem_image.2 ⟨a, Finset.mem_range.2 ha, rfl⟩
      have hvl' := hsl ⟨r, hr⟩ le_rfl _ hva
      obtain ⟨hvv, hzz⟩ := suffix_eq_of_mul_eq h (hvl.trans hvl'.symm)
      apply hnot
      have hz : z = p' := hzz
      subst hz
      exact mem_readSet hp' hvF

end OrthoDFA
