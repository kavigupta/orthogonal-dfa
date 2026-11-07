import OrthoDFA.Proofs.LoopStop

/-!
# The clustering, at two answer functions

Where two answer functions read alike on `clusterReads`, `ret` and `clusterAt` are the same at
every state of the schedule (`ret_eqOn`, `clusterAt_eqOn`); and `ret` and `clusterAt` read a run's
noise only through its answers, so a run's noise may be folded into the answer function
(`mem_ret_flat`, `clusterAt_flat`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S]
  (mq mq' : S → Ω → ℝ)

section Pieces

variable {mq mq'} {ω : Ω}

lemma screenCount_eqOn {P cands : Finset S} (hone : (1 : S) ∈ cands) {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq' w ω = 1)) :
    screenCount mq P v ω = screenCount mq' P v ω := by
  classical
  unfold screenCount
  refine congrArg Finset.card (Finset.filter_congr (fun p hp => ?_))
  have h1 := h _ (mem_readSet hp hv)
  have h0 : (mq p ω = 1 ↔ mq' p ω = 1) := by
    have := h _ (mem_readSet hp hone)
    rwa [mul_one] at this
  exact not_congr (iff_congr h1 h0)

lemma screenBase_eqOn {P cands : Finset S} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq' w ω = 1)) :
    screenBase mq P cands ω = screenBase mq' P cands ω := by
  classical
  unfold screenBase
  by_cases hne : (cands.erase 1).Nonempty
  · rw [dif_pos hne, dif_pos hne]
    exact Finset.inf'_congr hne rfl
      (fun v hv => screenCount_eqOn hone (Finset.mem_of_mem_erase hv) h)
  · rw [dif_neg hne, dif_neg hne]

lemma screened_eqOn (sc scd : ℕ) {P cands : Finset S} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq' w ω = 1)) :
    screened mq sc scd P cands ω = screened mq' sc scd P cands ω :=
  Finset.filter_congr fun v hv => by
    rw [screenCount_eqOn hone hv h, screenBase_eqOn hone h]

lemma one_mem_screened' (sc scd : ℕ) (P cands : Finset S) (hone : (1 : S) ∈ cands) :
    (1 : S) ∈ screened mq sc scd P cands ω := by
  classical
  refine Finset.mem_filter.2 ⟨hone, ?_⟩
  have : screenCount mq P 1 ω = 0 := by
    unfold screenCount
    rw [Finset.card_eq_zero]
    exact Finset.filter_eq_empty_iff.2 (fun p _ => by simp [mul_one])
  rw [this]
  simpa using Nat.zero_le _

lemma voteCount_eqOn {F : Finset S} {p : S} (h : ∀ v ∈ F, (mq (p * v) ω = 1 ↔ mq' (p * v) ω = 1)) :
    voteCount mq F p ω = voteCount mq' F p ω := by
  unfold voteCount
  rw [Finset.filter_congr fun v hv => h v hv]

lemma miscutBound_eqOn {a η₀ : ℝ} {lo hi : ℕ} {F R P : Finset S}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (mq (p * v) ω = 1 ↔ mq' (p * v) ω = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (mq (p * v) ω = 1 ↔ mq' (p * v) ω = 1))
    (hs : ∀ p ∈ P, (mq p ω = 1 ↔ mq' p ω = 1)) :
    miscutBound mq a lo hi F R P η₀ ω = miscutBound mq' a lo hi F R P η₀ ω := by
  classical
  have hv : ∀ p ∈ P, voteCount mq F p ω = voteCount mq' F p ω := fun p hp =>
    voteCount_eqOn (hF p hp)
  have hacc : acceptSide mq hi F P ω = acceptSide mq' hi F P ω := by
    unfold acceptSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hrej : rejectSide mq lo F P ω = rejectSide mq' lo F P ω := by
    unfold rejectSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hside : ∀ (accept : Bool) (A : Finset S), A ⊆ P →
      sideCap mq accept A R a η₀ ω = sideCap mq' accept A R a η₀ ω := by
    intro accept A hA
    unfold sideCap
    have h1 : againstOn mq accept A ω = againstOn mq' accept A ω := by
      unfold againstOn
      congr 1
      exact Finset.filter_congr fun p hp => by rw [hs p (hA hp)]
    have h2 : refApart mq A R ω = refApart mq' A R ω := by
      unfold refApart
      split_ifs with h
      · refine Finset.inf'_congr h rfl fun v hv => ?_
        congr 1
        exact Finset.filter_congr fun p hp => by rw [hR p (hA hp) v hv, hs p (hA hp)]
      · rfl
    rw [h1, h2]
  unfold miscutBound
  have hA : acceptSide mq' hi F P ω ⊆ P := by unfold acceptSide; exact Finset.filter_subset _ _
  have hR' : rejectSide mq' lo F P ω ⊆ P := by unfold rejectSide; exact Finset.filter_subset _ _
  rw [hacc, hrej, hside true _ hA, hside false _ hR']

lemma certifiesAt_eqOn {η₀ lim a : ℝ} {lo hi : ℕ} {F R P : Finset S}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (mq (p * v) ω = 1 ↔ mq' (p * v) ω = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (mq (p * v) ω = 1 ↔ mq' (p * v) ω = 1))
    (hs : ∀ p ∈ P, (mq p ω = 1 ↔ mq' p ω = 1)) :
    certifiesAt mq η₀ lim a lo hi F R P ω ↔ certifiesAt mq' η₀ lim a lo hi F R P ω := by
  unfold certifiesAt
  rw [miscutBound_eqOn hF hR hs]

omit [Stringlike S] in
lemma card_filter_le_congr {s : Finset S} {p q : S → Prop} [DecidablePred p] [DecidablePred q]
    (h : ∀ x ∈ s, p x ↔ q x) (c : ℝ) :
    ((s.filter p).card : ℝ) ≤ c ↔ ((s.filter q).card : ℝ) ≤ c := by
  rw [Finset.filter_congr h]

end Pieces

variable {J : Type*} [Fintype J]

/-- The cluster at `B` reads only `B`'s prefixes followed by its pool's suffixes. -/
lemma clusterAt_eqOn (populations : Finset J) (B : State) (x : Run Ω S J)
    (h : ∀ w ∈ readSet (prefixesAt populations B.npref x) (poolAt B.nsuff x),
      (mq w x.1 = 1 ↔ mq' w x.1 = 1)) :
    clusterAt mq populations x B = clusterAt mq' populations x B := by
  classical
  have hone : (1 : S) ∈ poolAt B.nsuff x := Finset.mem_insert_self _ _
  rw [clusterAt_eq, clusterAt_eq]
  unfold clusterBy screenedAt oracleNoise
  rw [screened_eqOn B.sc B.scd hone h]
  exact congrArg (fun t : Finset S => t.erase 1) (nearestClusterer.congr _ _ _ _ _
    (one_mem_screened' (mq := mq') B.sc B.scd _ _ hone)
    (fun p hp v hv => h _ (mem_readSet hp (Finset.filter_subset _ _ hv))))

lemma mem_ret_flat (populations : Finset J) (uni : J) (η₀ ι εc a : ℝ) (B : State)
    (x : Run Ω S J) :
    x ∈ ret mq populations uni η₀ ι εc a B
      ↔ (((), x.2) : Run Unit S J) ∈ ret (fun w _ => mq w x.1) populations uni η₀ ι εc a B :=
  Iff.rfl

lemma clusterAt_flat (populations : Finset J) (B : State) (x : Run Ω S J) :
    clusterAt mq populations x B
      = clusterAt (fun w (_ : Unit) => mq w x.1) populations (((), x.2) : Run Unit S J) B :=
  rfl

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

/-- `ret` at a state of `sched` reads only `clusterReads sched`. -/
lemma ret_eqOn (mq mq' : FreeMonoid α → Ω → ℝ) (sched : Finset State)
    (pops : Finset (Option (Fin K))) (uni : Option (Fin K)) (η₀ ι εc a : ℝ) {B : State}
    (hB : B ∈ sched) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (h : ∀ w ∈ clusterReads sched pops x, (mq w x.1 = 1 ↔ mq' w x.1 = 1)) :
    x ∈ ret mq pops uni η₀ ι εc a B ↔ x ∈ ret mq' pops uni η₀ ι εc a B := by
  classical
  have hrs : ∀ {P C : Finset (FreeMonoid α)}, P ⊆ roundPrefs sched pops x →
      C ⊆ insert 1 (roundSuffs sched x) → ∀ p ∈ P, ∀ v ∈ C, p * v ∈ clusterReads sched pops x :=
    fun hP hC p hp v hv =>
      Finset.mem_image.2 ⟨(p, v), Finset.mem_product.2 ⟨hP hp, hC hv⟩, rfl⟩
  set P := prefixesAt pops B.npref x
  set C := poolAt B.nsuff x
  have hone : (1 : FreeMonoid α) ∈ C := Finset.mem_insert_self _ _
  have hCs : C ⊆ insert 1 (roundSuffs sched x) := poolAt_sub_roundSuffs hB x
  have hPs : P ⊆ roundPrefs sched pops x := prefixesAt_sub_roundPrefs hB x
  have hcert : ∀ j ∈ pops, ∀ e ≤ B.npref, certOf P j (B.npref + e) x ⊆ roundPrefs sched pops x :=
    fun j hj e he => certOf_sub_roundPrefs hB hj he x
  have hcl : clusterAt mq pops x B = clusterAt mq' pops x B :=
    clusterAt_eqOn mq mq' pops B x fun w hw => by
      obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
      exact h _ (hrs hPs hCs p hp v hv)
  have hsub : clusterAt mq pops x B ⊆ C := by
    rw [clusterAt_eq]
    intro v hv
    exact Finset.filter_subset _ _
      (nearestClusterer.subset _ _ _ _ (one_mem_screened' (mq := mq) B.sc B.scd _ _ hone)
        (Finset.mem_of_mem_erase hv))
  have hins : insert 1 (clusterAt mq pops x B) ⊆ C := Finset.insert_subset hone hsub
  have hread : ∀ Q ⊆ roundPrefs sched pops x, ∀ p ∈ Q, ∀ v ∈ insert 1 (clusterAt mq pops x B),
      (mq (p * v) x.1 = 1 ↔ mq' (p * v) x.1 = 1) :=
    fun Q hQ p hp v hv => h _ (hrs hQ hCs p hp v (hins hv))
  have hseed : ∀ Q ⊆ roundPrefs sched pops x, ∀ p ∈ Q, (mq p x.1 = 1 ↔ mq' p x.1 = 1) :=
    fun Q hQ p hp => by
      have := hread Q hQ p hp 1 (Finset.mem_insert_self _ _)
      simpa using this
  simp only [ret, familyAt, refsAt, Set.mem_setOf_eq, oracleNoise, ← hcl]
  refine and_congr Iff.rfl (and_congr ?_ ?_)
  · refine forall₂_congr fun j hj => imp_congr_right fun _ => ?_
    have hc : certOf P j B.npref x ⊆ roundPrefs sched pops x := by
      have := hcert j hj 0 (Nat.zero_le _); simpa using this
    exact card_filter_le_congr (fun p hp => by
      rw [decided, decided, voteCount_eqOn fun v hv => hread _ hc p hp v hv]) _
  · refine forall₂_congr fun j hj => imp_congr_right fun _ =>
      exists_congr fun e => and_congr_right fun he => ?_
    have hc := hcert j hj e he
    exact certifiesAt_eqOn
      (fun p hp v hv => hread _ hc p hp v (Finset.mem_insert_of_mem hv))
      (fun p hp v hv => h _ (hrs hc hCs p hp v (Finset.mem_sdiff.1 hv).1))
      (fun p hp => hseed _ hc p hp)

/-- The cluster at a state of `sched` reads only `clusterReads sched`. -/
lemma clusterAt_eqOn_sched (mq mq' : FreeMonoid α → Ω → ℝ) (sched : Finset State)
    (pops : Finset (Option (Fin K))) {B : State} (hB : B ∈ sched)
    (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (h : ∀ w ∈ clusterReads sched pops x, (mq w x.1 = 1 ↔ mq' w x.1 = 1)) :
    clusterAt mq pops x B = clusterAt mq' pops x B :=
  clusterAt_eqOn mq mq' pops B x fun w hw => by
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
    exact h _ (Finset.mem_image.2 ⟨(p, v), Finset.mem_product.2
      ⟨prefixesAt_sub_roundPrefs hB x hp, poolAt_sub_roundSuffs hB x hv⟩, rfl⟩)

end OrthoDFA

end
