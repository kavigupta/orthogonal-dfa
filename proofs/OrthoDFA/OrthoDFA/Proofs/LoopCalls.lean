import OrthoDFA.Code.LoopCalls
import OrthoDFA.Proofs.LoopGood
import OrthoDFA.Proofs.LevelReads
import OrthoDFA.Proofs.HarvestCode
import OrthoDFA.Proofs.HarvestDraw
import OrthoDFA.Proofs.Nearest

/-!
# The loop's oracle calls, for every draw

With probes at most `L` long, `T` the probe budget, `M` the scan cap, `N` the harvest cap, and each
round's clustering reading at most `a` prefixes and `b` suffixes, a round calls the oracle at most

    K b M N (3 + ⌈log₂ L⌉)(T + 1) + a (b + 1) + b T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))

times (`roundCalls_le`), and the loop at most `K` times that (`loopCalls_le`). A level-3 step
adds at most one leaf, so a run's trees stay within `T + 2` leaves whatever is read
(`runReadCount_le`), and so do the hypotheses the harvests hand on. A family is among its
clustering's suffixes, so it has at most `b` members.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

/-- Splitting a leaf adds at most one leaf. -/
theorem DTree.length_leaves_splitAt_le (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), (T.splitAt d p).leaves.length ≤ T.leaves.length + 1
  | .leaf, [] => by simp [DTree.splitAt, DTree.leaves]
  | .leaf, _ :: _ => by simp [DTree.splitAt, DTree.leaves]
  | .node m r a, [] => by simp [DTree.splitAt, DTree.leaves]
  | .node m r a, false :: p => by
    have ih := DTree.length_leaves_splitAt_le d r p
    simp only [DTree.splitAt, DTree.leaves, List.length_append, List.length_map]
    omega
  | .node m r a, true :: p => by
    have ih := DTree.length_leaves_splitAt_le d a p
    simp only [DTree.splitAt, DTree.leaves, List.length_append, List.length_map]
    omega

variable [Fintype α] [DecidableEq α]

namespace Random

open Eps Kappa Spur

/-! ## Level 3 -/

omit [DecidableEq α] in
/-- A split re-sifts at most `|leaves|·|Σ|·|leaves|` strings. -/
theorem splitReads_card_le (C : Level3Settings) (read : FreeMonoid α → ARU) (s : Level3State α)
    (x : FreeMonoid α) :
    (splitReads read C s x).card
      ≤ s.tree.leaves.length * Fintype.card α * s.tree.leaves.length := by
  unfold splitReads
  rcases probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
  · simp
  · simp only
    split
    · rename_i t₀ w₀ _
      refine (resifts_card s _).trans ?_
      have hmid : (mids (s.tree.splitAt
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)).card
          ≤ s.tree.leaves.length := by
        have h1 := mids_card (s.tree.splitAt
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
        have h2 := DTree.length_leaves_splitAt_le
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p
        omega
      gcongr
    · simp
  · simp
  · simp
  · simp

/-- A run from `s` over `ys`, with `|leaves(s)| + |ys| ≤ n + 2`, reads at most
`|ys| (probeSifts L (n + 1) + (n + 2)|Σ|(n + 2))` strings. -/
theorem runReadCount_le (read : FreeMonoid α → ARU) (C : Level3Settings) {L : ℕ}
    (X : List (FreeMonoid α)) (n : ℕ) :
    ∀ (ys : List (FreeMonoid α)) (s : Level3State α), (∀ y ∈ ys, y ∈ X) →
      (∀ y ∈ ys, y.toList.length ≤ L) → RunInv read C X s →
      s.tree.leaves.length + ys.length ≤ n + 2 →
      runReadCount read C s ys
        ≤ ys.length * (probeSifts L * (n + 1) + (n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _, _ => by simp [runReadCount]
  | x :: ys, s, hX, hL, hs, hn => by
    set B := probeSifts L * (n + 1) + (n + 2) * Fintype.card α * (n + 2) with hB
    rw [List.length_cons] at hn
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    have hprobe : (probeReads read s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 1) :=
      (probeReads_card_mids read hxL).trans (Nat.mul_le_mul_left _ hmid)
    have hsplit : (splitReads read C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (splitReads_card_le C read s x).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_right _ (by omega)) (by omega))
    have hhead : (probeReads read s.tree s.edges C.walkStart x).card
        + (splitReads read C s x).card ≤ B := Nat.add_le_add hprobe hsplit
    simp only [runReadCount, List.length_cons]
    rcases ha : applyProbe read C s x with s' | e
    · obtain ⟨hs', hlv'⟩ := runInv_step read C (hX x (by simp)) hs ha
      have hrec := runReadCount_le read C X n ys s' (fun y hy => hX y (by simp [hy]))
        (fun y hy => hL y (by simp [hy])) hs' (by omega)
      simp only
      calc _ ≤ B + ys.length * B := Nat.add_le_add hhead hrec
        _ = (ys.length + 1) * B := by ring
    · simp only
      calc _ ≤ B := by omega
        _ ≤ (ys.length + 1) * B := Nat.le_mul_of_pos_left _ (Nat.succ_pos _)

/-- Level 3's run over `T` probes at most `L` long reads at most
`T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings. -/
theorem runReadCount_initial_le (read : FreeMonoid α → ARU) (C : Level3Settings) {L T : ℕ}
    (xs : Fin T → FreeMonoid α) (hL : ∀ i, (xs i).toList.length ≤ L) :
    runReadCount read C initialState (List.ofFn xs)
      ≤ T * (probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2)) := by
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  have := runReadCount_le read C (L := L) (List.ofFn xs) T (List.ofFn xs) initialState
    (fun _ h => h) (fun y hy => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy
      exact hL i) (runInv_start read C _) (by rw [h0, List.length_ofFn]; omega)
  rwa [List.length_ofFn] at this

/-! ## Harvests -/

omit [Fintype α] [DecidableEq α] in
/-- A probe at most `L` long reads at most `(3 + ⌈log₂ L⌉)|mids T|` strings. -/
theorem probeReads_card_clog (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    {x : FreeMonoid α} {L : ℕ} (hx : x.toList.length ≤ L) :
    (probeReads read T E k x).card ≤ (3 + Nat.clog 2 L) * (mids T).card := by
  obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := read) (T := T) E k hx
  calc (probeReads read T E k x).card ≤ (posReads read T x ps).card := Finset.card_le_card hsub
    _ ≤ ps.toFinset.card * (mids T).card := posReads_card _ _ _ _
    _ ≤ (3 + Nat.clog 2 L) * (mids T).card := Nat.mul_le_mul_right _ hps

omit [Fintype α] [DecidableEq α] in
theorem harvestCount_le (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (T : Set (FreeMonoid α)) {L : ℕ} :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, x.toList.length ≤ L) →
      harvestCount read k H site T xs ≤ xs.length * ((3 + Nat.clog 2 L) * (mids H.tree).card)
  | [], _ => by simp [harvestCount]
  | x :: xs, hL => by
    have hx := probeReads_card_clog read H.tree H.edges k (hL x (by simp))
    have hrec := harvestCount_le read k H site T xs fun y hy => hL y (by simp [hy])
    simp only [harvestCount, List.length_cons]
    split_ifs
    · calc _ ≤ (3 + Nat.clog 2 L) * (mids H.tree).card := by omega
        _ ≤ _ := Nat.le_mul_of_pos_left _ (Nat.succ_pos _)
    · calc _ ≤ (3 + Nat.clog 2 L) * (mids H.tree).card
            + xs.length * ((3 + Nat.clog 2 L) * (mids H.tree).card) := Nat.add_le_add hx hrec
        _ = _ := by ring

omit [Fintype α] [DecidableEq α] in
theorem drawManyCount_le (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (T : Set (FreeMonoid α)) {L m N' : ℕ}
    (yss : Fin m → (Fin N' → FreeMonoid α)) (hL : ∀ i t, (yss i t).toList.length ≤ L) :
    drawManyCount read k H site T yss ≤ m * (N' * ((3 + Nat.clog 2 L) * (mids H.tree).card)) := by
  unfold drawManyCount
  calc _ ≤ ∑ _i : Fin m, N' * ((3 + Nat.clog 2 L) * (mids H.tree).card) :=
        Finset.sum_le_sum fun i _ => by
          have := harvestCount_le read k H site T (List.ofFn (yss i)) (L := L) fun y hy => by
            obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hy
            exact hL i t
          rwa [List.length_ofFn] at this
    _ = _ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

end Random

/-! ## A round -/

omit [DecidableEq α] in
/-- A law that is `ν` scaled, or a point mass at `1`, keeps what `ν` almost surely has and `1`
has. -/
lemma ae_popLaw {ν : Measure (FreeMonoid α)} {P : FreeMonoid α → Prop} (hν : ∀ᵐ x ∂ν, P x)
    (h1 : P 1) : ∀ᵐ x ∂(popLaw ν), P x := by
  unfold popLaw
  split_ifs
  · exact (ae_dirac_iff (MeasurableSet.of_discrete)).2 h1
  · rw [ae_iff, Measure.smul_apply, ae_iff.1 hν, smul_zero]

omit [DecidableEq α] in
/-- Every probe a round draws is at most `L` long where `D`'s are. -/
lemma roundProbes_length {Ω : Type*} [MeasurableSpace Ω] {K : ℕ} (settings : LoopSettings K)
    {D : Measure (FreeMonoid α)} {L : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (h : LoopHist α)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) (u : Fin settings.level3.probeBudget → unitInterval)
    (i : Fin settings.level3.probeBudget) : (roundProbes settings D h x u i).toList.length ≤ L :=
  drawFrom_length_le _ (ae_popLaw (ae_restrict_of_ae hD) (by simp)) _

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- Where a state of the schedule returns, the round's family is among its clustering's suffixes. -/
lemma roundOf_family_sub (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : LoopHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : Fin settings.level3.probeBudget → unitInterval)
    (hB : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      x ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B) :
    (roundOf settings O D h x u).family
      ⊆ roundSuffs (settings.schedule (loopPopulations K h.pops.length)) x := by
  have hspec := Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
      (loopPopulations K h.pops.length) ∧ x ∈ ret O.mq (loopPopulations K h.pops.length) none
        settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    (by obtain ⟨B, h1, h2⟩ := hB; exact ⟨B, h1, h2⟩)
  intro v hv
  have hv' : v ∈ clusterAt O.mq (loopPopulations K h.pops.length) x (Classical.epsilon fun B =>
      B ∈ settings.schedule (loopPopulations K h.pops.length) ∧ x ∈ ret O.mq
        (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit settings.εcov
        settings.gateLevel B) := hv
  have hv1 : v ≠ 1 := (Finset.mem_erase.1 hv').1
  rw [clusterAt_eq] at hv'
  have := poolAt_sub_roundSuffs hspec.1 x
    (clusterAt_subset (rule := nearestClusterer) O _ _ x hv')
  rcases Finset.mem_insert.1 this with h1 | h1
  · exact absurd h1 hv1
  · exact h1

omit [IsProbabilityMeasure μ] in
/-- A round's hypothesis has at most `T + 1` midfixes, `T` the probe budget. -/
lemma roundOf_mids_le (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : LoopHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : Fin settings.level3.probeBudget → unitInterval) :
    (Random.mids (roundOf settings O D h x u).result.hypothesis.tree).card
      ≤ settings.level3.probeBudget + 1 := by
  have := (Random.runProbes_tree (roundOf settings O D h x u).read settings.level3
    (List.ofFn (roundProbes settings D h x u))).2.1
  have hm := Random.mids_card
    (Random.runProbes (roundOf settings O D h x u).read settings.level3 Random.initialState
      (List.ofFn (roundProbes settings D h x u))).1.tree
  rw [List.length_ofFn] at this
  have e : (roundOf settings O D h x u).result.hypothesis.tree
      = (Random.runProbes (roundOf settings O D h x u).read settings.level3 Random.initialState
        (List.ofFn (roundProbes settings D h x u))).1.tree := rfl
  rw [e]
  omega

/-- The bound on a round's calls. -/
def roundCallBound (K a b T M N L A : ℕ) : ℕ :=
  K * (b * (M * (N * ((3 + Nat.clog 2 L) * (T + 1))))) + a * (b + 1)
    + b * (T * (Random.Kappa.probeSifts L * (T + 1) + (T + 2) * A * (T + 2)))

theorem roundCalls_le (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (costs : List ℕ) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)
    {L a b : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ settings.level3.probeBudget + 1)
    (hcosts : ∀ c ∈ costs, c ≤ b) :
    roundCalls settings O D Dsf costs h d
      ≤ roundCallBound K a b settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) := by
  classical
  have hpop : ∀ i : Fin K, popDrawCalls D settings.level3.walkStart settings.level3.harvestReadCap
      settings.scanCap h.pops costs h.prefs (some i) (d.2.1.2.1 (some i))
        ≤ b * (settings.scanCap * (settings.level3.harvestReadCap
          * ((3 + Nat.clog 2 L) * (settings.level3.probeBudget + 1)))) := by
    intro i
    simp only [popDrawCalls]
    rcases hs : h.pops[i.val]? with _ | s
    · exact Nat.zero_le _
    simp only
    have hc : costs.getD i.val 0 ≤ b := by
      rw [List.getD_eq_getElem?_getD]
      rcases hci : costs[i.val]? with _ | c
      · simp
      · exact hcosts c (List.mem_of_getElem? hci)
    have hdm := Random.drawManyCount_le s.read settings.level3.walkStart s.hyp s.site
      (↑h.prefs) (L := L)
      (fun (n : Fin settings.scanCap) (t : Fin settings.level3.harvestReadCap) =>
        drawFrom D ((d.2.1.2.1 (some i) n).2 t))
      (fun _ _ => drawFrom_length_le _ hD _)
    have hmid := hpops s (List.mem_of_getElem? hs)
    calc _ ≤ b * (settings.scanCap * (settings.level3.harvestReadCap
          * ((3 + Nat.clog 2 L) * (Random.mids s.hyp.tree).card))) := Nat.mul_le_mul hc hdm
      _ ≤ _ := by gcongr
  have hsum : (∑ j, popDrawCalls D settings.level3.walkStart settings.level3.harvestReadCap
      settings.scanCap h.pops costs h.prefs j (d.2.1.2.1 j))
        ≤ K * (b * (settings.scanCap * (settings.level3.harvestReadCap
          * ((3 + Nat.clog 2 L) * (settings.level3.probeBudget + 1))))) := by
    rw [Fintype.sum_option]
    calc _ ≤ 0 + ∑ _i : Fin K, b * (settings.scanCap * (settings.level3.harvestReadCap
          * ((3 + Nat.clog 2 L) * (settings.level3.probeBudget + 1)))) :=
          Nat.add_le_add (le_of_eq rfl) (Finset.sum_le_sum fun i _ => hpop i)
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, zero_add]
  unfold roundCalls roundCallBound
  dsimp only
  rcases hdr : clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap h (d.1, d.2.1) with _ | dr
  · simp only [Nat.add_zero]
    exact hsum.trans (by omega)
  simp only
  obtain ⟨hpa, hsb⟩ := hab (loopPopulations K h.pops.length) (streams (fun _ => ∅) none dr)
  have hcl := Nat.mul_le_mul hpa ((Finset.card_insert_le (1 : FreeMonoid α)
    (roundSuffs (settings.schedule (loopPopulations K h.pops.length))
      (streams (fun _ => ∅) none dr))).trans (Nat.add_le_add_right hsb 1))
  split_ifs with hc
  · have h3 := Nat.mul_le_mul
      ((Finset.card_le_card (roundOf_family_sub settings O D h _ d.2.2 hc.2)).trans hsb)
      (Random.runReadCount_initial_le
        (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2).read settings.level3 _
        (roundProbes_length settings hD h (streams (fun _ => ∅) none dr) d.2.2))
    calc _ ≤ _ + (_ + _) := Nat.add_le_add hsum (Nat.add_le_add hcl h3)
      _ = _ := by ring
  · calc _ ≤ _ + (_ + 0) := Nat.add_le_add hsum (Nat.add_le_add hcl (le_refl 0))
      _ ≤ _ := by
        rw [Nat.add_zero, ← Nat.add_assoc]
        exact Nat.le_add_right _ _

/-! ## The loop -/

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)

/-- Every round the loop runs has a hypothesis of at most `T + 1` midfixes and a family of at
most `b` members. -/
lemma loopRounds_small {b : ℕ}
    (hb : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundSuffs (settings.schedule pops) x).card ≤ b) (n : ℕ) :
    ∀ rs ∈ loopRounds settings O D Dsf draws n, ∀ o ∈ rs,
      (Random.mids o.result.hypothesis.tree).card ≤ settings.level3.probeBudget + 1
        ∧ o.family.card ≤ b := by
  induction n with
  | zero =>
    intro rounds hr
    rw [loopRounds, Option.mem_def, Option.some.injEq] at hr
    subst hr
    simp
  | succ r ih =>
    intro rounds hr
    rw [loopRounds] at hr
    obtain ⟨rs, hrs, hr⟩ := Option.mem_bind_iff.1 hr
    have hrr := ih rs hrs
    split_ifs at hr with hc
    · obtain ⟨o, ho, rfl⟩ := Option.mem_map.1 hr
      intro o' ho'
      rcases List.mem_append.1 ho' with h1 | h1
      · exact hrr o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        unfold loopRound at ho
        obtain ⟨dr, -, hdr⟩ := Option.mem_bind_iff.1 ho
        split_ifs at hdr with hcond
        · rw [← Option.mem_some_iff.1 hdr]
          exact ⟨roundOf_mids_le settings O D _ _ _,
            (Finset.card_le_card (roundOf_family_sub settings O D _ _ _ hcond.2)).trans
              (hb _ _)⟩
        · cases hdr
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact hrr

/-- The bound on the loop's calls. -/
theorem loopCalls_le [IsProbabilityMeasure D] {L a b : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b) (n : ℕ) :
    loopCalls settings O D Dsf draws n
      ≤ n * roundCallBound K a b settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) := by
  classical
  induction n with
  | zero => simp [loopCalls]
  | succ r ih =>
    rw [loopCalls, Nat.succ_mul]
    refine Nat.add_le_add ih ?_
    rcases hrs : loopRounds settings O D Dsf draws r with _ | rs
    · exact Nat.zero_le _
    simp only
    split_ifs with hc
    · have hsm := loopRounds_small settings O D Dsf draws (fun pops x => (hab pops x).2) r rs hrs
      refine roundCalls_le settings O D Dsf _ _ _ hD hab (fun s hs => ?_) (fun c hc' => ?_)
      · obtain ⟨o, ho, hos⟩ := List.mem_filterMap.1 hs
        unfold LoopRound.sampler at hos
        obtain ⟨site, -, rfl⟩ := Option.map_eq_some_iff.1 hos
        exact (hsm o ho).1
      · obtain ⟨o, ho, hoc⟩ := List.mem_filterMap.1 hc'
        obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 hoc
        exact (hsm o ho).2
    · exact Nat.zero_le _

end OrthoDFA

end
