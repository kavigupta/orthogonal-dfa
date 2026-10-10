import OrthoDFA.Proofs.TallyNoise

/-!
# Adaptive reads

A query tree reads keys one at a time, each chosen by the values read before it, and each read
labelled by what the tree knows before it. Over independent reads, whether a key is read, and the
label of its first read, are fixed before its own value is: they depend only on the other keys'
values (`firsts_update`). So a sum over first reads of a per-read quantity has the mean of the sum
of that quantity's own means (`firsts_lintegral_le`), and the product over first reads of
per-read factors of mean at most `1` has mean at most `1` (`firsts_prod_le`). A key read again
within one run is counted once by `firsts`; with multiplicity, the sum is at most the most reads of
one key times the sum over first reads (`asks_sum_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- An adaptive computation: done with a value, or read a key under a label and go on by its
value. -/
inductive QTree (Λ ι V β : Type*) where
  | done (b : β)
  | ask (ℓ : Λ) (w : ι) (next : V → QTree Λ ι V β)

namespace QTree

variable {Λ ι V β γ : Type*}

/-- The run's value under the reads `rd`. -/
def run (rd : ι → V) : QTree Λ ι V β → β
  | done b => b
  | ask _ w next => run rd (next (rd w))

/-- The run's reads, in order, with their labels. -/
def asks (rd : ι → V) : QTree Λ ι V β → List (Λ × ι)
  | done _ => []
  | ask ℓ w next => (ℓ, w) :: asks rd (next (rd w))

/-- `t`, and then `f` of its value. -/
def bind : QTree Λ ι V β → (β → QTree Λ ι V γ) → QTree Λ ι V γ
  | done b, f => f b
  | ask ℓ w next, f => ask ℓ w fun v => bind (next v) f

/-- `t` with its value mapped. -/
def map (f : β → γ) (t : QTree Λ ι V β) : QTree Λ ι V γ := t.bind fun b => done (f b)

theorem run_bind (rd : ι → V) (f : β → QTree Λ ι V γ) :
    ∀ t : QTree Λ ι V β, (t.bind f).run rd = (f (t.run rd)).run rd
  | done _ => rfl
  | ask _ w next => run_bind rd f (next (rd w))

theorem asks_bind (rd : ι → V) (f : β → QTree Λ ι V γ) :
    ∀ t : QTree Λ ι V β, (t.bind f).asks rd = t.asks rd ++ (f (t.run rd)).asks rd
  | done _ => rfl
  | ask ℓ w next => by
    simp only [bind, asks, run, List.cons_append]
    rw [asks_bind rd f (next (rd w))]

theorem run_map (rd : ι → V) (f : β → γ) (t : QTree Λ ι V β) :
    (t.map f).run rd = f (t.run rd) := run_bind rd _ t

theorem asks_map (rd : ι → V) (f : β → γ) (t : QTree Λ ι V β) :
    (t.map f).asks rd = t.asks rd := by
  simp [map, asks_bind, asks]

variable [DecidableEq ι]

/-- The first read of each key not in `P`, in order, with its label. -/
def firsts (rd : ι → V) : Finset ι → QTree Λ ι V β → List (Λ × ι)
  | _, done _ => []
  | P, ask ℓ w next => if w ∈ P then firsts rd P (next (rd w))
      else (ℓ, w) :: firsts rd (insert w P) (next (rd w))

/-- `rd` with the key `w` read as `v`. -/
def upd (rd : ι → V) (w : ι) (v : V) : ι → V := fun y => if y = w then v else rd y

variable [Fintype V]

/-- The keys the tree can read. -/
def keys : QTree Λ ι V β → Finset ι
  | done _ => ∅
  | ask _ w next => insert w (Finset.univ.biUnion fun v => keys (next v))

theorem firsts_congr {rd rd' : ι → V} :
    ∀ (t : QTree Λ ι V β) (P : Finset ι), (∀ w ∈ t.keys, rd w = rd' w) →
      t.firsts rd P = t.firsts rd' P
  | done _, _, _ => rfl
  | ask ℓ w next, P, h => by
    have hw : rd w = rd' w := h w (Finset.mem_insert_self _ _)
    have hn : ∀ y ∈ (next (rd w)).keys, rd y = rd' y := fun y hy =>
      h y (Finset.mem_insert_of_mem (Finset.mem_biUnion.2 ⟨rd w, Finset.mem_univ _, hy⟩))
    simp only [firsts, hw] at hn ⊢
    rw [firsts_congr (next (rd' w)) P hn, firsts_congr (next (rd' w)) (insert w P) hn]

theorem firsts_keys :
    ∀ (rd : ι → V) (t : QTree Λ ι V β) (P : Finset ι) (q : Λ × ι), q ∈ t.firsts rd P →
      q.2 ∈ t.keys ∧ q.2 ∉ P
  | _, done _, _, _, h => by simp [firsts] at h
  | rd, ask ℓ w next, P, q, h => by
    have hsub : ∀ y ∈ (next (rd w)).keys, y ∈ (ask ℓ w next).keys := fun y hy =>
      Finset.mem_insert_of_mem (Finset.mem_biUnion.2 ⟨rd w, Finset.mem_univ _, hy⟩)
    simp only [firsts] at h
    split_ifs at h with hw
    · obtain ⟨h1, h2⟩ := firsts_keys rd (next (rd w)) P q h
      exact ⟨hsub _ h1, h2⟩
    · rcases List.mem_cons.1 h with rfl | h
      · exact ⟨Finset.mem_insert_self _ _, hw⟩
      · obtain ⟨h1, h2⟩ := firsts_keys rd (next (rd w)) (insert w P) q h
        exact ⟨hsub _ h1, fun hP => h2 (Finset.mem_insert_of_mem hP)⟩

theorem firsts_keys_eq :
    ∀ (rd : ι → V) (t : QTree Λ ι V β) (P : Finset ι),
      ((t.firsts rd P).map Prod.snd).Nodup
        ∧ ((t.firsts rd P).map Prod.snd).toFinset = ((t.asks rd).map Prod.snd).toFinset \ P
  | _, done _, _ => by simp [firsts, asks]
  | rd, ask ℓ w next, P => by
    simp only [firsts, asks, List.map_cons, List.toFinset_cons]
    split_ifs with hw
    · obtain ⟨h1, h2⟩ := firsts_keys_eq rd (next (rd w)) P
      refine ⟨h1, ?_⟩
      rw [h2]
      ext y
      simp only [Finset.mem_sdiff, Finset.mem_insert]
      constructor
      · rintro ⟨h, h'⟩; exact ⟨.inr h, h'⟩
      · rintro ⟨h | h, h'⟩
        · exact absurd (h ▸ hw) h'
        · exact ⟨h, h'⟩
    · obtain ⟨h1, h2⟩ := firsts_keys_eq rd (next (rd w)) (insert w P)
      refine ⟨List.nodup_cons.2 ⟨fun hm => ?_, h1⟩, ?_⟩
      · have := List.mem_toFinset.2 hm
        rw [h2] at this
        exact (Finset.mem_sdiff.1 this).2 (Finset.mem_insert_self _ _)
      · rw [List.map_cons, List.toFinset_cons, h2]
        ext y
        simp only [Finset.mem_insert, Finset.mem_sdiff]
        constructor
        · rintro (rfl | ⟨h, h'⟩)
          · exact ⟨.inl rfl, hw⟩
          · exact ⟨.inr h, fun hP => h' (.inr hP)⟩
        · rintro ⟨h | h, h'⟩
          · exact .inl h
          · by_cases hy : y = w
            · exact .inl hy
            · exact .inr ⟨h, fun h'' => h''.elim hy h'⟩

/-- Reads counted with multiplicity are at most the most reads of one key times the first reads. -/
theorem asks_sum_le (a : ι → V → ENNReal) (rd : ι → V) (t : QTree Λ ι V β) (M : ℕ)
    (hM : ∀ w, ((t.asks rd).map Prod.snd).count w ≤ M) :
    ((t.asks rd).map fun q => a q.2 (rd q.2)).sum
      ≤ M * ((t.firsts rd ∅).map fun q => a q.2 (rd q.2)).sum := by
  classical
  obtain ⟨hnd, hset⟩ := firsts_keys_eq rd t ∅
  rw [Finset.sdiff_empty] at hset
  have h1 : ((t.asks rd).map fun q => a q.2 (rd q.2)).sum
      = ∑ w ∈ ((t.asks rd).map Prod.snd).toFinset,
        ((t.asks rd).map Prod.snd).count w • a w (rd w) := by
    rw [show ((t.asks rd).map fun q => a q.2 (rd q.2))
        = ((t.asks rd).map Prod.snd).map fun w => a w (rd w) by simp [List.map_map]]
    exact Finset.sum_list_map_count _ _
  have h2 : ((t.firsts rd ∅).map fun q => a q.2 (rd q.2)).sum
      = ∑ w ∈ ((t.asks rd).map Prod.snd).toFinset, a w (rd w) := by
    rw [show ((t.firsts rd ∅).map fun q => a q.2 (rd q.2))
        = ((t.firsts rd ∅).map Prod.snd).map fun w => a w (rd w) by simp [List.map_map],
      ← hset, List.sum_toFinset _ hnd]
  rw [h1, h2, Finset.mul_sum]
  refine Finset.sum_le_sum fun w _ => ?_
  rw [nsmul_eq_mul]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hM w) zero_le

/-- The first reads are among the reads. -/
theorem firsts_sum_le (a : ι → V → ENNReal) (rd : ι → V) (t : QTree Λ ι V β) :
    ((t.firsts rd ∅).map fun q => a q.2 (rd q.2)).sum
      ≤ ((t.asks rd).map fun q => a q.2 (rd q.2)).sum := by
  classical
  obtain ⟨hnd, hset⟩ := firsts_keys_eq rd t ∅
  rw [Finset.sdiff_empty] at hset
  have h1 : ((t.asks rd).map fun q => a q.2 (rd q.2)).sum
      = ∑ w ∈ ((t.asks rd).map Prod.snd).toFinset,
        ((t.asks rd).map Prod.snd).count w • a w (rd w) := by
    rw [show ((t.asks rd).map fun q => a q.2 (rd q.2))
        = ((t.asks rd).map Prod.snd).map fun w => a w (rd w) by simp [List.map_map]]
    exact Finset.sum_list_map_count _ _
  have h2 : ((t.firsts rd ∅).map fun q => a q.2 (rd q.2)).sum
      = ∑ w ∈ ((t.asks rd).map Prod.snd).toFinset, a w (rd w) := by
    rw [show ((t.firsts rd ∅).map fun q => a q.2 (rd q.2))
        = ((t.firsts rd ∅).map Prod.snd).map fun w => a w (rd w) by simp [List.map_map],
      ← hset, List.sum_toFinset _ hnd]
  rw [h1, h2]
  refine Finset.sum_le_sum fun w hw => ?_
  have : 1 ≤ ((t.asks rd).map Prod.snd).count w :=
    List.count_pos_iff.2 (List.mem_toFinset.1 hw)
  calc a w (rd w) = 1 • a w (rd w) := by simp
    _ ≤ _ := nsmul_le_nsmul_left zero_le this

end QTree

/-- The reads `ARU` can take. -/
noncomputable def ARU.fintype : Fintype ARU := Fintype.ofFinite ARU

attribute [local instance] ARU.fintype

namespace QTree

variable {Λ ι β : Type*} [DecidableEq ι] [Inhabited ι]
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Over independent reads, the indicator of a key's value times any function of the other keys'
reads has the product mean. -/
theorem lintegral_ind_mul (read : ι → Ω → ARU) (hmeas : ∀ w, Measurable (read w))
    (hind : iIndepFun read μ) (w : ι) (v : ARU) (K : Finset ι) (F : (ι → ARU) → ENNReal)
    (hF : ∀ rd rd' : ι → ARU, (∀ y ∈ K, y ≠ w → rd y = rd' y) → F rd = F rd') :
    ∫⁻ ω, (if read w ω = v then 1 else 0) * F (fun y => read y ω) ∂μ
      = μ {ω | read w ω = v} * ∫⁻ ω, F (fun y => read y ω) ∂μ := by
  classical
  set R := K.erase w
  have hdisj : Disjoint ({w} : Finset ι) R := by simp [R]
  have hI := hind.indepFun_finset {w} R hdisj hmeas
  set Φ : (R → ARU) → ENNReal := fun z =>
    F fun y => if h : y ∈ R then z ⟨y, h⟩ else .accept
  have hΦ : ∀ ω, F (fun y => read y ω) = Φ (fun i : R => read i ω) := by
    intro ω
    refine hF _ _ fun y hy hyw => ?_
    simp only [Φ, R, Finset.mem_erase]
    rw [dif_pos ⟨hyw, hy⟩]
  set ψ : (({w} : Finset ι) → ARU) → ENNReal := fun z =>
    if z ⟨w, Finset.mem_singleton_self w⟩ = v then 1 else 0
  have hψ : ∀ ω, (if read w ω = v then (1 : ENNReal) else 0) = ψ (fun i : ({w} : Finset ι) =>
      read i ω) := fun ω => rfl
  have hmA : Measurable fun ω (i : ({w} : Finset ι)) => read i ω :=
    measurable_pi_lambda _ fun i => hmeas i
  have hmB : Measurable fun ω (i : R) => read i ω := measurable_pi_lambda _ fun i => hmeas i
  have hI' := hI.comp (φ := ψ) (ψ := Φ) (measurable_of_countable _) (measurable_of_countable _)
  simp_rw [hψ, hΦ]
  have hmul := lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun
    (f := fun ω => ψ (fun i => read i ω)) (g := fun ω => Φ (fun i => read i ω))
    ((measurable_of_countable ψ).comp hmA) ((measurable_of_countable Φ).comp hmB) hI'
  simp only [Pi.mul_apply] at hmul
  rw [hmul]
  congr 1
  rw [show (∫⁻ ω, ψ (fun i : ({w} : Finset ι) => read i ω) ∂μ)
      = ∫⁻ ω, {ω | read w ω = v}.indicator 1 ω ∂μ from
    lintegral_congr fun ω => by simp [ψ, Set.indicator_apply]]
  exact lintegral_indicator_one (hmeas w (measurableSet_singleton v))

theorem ind_measurable (read : ι → Ω → ARU) (hmeas : ∀ w, Measurable (read w)) (w : ι) (v : ARU) :
    Measurable fun ω => if read w ω = v then (1 : ENNReal) else 0 :=
  (measurable_of_countable (fun a : ARU => if a = v then (1 : ENNReal) else 0)).comp (hmeas w)

theorem lintegral_split (read : ι → Ω → ARU) (hmeas : ∀ w, Measurable (read w)) (w : ι)
    (g : Ω → ENNReal) (hg : Measurable g) :
    ∫⁻ ω, g ω ∂μ = ∑ v : ARU, ∫⁻ ω, (if read w ω = v then 1 else 0) * g ω ∂μ := by
  rw [← lintegral_finset_sum (f := fun v ω => (if read w ω = v then (1 : ENNReal) else 0) * g ω)
    _ fun v _ => (ind_measurable read hmeas w v).mul hg]
  congr 1
  ext ω
  rw [Finset.sum_eq_single (read w ω)]
  · simp
  · intro v _ hv; simp [Ne.symm hv]
  · simp

/-- Freshly read: a read family with the key `w` set to `v` is still independent. -/
theorem iIndepFun_upd (read : ι → Ω → ARU) (hind : iIndepFun read μ) (w : ι) (v : ARU) :
    iIndepFun (fun y ω => upd (fun y => read y ω) w v y) μ := by
  have := hind.comp (fun y (a : ARU) => if y = w then v else a) fun _ => measurable_of_countable _
  have h : (fun y ω => upd (fun y => read y ω) w v y)
      = fun i => (fun a => if i = w then v else a) ∘ read i := by
    funext y ω; simp [upd]
  rw [h]; exact this

theorem lintegral_prob (read : ι → Ω → ARU) (hmeas : ∀ w, Measurable (read w)) (w : ι)
    (g : ARU → ENNReal) :
    ∫⁻ ω, g (read w ω) ∂μ = ∑ v : ARU, μ {ω | read w ω = v} * g v := by
  rw [lintegral_split read hmeas w (fun ω => g (read w ω))
    ((measurable_of_countable g).comp (hmeas w))]
  refine Finset.sum_congr rfl fun v _ => ?_
  have hS : MeasurableSet {ω | read w ω = v} := hmeas w (measurableSet_singleton v)
  have : (fun ω => (if read w ω = v then (1 : ENNReal) else 0) * g (read w ω))
      = fun ω => g v * {ω | read w ω = v}.indicator 1 ω := by
    ext ω; by_cases h : read w ω = v <;> simp [h, Set.indicator_apply]
  rw [this, lintegral_const_mul _ (measurable_one.indicator hS), lintegral_indicator_one hS,
    mul_comm]

/-- The sum over first reads of a quantity `a`, along the run under `rd`. -/
noncomputable def fsum (a : Λ → ι → ARU → ENNReal) (rd : ι → ARU) (P : Finset ι)
    (t : QTree Λ ι ARU β) : ENNReal :=
  ((t.firsts rd P).map fun q => a q.1 q.2 (rd q.2)).sum

/-- The product over first reads of a factor `a`, along the run under `rd`. -/
noncomputable def fprod (a : Λ → ι → ARU → ENNReal) (rd : ι → ARU) (P : Finset ι)
    (t : QTree Λ ι ARU β) : ENNReal :=
  ((t.firsts rd P).map fun q => a q.1 q.2 (rd q.2)).prod

theorem fsum_congr (a : Λ → ι → ARU → ENNReal) {rd rd' : ι → ARU} (t : QTree Λ ι ARU β)
    (P : Finset ι) (h : ∀ w ∈ t.keys, rd w = rd' w) : fsum a rd P t = fsum a rd' P t := by
  unfold fsum
  rw [firsts_congr t P h]
  congr 1
  refine List.map_congr_left fun q hq => ?_
  rw [h q.2 (firsts_keys rd' t P q hq).1]

theorem fprod_congr (a : Λ → ι → ARU → ENNReal) {rd rd' : ι → ARU} (t : QTree Λ ι ARU β)
    (P : Finset ι) (h : ∀ w ∈ t.keys, rd w = rd' w) : fprod a rd P t = fprod a rd' P t := by
  unfold fprod
  rw [firsts_congr t P h]
  congr 1
  refine List.map_congr_left fun q hq => ?_
  rw [h q.2 (firsts_keys rd' t P q hq).1]

/-- On the reads where `w` is `v`, the rest of the run after reading `w` is the run with `w` set
to `v`. -/
theorem fsum_ask (a : Λ → ι → ARU → ENNReal) (rd : ι → ARU) (P : Finset ι) (ℓ : Λ) (w : ι)
    (next : ARU → QTree Λ ι ARU β) :
    fsum a rd P (ask ℓ w next)
      = (if w ∈ P then 0 else a ℓ w (rd w))
        + ∑ v : ARU, (if rd w = v then 1 else 0)
          * fsum a (upd rd w v) (insert w P) (next v) := by
  have hrest : fsum a rd (insert w P) (next (rd w))
      = ∑ v : ARU, (if rd w = v then 1 else 0) * fsum a (upd rd w v) (insert w P) (next v) := by
    rw [Finset.sum_eq_single (rd w)]
    · simp only [if_true, one_mul]
      unfold fsum
      congr 1
      have : upd rd w (rd w) = rd := by ext y; simp [upd]; intro h; rw [h]
      rw [this]
    · intro v _ hv; simp [Ne.symm hv]
    · simp
  by_cases hw : w ∈ P
  · rw [if_pos hw, zero_add, ← hrest]
    unfold fsum
    simp only [firsts, if_pos hw, Finset.insert_eq_of_mem hw]
  · rw [if_neg hw, ← hrest]
    unfold fsum
    simp [firsts, if_neg hw]

theorem fprod_ask (a : Λ → ι → ARU → ENNReal) (rd : ι → ARU) (P : Finset ι) (ℓ : Λ) (w : ι)
    (next : ARU → QTree Λ ι ARU β) :
    fprod a rd P (ask ℓ w next)
      = ∑ v : ARU, (if rd w = v then 1 else 0) * ((if w ∈ P then 1 else a ℓ w v)
          * fprod a (upd rd w v) (insert w P) (next v)) := by
  rw [Finset.sum_eq_single (rd w)]
  · simp only [if_true, one_mul]
    have : upd rd w (rd w) = rd := by ext y; simp [upd]; intro h; rw [h]
    rw [this]
    unfold fprod
    by_cases hw : w ∈ P
    · simp [firsts, if_pos hw, Finset.insert_eq_of_mem hw]
    · simp [firsts, if_neg hw]
  · intro v _ hv; simp [Ne.symm hv]
  · simp

theorem fsum_upd_local (a : Λ → ι → ARU → ENNReal) (w : ι) (v : ARU) (P : Finset ι)
    (t : QTree Λ ι ARU β) :
    ∀ rd rd' : ι → ARU, (∀ y ∈ t.keys, y ≠ w → rd y = rd' y) →
      fsum a (upd rd w v) P t = fsum a (upd rd' w v) P t := by
  intro rd rd' h
  refine fsum_congr a t P fun y hy => ?_
  by_cases hyw : y = w
  · simp [upd, hyw]
  · simp [upd, hyw, h y hy hyw]

theorem fprod_upd_local (a : Λ → ι → ARU → ENNReal) (w : ι) (v : ARU) (P : Finset ι)
    (t : QTree Λ ι ARU β) :
    ∀ rd rd' : ι → ARU, (∀ y ∈ t.keys, y ≠ w → rd y = rd' y) →
      fprod a (upd rd w v) P t = fprod a (upd rd' w v) P t := by
  intro rd rd' h
  refine fprod_congr a t P fun y hy => ?_
  by_cases hyw : y = w
  · simp [upd, hyw]
  · simp [upd, hyw, h y hy hyw]

theorem fsum_measurable (a : Λ → ι → ARU → ENNReal) (read : ι → Ω → ARU)
    (hmeas : ∀ w, Measurable (read w)) (P : Finset ι) (t : QTree Λ ι ARU β) :
    Measurable fun ω => fsum a (fun y => read y ω) P t := by
  classical
  set K := t.keys
  set Φ : (K → ARU) → ENNReal := fun z => fsum a (fun y => if h : y ∈ K then z ⟨y, h⟩ else .accept) P t
  have : (fun ω => fsum a (fun y => read y ω) P t) = fun ω => Φ (fun i : K => read i ω) := by
    ext ω; exact fsum_congr a t P fun y hy => by simp only [dif_pos (show y ∈ K from hy)]
  rw [this]
  exact (measurable_of_countable Φ).comp (measurable_pi_lambda _ fun i => hmeas i)

open scoped Classical in
/-- First reads are fresh: if each key's own mean of `a` is at most its mean of `b`, the mean of
the sum over first reads of `a` is at most that of `b`. -/
theorem firsts_lintegral_le (a b : Λ → ι → ARU → ENNReal) :
    ∀ (t : QTree Λ ι ARU β) (read : ι → Ω → ARU) (P : Finset ι), (∀ w, Measurable (read w)) →
      iIndepFun read μ →
      (∀ ℓ w, w ∉ P → ∫⁻ ω, a ℓ w (read w ω) ∂μ ≤ ∫⁻ ω, b ℓ w (read w ω) ∂μ) →
      ∫⁻ ω, fsum a (fun y => read y ω) P t ∂μ ≤ ∫⁻ ω, fsum b (fun y => read y ω) P t ∂μ
  | done _, _, _, _, _, _ => by simp [fsum, firsts]
  | ask ℓ w next, read, P, hmeas, hind, hab => by
    set read' : ARU → ι → Ω → ARU := fun v y ω => upd (fun y => read y ω) w v y
    have hmeas' : ∀ v y, Measurable (read' v y) := by
      intro v y
      by_cases hy : y = w
      · simp only [read', upd, hy, if_true]; exact measurable_const
      · simp only [read', upd, hy, if_false]; exact hmeas y
    have hind' : ∀ v, iIndepFun (read' v) μ := fun v => iIndepFun_upd read hind w v
    have hab' : ∀ v ℓ' y, y ∉ insert w P →
        ∫⁻ ω, a ℓ' y (read' v y ω) ∂μ ≤ ∫⁻ ω, b ℓ' y (read' v y ω) ∂μ := by
      intro v ℓ' y hy
      have hyw : y ≠ w := fun h => hy (h ▸ Finset.mem_insert_self _ _)
      simp only [read', upd, hyw, if_false]
      exact hab ℓ' y fun h => hy (Finset.mem_insert_of_mem h)
    have hupd : ∀ v ω, upd (fun y => read y ω) w v = fun y => read' v y ω := fun v ω => rfl
    have key : ∀ c : Λ → ι → ARU → ENNReal, ∫⁻ ω, fsum c (fun y => read y ω) P (ask ℓ w next) ∂μ
        = (if w ∈ P then 0 else ∫⁻ ω, c ℓ w (read w ω) ∂μ)
          + ∑ v : ARU, μ {ω | read w ω = v}
            * ∫⁻ ω, fsum c (fun y => read' v y ω) (insert w P) (next v) ∂μ := by
      intro c
      simp_rw [fsum_ask c _ P ℓ w next]
      rw [lintegral_add_left]
      · congr 1
        · split_ifs <;> simp
        · rw [lintegral_finset_sum]
          · refine Finset.sum_congr rfl fun v _ => ?_
            rw [lintegral_ind_mul read hmeas hind w v (next v).keys
              (fun rd => fsum c (upd rd w v) (insert w P) (next v))
              (fsum_upd_local c w v (insert w P) (next v))]
          · intro v _
            exact (ind_measurable read hmeas w v).mul
              (fsum_measurable c (read' v) (hmeas' v) _ _)
      · split_ifs
        · exact measurable_const
        · exact (measurable_of_countable _).comp (hmeas w)
    rw [key a, key b]
    refine add_le_add ?_ (Finset.sum_le_sum fun v _ => ?_)
    · split_ifs with hw
      · exact le_rfl
      · exact hab ℓ w hw
    · exact mul_le_mul_of_nonneg_left (firsts_lintegral_le a b (next v) (read' v) (insert w P)
        (hmeas' v) (hind' v) (hab' v)) zero_le

theorem fprod_measurable (a : Λ → ι → ARU → ENNReal) (read : ι → Ω → ARU)
    (hmeas : ∀ w, Measurable (read w)) (P : Finset ι) (t : QTree Λ ι ARU β) :
    Measurable fun ω => fprod a (fun y => read y ω) P t := by
  classical
  set K := t.keys
  set Φ : (K → ARU) → ENNReal := fun z =>
    fprod a (fun y => if h : y ∈ K then z ⟨y, h⟩ else .accept) P t
  have : (fun ω => fprod a (fun y => read y ω) P t) = fun ω => Φ (fun i : K => read i ω) := by
    ext ω; exact fprod_congr a t P fun y hy => by simp only [dif_pos (show y ∈ K from hy)]
  rw [this]
  exact (measurable_of_countable Φ).comp (measurable_pi_lambda _ fun i => hmeas i)

open scoped Classical in
/-- First reads are fresh: if each key's own mean of the factor `a` is at most `1`, so is the mean
of the product over first reads. -/
theorem firsts_prod_le (a : Λ → ι → ARU → ENNReal) :
    ∀ (t : QTree Λ ι ARU β) (read : ι → Ω → ARU) (P : Finset ι), (∀ w, Measurable (read w)) →
      iIndepFun read μ → (∀ ℓ w, w ∉ P → ∫⁻ ω, a ℓ w (read w ω) ∂μ ≤ 1) →
      ∫⁻ ω, fprod a (fun y => read y ω) P t ∂μ ≤ 1
  | done _, _, _, _, _, _ => by simp [fprod, firsts]
  | ask ℓ w next, read, P, hmeas, hind, ha => by
    set read' : ARU → ι → Ω → ARU := fun v y ω => upd (fun y => read y ω) w v y
    have hmeas' : ∀ v y, Measurable (read' v y) := by
      intro v y
      by_cases hy : y = w
      · simp only [read', upd, hy, if_true]; exact measurable_const
      · simp only [read', upd, hy, if_false]; exact hmeas y
    have hind' : ∀ v, iIndepFun (read' v) μ := fun v => iIndepFun_upd read hind w v
    have ha' : ∀ v ℓ' y, y ∉ insert w P → ∫⁻ ω, a ℓ' y (read' v y ω) ∂μ ≤ 1 := by
      intro v ℓ' y hy
      have hyw : y ≠ w := fun h => hy (h ▸ Finset.mem_insert_self _ _)
      simp only [read', upd, hyw, if_false]
      exact ha ℓ' y fun h => hy (Finset.mem_insert_of_mem h)
    simp_rw [fprod_ask a _ P ℓ w next]
    rw [lintegral_finset_sum (f := fun v ω => (if read w ω = v then (1 : ENNReal) else 0)
      * ((if w ∈ P then 1 else a ℓ w v)
        * fprod a (upd (fun y => read y ω) w v) (insert w P) (next v)))
      _ fun v _ => (ind_measurable read hmeas w v).mul
        (measurable_const.mul (fprod_measurable a (read' v) (hmeas' v) _ _))]
    have hstep : ∀ v : ARU, ∫⁻ ω, (if read w ω = v then 1 else 0) * ((if w ∈ P then 1 else a ℓ w v)
        * fprod a (upd (fun y => read y ω) w v) (insert w P) (next v)) ∂μ
        ≤ μ {ω | read w ω = v} * (if w ∈ P then 1 else a ℓ w v) := by
      intro v
      simp_rw [← mul_assoc, mul_comm _ (if w ∈ P then (1 : ENNReal) else a ℓ w v), mul_assoc]
      rw [lintegral_const_mul (f := fun ω => (if read w ω = v then (1 : ENNReal) else 0)
        * fprod a (upd (fun y => read y ω) w v) (insert w P) (next v)) _
        ((ind_measurable read hmeas w v).mul (fprod_measurable a (read' v) (hmeas' v) _ _))]
      rw [lintegral_ind_mul read hmeas hind w v (next v).keys
        (fun rd => fprod a (upd rd w v) (insert w P) (next v))
        (fprod_upd_local a w v (insert w P) (next v))]
      have := firsts_prod_le a (next v) (read' v) (insert w P) (hmeas' v) (hind' v) (ha' v)
      calc (if w ∈ P then 1 else a ℓ w v) * (μ {ω | read w ω = v}
            * ∫⁻ ω, fprod a (upd (fun y => read y ω) w v) (insert w P) (next v) ∂μ)
          ≤ (if w ∈ P then 1 else a ℓ w v) * (μ {ω | read w ω = v} * 1) := by gcongr
        _ = _ := by rw [mul_one, mul_comm]
    refine (Finset.sum_le_sum fun v _ => hstep v).trans ?_
    by_cases hw : w ∈ P
    · simp only [if_pos hw, mul_one]
      have h := lintegral_prob (μ := μ) read hmeas w (fun _ => (1 : ENNReal))
      simp only [mul_one, lintegral_const, measure_univ] at h
      rw [← h]
    · simp only [if_neg hw]
      rw [← lintegral_prob read hmeas w (a ℓ w)]
      exact ha ℓ w hw

end QTree

end OrthoDFA
