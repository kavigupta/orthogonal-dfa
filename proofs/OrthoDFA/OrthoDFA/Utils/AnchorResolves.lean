import OrthoDFA.FamilyRead

/-!
# A majority-pattern anchor resolves its atoms' states

Each atom `z_i` reads every candidate `v ∈ G` once, as `MQ(z_i·v)`, and the candidates are kept
whose reads form a most common pattern `b̂`. Write `ρ_i(v) = P(MQ(z_i·v) = b̂_i)`, `x ≥ 1/2 ≥ y`
its two values, on `z_i`'s side of `v` and off it, `A` the candidates reading `b̂` at every other
atom and `C ⊆ A` those reading `b̂_i` at `z_i` too. Then

    ∑_{v ∈ C} ρ_i(v) − (x + y − 2xy)|C| = 2xy(|C| − |A|/2) + ∑_{v ∈ A} f_v,

with `f_v` independent, of mean `0` and range at most `1`, and `|C| ≥ |A|/2` since flipping
`b̂_i` keeps no more candidates. With `x + y − 2xy = 1/2 + (1 − 2ηIn)(1 − 2ηOut)/2`, each atom's
state votes on its side of the pattern with margin `(1 − 2η₀)²/2` less a Hoeffding deviation.
-/

namespace OrthoDFA

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S]

/-- `MQ(s) = 1` exactly when `β`. -/
def readsAs (mq : S → Ω → ℝ) (s : S) (β : Bool) (ω : Ω) : Prop := mq s ω = 1 ↔ β = true

open scoped Classical in
/-- The candidates whose atom reads are `b`. -/
noncomputable def patternSet {m : ℕ} (mq : S → Ω → ℝ) (z : Fin m → S) (G : Finset S)
    (b : Fin m → Bool) (ω : Ω) : Finset S :=
  G.filter (fun v => ∀ i, readsAs mq (z i * v) (b i) ω)

/-- Accept for `true`, reject for `false`. -/
def sideRead (β : Bool) : ARU := if β then .accept else .reject

open scoped Classical in
/-- With `κ₀ = (1 − 2η₀)²/2`, `t = hoeffDev |G| (δ/(m 2^m + 1))`, at least `pAP` of `G`
class-preserving and the queries `z_i·v` distinct, but for chance `δ` over the noise, every most
common pattern `b̂`, with `C` the candidates reading it, has

    |C| ≥ (1 − η₀)^m pAP |G| − t

and at every atom `i`

    ∑_{v ∈ C} P(MQ(z_i·v) = b̂_i) ≥ (1/2 + κ₀)|C| − t.

So for any `w` that each candidate sends where it sends `z_i` (any `w` at `z_i`'s state) and any
band `kl < kh` over `W` with `g ≥ 0`,

    kh ≤ W(1/2 + κ₀ − t/|C| − g),   W(1/2 − κ₀ + t/|C| + g) ≤ kl,

a family drawn uniformly among the `W`-subsets of `C` reads `w` off side `b̂_i`, undecided or on
the other side, with chance at most `exp(−2Wg²)`, over the draw and the noise. The side `b̂_i` is
the majority's, which need not be `z_i`'s class: where most of `G` flips `z_i`'s state, the state
is read decidedly flipped. -/
def AnchorResolves : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] (O : Oracle μ S) {m : ℕ} (z : Fin m → S) (G : Finset S)
    (η₀ pAP δ : ℝ),
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < δ →
  (∀ i j, ∀ v ∈ G, ∀ v' ∈ G, z i * v = z j * v' → i = j ∧ v = v') →
  pAP * G.card ≤ (G.filter (fun v => ∀ p, p * v ∈ O.L ↔ p ∈ O.L)).card →
  let κ₀ := (1 - 2 * η₀) ^ 2 / 2
  let t := hoeffDev G.card (δ / (m * 2 ^ m + 1))
  1 - δ ≤ μ.real {ω | ∀ b : Fin m → Bool,
    (∀ b', (patternSet O.mq z G b' ω).card ≤ (patternSet O.mq z G b ω).card) →
    let C := patternSet O.mq z G b ω
    (1 - η₀) ^ m * pAP * G.card - t ≤ C.card ∧
    ∀ i, (1 / 2 + κ₀) * C.card - t ≤ ∑ v ∈ C, μ.real {ω' | readsAs O.mq (z i * v) (b i) ω'} ∧
      ∀ w, (∀ v ∈ G, w * v ∈ O.L ↔ z i * v ∈ O.L) →
      ∀ (W kl kh : ℕ) (g : ℝ), 0 ≤ g → kl < kh →
        (kh : ℝ) ≤ W * (1 / 2 + κ₀ - t / C.card - g) →
        W * (1 / 2 - κ₀ + t / C.card + g) ≤ kl →
        ∑ F ∈ C.powersetCard W, μ.real {ω' | familyRead O.mq F kl kh w ω' ≠ sideRead (b i)}
          ≤ C.card.choose W * Real.exp (-2 * W * g ^ 2)}

end OrthoDFA
