import OrthoDFA.FamilyRead

/-! A band's vote reads its rarer decided side at most `κ` times as often as undecided, whatever
its two rates. -/

namespace OrthoDFA

/-- With `Y ~ Bin(n, 1/2)`:

    κ = max( P(Y ≥ kh) / P(c ≤ Y < kh),  P(Y ≤ kl) / P(kl < Y ≤ c) ) -/
noncomputable def bandKappa (n kl c kh : ℕ) : ℝ :=
  max (binomSfGe n (1 / 2) kh / (binomSfGe n (1 / 2) c - binomSfGe n (1 / 2) kh))
    ((1 - binomSfGe n (1 / 2) (kl + 1))
      / (binomSfGe n (1 / 2) (kl + 1) - binomSfGe n (1 / 2) (c + 1)))

/-- Over `N ≤ n` queries at any two rates, the read at `kl` and `kh` is on its rarer decided side
at most `bandKappa n kl c kh` times as often as undecided, for any count `c` with `kl < c < kh` and
`c ≤ n`:

    min(P(X ≥ kh), P(X ≤ kl)) ≤ bandKappa n kl c kh · P(kl < X < kh) -/
def BandKappaBound : Prop :=
  ∀ (N n kl c kh : ℕ) (p r : ℝ), N ≤ n → kl < c → c < kh → c ≤ n →
    0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → RarerBounded N kl kh p r (bandKappa n kl c kh)

end OrthoDFA
