import RLStar.LowLevelClaims.FamilyRead

/-! A band's vote reads its rarer decided side at most `κ` times as often as undecided, whatever
its two rates. -/

namespace OrthoDFA

/-- With `Y ~ Bin(n, q)`:

    κ = max( P(Y ≥ kh) / P(c ≤ Y < kh),  P(Y ≤ kl) / P(kl < Y ≤ c) ) -/
noncomputable def bandKappa (n : ℕ) (q : ℝ) (kl c kh : ℕ) : ℝ :=
  max (binomSfGe n q kh / (binomSfGe n q c - binomSfGe n q kh))
    ((1 - binomSfGe n q (kl + 1)) / (binomSfGe n q (kl + 1) - binomSfGe n q (c + 1)))

/-- Over `N ≤ n` queries at any two rates, the read at `kl` and `kh` is on its rarer decided side
at most `bandKappa n q kl c kh` times as often as undecided, for any count `c` with `kl < c < kh`
and `c ≤ n` and any `q` strictly between `0` and `1`:

    min(P(X ≥ kh), P(X ≤ kl)) ≤ bandKappa n q kl c kh · P(kl < X < kh) -/
def BandKappaBound : Prop :=
  ∀ (N n kl c kh : ℕ) (q p r : ℝ), N ≤ n → kl < c → c < kh → c ≤ n → 0 < q → q < 1 →
    0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → RarerBounded N kl kh p r (bandKappa n q kl c kh)

end OrthoDFA
