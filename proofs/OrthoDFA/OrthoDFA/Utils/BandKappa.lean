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

/-- Take `N ≤ n` queries, `a` of them answering 1 with chance `p` and the rest with chance `r`,
and read their count `X` at `kl` and `kh`. For any count `c` with `kl < c < kh` and `c ≤ n`:

    min(P(X ≥ kh), P(X ≤ kl)) ≤ bandKappa n kl c kh · P(kl < X < kh) -/
def BandKappaBound : Prop :=
  ∀ (N n a kl c kh : ℕ) (p r : ℝ), a ≤ N → N ≤ n → kl < c → c < kh → c ≤ n →
    0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
    let dist : ARU → ℝ := fun rd => ∑ j ∈ Finset.range (N + 1),
      if readOf kl kh j = rd then
        ∑ x ∈ Finset.antidiagonal j,
          (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
            * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))
      else 0
    min (dist .accept) (dist .reject) ≤ bandKappa n kl c kh * dist .undecided

end OrthoDFA
