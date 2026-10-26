import OrthoDFA.Code.Probe

/-! What the probe reports of a hypothesis DFA's edges and of the strings it reads undecided. -/

namespace OrthoDFA

variable {α : Type*}

def ProbeCorrect : Prop :=
  ∀ {α : Type*} (read : FreeMonoid α → ARU) (H : HypothesisDFA α) (k : ℕ) (x : FreeMonoid α),
    EdgesOK read H.tree H.edges →
    let o := probe read H.tree H.edges k x
    -- A wrong edge is one the probe crossed: a string at its leaf whose successor sifts off its
    -- target,
    (∀ s c u t, o = .wrongEdge s c u t → s ∈ H.tree.leaves ∧ H.tree.sift read u = .inl s
        ∧ H.tree.sift read (u * FreeMonoid.of c) = .inl t
        ∧ ∃ t₀ w₀, H.edges (s, c) = some (t₀, w₀) ∧ t₀ ≠ t)
    -- and an unlearned edge is the one the walk stopped at, with such a string.
    ∧ (∀ s c u t, o = .unlearnedEdge s c u t → s ∈ H.tree.leaves ∧ H.tree.sift read u = .inl s
        ∧ H.tree.sift read (u * FreeMonoid.of c) = .inl t ∧ H.edges (s, c) = none)
    -- An undecided string it reports reads undecided, and is a prefix of the probe at least `k`
    -- long followed by a midfix;
    ∧ (∀ z ∈ o.undecidedStrings, read z = .undecided ∧ ∃ p, k ≤ p ∧ ∃ m, z = prefixOf x p * m)
    -- it reports at most one,
    ∧ o.undecidedStrings.length ≤ 1
    -- and where the walk and the sift part, it does not come out clean.
    ∧ (Disagrees read H k x → o.clean = false)

end OrthoDFA
