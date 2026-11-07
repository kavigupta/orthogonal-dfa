import OrthoDFA.Code.Level3

/-! What walking, splitting and learning edges promise. -/

namespace OrthoDFA

def WalkCorrect : Prop :=
  ∀ {α : Type*},
    (∀ (H : HypothesisDFA α) (s : List Bool) (cs : List α),
      let ss := walk H.edges s cs
      -- The walk starts at `s` and visits at most one leaf per letter after it.
      ss.head? = some s ∧ ss.length ≤ cs.length + 1
      -- Each step follows the learned edge on the next letter,
      ∧ (∀ i, i + 1 < ss.length →
          ∃ c w, cs[i]? = some c ∧ H.edges (ss.getD i [], c) = some (ss.getD (i + 1) [], w))
      -- and it stops short only at an unlearned edge.
      ∧ (ss.length ≤ cs.length →
          ∃ c, cs[ss.length - 1]? = some c ∧ H.edges (ss.getD (ss.length - 1) [], c) = none)
      -- `walkEnd` is where a walk that follows every letter ends.
      ∧ (∀ q, walkEnd H.edges s cs = some q ↔ ss.length = cs.length + 1 ∧ ss.getLast? = some q)
      -- Where every learned edge out of a leaf leads to a leaf, a walk from a leaf stays on them.
      ∧ ((∀ q ∈ H.tree.leaves, ∀ c t w, H.edges (q, c) = some (t, w) → t ∈ H.tree.leaves) →
          s ∈ H.tree.leaves → ∀ q ∈ ss, q ∈ H.tree.leaves))
    ∧ (∀ (x : FreeMonoid α) (i : ℕ),
      -- A prefix grows by the next letter, and stops at the whole string.
      (∀ c, x.toList[i]? = some c → prefixOf x (i + 1) = prefixOf x i * FreeMonoid.of c)
      ∧ (x.toList.length ≤ i → prefixOf x i = x))

def SplitAndLearnCorrect : Prop :=
  ∀ {α : Type*} [DecidableEq α] [Fintype α] (read : FreeMonoid α → ARU) (H : HypothesisDFA α),
    EdgesOK read H.tree H.edges →
    -- Splitting a leaf re-sifts the witnesses of the edges into it, so every learned edge stays
    -- witnessed,
    (∀ d p, p ∈ H.tree.leaves → EdgesOK read (H.split read d p).tree (H.split read d p).edges)
    -- and so does learning an edge from a string at its leaf whose successor sifts to its target.
    ∧ (∀ (u : FreeMonoid α) (s t : List Bool) (c : α), H.tree.sift read u = .inl s →
        H.tree.sift read (u * FreeMonoid.of c) = .inl t →
        EdgesOK read H.tree (Function.update H.edges (s, c) (some (t, u))))

/-- Every learned edge along a level-2 or level-3 run is at a leaf, so the leaf restriction in
`HypothesisDFA.splitQ` drops none, and the split is hypothesis_dfa.HypothesisDFA.split_leaf's,
which retargets every edge. -/
def RunsKeepEdgesAtLeaves : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (xs : List (FreeMonoid α)),
    -- Each state a run splits at ends the run over the probes before it.
    (∀ C : Level2.Settings,
      let s := (Level2.runProbes read C Level2.initialState xs).1
      EdgesAtLeaves ⟨s.tree, s.edges⟩)
    ∧ ∀ C : Random.Level3Settings,
      let s := (Random.runProbes read C Random.initialState xs).1
      EdgesAtLeaves ⟨s.tree, s.edges⟩

end OrthoDFA
