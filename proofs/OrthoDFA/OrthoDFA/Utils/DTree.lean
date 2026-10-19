import OrthoDFA.DTree

/-! What sifting a tree promises. -/

namespace OrthoDFA

def SiftWithPathCorrect : Prop :=
  ∀ {α : Type*} (cut : FreeMonoid α → Option Bool) (T : DTree α) (z : FreeMonoid α),
    let reads := (T.siftWithPath cut z).1
    let path := (T.siftWithPath cut z).2
    -- The `i`th read is `z` followed by the midfix of the node the first `i` answers lead to.
    (∀ i (h : i < reads.length), reads[i] = z * T.midfixAt ((reads.filterMap cut).take i))
    -- One read per level at most.
    ∧ reads.length ≤ T.depth
    -- A decided sift ends at a leaf, along the path its answers spell, with every read decided.
    ∧ (∀ p, path = some p → p ∈ T.leaves ∧ p = reads.filterMap cut ∧ ∀ w ∈ reads, cut w ≠ none)
    -- An undecided sift stops at its first undecided read.
    ∧ (path = none →
        ∃ w, reads.getLast? = some w ∧ cut w = none ∧ ∀ w' ∈ reads.dropLast, cut w' ≠ none)

def SiftAndSplitCorrect : Prop :=
  ∀ {α : Type*} (read : FreeMonoid α → ARU) (T : DTree α) (z : FreeMonoid α),
    let withPath := T.siftWithPath (fun w => (read w).cut) z
    -- `sift` lands on the leaf `siftWithPath` reaches under the read's cut,
    (∀ p, T.sift read z = .inl p ↔ withPath.2 = some p)
    -- or stops at its last read, the undecided one.
    ∧ (∀ w, T.sift read z = .inr w ↔ withPath.2 = none ∧ withPath.1.getLast? = some w)
    -- Strings at different leaves read decidedly apart where their paths part.
    ∧ (∀ (z' : FreeMonoid α) (p p' : List Bool),
        T.sift read z = .inl p → T.sift read z' = .inl p' → p ≠ p' →
          read (z * T.midfixAt (commonPrefix p p')) ≠ .undecided
          ∧ read (z' * T.midfixAt (commonPrefix p p')) ≠ .undecided
          ∧ read (z * T.midfixAt (commonPrefix p p')) ≠ read (z' * T.midfixAt (commonPrefix p p')))
    ∧ ∀ (d : FreeMonoid α) (p : List Bool), p ∈ T.leaves →
      -- Splitting the leaf `p` replaces it by its two children,
      (∀ q, q ∈ (T.splitAt d p).leaves ↔
          (q ∈ T.leaves ∧ q ≠ p) ∨ q = p ++ [false] ∨ q = p ++ [true])
      ∧ (T.splitAt d p).leaves.length = T.leaves.length + 1
      -- reads at `d` there,
      ∧ (T.splitAt d p).midfixAt p = d
      -- and changes only the sifts that landed on `p`, which now read `d` too.
      ∧ (T.splitAt d p).sift read z = (T.sift read z).elim
          (fun q => if q = p then ((DTree.node d .leaf .leaf).sift read z).map (p ++ ·) id
            else .inl q) .inr

end OrthoDFA
