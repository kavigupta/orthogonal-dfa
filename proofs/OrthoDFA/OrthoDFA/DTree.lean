import OrthoDFA.FamilyRead

/-! A tree that sifts strings to its leaves, reading each string followed by a node's midfix. -/

namespace OrthoDFA

variable {α : Type*}

/-- A tree whose nodes read the string followed by their midfix. -/
inductive DTree (α : Type*)
  | leaf
  | node (m : FreeMonoid α) (rej acc : DTree α)

namespace DTree

def leaves : DTree α → List (List Bool)
  | leaf => [[]]
  | node _ r a => r.leaves.map (false :: ·) ++ a.leaves.map (true :: ·)

def midfixAt : DTree α → List Bool → FreeMonoid α
  | leaf, _ => 1
  | node m _ _, [] => m
  | node _ r _, false :: p => r.midfixAt p
  | node _ _ a, true :: p => a.midfixAt p

def depth : DTree α → ℕ
  | leaf => 0
  | node _ r a => max r.depth a.depth + 1

/-- The strings sifting `z` reads, in order, and the path to its leaf, `none` where the last read
is undecided. -/
def siftWithPath (cut : FreeMonoid α → Option Bool) :
    DTree α → FreeMonoid α → List (FreeMonoid α) × Option (List Bool)
  | leaf, _ => ([], some [])
  | node m r a, z =>
    match cut (z * m) with
    | some true => (z * m :: (a.siftWithPath cut z).1, (a.siftWithPath cut z).2.map (true :: ·))
    | some false => (z * m :: (r.siftWithPath cut z).1, (r.siftWithPath cut z).2.map (false :: ·))
    | none => ([z * m], none)

/-- The path to the leaf `z` sifts to, or the string whose read was undecided. -/
def sift (read : FreeMonoid α → ARU) : DTree α → FreeMonoid α → List Bool ⊕ FreeMonoid α
  | leaf, _ => .inl []
  | node m r a, z =>
    match read (z * m) with
    | .accept => (a.sift read z).map (true :: ·) id
    | .reject => (r.sift read z).map (false :: ·) id
    | .undecided => .inr (z * m)

/-- The leaf at `p` replaced by a node reading at `d`; `MidfixTree.split` names the leaf by its
state rather than its path. -/
def splitAt (d : FreeMonoid α) : DTree α → List Bool → DTree α
  | leaf, [] => node d leaf leaf
  | node m r a, false :: p => node m (r.splitAt d p) a
  | node m r a, true :: p => node m r (a.splitAt d p)
  | t, _ => t

end DTree

/-- A read's side, or `none` where undecided. -/
def ARU.cut : ARU → Option Bool
  | .accept => some true
  | .reject => some false
  | .undecided => none

def commonPrefix : List Bool → List Bool → List Bool
  | a :: as, b :: bs => if a = b then a :: commonPrefix as bs else []
  | _, _ => []

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
