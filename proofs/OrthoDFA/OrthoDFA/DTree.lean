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

end OrthoDFA
