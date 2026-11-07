import Mathlib.Data.Fintype.Basic

/-!
# Programs that query an oracle

A `Query Q R β` asks an oracle `o : Q → R` one question at a time, each answer deciding what it
asks next, and returns a `β`. `eval o p` runs `p` against `o`, and `reads o p` is the set of
questions that run asks. Every piece of the learner that reads the oracle is written as such a
program, so the questions a run asks are those of the program itself, and a run against any
oracle agreeing with `o` on `reads o p` returns `eval o p` and asks `reads o p`
(`Query.eval_congr`). `bind` asks the first program's questions and then those of the second at
the first's answer, so its reads are the union of the two runs'.
-/

universe u v w w'

namespace OrthoDFA

/-- A program asking questions in `Q`, answered in `R`, returning a `β`. -/
inductive Query (Q : Type u) (R : Type v) (β : Type w) : Type (max u v w)
  | pure : β → Query Q R β
  | ask : Q → (R → Query Q R β) → Query Q R β

namespace Query

variable {Q : Type u} {R : Type v} {β : Type w} {γ : Type w'}

def bind : Query Q R β → (β → Query Q R γ) → Query Q R γ
  | pure b, f => f b
  | ask q k, f => ask q fun r => (k r).bind f

def map (f : β → γ) (p : Query Q R β) : Query Q R γ := p.bind fun b => pure (f b)

instance : Monad (Query.{u, v, w} Q R) where
  pure := pure
  bind := bind

/-- The question `q`, returning its answer. -/
def query (q : Q) : Query Q R R := ask q pure

/-- What `p` returns against the oracle `o`. -/
def eval (o : Q → R) : Query Q R β → β
  | pure b => b
  | ask q k => (k (o q)).eval o

/-- The questions `p` asks against the oracle `o`. -/
def reads [DecidableEq Q] (o : Q → R) : Query Q R β → Finset Q
  | pure _ => ∅
  | ask q k => insert q ((k (o q)).reads o)

/-- `p` with each of its questions `q` answered by running `h q`, a program over `Q'`. -/
def subst {Q' : Type*} {R' : Type*} (h : Q → Query Q' R' R) : Query Q R β → Query Q' R' β
  | pure b => pure b
  | ask q k => (h q).bind fun r => (k r).subst h

/-- `f i` for each `i` of `l` in turn, the earlier answer kept where `i` repeats; `default` off
`l`. -/
def piList {ι : Type*} [DecidableEq ι] [Inhabited γ] (f : ι → Query Q R γ) :
    List ι → Query Q R (ι → γ)
  | [] => pure fun _ => default
  | i :: l => (f i).bind fun r => (piList f l).map fun g => Function.update g i r

/-- `f i` for every `i`. -/
noncomputable def pi {ι : Type*} [Fintype ι] [DecidableEq ι] [Inhabited γ]
    (f : ι → Query Q R γ) : Query Q R (ι → γ) :=
  piList f Finset.univ.toList

/-- The answers at the questions `l`, `default` elsewhere. -/
def table [DecidableEq Q] [Inhabited R] (l : List Q) : Query Q R (Q → R) := piList query l

/-- `f` over `l` in turn, stopping at the first `some`. -/
def findSome {δ : Type*} {ε : Type w'} (f : δ → Query Q R (Option ε)) :
    List δ → Query Q R (Option ε)
  | [] => pure none
  | x :: xs => (f x).bind fun
    | some y => pure (some y)
    | none => findSome f xs

end Query

end OrthoDFA
