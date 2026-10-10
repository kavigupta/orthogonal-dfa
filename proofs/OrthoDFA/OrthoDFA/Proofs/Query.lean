import OrthoDFA.StartAtK

/-!
# Computations asking the cut one string at a time
-/

namespace OrthoDFA

/-- A computation that asks the cut about strings one at a time, each ask tagged. -/
inductive Qry (α : Type*) (β : Type*)
  | pure (b : β)
  | ask (w : FreeMonoid α) (tag : Bool) (k : Option Bool → Qry α β)

namespace Qry

variable {α β γ : Type*}

def run (cut : FreeMonoid α → Option Bool) : Qry α β → β
  | pure b => b
  | ask w _ k => (k (cut w)).run cut

def bind : Qry α β → (β → Qry α γ) → Qry α γ
  | pure b, f => f b
  | ask w g k, f => ask w g fun o => (k o).bind f

def map (f : β → γ) (q : Qry α β) : Qry α γ := q.bind fun b => pure (f b)

theorem run_bind (cut : FreeMonoid α → Option Bool) (f : β → Qry α γ) :
    ∀ q : Qry α β, (q.bind f).run cut = (f (q.run cut)).run cut
  | pure _ => rfl
  | ask w g k => run_bind cut f (k (cut w))

theorem run_map (cut : FreeMonoid α → Option Bool) (f : β → γ) (q : Qry α β) :
    (q.map f).run cut = f (q.run cut) := run_bind cut _ q

end Qry

variable {α : Type*} [Fintype α] [DecidableEq α]

end OrthoDFA
