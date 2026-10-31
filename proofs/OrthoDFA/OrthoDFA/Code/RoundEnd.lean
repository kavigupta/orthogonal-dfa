import OrthoDFA.Code.FamilyRead

/-! The ways a round of the learner's loop ends. -/

namespace OrthoDFA

variable {α : Type*}

inductive RoundEnd (H : Type*)
  | consistent
  | harvest (h : H)
  | failed

end OrthoDFA
