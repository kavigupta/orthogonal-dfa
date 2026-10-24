import OrthoDFA.Code.FamilyRead

namespace OrthoDFA

variable {α : Type*}

inductive RoundEnd (H : Type*)
  | consistent
  | harvest (h : H)
  | failed

end OrthoDFA
