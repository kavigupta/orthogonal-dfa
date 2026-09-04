import pickle, numpy as np
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps/round_00.pkl"
NAME = {0: "TAG", 1: "TAA", 2: "TGA", 3: "X", 4: "Y"}

with open(DUMP, "rb") as f:
    d = pickle.load(f)
dfa = d["dfa"]
print("round", d.get("round"), "est", d.get("est"), "decision_boundary", d.get("decision_boundary"))
print("states:", sorted(dfa.states))
print("initial:", dfa.initial_state)
print("accepting:", sorted(dfa.final_states))
print("input symbols:", sorted(dfa.input_symbols))
print()
print("transitions (state -> symbol -> next):")
for s in sorted(dfa.states):
    row = dfa.transitions[s]
    parts = []
    for sym in sorted(row, key=lambda k: (str(type(k)), k)):
        label = NAME.get(sym, sym) if isinstance(sym, int) else (NAME.get(int(sym), sym) if str(sym).isdigit() else sym)
        parts.append(f"{label}->{row[sym]}")
    acc = "ACCEPT" if s in dfa.final_states else "reject"
    init = " (init)" if s == dfa.initial_state else ""
    print(f"  {s}{init} [{acc}]: " + "  ".join(parts))

# Does it treat X and Y identically from every state?
def sym_key(row, want):
    for k in row:
        if (isinstance(k, int) and k == want) or (str(k) == str(want)):
            return row[k]
    return None
print()
xy_same = all(sym_key(dfa.transitions[s], 3) == sym_key(dfa.transitions[s], 4) for s in dfa.states)
print("X and Y transition identically from every state:", xy_same)

# What must a super-string contain/avoid to be accepted?  Show acceptance of a few probes.
def acc(seq): return bool(dfa.accepts_input(seq))
probes = {
  "empty": [],
  "all X (len6)": [3]*6,
  "all Y (len6)": [4]*6,
  "one TAG then X*": [0]+[3]*5,
  "one TAA then X*": [1]+[3]*5,
  "one TGA then X*": [2]+[3]*5,
  "X* then TAG": [3]*5+[0],
  "TAG,TAA,TGA,X*": [0,1,2]+[3]*3,
}
print("\nprobe acceptance:")
for k,v in probes.items():
    print(f"  {k:22s}: {'ACCEPT' if acc(v) else 'reject'}")
