"""Print one random DFA from the Ye et al. benchmark suite."""

import sys
import os

sys.path.insert(0, "/tmp/noisy_learning/source")

import numpy as np

from dfa import random_dfa
from benchmarking_noisy_dfa import minimize_dfa

np.random.seed(seed=2)

full_alphabet = "abcdefghijklmnopqrstuvwxyz"
alph_size = np.random.randint(4, 20)
alphabet = full_alphabet[:alph_size]

while True:
    dfa_rand = random_dfa(alphabet, min_state=20, max_states=60)
    dfa = minimize_dfa(dfa_rand)
    if len(dfa.states) > 20:
        break

print(f"States: {len(dfa.states)}")
print(f"Alphabet: {dfa.alphabet}")
print(f"Initial state: {dfa.init_state}")
print(f"Final states: {sorted(dfa.final_states)}")
print()
print("Transitions:")
for state in sorted(dfa.transitions.keys()):
    row = {letter: dfa.transitions[state][letter] for letter in dfa.alphabet}
    print(f"  {state}: {row}")
