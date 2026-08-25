import re
from dataclasses import dataclass
from typing import Tuple

from automata.fa.dfa import DFA
from automata.fa.nfa import NFA

from orthogonal_dfa.l_star.structures import NoiseModel, Oracle
from orthogonal_dfa.utils.dfa import al_dfa_symbols_to_int, dfa_symbols_to_num, p_to_al


@dataclass(frozen=True)
class BernoulliParityOracle(Oracle):
    noise_model: NoiseModel
    seed: int
    modulo: int = 2
    allowed_moduluses: Tuple[int] = (0,)
    alphabet_size: int = 2

    def membership_query(self, string: bytes) -> bool:
        correct = sum(string) % self.modulo in self.allowed_moduluses
        return self.noise_model.apply_noise(correct, string, self.seed)

    def target_dfa(self):
        """Sum mod ``modulo``: symbol ``s`` advances the running total by ``s``."""
        return DFA(
            states=set(range(self.modulo)),
            input_symbols=set(range(self.alphabet_size)),
            transitions={
                q: {s: (q + s) % self.modulo for s in range(self.alphabet_size)}
                for q in range(self.modulo)
            },
            initial_state=0,
            final_states=set(self.allowed_moduluses),
            allow_partial=False,
        )


@dataclass(frozen=True)
class BernoulliRegex(Oracle):
    noise_model: NoiseModel
    seed: int
    regex: str
    alphabet_size: int = 2

    def membership_query(self, string: bytes) -> bool:
        string_str = "".join(map(str, string))
        # print(string_str)
        correct = re.fullmatch(self.regex, string_str) is not None
        return self.noise_model.apply_noise(correct, string, self.seed)

    def target_dfa(self):
        """The regex compiled over the same int-as-str symbols it matches on.

        A language with a dead end (``1*``) compiles to a partial DFA, so
        ``to_complete`` gives the missing edges a trap state to land in.
        """
        symbols = {str(i) for i in range(self.alphabet_size)}
        nfa = NFA.from_regex(self.regex, input_symbols=symbols)
        return al_dfa_symbols_to_int(DFA.from_nfa(nfa, minify=True).to_complete())


@dataclass(frozen=True)
class AllFramesClosedOracle(Oracle):
    noise_model: NoiseModel
    seed: int
    stops: Tuple[int] = ("TAG", "TGA", "TAA")
    alphabet_size: int = 4

    def membership_query(self, string: bytes) -> bool:
        string_str = "".join("ACGT"[i] for i in string)
        correct = all(self.phase_closed(string_str, phase) for phase in range(3))
        return self.noise_model.apply_noise(correct, string, self.seed)

    def target_dfa(self):
        """The hand-built stop-codon DFA, which is this language already."""
        # Imported here: manual_dfa pulls in torch, which the oracle itself does
        # not need.
        from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_dfa

        return p_to_al(dfa_symbols_to_num(stop_codon_dfa(tuple(self.stops))))

    def phase_closed(self, string: str, phase: int) -> bool:
        string = string[phase:]
        for i in range(0, len(string), 3):
            codon = string[i : i + 3]
            if codon in self.stops:
                return True
        return False
