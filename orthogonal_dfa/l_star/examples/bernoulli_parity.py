import re
from dataclasses import dataclass
from typing import List, Tuple

from orthogonal_dfa.l_star.structures import NoiseModel, Oracle


@dataclass(frozen=True)
class BernoulliParityOracle(Oracle):
    noise_model: NoiseModel
    seed: int
    modulo: int = 2
    allowed_moduluses: Tuple[int] = (0,)

    def membership_query(self, string: List[int]) -> bool:
        correct = sum(string) % self.modulo in self.allowed_moduluses
        return self.noise_model.apply_noise(correct, string, self.seed)


@dataclass(frozen=True)
class BernoulliRegex(Oracle):
    noise_model: NoiseModel
    seed: int
    regex: str

    def membership_query(self, string: List[int]) -> bool:
        string_str = "".join(map(str, string))
        # print(string_str)
        correct = re.fullmatch(self.regex, string_str) is not None
        return self.noise_model.apply_noise(correct, string, self.seed)


@dataclass(frozen=True)
class AllFramesClosedOracle(Oracle):
    noise_model: NoiseModel
    seed: int
    stops: Tuple[int] = ("TAG", "TGA", "TAA")

    def membership_query(self, string: List[int]) -> bool:
        string_str = "".join("ACGT"[i] for i in string)
        correct = all(self.phase_closed(string_str, phase) for phase in range(3))
        return self.noise_model.apply_noise(correct, string, self.seed)

    def phase_closed(self, string: str, phase: int) -> bool:
        string = string[phase:]
        for i in range(0, len(string), 3):
            codon = string[i : i + 3]
            if codon in self.stops:
                return True
        return False
