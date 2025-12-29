from dataclasses import dataclass
import re
from typing import List, Tuple

from permacache import stable_hash

from orthogonal_dfa.l_star.structures import Oracle


def uniform_random(seed: object) -> float:
    hash_value = stable_hash(seed)
    hash_value = (int(hash_value, 16) % 100) / 100
    return hash_value


@dataclass(frozen=True)
class BernoulliParityOracle(Oracle):
    p_correct: float
    seed: int
    modulo: int = 2
    allowed_moduluses: Tuple[int] = (0,)

    def membership_query(self, string: List[int]) -> bool:
        correct = sum(string) % self.modulo in self.allowed_moduluses
        hash_input = uniform_random((string, self.seed))
        if hash_input < self.p_correct:
            return correct
        else:
            return not correct


@dataclass(frozen=True)
class BernoulliRegex(Oracle):
    p_correct: float
    seed: int
    regex: str

    def membership_query(self, string: List[int]) -> bool:
        string_str = "".join(map(str, string))
        # print(string_str)
        correct = re.fullmatch(self.regex, string_str) is not None
        hash_input = uniform_random((string, self.seed))
        if hash_input < self.p_correct:
            return correct
        else:
            return not correct
