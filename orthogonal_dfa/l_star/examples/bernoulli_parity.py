from dataclasses import dataclass
from typing import List

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

    def membership_query(self, string: List[int]) -> bool:
        correct = sum(string) % self.modulo == 0
        hash_input = uniform_random((string, self.seed))
        if hash_input < self.p_correct:
            return correct
        else:
            return not correct
