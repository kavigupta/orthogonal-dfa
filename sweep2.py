import contextlib, io, json, sys
import numpy as np
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle, BernoulliRegex
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle, Oracle
from orthogonal_dfa.l_star.learn import learn_dfa

SPEC = {
 "modulo":      (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.3, None),
 "subseq":      (lambda: BernoulliRegex(regex=r".*1010101.*"), 0.3, None),
 "two_subseq":  (lambda: BernoulliRegex(regex=r".*1111.*1111.*"), 0.3, None),
 "modulo_hard": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.2, None),
 "modulo_asym": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.15,
                 dict(p_0=0.10, p_1=0.40)),
}
class Counting(Oracle):
    def __init__(self, inner): self.inner, self.n = inner, 0
    @property
    def alphabet_size(self): return self.inner.alphabet_size
    def membership_query(self, w):
        self.n += 1; return self.inner.membership_query(w)
    def membership_queries(self, ws):
        self.n += len(ws); return self.inner.membership_queries(ws)

task, seed = sys.argv[1], int(sys.argv[2])
mk, sig, noise = SPEC[task]
inner = mk(); box = {}
def creator(nm, s):
    box["o"] = Counting(NoisyOracle(inner, nm, s)); return box["o"]
kw = dict(noise_model=AsymmetricBernoulli(**noise)) if noise else {}
with contextlib.redirect_stdout(io.StringIO()):
    dfa = learn_dfa(creator, min_signal_strength=sig, seed=seed, **kw)
print(json.dumps(dict(task=task, seed=seed, queries=box["o"].n, states=len(dfa.states))))
