from orthogonal_dfa.l_star.sampler import Sampler
def _has(word, sub):
    n=len(sub)
    return any(list(word[i:i+n])==list(sub) for i in range(len(word)-n+1))
class TailSampler(Sampler):
    def __init__(self, tails, motifs, length, share):
        self.tails, self.motifs, self.length, self.share = tails, motifs, length, share
    def sample(self, rng, alphabet_size):
        def draw(n): return rng.integers(0, alphabet_size, size=n).tolist()
        if rng.random() >= self.share: return draw(self.length)
        tail=list(self.tails[int(rng.integers(len(self.tails)))])
        for _ in range(50):
            cand=draw(self.length-len(tail))+tail
            if not any(_has(cand,m) for m in self.motifs): return cand
        return draw(self.length)
