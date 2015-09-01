from blocks.initialization import *

# L1-normalize along an axis (default: normalize columns, which for
# Linear bricks ensures each input is scaled by at most 1)
class NormalizedInitialization(NdarrayInitialization):
    def __init__(self, initialization, axis=0, **kwargs):
        self.initialization = initialization
        self.axis = axis

    def generate(self, rng, shape):
        x = self.initialization.generate(rng, shape)
        x /= abs(x).sum(axis=self.axis, keepdims=True)
        return x
